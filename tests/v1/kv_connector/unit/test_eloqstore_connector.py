# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from .utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
ELOQSTORE_PYTHON_SRC = WORKSPACE_ROOT / "eloqstore" / "python" / "src"
ELOQSTORE_CAPI_SO = WORKSPACE_ROOT / "eloqstore" / "build" / "libeloqstore_capi.so"


def _enable_eloqstore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ELOQSTORE_PYTHON_SRC))
    monkeypatch.setenv("ELOQSTORE_PY_LIB", str(ELOQSTORE_CAPI_SO))


def _make_config(tmpdir: str, **extra):
    return create_vllm_config(
        block_size=4,
        max_model_len=64,
        kv_connector="EloqStoreConnector",
        kv_connector_extra_config={
            "store_paths": [tmpdir],
            "num_threads": 2,
            "num_partitions": 4,
            # Keep the test pool below the default 64 KiB RLIMIT_MEMLOCK used
            # by many CI/dev shells so manager-side fixed-buffer registration
            # does not abort the process.
            "shared_memory_bytes": 32 << 10,
            "shared_memory_slot_size": 4 << 10,
            "shared_memory_slot_count": 8,
            **extra,
        },
    )


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_block_round_trip(monkeypatch: pytest.MonkeyPatch):
    _enable_eloqstore(monkeypatch)

    from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
        EloqStoreConnectorMetadata,
    )

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = _make_config(tmpdir)
        kv_cache_config = make_kv_cache_config(block_size=4, num_blocks=16)

        scheduler_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            kv_cache_config,
        )
        worker_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        try:
            assert worker_connector._table_name.startswith("vllm_kv__")
            assert worker_connector._runtime_options.shared_memory_name.startswith("/eloqstore-")
            assert worker_connector._runtime_options.ipc_path.startswith("ipc:///tmp/eloqstore-")

            key = worker_connector._block_key(torch.tensor([1, 2, 3, 4]), [], 4)
            assert key.startswith("kv:v1:")
            assert 0 <= worker_connector._partition_id_for_key(key) < 4

            request = create_request(request_id=7, num_tokens=10, block_size=4)
            source_kv = torch.arange(2 * 2 * 4 * 3, dtype=torch.float16).reshape(
                2, 2, 4, 3
            )
            source_kv_layer2 = torch.arange(
                100, 100 + 2 * 2 * 4 * 3, dtype=torch.float16
            ).reshape(2, 2, 4, 3)

            save_meta = EloqStoreConnectorMetadata()
            save_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0, 1],
                block_size=4,
                is_store=True,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(save_meta)
            worker_connector.register_kv_caches(
                {"layer0": source_kv, "layer2": source_kv_layer2}
            )
            worker_connector.save_kv_layer("layer0", source_kv, object())
            worker_connector.save_kv_layer("layer2", source_kv_layer2, object())
            worker_connector.wait_for_save()
            worker_connector.clear_connector_metadata()

            matched_after, is_async_after = scheduler_connector.get_num_new_matched_tokens(
                request, 0
            )
            assert matched_after == 8
            assert is_async_after is False

            target_kv = torch.zeros_like(source_kv)
            target_kv_layer2 = torch.zeros_like(source_kv_layer2)
            worker_connector.register_kv_caches(
                {"layer0": target_kv, "layer2": target_kv_layer2}
            )
            load_meta = EloqStoreConnectorMetadata()
            load_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0, 1],
                block_size=4,
                is_store=False,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(load_meta)
            forward_context = SimpleNamespace(
                no_compile_layers={
                    "layer0": SimpleNamespace(kv_cache=target_kv),
                    "layer2": SimpleNamespace(kv_cache=target_kv_layer2),
                },
                attn_metadata={"layer0": object(), "layer2": object()},
            )
            worker_connector.start_load_kv(forward_context)
            worker_connector.wait_for_layer_load("layer0")
            worker_connector.wait_for_layer_load("layer2")
            worker_connector.clear_connector_metadata()

            metrics = worker_connector.get_metrics()
            assert metrics["store_blocks"] == 2
            assert metrics["load_blocks"] == 2
            assert torch.equal(target_kv[:, :2, :, :], source_kv[:, :2, :, :])
            assert torch.equal(
                target_kv_layer2[:, :2, :, :], source_kv_layer2[:, :2, :, :]
            )
        finally:
            worker_connector.shutdown()
            scheduler_connector.shutdown()


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_cross_layer_block_round_trip(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_eloqstore(monkeypatch)

    from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
        EloqStoreConnectorMetadata,
    )

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = _make_config(tmpdir)
        kv_cache_config = make_kv_cache_config(block_size=4, num_blocks=16)

        scheduler_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            kv_cache_config,
        )
        worker_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        try:
            class _IdentityCrossLayerBackend:
                @staticmethod
                def get_kv_cache_stride_order(
                    include_num_layers_dimension: bool = False,
                ) -> tuple[int, ...]:
                    return tuple(range(5 if include_num_layers_dimension else 4))

            request = create_request(request_id=17, num_tokens=10, block_size=4)
            source_cross = torch.arange(2 * 2 * 16 * 4 * 3, dtype=torch.float16).reshape(
                2, 2, 16, 4, 3
            )

            save_meta = EloqStoreConnectorMetadata()
            save_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0, 1],
                block_size=4,
                is_store=True,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(save_meta)
            worker_connector.register_cross_layers_kv_cache(
                source_cross, _IdentityCrossLayerBackend
            )
            for layer_name in worker_connector._layer_order:
                worker_connector.save_kv_layer(
                    layer_name, worker_connector._registered_kv_caches[layer_name], object()
                )
            worker_connector.wait_for_save()
            worker_connector.clear_connector_metadata()

            target_cross = torch.zeros_like(source_cross)
            worker_connector.register_cross_layers_kv_cache(
                target_cross, _IdentityCrossLayerBackend
            )
            load_meta = EloqStoreConnectorMetadata()
            load_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0, 1],
                block_size=4,
                is_store=False,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(load_meta)
            forward_context = SimpleNamespace(no_compile_layers={}, attn_metadata={})
            worker_connector.start_load_kv(forward_context)
            for layer_name in worker_connector._layer_order:
                worker_connector.wait_for_layer_load(layer_name)
            worker_connector.clear_connector_metadata()

            assert torch.equal(target_cross[:, :, :2, :, :], source_cross[:, :, :2, :, :])
        finally:
            worker_connector.shutdown()
            scheduler_connector.shutdown()


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_prefill_emits_block_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_eloqstore(monkeypatch)

    from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
        EloqStoreConnectorMetadata,
    )

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = _make_config(tmpdir)
        scheduler = create_scheduler(vllm_config)
        connector = scheduler.connector
        assert connector is not None

        request = create_request(request_id=11, num_tokens=20, block_size=4)
        scheduler.add_request(request)

        block_ranges: list[tuple[int, int]] = []
        for _ in range(3):
            scheduler_output = scheduler.schedule()
            metadata = scheduler_output.kv_connector_metadata
            assert isinstance(metadata, EloqStoreConnectorMetadata)
            store_requests = [req for req in metadata.requests if req.is_store]
            assert len(store_requests) == 1
            store_request = store_requests[0]
            block_ranges.extend(
                (block_end, int(slot_mapping.numel()))
                for _, block_end, _, slot_mapping in connector._iter_request_blocks(
                    store_request
                )
            )

        assert block_ranges == [(4, 4), (8, 4), (12, 4), (16, 4), (20, 4)]


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_records_match_query_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_eloqstore(monkeypatch)

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = _make_config(tmpdir)
        kv_cache_config = make_kv_cache_config(block_size=4, num_blocks=32)
        connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            kv_cache_config,
        )
        worker = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        try:
            token_ids = list(range(20))
            request = create_request(request_id=51, num_tokens=20, block_size=4)
            request.prompt_token_ids = token_ids

            worker.register_kv_caches(
                {"layer0": torch.arange(2 * 1 * 4 * 2, dtype=torch.float16).reshape(2, 1, 4, 2)}
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
                EloqStoreConnectorMetadata,
            )

            save_meta = EloqStoreConnectorMetadata()
            save_meta.add_request(
                token_ids=token_ids,
                block_ids=[0],
                block_size=4,
                is_store=True,
                mm_hashes=[],
            )
            worker.bind_connector_metadata(save_meta)
            worker.save_kv_layer("layer0", worker._registered_kv_caches["layer0"], object())
            worker.wait_for_save()
            worker.clear_connector_metadata()

            matched = connector._get_num_matched_tokens_for_prompt(token_ids, [])
            assert matched == 4

            stats = connector.get_kv_connector_stats()
            assert stats is not None
            reduced = stats.reduce()
            assert reduced["match_queries"] == 1
            assert reduced["match_query_tokens"] == 20
            assert reduced["match_aligned_tokens"] == 16
            assert reduced["match_hit_tokens"] == 4
            assert reduced["match_hit_blocks"] == 1
            assert reduced["match_miss_blocks"] == 1
            assert reduced["match_reserved_tail_tokens"] == 1
            assert reduced["match_unaligned_tail_tokens"] == 3
        finally:
            worker.shutdown()
            connector.shutdown()


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_not_found_marks_invalid_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_eloqstore(monkeypatch)

    from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
        EloqStoreConnectorMetadata,
    )

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = _make_config(tmpdir, num_threads=1, num_partitions=1)
        kv_cache_config = make_kv_cache_config(block_size=4, num_blocks=16)
        worker_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        try:
            request = create_request(request_id=61, num_tokens=10, block_size=4)
            load_meta = EloqStoreConnectorMetadata()
            load_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[3, 7],
                block_size=4,
                is_store=False,
                mm_hashes=[],
            )
            worker_connector.register_kv_caches(
                {"layer0": torch.full((2, 8, 4, 3), 7, dtype=torch.float16)}
            )
            worker_connector.bind_connector_metadata(load_meta)
            forward_context = SimpleNamespace(
                no_compile_layers={
                    "layer0": SimpleNamespace(kv_cache=worker_connector._registered_kv_caches["layer0"])
                },
                attn_metadata={"layer0": object()},
            )
            worker_connector.start_load_kv(forward_context)
            worker_connector.wait_for_layer_load("layer0")

            assert worker_connector.get_block_ids_with_load_errors() == {3, 7}
            assert worker_connector.get_block_ids_with_load_errors() == set()
        finally:
            worker_connector.shutdown()
