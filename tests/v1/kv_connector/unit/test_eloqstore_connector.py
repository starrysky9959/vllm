# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from .utils import create_request, create_vllm_config, make_kv_cache_config

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
ELOQSTORE_PYTHON_SRC = WORKSPACE_ROOT / "eloqstore" / "python" / "src"
ELOQSTORE_CAPI_SO = WORKSPACE_ROOT / "eloqstore" / "build" / "libeloqstore_capi.so"


@pytest.mark.skipif(
    not ELOQSTORE_CAPI_SO.exists(), reason="eloqstore C API library is not built"
)
def test_eloqstore_connector_save_load_smoke(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ELOQSTORE_PYTHON_SRC))
    monkeypatch.setenv("ELOQSTORE_PY_LIB", str(ELOQSTORE_CAPI_SO))

    from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
        EloqStoreConnectorMetadata,
    )

    with TemporaryDirectory(prefix="eloqstore-connector-") as tmpdir:
        vllm_config = create_vllm_config(
            block_size=4,
            max_model_len=32,
            kv_connector="EloqStoreConnector",
            kv_connector_extra_config={
                "store_paths": [tmpdir],
                "table_name": "test_kv",
                "partition_id": 0,
                "num_threads": 1,
                "io_workers": 2,
            },
        )
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
            request = create_request(request_id=7, num_tokens=6, block_size=4)

            matched_before, is_async = scheduler_connector.get_num_new_matched_tokens(
                request, 0
            )
            assert matched_before == 0
            assert is_async is False

            source_kv = torch.arange(2 * 2 * 4 * 3, dtype=torch.float16).reshape(
                2, 2, 4, 3
            )
            save_meta = EloqStoreConnectorMetadata()
            save_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0],
                block_size=4,
                is_store=True,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(save_meta)
            worker_connector.save_kv_layer("layer0", source_kv, object())
            worker_connector.wait_for_save()
            worker_connector.clear_connector_metadata()

            matched_after, is_async_after = (
                scheduler_connector.get_num_new_matched_tokens(request, 0)
            )
            assert matched_after == 4
            assert is_async_after is False

            load_meta = EloqStoreConnectorMetadata()
            load_meta.add_request(
                token_ids=list(request.prompt_token_ids or []),
                block_ids=[0],
                block_size=4,
                is_store=False,
                mm_hashes=[],
            )
            worker_connector.bind_connector_metadata(load_meta)
            target_kv = torch.zeros_like(source_kv)
            forward_context = SimpleNamespace(
                no_compile_layers={"layer0": SimpleNamespace(kv_cache=target_kv)},
                attn_metadata={"layer0": object()},
            )
            worker_connector.start_load_kv(forward_context)
            worker_connector.wait_for_layer_load("layer0")
            worker_connector.clear_connector_metadata()

            assert torch.equal(target_kv[:, :1, :, :], source_kv[:, :1, :, :])
            assert torch.count_nonzero(target_kv[:, 1:, :, :]) == 0
        finally:
            worker_connector.shutdown()
            scheduler_connector.shutdown()
