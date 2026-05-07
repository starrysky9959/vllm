# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

try:
    from eloqstore import Client as EloqStoreClient
    from eloqstore import Options as EloqStoreOptions
except ImportError:  # pragma: no cover - surfaced at runtime when connector is used.
    EloqStoreClient = None
    EloqStoreOptions = None

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Token ids are kept only for key derivation. The actual data movement uses
    # slot_mapping to locate this request inside vLLM's paged KV cache.
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        # vLLM only reasons about external KV in block-aligned units. Truncate
        # the prompt to the aligned prefix and build the corresponding slot map.
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int64)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size, dtype=torch.int64)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class EloqStoreConnectorMetadata(KVConnectorMetadata):
    # Scheduler -> worker payload for a single engine step.
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


class EloqStoreConnector(KVConnectorBase_V1):
    """Direct vLLM KV connector backed by EloqStore.

    Design points:
    - Scheduler side decides whether a request should load or store.
    - Worker side extracts/injects per-layer KV tensors.
    - Persisted keys are derived from the aligned prompt prefix.
    - Data path targets one GPU<->CPU copy plus one CPU<->storage copy.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        if EloqStoreClient is None or EloqStoreOptions is None:
            raise ImportError(
                "eloqstore Python SDK is required to use EloqStoreConnector"
            )

        self._block_size = vllm_config.cache_config.block_size
        # Requests recorded here already have blocks allocated by the scheduler
        # and should be loaded on the next worker forward pass.
        self._requests_need_load: dict[str, Request] = {}
        # "ready" keys are written after all layer payloads for a prefix are durable.
        self._pending_ready_keys: set[str] = set()
        self._client = EloqStoreClient(self._build_client_options())
        self._executor = ThreadPoolExecutor(
            max_workers=max(
                1, int(self._kv_transfer_config.get_from_extra_config("io_workers", 2))
            ),
            thread_name_prefix="eloqstore-kv",
        )
        # One future per layer during load. Each future fills pinned CPU staging
        # buffers from EloqStore; GPU injection happens later in wait_for_layer_load.
        self._load_futures: dict[str, Future[list[tuple[ReqMeta, torch.Tensor]]]] = {}
        self._layer_attn_metadata: dict[str, AttentionMetadata] = {}
        self._layer_kv_cache_refs: dict[str, torch.Tensor] = {}
        # Save futures represent outstanding batch_put submissions.
        self._save_futures: list[Future[None]] = []

    def shutdown(self) -> None:
        self.wait_for_save()
        self._executor.shutdown(wait=True)
        self._client.close()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, EloqStoreConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("EloqStoreConnector.start_load_kv called without metadata")
            return

        self._load_futures.clear()
        self._layer_attn_metadata.clear()
        self._layer_kv_cache_refs.clear()
        requests_to_load = [request for request in metadata.requests if not request.is_store]
        if not requests_to_load:
            return

        for layer_name, layer in forward_context.no_compile_layers.items():
            kv_cache_layer = getattr(layer, "kv_cache", None)
            if kv_cache_layer is None:
                continue
            layer_attn_metadata = (
                attn_metadata[layer_name]
                if isinstance(attn_metadata, dict)
                else attn_metadata
            )
            self._layer_attn_metadata[layer_name] = layer_attn_metadata
            self._layer_kv_cache_refs[layer_name] = kv_cache_layer
            # Stage load into pinned CPU memory in the background so attention
            # can block only when it actually reaches this layer.
            self._load_futures[layer_name] = self._executor.submit(
                self._load_layer_staging,
                layer_name,
                kv_cache_layer.dtype,
                kv_cache_layer.element_size(),
                requests_to_load,
                layer_attn_metadata,
                kv_cache_layer.shape,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        future = self._load_futures.pop(layer_name, None)
        if future is None:
            return
        kv_cache_layer = self._layer_kv_cache_refs.pop(layer_name)
        layer_attn_metadata = self._layer_attn_metadata.pop(layer_name)
        for request, stage in future.result():
            # Reinterpret the pinned byte staging buffer as the original KV tensor,
            # then perform the single host->device copy before writing into slots.
            src_kv_cache_cpu = torch.frombuffer(
                stage.numpy(),
                dtype=kv_cache_layer.dtype,
                count=stage.numel() // kv_cache_layer.element_size(),
            ).reshape(
                self._loaded_kv_shape(
                    kv_cache_layer, request.slot_mapping, layer_attn_metadata
                )
            )
            src_kv_cache = src_kv_cache_cpu.to(
                device=kv_cache_layer.device, non_blocking=False
            )
            self._inject_kv_into_layer(
                kv_cache_layer,
                src_kv_cache,
                request.slot_mapping,
                layer_attn_metadata,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, EloqStoreConnectorMetadata)

        items: list[tuple[str, Any]] = []
        for request in metadata.requests:
            if not request.is_store or len(request.slot_mapping) == 0:
                continue

            # Extract only this request's aligned prefix from the layer KV cache.
            kv_cache = self._extract_kv_from_layer(
                kv_layer, request.slot_mapping, attn_metadata
            )
            # Save path target:
            #   GPU KV tensor -> pinned CPU staging -> EloqStore batch_put
            cpu_stage = torch.empty_like(kv_cache, device="cpu", pin_memory=True)
            cpu_stage.copy_(kv_cache, non_blocking=False)
            items.append(
                (
                    self._data_key(layer_name, request.token_ids, request.mm_hashes),
                    cpu_stage.view(torch.uint8).reshape(-1).numpy(),
                )
            )
            self._pending_ready_keys.add(
                self._ready_key(request.token_ids, request.mm_hashes)
            )

        if items:
            # Each layer is flushed as one batch_put. This keeps the EloqStore
            # API usage batch-oriented even though vLLM invokes us layer by layer.
            self._save_futures.append(
                self._executor.submit(self._client.batch_put, items)
            )

    def wait_for_save(self):
        for future in self._save_futures:
            future.result()
        self._save_futures.clear()
        if self._pending_ready_keys:
            # A prefix becomes externally visible only after all of its layer
            # payloads are committed. The ready key is the scheduler-side probe.
            ready_items = [(key, b"1") for key in self._pending_ready_keys]
            self._client.batch_put(ready_items)
            self._pending_ready_keys.clear()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._found_match_for_request(request):
            return 0, False

        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)
        return max(num_tokens_to_check - num_computed_tokens, 0), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        # This mirrors the ExampleConnector policy:
        # - requests with an external hit become load requests
        # - requests without a hit become store candidates
        meta = EloqStoreConnectorMetadata()
        total_need_load = 0

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            elif not self._found_match_for_prompt(token_ids, mm_hashes):
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    mm_hashes=mm_hashes,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue

            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]
            assert new_block_ids is not None

            meta.add_request(
                token_ids=token_ids,
                block_ids=new_block_ids[0],
                block_size=self._block_size,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def _build_client_options(self) -> Any:
        # Keep the connector configurable entirely through kv_connector_extra_config
        # so it can be instantiated by vLLM workers without extra wiring.
        extra = self._kv_transfer_config.kv_connector_extra_config
        store_paths = extra.get("store_paths")
        if store_paths is None:
            shared_path = extra.get("shared_storage_path")
            store_paths = [shared_path] if shared_path else []
        elif isinstance(store_paths, str):
            store_paths = [store_paths]

        option_fields = {
            "store_paths": list(store_paths or []),
            "options_path": extra.get("options_path"),
            "table_name": extra.get("table_name", "vllm_kv"),
            "partition_id": extra.get("partition_id", 0),
            "branch": extra.get("branch", "main"),
            "term": extra.get("term", 0),
            "partition_group_id": extra.get("partition_group_id", 0),
            "validate": extra.get("validate", True),
            "num_threads": extra.get("num_threads"),
            "data_page_size": extra.get("data_page_size"),
            "pages_per_file_shift": extra.get("pages_per_file_shift"),
            "data_append_mode": extra.get("data_append_mode"),
            "overflow_pointers": extra.get("overflow_pointers"),
            "enable_compression": extra.get("enable_compression"),
            "buffer_pool_size": extra.get("buffer_pool_size"),
            "manifest_limit": extra.get("manifest_limit"),
            "fd_limit": extra.get("fd_limit"),
        }
        return EloqStoreOptions(**option_fields)

    def _load_layer_staging(
        self,
        layer_name: str,
        dtype: torch.dtype,
        element_size: int,
        requests: list[ReqMeta],
        layer_attn_metadata: AttentionMetadata,
        kv_cache_shape: torch.Size,
    ) -> list[tuple[ReqMeta, torch.Tensor]]:
        loaded: list[tuple[ReqMeta, torch.Tensor]] = []
        shape_probe = torch.empty(kv_cache_shape, dtype=dtype, device="cpu")
        for request in requests:
            # Load each request into a dedicated pinned byte buffer sized for
            # this layer's aligned prefix. The GPU copy is deferred until
            # wait_for_layer_load.
            expected_shape = self._loaded_kv_shape(
                shape_probe, request.slot_mapping, layer_attn_metadata
            )
            numel = _numel(expected_shape)
            num_bytes = numel * element_size
            stage = torch.empty(
                num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            stage_np = stage.numpy()
            key = self._data_key(layer_name, request.token_ids, request.mm_hashes)
            written = self._client.get_into(key, stage_np)
            if written is None:
                raise KeyError(f"missing KV entry for key {key}")
            if written != num_bytes:
                raise ValueError(
                    f"KV entry size mismatch for {key}: expected {num_bytes}, got {written}"
                )
            loaded.append((request, stage))
        return loaded

    def _found_match_for_request(self, request: "Request") -> bool:
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        # Match granularity is block-aligned. Prefixes shorter than one aligned
        # block are treated as misses because vLLM will not allocate external
        # KV for them.
        num_tokens_to_check = align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        if num_tokens_to_check == 0:
            return False
        token_ids = torch.tensor(
            prompt_token_ids[:num_tokens_to_check], dtype=torch.int64
        )
        return self._client.exists(self._ready_key(token_ids, mm_hashes))

    def _prompt_hash(self, token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
        # The aligned prompt prefix is the stable logical identity of a stored KV.
        token_bytes = token_ids.cpu().numpy().tobytes()
        if mm_hashes:
            token_bytes += "-".join(mm_hashes).encode("utf-8")
        return safe_hash(token_bytes, usedforsecurity=False).hexdigest()

    def _ready_key(self, token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
        # Sentinel key used only for scheduler-side existence checks.
        return f"ready:{self._prompt_hash(token_ids, mm_hashes)}:{len(token_ids)}"

    def _data_key(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        # Actual layer payload key.
        return (
            f"kv:{self._prompt_hash(token_ids, mm_hashes)}:"
            f"{len(token_ids)}:{layer_name}"
        )

    def _extract_kv_from_layer(
        self,
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Normalize different attention backends into the same logical view:
        # "the KV slice for this request's aligned prefix".
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        if isinstance(attn_metadata, TritonAttentionMetadata):
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return layer[block_idxs, :, offsets]
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: torch.Tensor,
        src_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        # Inverse of _extract_kv_from_layer: write the loaded request KV back
        # into the paged cache slots that vLLM allocated for this request.
        dst_shape = dst_kv_cache_layer.shape
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = dst_shape[0], dst_shape[1]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            return
        if isinstance(attn_metadata, TritonAttentionMetadata):
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
            return
        num_pages, page_size = dst_shape[1], dst_shape[2]
        dst_kv_cache_layer = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache

    def _loaded_kv_shape(
        self,
        kv_cache_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> tuple[int, ...]:
        # Compute the logical per-request tensor shape that corresponds to
        # slot_mapping, independent of the backend-specific page layout.
        num_tokens = int(slot_mapping.numel())
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_cache_layer.shape[0], kv_cache_layer.shape[1]
            flat_dim = kv_cache_layer.reshape(num_pages * page_size, -1).shape[1]
            return (num_tokens, flat_dim)
        if isinstance(attn_metadata, TritonAttentionMetadata):
            return (
                num_tokens,
                kv_cache_layer.shape[1],
                kv_cache_layer.shape[3],
                kv_cache_layer.shape[4],
            )
        num_pages, page_size = kv_cache_layer.shape[1], kv_cache_layer.shape[2]
        flat_dim = kv_cache_layer.reshape(2, num_pages * page_size, -1).shape[2]
        return (2, num_tokens, flat_dim)


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return (num_tokens - 1) // block_size * block_size


def _numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel
