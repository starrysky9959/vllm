# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any
import re
import struct

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
    from eloqstore import RegisteredMemory as EloqStoreRegisteredMemory
except ImportError:  # pragma: no cover - surfaced at runtime when connector is used.
    EloqStoreClient = None
    EloqStoreOptions = None
    EloqStoreRegisteredMemory = None

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
_SCHEMA_VERSION = "v1"
_PAYLOAD_MAGIC = b"EKV1"
_PAYLOAD_HEADER = struct.Struct("<4sHQHH6x")
_PAYLOAD_HEADER_LEN = _PAYLOAD_HEADER.size
_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
_LAYOUT_CODES = {
    "generic": 0,
    "triton": 1,
    "mla": 2,
    "default_2plane": 3,
}


@dataclass
class ReqMeta:
    """Worker-side description of one request for the current engine step.

    token_ids:
        Block-aligned prompt prefix used only to derive stable EloqStore keys.
    slot_mapping:
        Physical slot positions inside vLLM's paged KV cache for this request.
    is_store:
        False means "load external KV into vLLM"; True means "persist vLLM KV".
    mm_hashes:
        Stable multimodal identities for attached features. These hashes are
        folded into the prompt key so two prompts with the same text tokens but
        different multimodal inputs do not collide.
    """

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
        chunk_tokens: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        """Build per-request worker metadata from scheduler block assignment."""
        # The worker can only touch tokens for which the scheduler already
        # allocated destination blocks. Limit the visible prefix by both token
        # count and allocated slot capacity, then round down to a full chunk.
        max_tokens_from_blocks = len(block_ids) * block_size
        valid_num_tokens = align_to_chunk_size(
            min(len(token_ids), max_tokens_from_blocks), chunk_tokens
        )
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
        chunk_tokens: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        """Append one request entry to the metadata payload."""
        self.requests.append(
            ReqMeta.make_meta(
                token_ids,
                block_ids,
                block_size,
                chunk_tokens,
                is_store,
                mm_hashes,
            )
        )


class EloqStoreConnector(KVConnectorBase_V1):
    """Direct vLLM KV connector backed by EloqStore.

    Design points:
    - Scheduler side decides whether a request should load or store.
    - Worker side extracts/injects per-layer KV tensors.
    - Persisted values are fixed-size chunks, not whole variable-length prefixes.
    - Persisted keys are derived from cumulative chunk prefixes.
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
        if (
            EloqStoreClient is None
            or EloqStoreOptions is None
            or EloqStoreRegisteredMemory is None
        ):
            raise ImportError(
                "eloqstore Python SDK is required to use EloqStoreConnector"
            )

        self._block_size = vllm_config.cache_config.block_size
        self._chunk_blocks = int(
            self._kv_transfer_config.get_from_extra_config("chunk_blocks", 1)
        )
        if self._chunk_blocks <= 0:
            raise ValueError("chunk_blocks must be positive")
        self._chunk_tokens = self._block_size * self._chunk_blocks
        self._model_name = self._resolve_model_name()
        self._table_name = self._resolve_table_name()
        self._kv_rank = self._resolve_kv_rank()
        self._layout_id = self._resolve_layout_id()
        self._layout_code = _layout_code(self._layout_id)
        self._key_dtype = self._resolve_key_dtype()
        # Requests recorded here already have blocks allocated by the scheduler
        # and should be loaded on the next worker forward pass.
        self._requests_need_load: dict[str, Request] = {}
        # "ready" keys are written after all layer payloads for a prefix are durable.
        self._pending_ready_keys: set[str] = set()
        # Cache of chunk-ready sentinels already observed or published by this
        # connector. This lets save skip redundant rewrites and reduces exists()
        # calls during scheduler match probing.
        self._known_ready_keys: set[str] = set()
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._registered_memory = EloqStoreRegisteredMemory(
            total_size=int(extra.get("registered_memory_total_size", 2 << 30)),
            chunk_size=int(extra.get("registered_memory_chunk_size", 1 << 30)),
            segment_size=int(extra.get("segment_size", 512 << 10)),
        )
        self._client = EloqStoreClient(self._build_client_options())
        # Reusable staging is now EloqStore's registered memory pool. The pool
        # is registered with io_uring and shared with Python tensor views.
        self._stage_pool_lock = Lock()
        # One native async read handle list per layer during load. GPU injection
        # happens later in wait_for_layer_load.
        self._load_handles: dict[str, list[tuple[torch.Tensor, Any, int, str]]] = {}
        self._layer_attn_metadata: dict[str, AttentionMetadata] = {}
        self._layer_kv_cache_refs: dict[str, torch.Tensor] = {}
        # Save futures represent outstanding batch_put submissions.
        self._save_handles: list[Any] = []

    def shutdown(self) -> None:
        """Drain async writes, stop background workers, and close the client."""
        self.wait_for_save()
        self._close_load_handles()
        self._client.close()
        self._registered_memory.close()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start the worker-side load phase for the current forward step.

        This does not write anything back to GPU yet. It only submits native
        EloqStore async reads into registered memory. Actual CPU->GPU copies are deferred to
        wait_for_layer_load(layer_name) so I/O can overlap with model
        execution until the layer is actually needed.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, EloqStoreConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("EloqStoreConnector.start_load_kv called without metadata")
            return

        self._close_load_handles()
        self._layer_attn_metadata.clear()
        self._layer_kv_cache_refs.clear()
        requests_to_load = [
            request
            for request in metadata.requests
            if not request.is_store and len(request.slot_mapping) > 0
        ]
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
            handles = self._submit_layer_loads(
                layer_name,
                kv_cache_layer.dtype,
                kv_cache_layer.element_size(),
                requests_to_load,
                layer_attn_metadata,
                kv_cache_layer.shape,
            )
            if handles:
                self._load_handles[layer_name] = handles

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Finish one layer load by injecting staged CPU data into GPU KV cache."""
        handles = self._load_handles.pop(layer_name, None)
        if handles is None:
            return
        kv_cache_layer = self._layer_kv_cache_refs.pop(layer_name)
        layer_attn_metadata = self._layer_attn_metadata.pop(layer_name)
        next_handle_index = 0
        try:
            for index, (slot_mapping, handle, payload_bytes, key) in enumerate(handles):
                next_handle_index = index + 1
                large = None
                try:
                    handle.wait()
                    large = handle.result_large()
                    if large is None:
                        raise KeyError(f"missing KV entry for key {key}")
                    if len(large) != _PAYLOAD_HEADER_LEN + payload_bytes:
                        raise ValueError(
                            f"KV entry size mismatch for {key}: "
                            f"expected {_PAYLOAD_HEADER_LEN + payload_bytes}, got {len(large)}"
                        )
                    _validate_payload_header_buffer(
                        large,
                        expected_payload_len=payload_bytes,
                        expected_dtype=kv_cache_layer.dtype,
                        expected_layout_code=self._layout_code,
                        key=key,
                    )
                    shape = self._loaded_kv_shape(
                        kv_cache_layer, slot_mapping, layer_attn_metadata
                    )
                    src_kv_cache = self._large_buffer_to_gpu_tensor(
                        large,
                        _PAYLOAD_HEADER_LEN,
                        shape,
                        kv_cache_layer.dtype,
                        kv_cache_layer.device,
                    )
                    self._inject_kv_into_layer(
                        kv_cache_layer,
                        src_kv_cache,
                        slot_mapping,
                        layer_attn_metadata,
                    )
                finally:
                    handle.close()
                    if large is not None:
                        large.close()
        except Exception:
            for _, handle, _, _ in handles[next_handle_index:]:
                handle.close()
            raise

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Queue persistence of one layer's request-local KV slices to EloqStore.

        vLLM calls this layer by layer. For each request marked as store, we
        cut out only that request's block-aligned prefix from the current layer,
        copy it once to pinned CPU memory, and submit a batch_put in the
        background.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, EloqStoreConnectorMetadata)

        items: list[tuple[str, Any]] = []
        large_buffers: list[Any] = []
        for request in metadata.requests:
            if not request.is_store or len(request.slot_mapping) == 0:
                continue

            for chunk_end, chunk_token_ids, chunk_slot_mapping in self._iter_request_chunks(
                request
            ):
                ready_key = self._ready_key(chunk_token_ids, request.mm_hashes, chunk_end)
                if self._chunk_is_persisted(ready_key):
                    continue

                # Extract only this request's current chunk from the layer KV cache.
                kv_cache = self._extract_kv_from_layer(
                    kv_layer, chunk_slot_mapping, attn_metadata
                )
                # Save path target:
                #   GPU KV tensor -> EloqStore registered memory -> fixed write
                payload_bytes = kv_cache.numel() * kv_cache.element_size()
                large = self._client.allocate_large_value(
                    _PAYLOAD_HEADER_LEN + payload_bytes
                )
                _pack_payload_header_buffer(
                    large,
                    payload_len=payload_bytes,
                    dtype=kv_cache.dtype,
                    layout_code=self._layout_code,
                )
                self._copy_tensor_to_large_buffer(
                    large, _PAYLOAD_HEADER_LEN, kv_cache
                )
                large_buffers.append(large)
                items.append(
                    (
                        self._data_key(
                            layer_name, chunk_token_ids, request.mm_hashes, chunk_end
                        ),
                        large,
                    )
                )
                self._pending_ready_keys.add(ready_key)

        if items:
            # Each layer is flushed as one batch_put. This keeps the EloqStore
            # API usage batch-oriented even though vLLM invokes us layer by layer.
            items.sort(key=lambda item: item[0])
            self._save_handles.append(self._client.batch_put_large_async(items))
        else:
            for large in large_buffers:
                large.close()

    def wait_for_save(self):
        """Wait for all outstanding layer writes and then publish ready keys."""
        for handle in self._save_handles:
            try:
                handle.wait()
            finally:
                handle.close()
        self._save_handles.clear()
        if self._pending_ready_keys:
            # A prefix becomes externally visible only after all of its layer
            # payloads are committed. The ready key is the scheduler-side probe.
            ready_items = sorted(
                ((key, b"1") for key in self._pending_ready_keys),
                key=lambda item: item[0],
            )
            self._client.batch_put(ready_items)
            self._known_ready_keys.update(self._pending_ready_keys)
            self._pending_ready_keys.clear()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Tell the scheduler how many prefix tokens can be reused externally.

        The scheduler asks this before allocating work for a request. We answer
        in block-aligned units only:
        - if no ready sentinel exists in EloqStore, return 0
        - otherwise report how many additional prefix tokens can be skipped
          because they are already stored externally

        num_computed_tokens is the number of prefix tokens already available
        locally in vLLM. The return value is therefore:

            aligned_external_prefix - num_computed_tokens

        clamped at zero.
        """
        matched_tokens = self._get_num_matched_tokens_for_request(request)
        return max(matched_tokens - num_computed_tokens, 0), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Remember which requests should be loaded after scheduler allocation.

        At this point the scheduler has decided that some prefix tokens can be
        restored from external storage and has already allocated destination
        blocks for them. The worker still needs the full Request object later,
        so we keep it here until build_connector_meta serializes the
        step-specific payload.
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the scheduler->worker payload for one engine step.

        The output says, request by request, whether the worker should load KV
        from EloqStore or store newly computed KV into EloqStore.
        """
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
                    chunk_tokens=self._chunk_tokens,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            elif self._get_num_matched_tokens_for_prompt(token_ids, mm_hashes) == 0:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    chunk_tokens=self._chunk_tokens,
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
                chunk_tokens=self._chunk_tokens,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def _build_client_options(self) -> Any:
        """Create EloqStore client options from kv_connector_extra_config."""
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
            "table_name": self._table_name,
            "partition_id": extra.get("partition_id", 0),
            "branch": extra.get("branch", "main"),
            "term": extra.get("term", 0),
            "partition_group_id": extra.get("partition_group_id", 0),
            "validate": extra.get("validate", True),
            "num_threads": extra.get("num_threads"),
            "data_page_size": extra.get("data_page_size"),
            "pages_per_file_shift": extra.get("pages_per_file_shift"),
            "data_append_mode": extra.get("data_append_mode", True),
            "overflow_pointers": extra.get("overflow_pointers"),
            "enable_compression": extra.get("enable_compression", False),
            "buffer_pool_size": extra.get("buffer_pool_size"),
            "manifest_limit": extra.get("manifest_limit"),
            "fd_limit": extra.get("fd_limit"),
            "segment_size": extra.get("segment_size", 512 << 10),
            "registered_memory_chunk_size": extra.get(
                "registered_memory_chunk_size", 1 << 30
            ),
            "segments_per_file_shift": extra.get("segments_per_file_shift", 7),
            "registered_memory": self._registered_memory,
        }
        return EloqStoreOptions(**option_fields)

    def _resolve_model_name(self) -> str:
        """Choose a stable model identity for storage namespacing."""
        model_config = self._vllm_config.model_config
        served = getattr(model_config, "served_model_name", None)
        if isinstance(served, str) and served:
            return served
        model = getattr(model_config, "model", None)
        if isinstance(model, str) and model:
            return model
        return "unknown_model"

    def _resolve_table_name(self) -> str:
        """Derive the EloqStore table name unless explicitly overridden."""
        explicit = self._kv_transfer_config.get_from_extra_config("table_name", None)
        if explicit:
            return str(explicit)
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", self._model_name).strip("._-")
        if not sanitized:
            sanitized = "unknown_model"
        return f"vllm_kv__{sanitized}"

    def _resolve_kv_rank(self) -> int:
        """Return the KV rank used as part of the explicit key namespace."""
        extra = self._kv_transfer_config.kv_connector_extra_config
        if "kv_rank" in extra:
            return int(extra["kv_rank"])
        parallel_config = self._vllm_config.parallel_config
        return int(getattr(parallel_config, "rank", 0) or 0)

    def _resolve_layout_id(self) -> str:
        """Return the explicit KV layout namespace for keys and payloads."""
        return str(
            self._kv_transfer_config.get_from_extra_config("layout_id", "generic")
        )

    def _resolve_key_dtype(self) -> str:
        """Return the dtype namespace used in keys before worker tensors exist."""
        cache_dtype = getattr(self._vllm_config.cache_config, "cache_dtype", None)
        dtype = str(cache_dtype)
        if dtype and dtype != "auto" and dtype != "None":
            return _normalize_dtype_name(dtype)
        model_dtype = getattr(self._vllm_config.model_config, "dtype", None)
        return _normalize_dtype_name(str(model_dtype))

    def _submit_layer_loads(
        self,
        layer_name: str,
        dtype: torch.dtype,
        element_size: int,
        requests: list[ReqMeta],
        layer_attn_metadata: AttentionMetadata,
        kv_cache_shape: torch.Size,
    ) -> list[tuple[torch.Tensor, Any, int, str]]:
        """Submit native async reads for one layer's serialized KV chunks."""
        handles: list[tuple[torch.Tensor, Any, int, str]] = []
        shape_probe = torch.empty(kv_cache_shape, dtype=dtype, device="cpu")
        for request in requests:
            for chunk_end, chunk_token_ids, chunk_slot_mapping in self._iter_request_chunks(
                request
            ):
                # Load each chunk into a dedicated pinned byte buffer. The GPU
                # copy is deferred until wait_for_layer_load.
                expected_shape = self._loaded_kv_shape(
                    shape_probe, chunk_slot_mapping, layer_attn_metadata
                )
                numel = _numel(expected_shape)
                payload_bytes = numel * element_size
                key = self._data_key(
                    layer_name, chunk_token_ids, request.mm_hashes, chunk_end
                )
                handles.append(
                    (chunk_slot_mapping, self._client.get_large_async(key), payload_bytes, key)
                )
        return handles

    def _close_load_handles(self) -> None:
        """Close any outstanding native read handles during step reset/shutdown."""
        for handles in self._load_handles.values():
            for _, handle, _, _ in handles:
                handle.close()
        self._load_handles.clear()

    def _found_match_for_request(self, request: "Request") -> bool:
        """Check whether this request has at least one chunk externally available."""
        return self._get_num_matched_tokens_for_request(request) > 0

    def _get_num_matched_tokens_for_request(self, request: "Request") -> int:
        """Return the contiguous chunk-aligned prefix length available externally."""
        return self._get_num_matched_tokens_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _get_num_matched_tokens_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> int:
        """Return how many leading tokens are available as contiguous chunks."""
        # vLLM still needs to execute at least one prompt token in the prefill
        # step; reporting the entire prompt as externally reusable can leave the
        # scheduler with zero new tokens to compute and trip internal asserts.
        aligned_tokens = align_to_chunk_size(
            max(len(prompt_token_ids) - 1, 0), self._chunk_tokens
        )
        if aligned_tokens == 0:
            return 0

        token_ids = torch.tensor(prompt_token_ids[:aligned_tokens], dtype=torch.int64)
        matched = 0
        for chunk_end in range(
            self._chunk_tokens, aligned_tokens + 1, self._chunk_tokens
        ):
            chunk_prefix = token_ids[:chunk_end]
            ready_key = self._ready_key(chunk_prefix, mm_hashes, chunk_end)
            if not self._chunk_is_ready(ready_key):
                break
            self._known_ready_keys.add(ready_key)
            matched = chunk_end
        return matched

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        """Check whether the prompt has at least one externally ready chunk."""
        return self._get_num_matched_tokens_for_prompt(prompt_token_ids, mm_hashes) > 0

    def _prompt_hash(self, token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
        """Hash text tokens plus multimodal identities into a stable key."""
        # The aligned prompt prefix is the stable logical identity of a stored KV.
        token_bytes = token_ids.cpu().numpy().tobytes()
        if mm_hashes:
            token_bytes += "-".join(mm_hashes).encode("utf-8")
        return safe_hash(token_bytes, usedforsecurity=False).hexdigest()

    def _ready_key(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        chunk_end: int,
    ) -> str:
        """Return the sentinel key for one persisted chunk prefix."""
        # Sentinel key used only for scheduler-side existence checks.
        return (
            f"ready:{_SCHEMA_VERSION}:{self._kv_rank:08x}:{self._layout_id}:"
            f"{self._key_dtype}:{self._block_size}:{self._chunk_blocks}:"
            f"{self._prompt_hash(token_ids, mm_hashes)}:{chunk_end}"
        )

    def _data_key(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        chunk_end: int,
    ) -> str:
        """Return the payload key for one chunk prefix and one model layer."""
        # Actual layer payload key. The payload itself stores only the last
        # chunk [chunk_end - chunk_tokens : chunk_end], but the key is derived
        # from the cumulative prefix so longer prompts can reuse earlier chunks.
        return (
            f"kv:{_SCHEMA_VERSION}:{self._kv_rank:08x}:{self._layout_id}:"
            f"{self._key_dtype}:{self._block_size}:{self._chunk_blocks}:"
            f"{self._prompt_hash(token_ids, mm_hashes)}:"
            f"{chunk_end}:{layer_name}"
        )

    def _iter_request_chunks(
        self,
        request: ReqMeta,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Split one request into fixed-size storage chunks.

        Returns tuples of:
        - chunk_end token index in the full prompt prefix
        - cumulative token_ids prefix up to chunk_end (used for key derivation)
        - slot_mapping for this chunk only (used for extract/inject)
        """
        chunks: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for chunk_start in range(0, len(request.token_ids), self._chunk_tokens):
            chunk_end = chunk_start + self._chunk_tokens
            if chunk_end > len(request.token_ids):
                break
            chunks.append(
                (
                    chunk_end,
                    request.token_ids[:chunk_end],
                    request.slot_mapping[chunk_start:chunk_end],
                )
            )
        return chunks

    def _chunk_is_ready(self, ready_key: str) -> bool:
        """Check whether a chunk-ready sentinel already exists."""
        if ready_key in self._known_ready_keys:
            return True
        return self._client.exists(ready_key)

    def _chunk_is_persisted(self, ready_key: str) -> bool:
        """Check whether a chunk is already durably stored from an earlier save."""
        if ready_key in self._known_ready_keys:
            return True
        exists = self._client.exists(ready_key)
        if exists:
            self._known_ready_keys.add(ready_key)
        return exists

    def _acquire_stage_buffer(self, num_bytes: int) -> torch.Tensor:
        """Legacy staging is disabled; use EloqStore registered memory."""
        raise RuntimeError("EloqStoreConnector uses registered memory, not stage_pool")

    def _release_stage_buffer(self, stage: torch.Tensor) -> None:
        """Legacy staging is disabled; large buffers recycle themselves."""
        return

    def _copy_tensor_to_large_buffer(
        self, large: Any, offset: int, tensor: torch.Tensor
    ) -> None:
        """Copy a GPU/CPU tensor into EloqStore registered-memory fragments."""
        elem_size = tensor.element_size()
        src = tensor.reshape(-1)
        copied_elems = 0
        skip = offset
        for view in large.memoryviews():
            if skip >= len(view):
                skip -= len(view)
                continue
            usable = view[skip:]
            skip = 0
            usable_bytes = len(usable) - (len(usable) % elem_size)
            if usable_bytes <= 0:
                continue
            elems = min(usable_bytes // elem_size, src.numel() - copied_elems)
            if elems <= 0:
                break
            dst = torch.frombuffer(usable, dtype=tensor.dtype, count=elems)
            dst.copy_(src[copied_elems : copied_elems + elems], non_blocking=False)
            copied_elems += elems
            if copied_elems >= src.numel():
                break
        if copied_elems != src.numel():
            raise RuntimeError("large registered-memory buffer is too small")

    def _large_buffer_to_gpu_tensor(
        self,
        large: Any,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Copy registered-memory fragments to one GPU tensor without CPU concat."""
        elem_size = torch.empty((), dtype=dtype).element_size()
        total_elems = _numel(shape)
        dst = torch.empty(total_elems, dtype=dtype, device=device)
        copied_elems = 0
        skip = offset
        for view in large.memoryviews():
            if skip >= len(view):
                skip -= len(view)
                continue
            usable = view[skip:]
            skip = 0
            usable_bytes = len(usable) - (len(usable) % elem_size)
            if usable_bytes <= 0:
                continue
            elems = min(usable_bytes // elem_size, total_elems - copied_elems)
            if elems <= 0:
                break
            src = torch.frombuffer(usable, dtype=dtype, count=elems)
            dst[copied_elems : copied_elems + elems].copy_(src, non_blocking=False)
            copied_elems += elems
            if copied_elems >= total_elems:
                break
        if copied_elems != total_elems:
            raise RuntimeError("large registered-memory buffer is truncated")
        return dst.reshape(shape)

    def _extract_kv_from_layer(
        self,
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Extract this request's KV slice from one layer's paged cache tensor."""
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
        """Write a loaded KV slice back into the destination paged cache layer."""
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
        """Compute the logical tensor shape for one loaded request/layer payload."""
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
    """Round down to the largest positive block-aligned prefix length."""
    if num_tokens <= 0:
        return 0
    return (num_tokens - 1) // block_size * block_size


def align_to_chunk_size(num_tokens: int, chunk_tokens: int) -> int:
    """Round down to the largest full chunk prefix length."""
    if num_tokens <= 0:
        return 0
    return num_tokens // chunk_tokens * chunk_tokens


def _numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements implied by a shape tuple."""
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _normalize_dtype_name(dtype: str) -> str:
    normalized = dtype.lower().replace("torch.", "")
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }
    return aliases.get(normalized, normalized)


def _dtype_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_CODES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported KV dtype for EloqStore payload: {dtype}") from exc


def _layout_code(layout_id: str) -> int:
    try:
        return _LAYOUT_CODES[layout_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported EloqStore KV layout_id {layout_id!r}; "
            f"known layouts: {sorted(_LAYOUT_CODES)}"
        ) from exc


def _pack_payload_header(
    stage: torch.Tensor,
    *,
    payload_len: int,
    dtype: torch.dtype,
    layout_code: int,
) -> None:
    _PAYLOAD_HEADER.pack_into(
        stage.numpy(),
        0,
        _PAYLOAD_MAGIC,
        _PAYLOAD_HEADER_LEN,
        payload_len,
        _dtype_code(dtype),
        layout_code,
    )


def _pack_payload_header_buffer(
    large: Any,
    *,
    payload_len: int,
    dtype: torch.dtype,
    layout_code: int,
) -> None:
    header = bytearray(_PAYLOAD_HEADER_LEN)
    _PAYLOAD_HEADER.pack_into(
        header,
        0,
        _PAYLOAD_MAGIC,
        _PAYLOAD_HEADER_LEN,
        payload_len,
        _dtype_code(dtype),
        layout_code,
    )
    remaining = memoryview(header)
    for view in large.memoryviews():
        if not remaining:
            break
        n = min(len(view), len(remaining))
        view[:n] = remaining[:n]
        remaining = remaining[n:]
    if remaining:
        raise RuntimeError("large registered-memory buffer cannot hold header")


def _validate_payload_header(
    stage: torch.Tensor,
    *,
    expected_payload_len: int,
    expected_dtype: torch.dtype,
    expected_layout_code: int,
    key: str,
) -> None:
    magic, header_len, payload_len, dtype_code, layout_code = _PAYLOAD_HEADER.unpack_from(
        stage.numpy(), 0
    )
    if magic != _PAYLOAD_MAGIC:
        raise ValueError(f"invalid EloqStore KV payload magic for {key}: {magic!r}")
    if header_len != _PAYLOAD_HEADER_LEN:
        raise ValueError(
            f"unsupported EloqStore KV header length for {key}: {header_len}"
        )
    if payload_len != expected_payload_len:
        raise ValueError(
            f"KV payload length mismatch for {key}: "
            f"expected {expected_payload_len}, got {payload_len}"
        )
    expected_dtype_code = _dtype_code(expected_dtype)
    if dtype_code != expected_dtype_code:
        raise ValueError(
            f"KV dtype mismatch for {key}: "
            f"expected {expected_dtype_code}, got {dtype_code}"
        )
    if layout_code != expected_layout_code:
        raise ValueError(
            f"KV layout mismatch for {key}: "
            f"expected {expected_layout_code}, got {layout_code}"
        )


def _validate_payload_header_buffer(
    large: Any,
    *,
    expected_payload_len: int,
    expected_dtype: torch.dtype,
    expected_layout_code: int,
    key: str,
) -> None:
    header = bytearray()
    for view in large.memoryviews():
        needed = _PAYLOAD_HEADER_LEN - len(header)
        if needed <= 0:
            break
        header.extend(bytes(view[:needed]))
    if len(header) != _PAYLOAD_HEADER_LEN:
        raise ValueError(f"truncated EloqStore KV payload header for {key}")
    magic, header_len, payload_len, dtype_code, layout_code = _PAYLOAD_HEADER.unpack_from(
        header, 0
    )
    if magic != _PAYLOAD_MAGIC:
        raise ValueError(f"invalid EloqStore KV payload magic for {key}: {magic!r}")
    if header_len != _PAYLOAD_HEADER_LEN:
        raise ValueError(
            f"unsupported EloqStore KV header length for {key}: {header_len}"
        )
    if payload_len != expected_payload_len:
        raise ValueError(
            f"KV payload length mismatch for {key}: "
            f"expected {expected_payload_len}, got {payload_len}"
        )
    expected_dtype_code = _dtype_code(expected_dtype)
    if dtype_code != expected_dtype_code:
        raise ValueError(
            f"KV dtype mismatch for {key}: "
            f"expected {expected_dtype_code}, got {dtype_code}"
        )
    if layout_code != expected_layout_code:
        raise ValueError(
            f"KV layout mismatch for {key}: "
            f"expected {expected_layout_code}, got {layout_code}"
        )
