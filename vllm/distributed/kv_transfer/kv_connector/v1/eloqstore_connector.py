#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass, field
import hashlib
import mmap
import os
import re
import resource
import time
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_stats import (
    EloqStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPromMetrics
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.v1.core.sched.output import SchedulerOutput

try:
    from eloqstore import (
        KVCacheManager,
        KVCacheManagerOptions,
        KVCacheWorker,
        KVCacheWorkerOptions,
    )
except ImportError:  # pragma: no cover
    KVCacheManager = None
    KVCacheManagerOptions = None
    KVCacheWorker = None
    KVCacheWorkerOptions = None

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric, PromMetricT
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
_SCHEMA_VERSION = "v1"


def align_to_block_size(value: int, block_size: int) -> int:
    return value - (value % block_size)


@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]
    start_block: int = 0

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
        start_token: int = 0,
        token_limit: int | None = None,
    ) -> "ReqMeta":
        max_tokens_from_blocks = len(block_ids) * block_size
        token_limit = len(token_ids) if token_limit is None else token_limit
        valid_num_tokens = align_to_block_size(
            min(len(token_ids), token_limit, start_token + max_tokens_from_blocks),
            block_size,
        )
        token_ids_tensor = torch.as_tensor(token_ids, dtype=torch.long)[:valid_num_tokens]
        block_ids_tensor = torch.as_tensor(block_ids, dtype=torch.long)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        start_block = start_token // block_size
        num_request_blocks = max(valid_num_tokens // block_size - start_block, 0)
        slot_mapping = slot_mapping.flatten()[: num_request_blocks * block_size]
        return ReqMeta(
            token_ids=token_ids_tensor,
            block_ids=block_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
            start_block=start_block,
        )


@dataclass
class _PendingLoadRequest:
    request: "Request"
    num_external_tokens: int


@dataclass
class _WorkerSharedBufferState:
    descriptor: str | None = None
    attached: bool = False
    cuda_registered: bool = False
    shm_path: str | None = None
    mapped_bytes: int = 0
    slot_size: int = 0
    slot_count: int = 0
    slot_alignment: int = 0
    shard_count: int = 0
    partition_count: int = 0
    mmap_obj: mmap.mmap | None = None
    fd: int | None = None
    cuda_base_ptr: int = 0


@dataclass
class _PendingRuntimeRequest:
    block_key: str
    kind: str
    block_id: int
    payload_bytes: int
    layer_slices: dict[str, tuple[int, int]]
    request_id: int
    partition_id: int
    shard_id: int
    slot_id: int
    slot_generation: int


@dataclass
class _PreparedBlockPlan:
    block_key: str
    block_id: int
    partition_id: int


@dataclass
class _BlockRuntimePayload:
    block_key: str
    block_id: int
    payload_bytes: int
    layer_slices: dict[str, tuple[int, int]]
    buffer: bytearray | None = None


@dataclass
class EloqStoreConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
        start_token: int = 0,
        token_limit: int | None = None,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                token_ids,
                block_ids,
                block_size,
                is_store,
                mm_hashes,
                start_token,
                token_limit,
            )
        )


class EloqStoreConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=cast(Any, kv_cache_config),
        )
        if (
            KVCacheManager is None
            or KVCacheManagerOptions is None
            or KVCacheWorker is None
            or KVCacheWorkerOptions is None
        ):
            raise ImportError("eloqstore runtime SDK is required to use EloqStoreConnector")

        self._role = role
        self._block_size = vllm_config.cache_config.block_size
        self._model_name = self._resolve_model_name()
        self._table_name = self._resolve_table_name()
        self._layer_order = self._resolve_layer_order()
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        self._requests_need_load: dict[str, _PendingLoadRequest] = {}
        self._requests_need_store: dict[str, "Request"] = {}
        self._stats = EloqStoreConnectorStats()
        self._runtime_options: Any | None = None
        self._kv_cache_manager: Any | None = None
        self._kv_cache_worker: Any | None = None
        self._buffer_pool_descriptor: str | None = None
        self._worker_shared_buffer_state = _WorkerSharedBufferState()
        self._pending_runtime_requests: dict[int, _PendingRuntimeRequest] = {}
        self._pending_load_request_ids: set[int] = set()
        self._pending_save_blocks: dict[str, _BlockRuntimePayload] = {}
        self._prepared_load_blocks: list[_PreparedBlockPlan] = []
        self._prepared_save_blocks: list[_PreparedBlockPlan] = []
        self._load_error_block_ids: set[int] = set()
        self._init_runtime()

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def _init_runtime(self) -> None:
        options = self._build_runtime_options()
        self._runtime_options = options
        if self._role == KVConnectorRole.SCHEDULER:
            assert KVCacheManager is not None
            runtime = KVCacheManager(options)
            runtime.start()
            if options.eager_io_uring_register:
                self._validate_memlock_budget(options.shared_memory_bytes)
                runtime.register_io_uring_buffers()
            self._buffer_pool_descriptor = runtime.export_buffer_pool()
            self._kv_transfer_config.kv_connector_extra_config[
                "shared_memory_descriptor"
            ] = self._buffer_pool_descriptor
            self._kv_cache_manager = runtime
            return

        assert KVCacheWorker is not None
        runtime = KVCacheWorker(options)
        self._kv_cache_worker = runtime

    def _resolve_worker_descriptor(self) -> str:
        descriptor = self._kv_transfer_config.get_from_extra_config(
            "shared_memory_descriptor", None
        )
        if descriptor:
            return str(descriptor)
        return self._descriptor_from_runtime_options()

    def shutdown(self) -> None:
        self._detach_worker_shared_buffer()
        if self._kv_cache_worker is not None:
            self._kv_cache_worker.close()
            self._kv_cache_worker = None
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.close()
            self._kv_cache_manager = None
        self._worker_shared_buffer_state = _WorkerSharedBufferState()
        self._pending_runtime_requests.clear()
        self._pending_load_request_ids.clear()
        self._pending_save_blocks.clear()
        self._prepared_load_blocks.clear()
        self._prepared_save_blocks.clear()
        self._load_error_block_ids.clear()

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        self._prepare_block_plans(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._prepared_load_blocks.clear()
        self._prepared_save_blocks.clear()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._registered_kv_caches = {
            layer_name: kv_caches[layer_name]
            for layer_name in self._layer_order
            if layer_name in kv_caches
        }
        self._ensure_worker_runtime_attached()
        self._ensure_worker_cuda_registration()

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[Any]
    ) -> None:
        try:
            stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            inv_order = [stride_order.index(i) for i in range(len(stride_order))]
            layer_major_kv_cache = kv_cache.permute(*inv_order)
        except (AttributeError, NotImplementedError):
            layer_major_kv_cache = kv_cache

        self._registered_kv_caches = {
            layer_name: layer_major_kv_cache[layer_idx]
            for layer_idx, layer_name in enumerate(self._layer_order)
        }
        self._ensure_worker_runtime_attached()
        self._ensure_worker_cuda_registration()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("start_load_kv is worker-only for EloqStoreConnector")
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, EloqStoreConnectorMetadata)
        self._ensure_worker_runtime_attached()
        self._drain_runtime_completions()
        self._pending_load_request_ids.clear()
        self._load_error_block_ids.clear()
        for block_plan in self._prepared_load_blocks:
            # Scheduler has already decided which blocks are loadable before
            # this metadata reaches the worker. Re-probing existence here
            # would add one extra IPC round-trip per block for no value.
            payload = self._build_block_runtime_payload(
                block_plan.block_key,
                block_plan.block_id,
                allocate_buffer=False,
            )
            try:
                submitted = self._submit_load(
                    payload.block_key,
                    partition_id=block_plan.partition_id,
                    payload_bytes=payload.payload_bytes,
                )
            except Exception as exc:
                logger.warning(
                    "EloqStore load submit failed for key=%s block=%s: %s",
                    payload.block_key,
                    payload.block_id,
                    exc,
                )
                self._load_error_block_ids.add(payload.block_id)
                continue
            self._pending_load_request_ids.add(submitted.request_id)
            self._pending_runtime_requests[submitted.request_id] = _PendingRuntimeRequest(
                block_key=payload.block_key,
                kind="load",
                block_id=payload.block_id,
                payload_bytes=payload.payload_bytes,
                layer_slices=dict(payload.layer_slices),
                request_id=submitted.request_id,
                partition_id=submitted.partition_id,
                shard_id=submitted.shard_id,
                slot_id=submitted.slot_id,
                slot_generation=submitted.slot_generation,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError(
                "wait_for_layer_load is worker-only for EloqStoreConnector"
            )
        if layer_name not in self._registered_kv_caches or not self._pending_load_request_ids:
            self._drain_runtime_completions()
            return
        self._drain_runtime_completions(expected_request_ids=self._pending_load_request_ids)
        self._pending_load_request_ids.clear()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        del attn_metadata, kwargs
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("save_kv_layer is worker-only for EloqStoreConnector")
        metadata = self._get_connector_metadata() if self.has_connector_metadata() else None
        if not isinstance(metadata, EloqStoreConnectorMetadata):
            return
        self._ensure_worker_runtime_attached()
        for block_plan in self._prepared_save_blocks:
            payload = self._pending_save_blocks.get(block_plan.block_key)
            if payload is None:
                payload = self._build_block_runtime_payload(
                    block_plan.block_key,
                    block_plan.block_id,
                    allocate_buffer=True,
                )
                self._pending_save_blocks[block_plan.block_key] = payload
            if layer_name not in payload.layer_slices:
                continue
            self._stage_layer_bytes_into_payload(
                payload,
                layer_name,
                kv_layer,
                block_plan.block_id,
            )

    def wait_for_save(self):
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("wait_for_save is worker-only for EloqStoreConnector")
        self._ensure_worker_runtime_attached()
        expected_request_ids: set[int] = set()
        for payload in self._pending_save_blocks.values():
            if payload.payload_bytes <= 0:
                raise RuntimeError(
                    f"refusing to save empty payload for block {payload.block_key}"
                )
            partition_id = self._partition_id_for_key(payload.block_key)
            submitted = self._submit_save(
                payload.block_key,
                partition_id=partition_id,
                payload_bytes=payload.payload_bytes,
            )
            self._copy_staged_payload_into_slot(payload, submitted.slot_id)
            self._mark_save_ready(submitted.request_id)
            expected_request_ids.add(submitted.request_id)
            self._pending_runtime_requests[submitted.request_id] = _PendingRuntimeRequest(
                block_key=payload.block_key,
                kind="save",
                block_id=payload.block_id,
                payload_bytes=payload.payload_bytes,
                layer_slices=dict(payload.layer_slices),
                request_id=submitted.request_id,
                partition_id=submitted.partition_id,
                shard_id=submitted.shard_id,
                slot_id=submitted.slot_id,
                slot_generation=submitted.slot_generation,
            )
        self._drain_runtime_completions(expected_request_ids=expected_request_ids)
        self._pending_save_blocks.clear()

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid = set(self._load_error_block_ids)
        self._load_error_block_ids.clear()
        return invalid

    def _attach_worker_shared_buffer(self, runtime: Any, descriptor: str) -> None:
        runtime.attach_buffer_pool(descriptor)
        self._buffer_pool_descriptor = descriptor
        parts = descriptor.split("|")
        if len(parts) < 9:
            raise ValueError("buffer pool descriptor format is invalid")
        shm_path = parts[1]
        mapped_bytes = int(parts[2])
        if not os.path.exists(shm_path):
            self._worker_shared_buffer_state = _WorkerSharedBufferState(
                descriptor=descriptor,
                attached=True,
                cuda_registered=False,
                shm_path=shm_path,
                mapped_bytes=mapped_bytes,
                slot_size=int(parts[3]),
                slot_count=int(parts[4]),
                slot_alignment=int(parts[5]),
                shard_count=int(parts[6]),
                partition_count=int(parts[8]),
                mmap_obj=None,
                fd=None,
            )
            return
        fd = os.open(shm_path, os.O_RDWR)
        mm = mmap.mmap(fd, mapped_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._worker_shared_buffer_state = _WorkerSharedBufferState(
            descriptor=descriptor,
            attached=True,
            cuda_registered=False,
            shm_path=shm_path,
            mapped_bytes=mapped_bytes,
            slot_size=int(parts[3]),
            slot_count=int(parts[4]),
            slot_alignment=int(parts[5]),
            shard_count=int(parts[6]),
            partition_count=int(parts[8]),
            mmap_obj=mm,
            fd=fd,
        )

    def _ensure_worker_runtime_attached(self) -> None:
        if self._role != KVConnectorRole.WORKER:
            return
        if self._kv_cache_worker is None:
            return
        if self._worker_shared_buffer_state.attached:
            self._maybe_map_worker_shared_buffer()
            return
        descriptor = self._resolve_worker_descriptor()
        if not descriptor:
            return
        self._attach_worker_shared_buffer(self._kv_cache_worker, descriptor)
        self._maybe_map_worker_shared_buffer()

    def _maybe_map_worker_shared_buffer(self) -> None:
        state = self._worker_shared_buffer_state
        if not state.attached or state.mmap_obj is not None:
            return
        if not state.shm_path or not os.path.exists(state.shm_path):
            return
        fd = os.open(state.shm_path, os.O_RDWR)
        mm = mmap.mmap(
            fd,
            state.mapped_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        state.fd = fd
        state.mmap_obj = mm

    def _descriptor_from_runtime_options(self) -> str:
        options = self._runtime_options
        if options is None or not getattr(options, "shared_memory_name", ""):
            return ""
        shm_name = str(options.shared_memory_name)
        shm_path = shm_name if shm_name.startswith("/dev/shm/") else f"/dev/shm{shm_name if shm_name.startswith('/') else '/' + shm_name}"
        return (
            f"{options.shared_memory_name}|{shm_path}|{options.shared_memory_bytes}|"
            f"{options.slot_size}|{options.slot_count}|{options.slot_alignment}|"
            f"{options.num_threads}|{options.submission_queue_depth}|{options.partition_count}"
        )

    def _ensure_worker_cuda_registration(self) -> None:
        if self._role != KVConnectorRole.WORKER:
            return
        if not self._worker_shared_buffer_state.attached:
            return
        if self._worker_shared_buffer_state.cuda_registered:
            return
        if not torch.cuda.is_available():
            return
        mm = self._worker_shared_buffer_state.mmap_obj
        if mm is None:
            return
        cudart = self._load_cudart()
        if cudart is None:
            logger.warning("Unable to load cudart for cudaHostRegister; continuing without pinned registration")
            return
        base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        rc = cudart.cudaHostRegister(
            ctypes.c_void_p(base_ptr),
            ctypes.c_size_t(self._worker_shared_buffer_state.mapped_bytes),
            ctypes.c_uint(0),
        )
        if rc != 0:
            logger.warning("cudaHostRegister failed with error code %s; continuing without pinned registration", rc)
            return
        self._worker_shared_buffer_state.cuda_registered = True
        self._worker_shared_buffer_state.cuda_base_ptr = base_ptr

    def _partition_count(self) -> int:
        if self._worker_shared_buffer_state.partition_count > 0:
            return self._worker_shared_buffer_state.partition_count
        extra = self._kv_transfer_config.kv_connector_extra_config
        num_threads = int(extra.get("num_threads") or 1)
        return int(extra.get("partition_count") or extra.get("num_partitions") or num_threads)

    def _prepare_block_plans(self, connector_metadata: KVConnectorMetadata) -> None:
        self._prepared_load_blocks.clear()
        self._prepared_save_blocks.clear()
        if not isinstance(connector_metadata, EloqStoreConnectorMetadata):
            return
        for request in connector_metadata.requests:
            target = self._prepared_save_blocks if request.is_store else self._prepared_load_blocks
            for block_id, block_end, block_token_ids, _ in self._iter_request_blocks(request):
                block_key = self._block_key(block_token_ids, request.mm_hashes, block_end)
                target.append(
                    _PreparedBlockPlan(
                        block_key=block_key,
                        block_id=block_id,
                        partition_id=self._partition_id_for_key(block_key),
                    )
                )

    def _validate_memlock_budget(self, shared_memory_bytes: int) -> None:
        if shared_memory_bytes <= 0:
            return
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        except (AttributeError, OSError, ValueError):
            return
        if soft_limit < 0 or shared_memory_bytes <= soft_limit:
            return
        soft_kib = soft_limit // 1024
        hard_kib = hard_limit // 1024 if hard_limit >= 0 else -1
        required_kib = (shared_memory_bytes + 1023) // 1024
        raise RuntimeError(
            "EloqStore shared pinned memory pool exceeds RLIMIT_MEMLOCK: "
            f"requested={required_kib} KiB soft_limit={soft_kib} KiB "
            f"hard_limit={hard_kib} KiB. Reduce shared_memory_bytes or raise memlock "
            "before enabling io_uring buffer registration."
        )

    def _partition_id_for_key(self, block_key: str) -> int:
        partition_count = max(self._partition_count(), 1)
        digest = hashlib.sha256(block_key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little") % partition_count

    def _submit_save(
        self,
        key: str,
        partition_id: int,
        payload_bytes: int,
    ):
        if self._role == KVConnectorRole.SCHEDULER:
            return self._kv_cache_manager.submit_save(key, partition_id, payload_bytes)
        self._ensure_worker_runtime_attached()
        if self._kv_cache_worker is None:
            raise RuntimeError("kv cache worker runtime is not available for submit_save")
        return self._kv_cache_worker.submit_save(key, partition_id, payload_bytes)

    def _submit_load(
        self,
        key: str,
        partition_id: int,
        payload_bytes: int,
    ):
        if self._role == KVConnectorRole.SCHEDULER:
            return self._kv_cache_manager.submit_load(key, partition_id, payload_bytes)
        self._ensure_worker_runtime_attached()
        if self._kv_cache_worker is None:
            raise RuntimeError("kv cache worker runtime is not available for submit_load")
        return self._kv_cache_worker.submit_load(key, partition_id, payload_bytes)

    def _mark_save_ready(self, request_id: int) -> None:
        if self._role == KVConnectorRole.SCHEDULER:
            self._kv_cache_manager.mark_save_ready(request_id)
            return
        if self._kv_cache_worker is None:
            raise RuntimeError("kv cache worker runtime is not available for mark_save_ready")
        self._kv_cache_worker.mark_save_ready(request_id)

    def _poll_completion(self):
        if self._role == KVConnectorRole.SCHEDULER:
            return self._kv_cache_manager.poll_completion()
        if self._kv_cache_worker is None:
            return None
        return self._kv_cache_worker.poll_completion()

    def _drain_runtime_completions(
        self,
        expected_request_ids: set[int] | None = None,
    ) -> None:
        pending_expected = set(expected_request_ids or ())
        while True:
            completion = self._poll_completion()
            if completion is None:
                if pending_expected:
                    time.sleep(0.001)
                    continue
                break
            pending = self._pending_runtime_requests.pop(completion.request_id, None)
            if pending is None:
                continue
            pending_expected.discard(completion.request_id)
            if completion.slot_generation != pending.slot_generation:
                logger.warning(
                    "Ignoring stale EloqStore completion for key=%s block=%s: expected slot_generation=%s got=%s",
                    pending.block_key,
                    pending.block_id,
                    pending.slot_generation,
                    completion.slot_generation,
                )
                continue
            if completion.status != 2:
                logger.warning(
                    "EloqStore %s failed for key=%s block=%s: status=%s",
                    pending.kind,
                    pending.block_key,
                    pending.block_id,
                    completion.status,
                )
                if pending.kind == "load":
                    self._load_error_block_ids.add(pending.block_id)
                continue
            if pending.kind == "load":
                self._copy_slot_into_block(
                    block_key=pending.block_key,
                    block_id=pending.block_id,
                    slot_id=pending.slot_id,
                    payload_bytes=completion.payload_bytes,
                    layer_slices=pending.layer_slices,
                )
                self._stats.record_load(1, completion.payload_bytes)
            else:
                self._stats.record_store(1, completion.payload_bytes)
            if not pending_expected and expected_request_ids is not None:
                break

    def _detach_worker_shared_buffer(self) -> None:
        state = self._worker_shared_buffer_state
        if state.cuda_registered and state.cuda_base_ptr:
            cudart = self._load_cudart()
            if cudart is not None:
                cudart.cudaHostUnregister(ctypes.c_void_p(state.cuda_base_ptr))
        if state.mmap_obj is not None:
            state.mmap_obj.close()
        if state.fd is not None:
            os.close(state.fd)

    def _slot_slice(self, slot_id: int, payload_bytes: int) -> memoryview:
        self._ensure_worker_runtime_attached()
        state = self._worker_shared_buffer_state
        if state.mmap_obj is None:
            raise RuntimeError("worker shared buffer is not attached")
        slot_offset = slot_id * state.slot_size
        # Normalize the mmap slice to a plain byte view so Python accepts
        # assignments from bytearray and torch.frombuffer sees uint8 storage.
        return memoryview(state.mmap_obj)[slot_offset : slot_offset + payload_bytes].cast("B")

    def _extract_block_tensor(self, kv_layer: torch.Tensor, block_id: int) -> torch.Tensor:
        if kv_layer.shape[0] == 2:
            return kv_layer[:, block_id, ...]
        return kv_layer[block_id, ...]

    def _layer_payload_bytes(self, layer_name: str, block_id: int) -> int:
        kv_layer = self._registered_kv_caches.get(layer_name)
        if kv_layer is None:
            raise RuntimeError(f"registered kv cache missing layer {layer_name}")
        block_tensor = self._extract_block_tensor(kv_layer, block_id)
        return int(block_tensor.numel() * block_tensor.element_size())

    def _build_block_layout(
        self,
        block_key: str,
        block_id: int,
    ) -> tuple[int, dict[str, tuple[int, int]]]:
        layer_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for layer_name in self._layer_order:
            if layer_name not in self._registered_kv_caches:
                continue
            payload_bytes = self._layer_payload_bytes(layer_name, block_id)
            layer_slices[layer_name] = (offset, payload_bytes)
            offset += payload_bytes
        return offset, layer_slices

    def _build_block_runtime_payload(
        self,
        block_key: str,
        block_id: int,
        *,
        allocate_buffer: bool,
    ) -> _BlockRuntimePayload:
        payload_bytes, layer_slices = self._build_block_layout(block_key, block_id)
        return _BlockRuntimePayload(
            block_key=block_key,
            block_id=block_id,
            payload_bytes=payload_bytes,
            layer_slices=layer_slices,
            # Save assembles bytes from multiple layer tensors, so it needs a
            # staging buffer. Load only needs the byte layout and can avoid the
            # extra host allocation.
            buffer=bytearray(payload_bytes) if allocate_buffer else None,
        )

    def _stage_layer_bytes_into_payload(
        self,
        payload: _BlockRuntimePayload,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_id: int,
    ) -> None:
        offset, payload_bytes = payload.layer_slices[layer_name]
        if payload.buffer is None:
            raise RuntimeError(
                f"save payload buffer is not allocated for block {payload.block_key}"
            )
        block_tensor = self._extract_block_tensor(kv_layer, block_id).detach().contiguous()
        cpu_tensor = block_tensor.to("cpu")
        payload.buffer[offset : offset + payload_bytes] = cpu_tensor.view(torch.uint8).numpy().tobytes()

    def _copy_staged_payload_into_slot(self, payload: _BlockRuntimePayload, slot_id: int) -> None:
        state = self._worker_shared_buffer_state
        if state.mmap_obj is None:
            raise RuntimeError("worker shared buffer is not attached")
        if payload.buffer is None:
            raise RuntimeError(
                f"save payload buffer is not allocated for block {payload.block_key}"
            )
        slot_offset = slot_id * state.slot_size
        state.mmap_obj.seek(slot_offset)
        state.mmap_obj.write(payload.buffer)

    def _copy_slot_into_block(
        self,
        block_key: str,
        block_id: int,
        slot_id: int,
        payload_bytes: int,
        layer_slices: dict[str, tuple[int, int]],
    ) -> None:
        expected_payload_bytes = sum(layer_bytes for _, layer_bytes in layer_slices.values())
        if expected_payload_bytes != payload_bytes:
            raise RuntimeError(
                f"unexpected payload size for block {block_key}: expected {expected_payload_bytes}, got {payload_bytes}"
            )
        # Reuse the layout captured at submit-load time so completion handling
        # does not recompute offsets or allocate another staging buffer.
        for layer_name, (offset, layer_bytes) in layer_slices.items():
            kv_layer = self._registered_kv_caches.get(layer_name)
            if kv_layer is None:
                continue
            block_tensor = self._extract_block_tensor(kv_layer, block_id)
            cpu_tensor = torch.frombuffer(
                self._slot_slice(slot_id, payload_bytes)[offset : offset + layer_bytes],
                dtype=torch.uint8,
                count=layer_bytes,
            ).view(block_tensor.dtype).reshape(tuple(block_tensor.shape))
            block_tensor.copy_(cpu_tensor.to(device=block_tensor.device, dtype=block_tensor.dtype))

    def _load_cudart(self):
        libname = ctypes.util.find_library("cudart")
        if not libname:
            return None
        cudart = ctypes.CDLL(libname)
        cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        cudart.cudaHostRegister.restype = ctypes.c_int
        cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        cudart.cudaHostUnregister.restype = ctypes.c_int
        return cudart

    def _block_exists(self, block_key: str) -> bool:
        # Existence checks belong to the scheduler-side manager. Worker-side
        # save/load should consume scheduler-approved metadata rather than add
        # another IPC probe per block.
        partition_id = self._partition_id_for_key(block_key)
        if self._role != KVConnectorRole.SCHEDULER:
            return False
        return bool(self._kv_cache_manager.contains_key(block_key, partition_id))

    def _get_num_matched_tokens_for_request(self, request: "Request") -> int:
        return self._get_num_matched_tokens_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _get_num_matched_tokens_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> int:
        max_probe_tokens = max(len(prompt_token_ids) - 1, 0)
        aligned_probe_tokens = align_to_block_size(max_probe_tokens, self._block_size)
        matched_tokens = 0
        hit_blocks = 0
        miss_blocks = 0
        for token_end in range(self._block_size, aligned_probe_tokens + 1, self._block_size):
            block_token_ids = torch.as_tensor(
                prompt_token_ids[token_end - self._block_size : token_end],
                dtype=torch.long,
            )
            block_key = self._block_key(block_token_ids, mm_hashes, token_end)
            if not self._block_exists(block_key):
                miss_blocks += 1
                break
            hit_blocks += 1
            matched_tokens = token_end
        self._stats.record_match_query(
            query_tokens=len(prompt_token_ids),
            aligned_tokens=aligned_probe_tokens,
            hit_tokens=matched_tokens,
            hit_blocks=hit_blocks,
            miss_blocks=miss_blocks,
            reserved_tail_tokens=1 if prompt_token_ids else 0,
            unaligned_tail_tokens=max(len(prompt_token_ids) - 1 - aligned_probe_tokens, 0),
        )
        return matched_tokens

    def get_metrics(self) -> dict[str, float | int]:
        return self._stats.reduce()

    def get_kv_connector_stats(self) -> EloqStoreConnectorStats | None:
        if self._stats.is_empty():
            return None
        return self._stats.clone_and_reset()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> EloqStoreConnectorStats | None:
        return EloqStoreConnectorStats(data=data) if data is not None else EloqStoreConnectorStats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics | None:
        del vllm_config, metric_types, labelnames, per_engine_labelvalues
        return None

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._role != KVConnectorRole.SCHEDULER:
            raise RuntimeError(
                "get_num_new_matched_tokens is scheduler-only for EloqStoreConnector"
            )
        matched_tokens = self._get_num_matched_tokens_for_request(request)
        return max(matched_tokens - num_computed_tokens, 0), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        del blocks
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = _PendingLoadRequest(
                request=request,
                num_external_tokens=num_external_tokens,
            )
        elif request.num_computed_tokens < request.num_prompt_tokens:
            self._requests_need_store[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self._role != KVConnectorRole.SCHEDULER:
            raise RuntimeError(
                "build_connector_meta is scheduler-only for EloqStoreConnector"
            )
        meta = EloqStoreConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            num_new_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
            pending_load = self._requests_need_load.get(new_req.req_id)
            token_limit = min(new_req.num_computed_tokens + num_new_tokens, len(token_ids))
            if pending_load is not None:
                token_limit = min(token_limit, pending_load.num_external_tokens)
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                    start_token=0,
                    token_limit=token_limit,
                )
            elif token_limit > new_req.num_computed_tokens:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    mm_hashes=mm_hashes,
                    start_token=new_req.num_computed_tokens,
                    token_limit=token_limit,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            request = self._requests_need_store.get(req_id)
            if request is None:
                continue
            if new_block_ids is None:
                meta.add_request(
                    token_ids=request.all_token_ids[:num_computed_tokens],
                    block_ids=[],
                    block_size=self._block_size,
                    is_store=True,
                    mm_hashes=[f.identifier for f in request.mm_features],
                    start_token=num_computed_tokens,
                    token_limit=num_computed_tokens,
                )
                continue
            total_tokens = min(
                num_computed_tokens + num_new_tokens,
                request.num_prompt_tokens,
            )
            meta.add_request(
                token_ids=request.all_token_ids[:total_tokens],
                block_ids=new_block_ids[0],
                block_size=self._block_size,
                is_store=True,
                mm_hashes=[f.identifier for f in request.mm_features],
                start_token=num_computed_tokens,
                token_limit=total_tokens,
            )
        self._requests_need_load.clear()
        return meta

    def _build_runtime_options(self) -> Any:
        extra = self._kv_transfer_config.kv_connector_extra_config
        store_paths = extra.get("store_paths") or []
        if isinstance(store_paths, str):
            store_paths = [store_paths]
        shared_memory_bytes = int(extra.get("shared_memory_bytes", 512 << 20))
        slot_size = int(extra.get("shared_memory_slot_size", 4 << 20))
        slot_count = int(extra.get("shared_memory_slot_count", 128))
        slot_alignment = int(extra.get("shared_memory_slot_alignment", 4096))
        submission_queue_depth = int(extra.get("submission_queue_depth", 128))
        eager_io_uring_register = bool(extra.get("eager_io_uring_register", True))
        runtime_token = self._runtime_token(store_paths, extra)
        ipc_path = str(extra.get("ipc_path") or f"ipc:///tmp/eloqstore-{runtime_token}.sock")
        shm_name = str(extra.get("shared_memory_name") or f"/eloqstore-{runtime_token}")
        num_threads = int(extra.get("num_threads") or 1)
        partition_count = int(extra.get("partition_count") or extra.get("num_partitions") or num_threads)
        if self._role == KVConnectorRole.SCHEDULER:
            assert KVCacheManagerOptions is not None
            return KVCacheManagerOptions(
                store_paths=list(store_paths),
                table_name=self._table_name,
                branch=str(extra.get("branch", "main")),
                ipc_path=ipc_path,
                shared_memory_name=shm_name,
                term=int(extra.get("term", 0)),
                partition_group_id=int(extra.get("partition_group_id", 0)),
                num_threads=num_threads,
                partition_count=partition_count,
                shared_memory_bytes=shared_memory_bytes,
                slot_size=slot_size,
                slot_count=slot_count,
                slot_alignment=slot_alignment,
                submission_queue_depth=submission_queue_depth,
                eager_io_uring_register=eager_io_uring_register,
            )
        assert KVCacheWorkerOptions is not None
        return KVCacheWorkerOptions(
            ipc_path=ipc_path,
            shared_memory_name=shm_name,
            num_threads=num_threads,
            partition_count=partition_count,
            shared_memory_bytes=shared_memory_bytes,
            slot_size=slot_size,
            slot_count=slot_count,
            slot_alignment=slot_alignment,
            submission_queue_depth=submission_queue_depth,
        )

    def _runtime_token(self, store_paths: list[str], extra: dict[str, Any]) -> str:
        seed = "|".join(
            [self._table_name, str(extra.get("branch", "main")), *sorted(map(str, store_paths))]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def _resolve_model_name(self) -> str:
        model_config = self._vllm_config.model_config
        served = getattr(model_config, "served_model_name", None)
        if isinstance(served, str) and served:
            return served
        model = getattr(model_config, "model", None)
        if isinstance(model, str) and model:
            return model
        return "unknown_model"

    def _resolve_table_name(self) -> str:
        explicit = self._kv_transfer_config.get_from_extra_config("table_name", None)
        if explicit:
            return str(explicit)
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", self._model_name).strip("._-")
        if not sanitized:
            sanitized = "unknown_model"
        return f"vllm_kv__{sanitized}"

    def _resolve_layer_order(self) -> list[str]:
        if self._kv_cache_config is None:
            return []
        return [
            layer_name
            for group in self._kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        ]

    def _block_key(
        self,
        block_token_ids: torch.Tensor,
        mm_hashes: list[str],
        block_end: int,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(block_token_ids.cpu().numpy().tobytes())
        for mm_hash in mm_hashes:
            digest.update(mm_hash.encode("utf-8"))
        prefix_hash = safe_hash(digest.digest(), usedforsecurity=False).hexdigest()
        return f"kv:{_SCHEMA_VERSION}:{prefix_hash}:{block_end}"

    def _iter_request_blocks(
        self, request: ReqMeta
    ) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
        blocks: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        if request.token_ids.numel() == 0 or request.slot_mapping.numel() == 0:
            return blocks
        block_count = request.slot_mapping.numel() // self._block_size
        for local_index in range(block_count):
            token_start = (request.start_block + local_index) * self._block_size
            token_end = token_start + self._block_size
            slot_start = local_index * self._block_size
            slot_end = slot_start + self._block_size
            blocks.append(
                (
                    int(request.block_ids[local_index]),
                    token_end,
                    request.token_ids[token_start:token_end],
                    request.slot_mapping[slot_start:slot_end],
                )
            )
        return blocks
