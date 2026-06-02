# EloqStore KV Connector

This document describes how `EloqStoreConnector` uses EloqStore as an external
KV-cache store for vLLM, why the implementation has several moving parts, and
which parts are essential versus candidates for simplification.

The connector lives at:

```text
vllm/distributed/kv_transfer/kv_connector/v1/eloqstore_connector.py
```

## Summary

The connector persists vLLM prompt KV cache chunks into EloqStore and later
restores the longest contiguous prefix that is already present in EloqStore.

The intended data path is:

```text
save:
GPU KV cache -> connector-owned registered host buffer -> EloqStore -> storage

load:
storage -> EloqStore -> connector-owned registered host buffer -> GPU KV cache
```

There is still one GPU/CPU copy in each direction because vLLM KV cache is GPU
resident and EloqStore is a CPU-side storage engine. The connector avoids an
extra CPU staging copy by using one host buffer that is visible to both sides:

```text
posix_memalign host memory
  -> registered with EloqStore/io_uring as fixed I/O buffers
  -> registered with CUDA using cudaHostRegister
```

The connector does not use EloqStore `GlobalRegisteredMemory` for vLLM payloads.
It uses EloqStore's caller-managed pinned large-value API, passing `(char *,
size_t)` buffers owned by the connector.

## Why It Looks Complex

Some complexity is inherent to the vLLM KV connector API, not EloqStore itself.
The connector must bridge several independently constrained systems:

- vLLM scheduler decides prefix reuse before the worker touches GPU tensors.
- vLLM worker calls save/load hooks layer by layer, not once per request.
- vLLM KV cache is paged, so request tokens must be translated through
  `slot_mapping`.
- EloqStore stores byte values keyed by table/partition/key, while vLLM thinks in
  model layers, blocks, tokens, dtype, and layout.
- `io_uring` fixed-buffer I/O requires registered, aligned host memory.
- CUDA asynchronous copies require host memory registered with CUDA.
- A stored chunk must not become visible to the scheduler until all layer payloads
  for that chunk are durable.

The current implementation handles all of those constraints in one connector.
That makes the code look heavier than a simple key-value wrapper, but most of the
weight comes from matching vLLM's control plane to EloqStore's byte-oriented data
plane safely.

## Essential Pieces

These parts are required for correctness or for the current performance target.

### Chunking

The connector stores fixed-size prompt chunks instead of one growing value per
whole prefix.

```text
chunk_tokens = block_size * chunk_blocks
```

Fixed chunks are needed because:

- vLLM allocates KV cache in blocks.
- chunked prefill computes long prompts over multiple scheduler steps.
- rewriting a whole prefix after every step would duplicate old data.
- scheduler matching needs the longest contiguous reusable prefix.

Only full chunks are stored and matched. A request shorter than one full chunk is
not externally reusable.

### Ready Keys

EloqStore stores one payload value per `(layer, chunk)`, but scheduler matching
must know whether a chunk is complete across all layers. The connector therefore
writes a small `ready:` sentinel only after every layer payload for the chunk has
finished writing.

Without ready keys, the scheduler could observe a partially persisted chunk and
schedule a load that later misses one or more layers.

### Explicit Key Namespace

Payload and ready keys include the fields that define KV compatibility:

```text
schema version
kv_rank
layout_id
dtype
block_size
chunk_blocks
prompt hash up to chunk boundary
chunk_end
layer name for payload keys
```

This prevents incompatible KV layouts from sharing entries in the same model
table. The table name gives coarse model-level isolation, and the key fields give
fine-grained runtime-layout isolation.

### Caller-Managed Registered Host Pool

The connector owns a `PinnedMemoryPool` from the EloqStore Python SDK. Despite
the name, the memory is not allocated with `cudaHostAlloc` or
`torch.empty(pin_memory=True)`. The pool allocates ordinary page-aligned host
memory with `posix_memalign`.

The setup order is:

```text
1. connector allocates page-aligned host chunks
2. connector passes chunks to EloqStore through Options.pinned_memory_pool
3. EloqStore registers the chunks with io_uring
4. connector calls cudaHostRegister on the same chunks
```

The ordering matters because `io_uring` fixed-buffer registration expects normal
kernel-managed user memory. CUDA-allocated pinned memory may be accessible from
the CPU but still fail with fixed-buffer I/O, especially on WSL2.

Suballocations from the pool are also rounded up to 4 KiB. This is necessary
because the large-value write path uses fixed/direct segment writes. Even if the
top-level chunk is aligned, a non-aligned suballocation can make
`io_uring_prep_write_fixed` fail with `EINVAL`.

### Payload Value

Each stored value is:

```text
[RawKvPayload]
```

The hot path intentionally does not prepend a connector payload header. Dtype,
layout, block size, chunk size, layer name, and prefix identity are already part
of the key namespace or known from the current runtime tensor shape. Repeating
them in every stored value adds per-chunk CPU writes, read-side validation, and a
payload offset without adding useful information.

If future code needs non-redundant metadata, it should use EloqStore metadata or
a separate cold/debug path instead of adding bytes to the hot payload.

### Partition Routing

EloqStore routes by table partition. The connector hashes each `kv:` and
`ready:` key to a deterministic partition:

```text
partition_id = base_partition_id + hash(key) % num_partitions
```

This lets one connector spread traffic across EloqStore shard threads. The pool
is split into `num_pools`, normally matching `num_threads`, so each partition can
allocate from the pool associated with the shard that will process the request.

### Layer-Aware Load Pipelining

vLLM invokes load completion when each layer is reached. The connector submits
EloqStore reads ahead of use and injects data layer by layer. This allows storage
I/O for later layers to overlap with GPU execution for earlier layers.

This is an optimization, but it also fits vLLM's hook structure. Loading the
entire prefix for all layers synchronously before forward would be simpler but
would add avoidable TTFT latency.

## Current Save Flow

`save_kv_layer(layer_name, kv_layer, attn_metadata)` is called once per layer.

For every request marked as store:

1. Split the request into full external chunks.
2. Derive the payload key and ready key for each chunk.
3. Skip the chunk if its ready key is already known or present.
4. Extract this request's chunk-local KV slice from the layer tensor using
   `slot_mapping`.
5. Allocate one page-aligned caller-managed host buffer.
6. Copy raw GPU KV bytes into the buffer at offset 0.
7. Submit `batch_put_pinned_large_async` with `(key, ptr, nbytes, metadata)`.
8. Track the ready key in `_pending_ready_keys`.

`wait_for_save()` then:

1. Waits for all outstanding async EloqStore writes.
2. Closes the native handles.
3. Returns all save-side pool suballocations with `free_all()`.
4. Writes ready sentinels.
5. Caches ready keys locally to avoid redundant probes.

Ready keys are intentionally written after payload writes complete. This is the
visibility barrier between the worker data plane and scheduler match probing.

## Current Load Flow

`get_num_new_matched_tokens(request, num_computed_tokens)` runs on the scheduler
side. It checks ready keys chunk by chunk from the start of the prompt and
returns the longest contiguous chunk-aligned prefix.

The connector intentionally never reports the entire prompt as externally
reusable. It reserves at least one prompt token for local execution before
alignment because vLLM's current prefill path expects some local work.

`start_load_kv(forward_context)` runs on the worker side. It:

1. Receives scheduler metadata describing which request chunks need load.
2. Records layer names, layer KV cache tensors, and attention metadata.
3. Prefetches reads using the connector's internal pipeline depth.

For each layer, `_submit_layer_loads(...)`:

1. Derives each `(layer, chunk)` payload key.
2. Allocates one page-aligned caller-managed host buffer per chunk.
3. Submits `get_pinned_large_only_into_async(key, buffer)`.
4. Records the native handle, expected payload size, key, and buffer.

`wait_for_layer_load(layer_name)` then:

1. Waits for that layer's native async reads.
2. Copies raw payload bytes to a GPU tensor.
3. Injects the tensor into vLLM's paged KV cache using `slot_mapping`.
4. Submits additional layer prefetch work to keep the pipeline full.

## Data Structures In The Connector

The main state buckets are:

```text
_requests_need_load
    Scheduler-side requests that matched external KV and must be loaded.

_requests_need_store
    Scheduler-side requests that missed external KV and should be saved as
    chunked prefill progresses.

_pending_ready_keys
    Ready sentinels to publish after all current writes finish.

_known_ready_keys
    Small in-process cache for ready keys already seen or published.

_pinned_pool
    Connector-owned host memory registered with both EloqStore and CUDA.

_save_handles
    Outstanding EloqStore async write handles for the current save phase.

_load_handles
    Outstanding EloqStore async read handles grouped by layer.
```

## API Usage

The connector uses the EloqStore Python SDK in caller-managed pinned mode:

```python
pool = PinnedMemoryPool(total_size=..., chunk_size=..., num_pools=num_threads)

client = Client(Options(
    store_paths=[...],
    table_name="vllm_kv__...",
    partition_id=base_partition_id,
    num_threads=num_threads,
    pinned_memory_pool=pool,
    gc_global_mem_size_per_shard=32 << 20,
    pinned_tail_scratch_slots=8,
    segment_size=512 << 10,
    data_append_mode=True,
))

pool.register_cuda_host()
```

Save uses:

```python
buf = pool.allocate(total_size, pool_index=partition_id % pool.num_pools)
client.batch_put_pinned_large_async(
    [(key, buf.ptr, buf.nbytes, b"")],
    partition_id=partition_id,
)
```

Load uses:

```python
buf = pool.allocate(total_size, pool_index=partition_id % pool.num_pools)
handle = client.get_pinned_large_only_into_async(
    key,
    buf,
    partition_id=partition_id,
)
```

The connector uses `*_only_*` read APIs because it already knows the expected
value size from the runtime KV shape. Metadata is not needed for hot-path loads.

## Configuration

Common `kv_connector_extra_config` fields:

```json
{
  "store_paths": ["/path/to/eloqstore"],
  "partition_id": 0,
  "num_partitions": 4,
  "num_threads": 1,
  "chunk_blocks": 64,
  "registered_memory_total_size": 536870912,
  "registered_memory_chunk_size": 536870912,
  "cuda_host_register": true
}
```

Important fields:

- `chunk_blocks`: number of vLLM KV blocks in one external chunk.
- `num_partitions`: number of EloqStore partitions used for key routing.
- `num_threads`: EloqStore shard threads and host pool count.
- `registered_memory_total_size`: total caller-managed host pool size.
- `registered_memory_chunk_size`: top-level host chunk size registered with
  EloqStore/io_uring.
- `cuda_host_register`: whether to register the host pool with CUDA.

The `registered_memory_*` names are historical. In the current caller-managed
path they configure the connector-owned pinned pool, not EloqStore
`GlobalRegisteredMemory`.

The connector keeps `pipeline_depth`, `segment_size`, `data_append_mode`,
`enable_compression`, `segments_per_file_shift`,
`gc_global_mem_size_per_shard`, and `pinned_tail_scratch_slots` as internal
defaults rather than regular vLLM-facing tuning knobs.

## Complexity Review

The following pieces are necessary for the current design:

- Chunk-level storage and matching.
- Ready keys as a completion barrier.
- Explicit key namespace for layout compatibility.
- Caller-managed `posix_memalign` pool with 4 KiB-aligned suballocations.
- `cudaHostRegister` on the same host pages used by EloqStore fixed I/O.
- Per-layer save/load hooks because vLLM exposes layer-by-layer callbacks.
- Slot mapping conversion for paged KV cache.
- Pool recycling after async writes complete.

The following pieces are likely over-designed or should be revisited:

- Metrics are intentionally limited to aggregate chunk and payload-byte counters;
  diagnostic timing should stay off the hot path unless a concrete need appears.
- `pipeline_depth` remains an internal default and should only become user-facing
  if benchmarks show it needs workload-specific tuning.
- `pinned_tail_scratch_slots` is an EloqStore safety mechanism for tail segments;
  the connector's 4 KiB-aligned suballocations reduce, but do not eliminate, the
  need to reason about segment tails.
- `num_partitions` plus `num_threads` configuration is flexible but exposes more
  tuning knobs than most users need.

A future simplification pass can consider a single-thread/single-partition
default path with advanced tuning opt-in.

## Known Failure Modes

### `ToKvError: -22` / `EINVAL`

This usually means fixed/direct I/O rejected the submitted buffer. Common causes:

- top-level host memory was allocated by CUDA rather than ordinary kernel-managed
  memory;
- suballocated buffer address was not 4 KiB aligned;
- buffer range crossed outside the registered chunk;
- `buf_index` did not match the registered chunk containing the pointer.

The current pool avoids the first two by using `posix_memalign` and aligned
suballocations.

### Pool Exhaustion

Long prompts and chunked prefill can allocate many per-layer chunk buffers before
the save phase drains. `wait_for_save()` must free the pool after native writes
complete. If `MemoryError` still occurs, increase `registered_memory_total_size`
or reduce `chunk_blocks`/concurrency.

### Full-Prompt Reuse

The connector does not claim the entire prompt as external KV, even if all chunks
exist. It leaves one token for vLLM to compute locally to satisfy current prefill
assumptions.

## Tested Smoke Path

The current caller-managed path has been validated with:

```text
Qwen3-4B
random input length: 10000
random output length: 8
num prompts: 2
max concurrency: 1
```

The successful run stored 20k prompt tokens with no `ToKvError: -22`, `IoFail`,
or pool `MemoryError`.
