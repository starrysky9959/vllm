# EloqStore KV Connector Integration

This note describes how to integrate a new KV-cache storage system into vLLM's
KV connector framework, using the local EloqStore Python SDK as the concrete
example.

The target is a **direct vLLM connector**. This path bypasses LMCache and plugs
EloqStore straight into `vllm.distributed.kv_transfer.kv_connector.v1`.

## Goal

Use EloqStore as an external KV-cache store for vLLM requests:

- save KV blocks or KV slices produced by vLLM workers
- look up how many prefix tokens are already available
- load matching KV data back into vLLM's paged KV cache before compute
- support async save/load where useful

This is a connector problem, not just a storage-backend problem. The connector
must implement both:

- scheduler-side request matching and metadata planning
- worker-side save/load into vLLM's paged KV tensors

The performance target for the direct EloqStore connector is:

- exactly one `GPU -> pinned CPU` copy on save
- exactly one `CPU -> storage` write on save
- exactly one `storage -> pinned CPU` read on load
- exactly one `pinned CPU -> GPU` copy on load

The connector should be designed to avoid any extra CPU-side repacking copies in
the hot path.

## Core Concepts

Before talking about EloqStore, it helps to make the vLLM connector model
explicit.

### Scheduler

The scheduler is the control plane.

It does not move KV bytes itself. Instead, it decides:

- how many prefix tokens are already reusable
- which requests should load external KV
- which blocks should be allocated for those tokens
- whether completed requests should save KV to external storage

In connector terms, the scheduler answers:

- what should be loaded
- what should be saved
- which metadata should be sent to workers

### Worker

The worker is the data plane.

It owns the actual model execution state and can see the real KV cache tensors.
It is responsible for:

- loading external KV data into vLLM's KV cache tensors
- extracting KV data from vLLM's KV cache tensors
- calling the external system, such as EloqStore

In connector terms, the worker answers:

- how to move bytes into vLLM
- how to move bytes out of vLLM

### Paged KV Cache

vLLM does not treat KV cache as one flat tensor per request. It manages KV in a
paged/block-based layout.

That means a request's prefix is mapped onto a set of KV cache blocks rather
than one contiguous request-owned buffer.

This matters because an external connector must restore data into the exact
block/page positions that vLLM allocated for the request.

### Block

A block is the unit of KV allocation in vLLM.

At a high level:

- one block covers a fixed number of tokens
- a request prefix occupies one or more blocks
- connector matching is usually most natural at block-aligned boundaries

This is why connector-side prefix matching is typically phrased as "how many
aligned tokens can be reused?" rather than "is the entire request cached?"

### Slot Mapping

`slot_mapping` tells the worker where each token's KV should go inside vLLM's
paged KV layout.

When loading:

- external KV bytes are decoded into token-ordered KV slices
- `slot_mapping` tells the connector which positions inside the destination
  paged KV tensors should receive those slices

When saving:

- the connector uses `slot_mapping` to extract the request's token-aligned KV
  slices from the paged KV tensors

This is the key bridge between:

- external storage values
- vLLM's internal paged cache layout

### Prefix Match

Prefix match means:

- some prefix of the current request already exists in external KV storage
- those cached tokens can be loaded instead of recomputed

The scheduler-side connector computes how many prefix tokens match and reports
that back to vLLM.

In practice, a connector usually matches only at aligned boundaries, such as
full blocks, because that lines up with vLLM's KV allocation model.

### What `save_kv_layer` Actually Saves

`save_kv_layer()` does not save an abstract "request object".

It is called with:

- a layer name
- that layer's paged KV tensor
- attention metadata

The connector must:

1. identify which requests need to be stored
2. use request metadata plus slot mapping to locate the relevant KV slices
3. extract those slices from the current layer tensor
4. serialize and write them to the external store

So the storage object is defined by connector policy, not by vLLM directly.
You can choose to store:

- one value per layer per prefix
- one value per whole prefix across layers
- one value per block group

But the source data always comes from vLLM's layer-local paged KV buffers.

## Performance Target

The direct EloqStore connector should not target theoretical zero-copy end to
end, because EloqStore is a CPU-side storage system and vLLM KV cache is
normally GPU-resident.

The practical target is a **two-copy save path** and a **two-copy load path**.

### Save path target

1. extract request-owned KV data from a GPU `kv_layer`
2. copy once into a pinned CPU staging buffer
3. pass that staging buffer to EloqStore for one storage write

Target:

- `GPU -> pinned CPU`: once
- `CPU -> EloqStore/storage`: once

Avoid:

- GPU -> pageable CPU
- pinned CPU -> temporary Python `bytes` -> second Python `bytes`
- per-layer intermediate repacking buffers

### Load path target

1. read from EloqStore into a pinned CPU staging buffer
2. copy once from pinned CPU into the destination GPU KV tensor positions

Target:

- `storage -> pinned CPU`: once
- `pinned CPU -> GPU`: once

Avoid:

- storage -> temporary Python bytes -> second CPU buffer
- pinned CPU -> pageable CPU -> GPU

## What This Implies for the Connector

This target constrains the connector design more than the abstract connector API
does.

### Do not treat Python `bytes` as the hot-path format

If the connector serializes GPU data into fresh Python `bytes` objects on every
save, it will add at least one extra CPU-side copy. That breaks the target.

The connector should prefer:

- pinned CPU staging tensors
- memoryviews or pointer-based views over those tensors
- SDK extensions that accept buffer-like objects without repacking

### Use pinned CPU staging buffers explicitly

The connector should allocate reusable pinned CPU buffers for:

- save staging
- load staging

These buffers should be sized around the chosen value granularity:

- per block
- per cross-layer block group
- or per packed request-prefix blob

### GPU/CPU transfer should be stream-aware

The worker-side implementation should use CUDA streams and async copies where
possible so that:

- save can overlap with compute
- load can begin before the full forward step reaches the relevant layer

Pinned buffers are useful here because pageable CPU buffers reduce the benefit
of async DMA.

## Relevant vLLM Interfaces

Start from these files:

- `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

The base contract in `base.py` is split into two planes.

Scheduler-side methods:

- `get_num_new_matched_tokens`
- `update_state_after_alloc`
- `build_connector_meta`
- `update_connector_output`
- `request_finished`

Worker-side methods:

- `register_kv_caches`
- `start_load_kv`
- `wait_for_layer_load`
- `save_kv_layer`
- `wait_for_save`
- `get_finished`

The worker-side code operates on vLLM's real paged KV buffers. The connector is
responsible for translating between:

- vLLM block/page layout
- external storage keys and values

## What "Direct EloqStore" Means

LMCache already provides a vLLM adapter, but that route adds another layer:

`vLLM -> LMCache connector -> LMCache cache engine -> backend`

The direct path is:

`vLLM -> EloqStoreConnector -> EloqStore Python SDK`

This can reduce abstraction overhead and lets the storage layout follow vLLM's
native paged KV structure instead of LMCache chunk structure.

## EloqStore Python SDK Surface

The local SDK entry points are in:

- `/home/starrysky/projects/llm/eloqstore/python/src/eloqstore/client.py`

Main API:

- `Options(...)`
- `Client(options)`
- `put(key, value)`
- `get(key) -> bytes | None`
- `batch_put(items)`
- `batch_get(keys) -> list[bytes | None]`
- `delete(key)`
- `batch_delete(keys)`
- `exists(key)`

Important current behavior:

- `batch_put` is a real native batch call.
- `batch_get` is currently just `return [self.get(key) for key in keys]`.

Important implication for the connector:

- the current SDK surface is convenient for correctness
- but it does not yet guarantee the hot path stays at the desired copy budget
- especially if `get()` / `batch_get()` materialize fresh Python-owned `bytes`
  objects before the connector can stage them into pinned CPU memory

That means the current Python SDK already helps on write batching, but read
batching is still serialized at the Python layer. If read throughput matters,
adding a native batch-get path should be treated as a first-class optimization.

## Recommended Connector Shape

Implement a new connector under:

- `vllm/distributed/kv_transfer/kv_connector/v1/eloqstore_connector.py`

Recommended structure:

- `EloqStoreConnectorMetadata(KVConnectorMetadata)`
- `EloqStoreWorkerMetadata(KVConnectorWorkerMetadata)` if worker feedback is
  needed
- `EloqStoreConnector(KVConnectorBase_V1)`

Use `ExampleConnector` as the simplest end-to-end reference for how KV data is
extracted from and injected into vLLM block tensors.

## Storage Granularity

This is the main design choice.

### Option A: Store per-layer request slices

Each key maps to one layer's KV payload for one request prefix.

Pros:

- simplest implementation
- directly follows `ExampleConnector`
- easy to debug

Cons:

- many keys
- more per-layer Python overhead
- weaker batching

### Option B: Store a whole request prefix blob

Each key maps to all layers for one request prefix.

Pros:

- fewer keys
- better write batching
- easier to exploit sequential read/write

Cons:

- larger value size
- requires explicit packing/unpacking of all layer payloads
- partial-layer async load becomes harder

### Option C: Store by block group / cross-layer block

If the connector can align with vLLM cross-layer block layout, this is likely
the best long-term structure.

Pros:

- closer to vLLM's actual block abstraction
- potentially better reuse and more precise prefix matching

Cons:

- highest implementation complexity
- requires careful layout planning

For a first implementation, use **Option A** or **Option B**. Option A is the
fastest route to correctness; Option B is the cleaner route if you already care
about performance.

## Key Design

Keys should be deterministic and fully derived from request content plus storage
format version.

Suggested key components:

- connector version
- model identifier
- KV layout identifier
- dtype
- block size
- layer name or layer id
- request prefix hash
- multimodal hashes if present
- tensor-parallel rank and pipeline rank if needed

Example string key:

```text
eloq:v1:model=<model>:dtype=<dtype>:layout=<layout>:tp=<tp>:pp=<pp>:prefix=<hash>:layer=<layer>
```

If you choose whole-request blobs, omit `layer=<layer>` and keep one key per
prefix.

Do not use Python object identity or transient request ids as durable keys.
They are not stable across workers or retries.

## Value Format

For direct vLLM integration, avoid LMCache's extra wrapper format.

Recommended value layout:

1. a tiny custom header
2. raw tensor payload bytes

The header should contain only what cannot be safely inferred globally:

- format version
- number of tensors or layers packed
- per-segment offsets if using multi-layer blobs
- payload length
- optional checksum

Do not redundantly store full shape/dtype/fmt per value if they are fixed for a
given connector instance and model config.

The value format should also be chosen to minimize repacking on the CPU side.

That means:

- no JSON metadata on the hot path
- no nested Python object structures per value
- no per-layer Python reassembly if a single packed value can be written once

If multiple layer payloads are packed together, the header should be fixed-size
and cheap to parse from a pinned staging buffer.

## Scheduler-Side Responsibilities

### `get_num_new_matched_tokens`

This method decides how many prefix tokens already exist in EloqStore.

A minimal first version can:

- hash the prefix currently being considered
- check for existence of the corresponding key set
- return the largest aligned prefix match

This method should be side-effect free.

### `update_state_after_alloc`

After vLLM allocates temporary destination blocks, save the chosen block ids or
slot mappings into connector state so the worker can later inject the loaded KV.

### `build_connector_meta`

Pass the worker everything it needs to load or save:

- request id
- token ids or prefix hash
- block ids
- slot mapping
- store/load mode
- key strings

Follow the style of `ExampleConnectorMetadata`.

### `request_finished`

When a request finishes, decide whether to:

- synchronously save now
- schedule async save and return `True`
- skip save

If async save is used, `get_finished()` must eventually report completion.

## Worker-Side Responsibilities

### `register_kv_caches`

Optional. Use this if you need stable access to the worker's KV cache tensors.

### `save_kv_layer`

This is where data leaves vLLM.

The implementation should:

1. inspect connector metadata for requests that need store
2. extract the relevant KV slice from `kv_layer`
3. copy it into a reusable pinned CPU staging buffer
4. issue one write into EloqStore for that staged payload

Use the extraction logic pattern from `ExampleConnector.save_kv_layer()`.

If using per-layer keys:

- one extracted tensor -> one EloqStore value

If using whole-request blobs:

- stage per-layer payloads in memory
- flush one packed value in `wait_for_save()`

The intended save hot path is:

`GPU kv_layer -> pinned CPU staging -> EloqStore write`

and not:

`GPU kv_layer -> CPU tensor -> Python bytes -> repacked bytes -> EloqStore write`

### `wait_for_save`

If save is async, this is where outstanding writes must be joined before vLLM
reuses the paged KV buffer.

### `start_load_kv`

This is where data comes back into vLLM.

The implementation should:

1. read connector metadata for load requests
2. issue EloqStore `get` / `batch_get`
3. place the result into a pinned CPU staging buffer
4. inject it into the correct vLLM paged KV locations on GPU

Use the injection pattern from `ExampleConnector.start_load_kv()`.

### `wait_for_layer_load`

If load is async and layer-pipelined, block here until layer `i` is ready.

For a first synchronous implementation, this can be a no-op because
`start_load_kv()` finishes the whole load before returning.

The intended load hot path is:

`EloqStore read -> pinned CPU staging -> GPU kv_layer`

## How To Use the EloqStore SDK

Minimal initialization:

```python
from eloqstore import Client, Options

opts = Options(
    store_paths=["/path/to/eloqstore"],
    table_name="vllm_kv",
    num_threads=4,
)
client = Client(opts)
```

Suggested connector lifecycle:

- create one `Client` per worker process
- create a separate scheduler-side lightweight client only if the scheduler
  needs direct existence probes
- otherwise keep scheduler-side matching metadata-driven and let workers own I/O

To hit the copy budget above, the connector should avoid turning staged payloads
into extra Python-owned copies before the SDK call.

For writes:

```python
items = [(key1, value1), (key2, value2)]
client.batch_put(items)
```

For reads:

```python
values = client.batch_get(keys)
```

Because `batch_get` is currently Python-serialized, a production connector will
likely need one of:

- SDK native batch-get support
- connector-side thread pool around `client.get`
- larger packed values to reduce key count

For performance, the SDK should ideally grow two capabilities:

- write APIs that consume buffer views over pinned CPU memory without forcing an
  extra Python copy
- read APIs that fill caller-provided CPU buffers, or at least expose a buffer
  view that can be copied directly into pinned staging memory once

## Async Model

The EloqStore Python client is synchronous today. To fit vLLM's connector API,
wrap storage calls in a worker-local executor:

- `ThreadPoolExecutor`
- one queue for save
- one queue for load if needed

Typical pattern:

- `save_kv_layer()` submits storage work
- `wait_for_save()` waits on futures
- `start_load_kv()` submits or performs loads
- `wait_for_layer_load()` waits per layer if pipelining is enabled

If the EloqStore client is not explicitly documented as thread-safe, use one of:

- one client per worker thread
- or one shared client protected by a lock

The first version should favor correctness over aggressive parallelism.

For the optimized version, async execution should be layered like this:

- CUDA stream handles `GPU <-> pinned CPU`
- worker-local executor handles EloqStore blocking SDK calls
- staging buffer lifetime is tied to the completion of both stages

## Prefix Matching Strategy

To make `get_num_new_matched_tokens()` useful, you need a stable prefix naming
scheme.

Recommended approach:

1. tokenize request as usual
2. compute prefix hashes for aligned token counts
3. only store prefixes aligned to block size
4. match the longest stored aligned prefix

This aligns naturally with vLLM's block allocation model and keeps matching
cheap.

## What To Optimize First

If the first connector works but is slow, prioritize in this order:

1. eliminate extra CPU-side copies beyond the target two-copy path
2. avoid per-layer tiny puts by packing larger values
3. add native batch-get to EloqStore Python SDK
4. overlap load with compute through `start_load_kv` / `wait_for_layer_load`
5. use a lighter value header and shared global metadata

Do not start with generic database features. For direct vLLM integration, the
critical path is extracting and injecting block data efficiently.

## Recommended Minimal Implementation Plan

Phase 1: correctness

- create `EloqStoreConnector`
- implement per-layer store/load using `ExampleConnector` logic
- use synchronous `put/get`
- support block-aligned prefix matching only

Phase 2: batching

- switch writes to `batch_put`
- group reads by request and layer
- move storage calls to a background executor
- introduce reusable pinned CPU staging buffers

Phase 3: performance

- enforce the two-copy hot path
- add packed multi-layer values
- add native batch-get support in SDK
- add buffer-oriented SDK entry points if needed
- overlap load with forward execution

## Validation Checklist

Before tuning performance, verify:

- loaded KV reproduces baseline outputs
- prefix match length is correct
- async save does not race buffer reuse
- worker restarts do not corrupt key naming
- TP/PP rank separation is encoded in keys
- mismatched model/layout versions are rejected

## Summary

To integrate a new KV-cache store into vLLM:

1. implement `KVConnectorBase_V1`
2. translate vLLM paged KV blocks into your storage values
3. implement scheduler-side prefix matching and worker-side save/load
4. keep key/value format stable and versioned

To integrate EloqStore specifically:

1. create a direct connector under `kv_connector/v1`
2. initialize one or more EloqStore Python clients with `Options` / `Client`
3. use `batch_put` for store
4. add stronger read batching beyond the current Python `batch_get`
5. optimize around vLLM block layout, not LMCache chunk layout

This direct path is viable, but it is more engineering work than plugging
EloqStore in behind LMCache because the connector must speak vLLM's paged KV
protocol directly.
