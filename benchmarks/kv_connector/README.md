# KV Connector Test Guide

This directory does not add a custom benchmark runner. Use the existing vLLM
tests and benchmark-style integration tests already in the repo.

## Prerequisites

Run from:

```bash
cd /home/starrysky/projects/llm
```

Build the EloqStore C API once:

```bash
cmake --build /home/starrysky/projects/llm/eloqstore/build --target eloqstore_capi -j2
```

The current vLLM tree now probes pinned-memory availability on WSL instead of
hard-disabling it, so `UVA is not available` should no longer block startup if
`torch.empty(..., pin_memory=True)` succeeds on your machine.

## CPU Offloading

The repo already provides a real integration test for `SimpleCPUOffloadConnector`:

- test file: [test_integration.py](/home/starrysky/projects/llm/vllm/tests/v1/simple_kv_offload/test_integration.py)

### Accuracy

Small-model accuracy:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/simple_kv_offload/test_integration.py -k "accuracy and not lazy"
```

Lazy offload accuracy:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/simple_kv_offload/test_integration.py -k "accuracy_lazy"
```

### Latency / 10GB host memory

The existing perf test already uses `10 << 30` bytes for eager CPU offload:

- `test_simple_cpu_offload_perf_latency`

Run it with:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/simple_kv_offload/test_integration.py -k "perf_latency and not lazy"
```

That path constructs:

```python
KVTransferConfig(
    kv_connector="SimpleCPUOffloadConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 10 << 30,
        "lazy_offload": False,
    },
)
```

Lazy mode uses `80 << 30` bytes and is a separate test:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/simple_kv_offload/test_integration.py -k "perf_latency_lazy"
```

## EloqStore Connector

There are two existing test layers for `EloqStoreConnector`.

### Unit / connector logic

File:

- [test_eloqstore_connector.py](/home/starrysky/projects/llm/vllm/tests/v1/kv_connector/unit/test_eloqstore_connector.py)

Run:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/kv_connector/unit/test_eloqstore_connector.py
```

This covers:

- scheduler-side prefix match
- worker-side save/load
- cross-layer block layout
- missing-key load failure handling

### End-to-end functional test

File:

- [test_eloqstore_e2e.py](/home/starrysky/projects/llm/vllm/tests/v1/kv_connector/test_eloqstore_e2e.py)

Run:

```bash
cd /home/starrysky/projects/llm/vllm
pytest -q tests/v1/kv_connector/test_eloqstore_e2e.py
```

This test uses:

- `RecordingEloqStoreConnector`
- a local tiny model snapshot
- `shared_memory_bytes=64 << 20`
- `shared_memory_slot_size=4 << 20`
- `shared_memory_slot_count=16`

and checks that:

- `save_kv_layer` happens
- `start_load_kv` happens
- scheduler-side match query returns a cache hit

## Comparing CPU Offloading vs EloqStore

Current repo status:

- CPU offloading already has a built-in latency-style integration test.
- EloqStore currently has built-in unit and end-to-end functional tests.
- There is not yet a repo-native EloqStore latency benchmark matching
  `test_simple_cpu_offload_perf_latency`.

So today, the comparison workflow is:

1. Run the CPU offload latency test above.
2. Run EloqStore unit and e2e tests above to validate correctness.
3. If you want a fair latency comparison, add it on top of the existing vLLM
   test style instead of introducing an out-of-tree runner.
