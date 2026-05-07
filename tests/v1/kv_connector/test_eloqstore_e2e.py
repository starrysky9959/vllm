# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)


def _resolve_local_model() -> str:
    return str(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--hmellor--tiny-random-LlamaForCausalLM"
        / "snapshots"
        / "9408c553e5c189a7dcdc5a5dbd2feb476b061759"
    )


@pytest.mark.optional
@pytest.mark.slow_test
def test_eloqstore_connector_end_to_end():
    model = _resolve_local_model()

    with TemporaryDirectory(prefix="vllm-eloqstore-e2e-") as tmpdir:
        root = Path(tmpdir)
        event_file = root / "events.log"
        store_dir = root / "store"
        event_file.write_text("", encoding="utf-8")
        store_dir.mkdir()

        kv_transfer_config = KVTransferConfig(
            kv_connector="RecordingEloqStoreConnector",
            kv_connector_module_path=(
                "tests.v1.kv_connector.eloqstore_recording_connector"
            ),
            kv_role="kv_both",
            kv_connector_extra_config={
                "store_paths": [str(store_dir)],
                "table_name": "vllm_e2e",
                "partition_id": 0,
                "num_threads": 1,
                "io_workers": 2,
                "event_file": str(event_file),
            },
        )

        llm = LLM(
            model=model,
            kv_transfer_config=kv_transfer_config,
            enable_prefix_caching=True,
            enforce_eager=True,
            gpu_memory_utilization=0.80,
            kv_cache_memory_bytes=1 << 30,
            max_model_len=256,
        )
        try:
            sampling_params = SamplingParams(max_tokens=1, temperature=0)
            prompt = TokensPrompt(prompt_token_ids=[42] * 32)

            first = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            time.sleep(2)
            assert llm.reset_prefix_cache() is True
            second = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

            assert first.outputs[0].text == second.outputs[0].text

            events = event_file.read_text(encoding="utf-8").splitlines()
            assert any("WORKER:save_kv_layer" in line for line in events)
            assert any("WORKER:start_load_kv" in line for line in events)
            assert any("result=(16, False)" in line for line in events)
        finally:
            del llm
