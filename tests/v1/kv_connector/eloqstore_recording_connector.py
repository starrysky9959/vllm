# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.eloqstore_connector import (
    EloqStoreConnector,
)


class RecordingEloqStoreConnector(EloqStoreConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._event_file = (
            vllm_config.kv_transfer_config.kv_connector_extra_config["event_file"]
        )
        self._role_name = role.name
        self._log(
            "init "
            f"table={self._table_name} "
            f"layers={len(self._layer_order)} "
            f"has_kv_cache_config={kv_cache_config is not None}"
        )

    def _log(self, message: str) -> None:
        with open(self._event_file, "a", encoding="utf-8") as f:
            f.write(f"{self._role_name}:{message}\n")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        self._log(
            f"match request={request.request_id} "
            f"computed={num_computed_tokens} result={result}"
        )
        return result

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        self._log("start_load_kv")
        return super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._log(f"wait_for_layer_load {layer_name}")
        return super().wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        self._log(
            f"save_kv_layer {layer_name} shape={tuple(kv_layer.shape)} "
            f"registered_layers={sorted(self._registered_kv_caches.keys())}"
        )
        return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        pending = {
            key: payload.payload_bytes for key, payload in self._pending_save_blocks.items()
        }
        self._log(f"wait_for_save pending={pending}")
        return super().wait_for_save()
