# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


@dataclass
class EloqStoreConnectorStats(KVConnectorStats):
    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        self.data = {
            "match_query_tokens": 0,
            "match_aligned_tokens": 0,
            "match_hit_tokens": 0,
            "match_miss_queries": 0,
            "match_hit_blocks": 0,
            "match_miss_blocks": 0,
            "match_unaligned_tail_tokens": 0,
            "match_reserved_tail_tokens": 0,
            "match_queries": 0,
            "load_blocks": 0,
            "load_payload_bytes": 0,
            "store_blocks": 0,
            "store_payload_bytes": 0,
        }

    def clone_and_reset(self) -> "EloqStoreConnectorStats":
        old = copy.copy(self)
        old.data = dict(self.data)
        self.reset()
        return old

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for key, value in other.data.items():
            self.data[key] = self.data.get(key, 0) + value
        return self

    def reduce(self) -> dict[str, int | float]:
        query_tokens = int(self.data["match_query_tokens"])
        aligned_tokens = int(self.data["match_aligned_tokens"])
        hit_tokens = int(self.data["match_hit_tokens"])
        hit_rate = 0.0 if query_tokens == 0 else hit_tokens / query_tokens
        aligned_hit_rate = 0.0 if aligned_tokens == 0 else hit_tokens / aligned_tokens
        return {
            **self.data,
            "match_hit_rate": round(hit_rate, 4),
            "match_aligned_hit_rate": round(aligned_hit_rate, 4),
        }

    def is_empty(self) -> bool:
        return not any(self.data.values())

    def record_match_query(
        self,
        *,
        query_tokens: int,
        aligned_tokens: int,
        hit_tokens: int,
        hit_blocks: int,
        miss_blocks: int,
        reserved_tail_tokens: int,
        unaligned_tail_tokens: int,
    ) -> None:
        self.data["match_queries"] += 1
        self.data["match_query_tokens"] += query_tokens
        self.data["match_aligned_tokens"] += aligned_tokens
        self.data["match_hit_tokens"] += hit_tokens
        self.data["match_hit_blocks"] += hit_blocks
        self.data["match_miss_blocks"] += miss_blocks
        self.data["match_reserved_tail_tokens"] += reserved_tail_tokens
        self.data["match_unaligned_tail_tokens"] += unaligned_tail_tokens
        if hit_tokens == 0:
            self.data["match_miss_queries"] += 1

    def record_load(self, blocks: int, payload_bytes: int) -> None:
        self.data["load_blocks"] += blocks
        self.data["load_payload_bytes"] += payload_bytes

    def record_store(self, blocks: int, payload_bytes: int) -> None:
        self.data["store_blocks"] += blocks
        self.data["store_payload_bytes"] += payload_bytes
