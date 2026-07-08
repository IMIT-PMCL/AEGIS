"""Append-only, checksummed provenance log.

Every gate decision -- including negatives -- is written here as one JSON record
per line (JSONL). Records are content-addressed with a SHA-256 so the log is
tamper-evident and reproducible. No infrastructure identifiers, hostnames, paths
or credentials are recorded; only the scientific decision and its parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


def _canonical(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(obj: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class ProvenanceLog:
    """A durable, append-only record of pre-registrations and gate decisions."""

    path: str
    _records: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)

    def append(self, kind: str, payload: Dict[str, Any]) -> str:
        record = {
            "kind": kind,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload,
        }
        record["sha256"] = sha256_of(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(_canonical(record) + "\n")
        self._records.append(record)
        return record["sha256"]

    def read(self):
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
