from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataRef:
    kind: str                   # projections | sinograms | reconstruction | volume
    storage: str                # folder | tiff_stack | file | virtual
    path: Optional[str] = None
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    outputs: dict[str, DataRef]
    metadata: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
