from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    project_dir: Path
    run_dir: Path
    temp_dir: Path
    cache_dir: Path
    settings: dict[str, Any] = field(default_factory=dict)

    def node_dir(self, node_id: str) -> Path:
        p = self.run_dir / node_id
        p.mkdir(parents=True, exist_ok=True)
        return p
