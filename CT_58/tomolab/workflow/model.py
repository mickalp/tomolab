from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeInstance:
    id: str
    type_name: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Edge:
    source_node: str
    source_output: str
    target_node: str
    target_input: str


@dataclass
class Pipeline:
    name: str
    nodes: list[NodeInstance] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
