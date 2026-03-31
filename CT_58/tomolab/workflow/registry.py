from __future__ import annotations
from typing import Dict, Type

from .nodes.base import BaseNode


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: Dict[str, Type[BaseNode]] = {}

    def register(self, node_cls: Type[BaseNode]) -> None:
        self._nodes[node_cls.type_name] = node_cls

    def create(self, type_name: str) -> BaseNode:
        if type_name not in self._nodes:
            raise KeyError(f"Unknown node type: {type_name}")
        return self._nodes[type_name]()

    def list_types(self) -> list[str]:
        return sorted(self._nodes.keys())
