from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from ..types import DataRef, NodeResult
from ..context import RunContext


class BaseNode(ABC):
    type_name: str = "base"
    display_name: str = "Base Node"
    input_ports: dict[str, str] = {}
    output_ports: dict[str, str] = {}

    @abstractmethod
    def validate(self, params: dict[str, Any], inputs: dict[str, DataRef]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        node_id: str,
        params: dict[str, Any],
        inputs: dict[str, DataRef],
        context: RunContext,
    ) -> NodeResult:
        raise NotImplementedError
