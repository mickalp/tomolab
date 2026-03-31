from __future__ import annotations
from pathlib import Path

from .base import BaseNode
from ..types import DataRef, NodeResult


class LoadProjectionsNode(BaseNode):
    type_name = "load_projections"
    display_name = "Load Projections"
    input_ports = {}
    output_ports = {"projections": "projections"}

    def validate(self, params, inputs):
        errors = []
        path = params.get("path")
        if not path:
            errors.append("Missing projections folder path.")
        elif not Path(path).exists():
            errors.append(f"Projections folder does not exist: {path}")
        return errors

    def run(self, *, node_id, params, inputs, context):
        path = Path(params["path"])
        return NodeResult(
            outputs={
                "projections": DataRef(
                    kind="projections",
                    storage="folder",
                    path=str(path),
                    metadata={
                        "glob_pattern": params.get("glob_pattern", "tomo_*.tif"),
                        "recursive": params.get("recursive", False),
                    },
                )
            }
        )
