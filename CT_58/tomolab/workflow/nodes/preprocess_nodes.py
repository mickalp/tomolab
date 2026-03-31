from __future__ import annotations
from pathlib import Path

from .base import BaseNode
from ..types import DataRef, NodeResult
from ...projections import ProjectionsToSinogramsSpec, build_sinograms_from_projection_dir
from ...stack import correct_tiff_stack
from ...engine import Params


class ProjectionsToSinogramsNode(BaseNode):
    type_name = "projections_to_sinograms"
    display_name = "Projections to Sinograms"
    input_ports = {"projections": "projections"}
    output_ports = {"sinograms": "sinograms"}

    def validate(self, params, inputs):
        return [] if "projections" in inputs else ["Missing input: projections"]

    def run(self, *, node_id, params, inputs, context):
        src = inputs["projections"]
        out_path = context.node_dir(node_id) / "sinograms_stack.tif"

        spec = ProjectionsToSinogramsSpec(
            projections_dir=src.path,
            output_mode="stack",
            output_sinogram_stack_tiff=str(out_path),
            glob_pattern=params.get("glob_pattern", src.metadata.get("glob_pattern", "tomo_*.tif")),
            recursive=params.get("recursive", src.metadata.get("recursive", False)),
            overwrite=True,
            temp_dir=str(context.temp_dir),
        )
        meta = build_sinograms_from_projection_dir(spec)

        return NodeResult(
            outputs={
                "sinograms": DataRef(
                    kind="sinograms",
                    storage="tiff_stack",
                    path=meta["output"],
                    metadata=meta,
                )
            },
            metadata=meta,
        )


class RingRemovalNode(BaseNode):
    type_name = "ring_removal"
    display_name = "Ring Removal"
    input_ports = {"sinograms": "sinograms"}
    output_ports = {"corrected_sinograms": "sinograms"}

    def validate(self, params, inputs):
        return [] if "sinograms" in inputs else ["Missing input: sinograms"]

    def run(self, *, node_id, params, inputs, context):
        src = inputs["sinograms"]
        out_path = context.node_dir(node_id) / "corrected_sinograms.tif"

        p = Params(
            mode=params.get("mode", "auto"),
            apply_correction=True,
            correction=params.get("correction", "algotom"),
            snr=params.get("snr", 3.5),
            la_size=params.get("la_size", 50),
            sm_size=params.get("sm_size", 12),
        )

        meta = correct_tiff_stack(
            input_tiff=src.path,
            output_sino_tiff=str(out_path),
            params=p,
            overwrite=True,
            workers=params.get("workers", 12),
        )

        return NodeResult(
            outputs={
                "corrected_sinograms": DataRef(
                    kind="sinograms",
                    storage="tiff_stack",
                    path=str(out_path),
                    metadata=meta,
                )
            },
            metadata=meta,
        )
