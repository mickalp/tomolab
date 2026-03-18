# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:44:29 2026

@author: michal.kalapus
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from ringremoval.engine import Params
from ringremoval.projections import (
    ProjectionsToSinogramsSpec,
    build_sinogram_stack_from_projection_dir,
    sinograms_to_projection_files,
)
from ringremoval.stack import correct_tiff_stack

from .path_rules import resolve_output_dir


@dataclass
class ProjectionJob:
    input_dir: str
    output_mode: str = "down"          # custom | inside | up | down
    folder_name: str = "ring_corrected"
    custom_output_dir: str | None = None
    glob_pattern: str = "tomo_*.tif"
    recursive: bool = False
    overwrite: bool = False
    keep_temp: bool = False
    temp_dir: str | None = None
    workers: int = 0


def _emit(cb: Optional[Callable[[str], None]], message: str) -> None:
    if cb:
        cb(message)


def process_projection_job(
    job: ProjectionJob,
    params: Params,
    log: Optional[Callable[[str], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Pipeline:
      projections folder
        -> temporary sinogram stack
        -> corrected sinogram stack
        -> corrected projection folder
    """
    input_dir = Path(job.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out_dir = resolve_output_dir(
        input_dir=str(input_dir),
        mode=job.output_mode,
        folder_name=job.folder_name,
        custom_dir=job.custom_output_dir,
    )

    _emit(log, f"Input: {input_dir}")
    _emit(log, f"Output: {out_dir}")

    temp_root_ctx = None
    if job.keep_temp:
        temp_root = Path(job.temp_dir or (out_dir / "_temp")).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_root_ctx = tempfile.TemporaryDirectory(dir=job.temp_dir)
        temp_root = Path(temp_root_ctx.name)

    try:
        sino_stack = temp_root / "sinograms_stack.tif"
        corrected_sino_stack = temp_root / "sinograms_corrected_stack.tif"

        _emit(log, "Step 1/3: Building sinograms from projection folder...")
        sino_spec = ProjectionsToSinogramsSpec(
            projections_dir=str(input_dir),
            output_mode="stack",
            output_sinogram_stack_tiff=str(sino_stack),
            glob_pattern=job.glob_pattern,
            recursive=job.recursive,
            overwrite=True,
            temp_dir=str(temp_root),
        )
        sino_meta = build_sinogram_stack_from_projection_dir(sino_spec)

        if progress:
            progress(1, 3)

        _emit(log, "Step 2/3: Correcting ring artefacts in sinograms...")

        def on_stack_progress(done: int, total: int, page_idx: int, meta: dict) -> None:
            if "error" in meta:
                _emit(log, f"  FAIL page={page_idx}: {meta['error']}")
            else:
                _emit(log, f"  OK page={page_idx + 1}/{total}")

        corr_meta = correct_tiff_stack(
            input_tiff=str(sino_stack),
            output_sino_tiff=str(corrected_sino_stack),
            params=params,
            overwrite=True,
            workers=job.workers,
            on_progress=on_stack_progress,
        )

        if progress:
            progress(2, 3)

        _emit(log, "Step 3/3: Converting corrected sinograms back to projections...")
        proj_meta = sinograms_to_projection_files(
            input_mode="stack",
            input_path=str(corrected_sino_stack),
            output_dir=str(out_dir),
            projection_template="tomo_{index:04d}.tif",
            overwrite=job.overwrite,
            temp_dir=str(temp_root),
        )

        if progress:
            progress(3, 3)

        summary = {
            "input_dir": str(input_dir),
            "output_dir": str(out_dir),
            "sinogram_build": sino_meta,
            "correction": corr_meta,
            "projection_export": proj_meta,
            "params": asdict(params),
            "job": asdict(job),
        }

        summary_path = out_dir / "ringremoval_job_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _emit(log, f"Saved summary: {summary_path}")

        return summary

    finally:
        if temp_root_ctx is not None:
            temp_root_ctx.cleanup()