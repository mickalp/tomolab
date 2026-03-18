# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:44:25 2026

@author: michal.kalapus
"""

from __future__ import annotations

from pathlib import Path


def resolve_output_dir(
    input_dir: str,
    mode: str,
    folder_name: str = "ring_corrected",
    custom_dir: str | None = None,
) -> Path:
    """
    Resolve the final output directory for one input projection folder.

    mode:
      - "custom" : use custom_dir directly
      - "inside" : input_dir / folder_name
      - "up"     : input_dir.parent / folder_name
      - "down"   : same as inside for now
    """
    p = Path(input_dir).resolve()

    if mode == "custom":
        if not custom_dir:
            raise ValueError("custom_dir is required when mode='custom'")
        out = Path(custom_dir).resolve()

    elif mode == "inside":
        out = p / folder_name

    elif mode == "up":
        if p.parent == p:
            raise ValueError(f"Cannot create parent-level output for: {p}")
        out = p.parent / folder_name

    elif mode == "down":
        out = p / folder_name

    else:
        raise ValueError(f"Unknown output mode: {mode!r}")

    out.mkdir(parents=True, exist_ok=True)
    return out