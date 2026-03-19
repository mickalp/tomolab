from __future__ import annotations

"""
Build sinograms from a directory of projection TIFFs.

Input:
  - projections: N files, each (H, W)

Output (two modes):
  1) output_mode="stack" (default):
       - a multi-page TIFF with H pages
       - each page is a 2D sinogram (N, W) = (angles x detector-channels)

  2) output_mode="files":
       - one TIFF per sinogram page written into output_sinogram_dir
       - each file is a 2D sinogram (N, W)

This is compatible with a typical pipeline where ring removal runs per sinogram (per detector row).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import os
import re
import tempfile

import numpy as np
import tifffile as tiff


_RX_TOMO = re.compile(r"^tomo_(\d+)\.(tif|tiff)$", re.IGNORECASE)


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tomo_sort_key(p: Path) -> int:
    """Sort tomo_0001.tif, tomo_0002.tif, ... numerically."""
    m = _RX_TOMO.match(p.name)
    return int(m.group(1)) if m else 10**18


def _natural_key(p: Path):
    """Natural-ish sort key: img2.tif < img10.tif (fallback)."""
    parts = re.split(r"(\d+)", p.name)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def sinograms_to_projection_files(
    *,
    input_mode: str,
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    projection_template: str = "tomo_{index:04d}.tif",
    overwrite: bool = False,
    dtype: Optional[np.dtype] = None,
    temp_dir: Optional[str] = None,
    bigtiff: bool = True,
) -> dict:
    """
    Convert corrected sinograms -> corrected projections (NO reconstruction).

    INPUTS:
      input_mode:
        - "stack": input_path is a multi-page TIFF where each page is a sinogram (N,W)
        - "files": input_path is a directory containing H sinogram TIFF files, each (N,W)

      output_dir:
        - directory where corrected projections will be saved
        - will save N files: tomo_0000.tif ... (or template)

    OUTPUT:
      N projection TIFF files, each (H,W)

    Notes:
      - This is the exact inverse of projections->sinograms transform:
          sinos[y, k, x] = proj[y, x, k]
        so:
          proj_k[y, x] = sinos[y, k, x]
    """
    input_mode = (input_mode or "stack").lower()
    if input_mode not in {"stack", "files"}:
        raise ValueError(f"input_mode must be 'stack' or 'files', got {input_mode!r}")

    input_path = Path(input_path)
    out_dir = _ensure_dir(output_dir)

    if input_mode == "stack":
        if not input_path.exists():
            raise FileNotFoundError(f"Sinogram stack not found: {input_path}")

        with tiff.TiffFile(str(input_path)) as tf:
            H = len(tf.pages)
            if H == 0:
                raise ValueError(f"Empty sinogram stack: {input_path}")

            first = tf.pages[0].asarray()
            if first.ndim != 2:
                raise ValueError(f"Expected 2D sinogram pages (N,W). Got {first.shape} in page 0")
            N, W = map(int, first.shape)

            if dtype is None:
                dtype = np.asarray(first).dtype

            for k in range(N):
                fpath = out_dir / projection_template.format(index=k)
                if fpath.exists() and not overwrite:
                    raise FileExistsError(f"Output exists (use overwrite=True): {fpath}")

            with tempfile.TemporaryDirectory(dir=temp_dir) as td:
                mm_path = Path(td) / "projections_memmap.dat"
                projs = np.memmap(mm_path, mode="w+", dtype=dtype, shape=(N, H, W))

                try:
                    for y in range(H):
                        arr = tf.pages[y].asarray()
                        if arr.shape != (N, W):
                            raise ValueError(
                                f"Sinogram page shape mismatch at y={y}: got {arr.shape}, expected {(N, W)}"
                            )
                        projs[:, y, :] = np.asarray(arr, dtype=dtype)

                    projs.flush()

                    for k in range(N):
                        out_path = out_dir / projection_template.format(index=k)

                        # Make a true detached array copy so no memmap-backed view survives.
                        img = np.array(projs[k, :, :], copy=True)

                        img = np.nan_to_num(img, nan=0.0, posinf=65535.0, neginf=0.0)
                        img = np.clip(img, 0, 65535).astype(np.uint16, copy=False)

                        tiff.imwrite(str(out_path), img)
                        del img

                finally:
                    try:
                        projs.flush()
                    except Exception:
                        pass

                    mm = getattr(projs, "_mmap", None)
                    del projs
                    if mm is not None:
                        mm.close()

    else:
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Sinogram directory not found: {input_path}")

        sino_files = sorted(
            list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")),
            key=_natural_key,
        )
        if not sino_files:
            raise FileNotFoundError(f"No .tif/.tiff sinogram files found in: {input_path}")

        first = tiff.imread(str(sino_files[0]))
        if first.ndim != 2:
            raise ValueError(f"Expected 2D sinograms (N,W). Got {first.shape} in {sino_files[0].name}")
        N, W = map(int, first.shape)
        H = len(sino_files)

        if dtype is None:
            dtype = np.asarray(first).dtype

        for k in range(N):
            fpath = out_dir / projection_template.format(index=k)
            if fpath.exists() and not overwrite:
                raise FileExistsError(f"Output exists (use overwrite=True): {fpath}")

        with tempfile.TemporaryDirectory(dir=temp_dir) as td:
            mm_path = Path(td) / "projections_memmap.dat"
            projs = np.memmap(mm_path, mode="w+", dtype=dtype, shape=(N, H, W))

            try:
                for y, f in enumerate(sino_files):
                    arr = tiff.imread(str(f))
                    if arr.shape != (N, W):
                        raise ValueError(
                            f"Sinogram file shape mismatch at {f.name}: got {arr.shape}, expected {(N, W)}"
                        )
                    projs[:, y, :] = np.asarray(arr, dtype=dtype)

                projs.flush()

                for k in range(N):
                    out_path = out_dir / projection_template.format(index=k)

                    img = np.array(projs[k, :, :], copy=True)
                    img = np.nan_to_num(img, nan=0.0, posinf=65535.0, neginf=0.0)
                    img = np.clip(img, 0, 65535).astype(np.uint16, copy=False)

                    tiff.imwrite(str(out_path), img)
                    del img

            finally:
                try:
                    projs.flush()
                except Exception:
                    pass

                mm = getattr(projs, "_mmap", None)
                del projs
                if mm is not None:
                    mm.close()

    return {
        "input_mode": input_mode,
        "input": str(input_path),
        "output_dir": str(out_dir),
        "n_projections": N,
        "projection_shape": (H, W),
    }


def list_projection_files(
    proj_dir: str | os.PathLike,
    *,
    extensions: Tuple[str, ...] = (".tif", ".tiff"),
    recursive: bool = False,
    glob_pattern: Optional[str] = None,
    sort_key: Optional[Callable[[Path], object]] = None,
) -> List[Path]:
    d = Path(proj_dir)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Projection directory does not exist or is not a directory: {d}")

    if glob_pattern:
        it = d.rglob(glob_pattern) if recursive else d.glob(glob_pattern)
        files = [p for p in it if p.is_file()]
    else:
        exts = {e.lower() for e in extensions}
        it = d.rglob("*") if recursive else d.glob("*")
        files = [p for p in it if p.is_file() and p.suffix.lower() in exts]

    files.sort(key=sort_key or _natural_key)
    if not files:
        raise FileNotFoundError(f"No projection TIFFs found in {d}")
    return files


@dataclass
class ProjectionsToSinogramsSpec:
    projections_dir: str

    # Output mode:
    #   - "stack": write one multi-page TIFF at output_sinogram_stack_tiff (DEFAULT)
    #   - "files": write one TIFF per sinogram into output_sinogram_dir
    output_mode: str = "stack"  # "stack" | "files"

    # For output_mode="stack"
    output_sinogram_stack_tiff: str = "sinograms_stack.tif"

    # For output_mode="files"
    output_sinogram_dir: Optional[str] = None
    filename_template: str = "sino_{index:04d}.tif"

    # File discovery
    extensions: Tuple[str, ...] = (".tif", ".tiff")
    recursive: bool = False
    glob_pattern: Optional[str] = "tomo_*.tif"  # your naming pattern

    # Numeric / IO
    dtype: np.dtype = np.float32
    overwrite: bool = False
    bigtiff: bool = True
    imagej_axes: bool = True  # write metadata axes="YX" (many tools ignore, harmless)

    # Use a memmap temp file to avoid keeping (H,N,W) fully in RAM
    temp_dir: Optional[str] = None


def build_sinograms_from_projection_dir(
    spec: ProjectionsToSinogramsSpec,
    *,
    sort_key: Optional[Callable[[Path], object]] = tomo_sort_key,
) -> dict:
    """
    Read projection TIFFs (H,W) from spec.projections_dir and create sinograms.

    Output modes:
      - "stack" (default): write one multi-page TIFF where each page is a sinogram (N,W)
      - "files": write one TIFF per sinogram into output_sinogram_dir

    Returns a dict with metadata and outputs.
    """
    mode = (spec.output_mode or "stack").lower()
    if mode not in {"stack", "files"}:
        raise ValueError(f"Invalid output_mode={spec.output_mode!r}. Use 'stack' or 'files'.")

    files = list_projection_files(
        spec.projections_dir,
        extensions=spec.extensions,
        recursive=spec.recursive,
        glob_pattern=spec.glob_pattern,
        sort_key=sort_key,
    )

    first = tiff.imread(str(files[0]))
    if first.ndim != 2:
        raise ValueError(f"Expected 2D projections. Got {first.ndim}D in {files[0].name}: {first.shape}")

    H, W = map(int, first.shape)
    N = len(files)

    out_stack_path = Path(spec.output_sinogram_stack_tiff)
    out_dir = Path(spec.output_sinogram_dir) if spec.output_sinogram_dir else None

    if mode == "stack":
        out_stack_path.parent.mkdir(parents=True, exist_ok=True)
        if out_stack_path.exists() and not spec.overwrite:
            raise FileExistsError(f"Output exists (use overwrite=True): {out_stack_path}")
        output_location = str(out_stack_path)
    else:
        if out_dir is None:
            out_dir = out_stack_path
        out_dir.mkdir(parents=True, exist_ok=True)
        output_location = str(out_dir)

    with tempfile.TemporaryDirectory(dir=spec.temp_dir) as td:
        mm_path = Path(td) / "sinograms_memmap.dat"
        sinos = np.memmap(mm_path, mode="w+", dtype=spec.dtype, shape=(H, N, W))

        try:
            for k, f in enumerate(files):
                img = tiff.imread(str(f))
                if img.shape != (H, W):
                    raise ValueError(f"Shape mismatch at {f.name}: got {img.shape}, expected {(H, W)}")
                sinos[:, k, :] = np.asarray(img, dtype=spec.dtype)

            meta = {"axes": "YX"} if spec.imagej_axes else None

            if mode == "stack":
                with tiff.TiffWriter(str(out_stack_path), bigtiff=spec.bigtiff) as tw:
                    for y in range(H):
                        tw.write(
                            np.asarray(sinos[y, :, :], dtype=spec.dtype),
                            contiguous=True,
                            metadata=meta,
                        )
            else:
                assert out_dir is not None
                for y in range(H):
                    fname = spec.filename_template.format(index=y)
                    fpath = out_dir / fname
                    if fpath.exists() and not spec.overwrite:
                        raise FileExistsError(f"Output exists (use overwrite=True): {fpath}")
                    tiff.imwrite(
                        str(fpath),
                        np.asarray(sinos[y, :, :], dtype=spec.dtype),
                        metadata=meta,
                    )
        finally:
            try:
                sinos.flush()
            except Exception:
                pass

            mm = getattr(sinos, "_mmap", None)
            del sinos
            if mm is not None:
                mm.close()

    result = {
        "output_mode": mode,
        "output": output_location,
        "n_projections": N,
        "projection_shape": (H, W),
        "sinogram_pages": H,
        "sinogram_shape_per_page": (N, W),
        "projection_files": [str(p) for p in files],
    }
    if mode == "files":
        assert out_dir is not None
        result["sinogram_files"] = [
            str(out_dir / spec.filename_template.format(index=y)) for y in range(H)
        ]
    return result


# Backward-compatible alias
build_sinogram_stack_from_projection_dir = build_sinograms_from_projection_dir