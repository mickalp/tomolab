from __future__ import annotations

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np

from .engine import Params

logger = logging.getLogger(__name__)


@dataclass
class StackSpec:
    input_tiff: str
    output_recon_tiff: str

    # Optional: save corrected sinograms as a stack too
    output_sino_tiff: Optional[str] = None

    # Slice selection (after reading stack)
    start: Optional[int] = None      # inclusive
    end: Optional[int] = None        # exclusive
    step: int = 1
    indices: Optional[List[int]] = None  # explicit indices overrides start/end
    center_mode: str = "once"

    overwrite: bool = False


def _import_tifffile():
    """
    We use tifffile because it’s the most reliable way to read/write
    ImageJ-compatible multi-page TIFF stacks.
    """
    try:
        import tifffile  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency 'tifffile'. Install it with:\n"
            "  pip install tifffile\n"
            "It is required for multi-page TIFF stack IO."
        ) from e
    return tifffile


def _select_indices(n_pages: int, spec: StackSpec) -> List[int]:
    if spec.indices is not None and len(spec.indices) > 0:
        out = []
        for i in spec.indices:
            if i < 0 or i >= n_pages:
                raise IndexError(f"Index {i} out of range for stack with {n_pages} pages")
            out.append(i)
        return out

    start = spec.start if spec.start is not None else 0
    end = spec.end if spec.end is not None else n_pages
    step = spec.step if spec.step else 1

    start = max(0, start)
    end = min(n_pages, end)

    return list(range(start, end, step))


def _correct_stack_page_worker(
    input_tiff: str,
    page_idx: int,
    params_payload,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    Worker used by correct_tiff_stack (multiprocessing-safe).
    Reads one page, corrects it, returns (page_idx, corrected_sino, meta).

    NOTE: Keep imports inside this function for faster parent startup and to
    reduce Windows multiprocessing issues.
    """
    from .engine import Params, correct_sinogram_array

    params = Params(**params_payload) if isinstance(params_payload, dict) else params_payload

    tifffile = _import_tifffile()
    with tifffile.TiffFile(str(input_tiff)) as tf:
        sino = tf.pages[page_idx].asarray()

    sino = np.asarray(sino, dtype=np.float32)
    if sino.ndim != 2:
        raise ValueError(f"Page {page_idx} is not 2D, got shape {sino.shape}")

    if params.transpose:
        sino = sino.T

    sino_corr, meta = correct_sinogram_array(sino, params)
    return page_idx, sino_corr.astype(np.float32, copy=False), meta


def process_tiff_stack(
    spec: StackSpec,
    params: Params,
    on_progress: Optional[Callable[[int, int, int, Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Process a multi-page TIFF where each page is a 2D sinogram.

    Writes:
      - output_recon_tiff : multi-page recon stack
      - output_sino_tiff  : optional multi-page corrected sinogram stack

    Progress callback:
      on_progress(done, total, page_index, meta)
    """
    tifffile = _import_tifffile()
    from .engine import process_sinogram_array  # lazy import (faster CLI startup)

    in_path = Path(spec.input_tiff)
    if not in_path.exists():
        raise FileNotFoundError(f"Input stack not found: {in_path}")

    out_recon = Path(spec.output_recon_tiff)
    out_recon.parent.mkdir(parents=True, exist_ok=True)

    out_sino = Path(spec.output_sino_tiff) if spec.output_sino_tiff else None
    if out_sino:
        out_sino.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite checks
    if out_recon.exists() and not spec.overwrite:
        raise FileExistsError(f"Output exists (use overwrite=True): {out_recon}")
    if out_sino and out_sino.exists() and not spec.overwrite:
        raise FileExistsError(f"Output exists (use overwrite=True): {out_sino}")

    # Read metadata and pages count
    with tifffile.TiffFile(str(in_path)) as tf:
        n_pages = len(tf.pages)

    selected = _select_indices(n_pages, spec)
    total = len(selected)
    if spec.center_mode not in ("once", "each"):
        raise ValueError(f"Invalid center_mode: {spec.center_mode}")

    shared_center = None

    if total == 0:
        logger.warning("No pages selected from stack.")
        return {"total_pages": n_pages, "selected_pages": 0, "processed": 0, "failures": []}

    logger.info("Stack pages: %d, selected: %d", n_pages, total)

    failures: List[Dict[str, Any]] = []
    done = 0

    # Stream read + stream write (no need to load entire stack into memory)
    with tifffile.TiffFile(str(in_path)) as tf, \
            tifffile.TiffWriter(str(out_recon), bigtiff=True) as tw_recon, \
            (tifffile.TiffWriter(str(out_sino), bigtiff=True) if out_sino else _NullContext()) as tw_sino:

        for page_idx in selected:
            if should_cancel and should_cancel():
                logger.warning("Stack processing cancelled by user.")
                break

            try:
                sino = tf.pages[page_idx].asarray()
                sino = np.asarray(sino, dtype=np.float32)

                if sino.ndim != 2:
                    raise ValueError(f"Page {page_idx} is not 2D, got shape {sino.shape}")

                if params.transpose:
                    sino = sino.T

                center_override = None
                if params.center is None and spec.center_mode == "once" and shared_center is not None:
                    center_override = shared_center

                sino_corr, recon, meta = process_sinogram_array(sino, params, center_override=center_override)

                # Write a page to output stacks (ImageJ-friendly)
                tw_recon.write(
                    recon.astype(np.float32),
                    contiguous=True,
                    metadata={"axes": "YX"},
                )
                if out_sino:
                    tw_sino.write(
                        sino_corr.astype(np.float32),
                        contiguous=True,
                        metadata={"axes": "YX"},
                    )

                if params.center is None and spec.center_mode == "once" and shared_center is None:
                    shared_center = float(meta["center_used"])
                    logger.info("Stack: using shared center=%g for subsequent pages.", shared_center)

                done += 1
                if on_progress:
                    on_progress(done, total, page_idx, meta)

            except Exception as e:
                failures.append({"page": page_idx, "error": repr(e)})
                done += 1
                if on_progress:
                    on_progress(done, total, page_idx, {"error": repr(e)})

    return {
        "total_pages": n_pages,
        "selected_pages": total,
        "processed": done,
        "failures": failures,
        "output_recon_tiff": str(out_recon),
        "output_sino_tiff": str(out_sino) if out_sino else None,
    }


def correct_tiff_stack(
    input_tiff: str,
    output_sino_tiff: str,
    params: Params,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    indices: Optional[List[int]] = None,
    overwrite: bool = False,
    workers: int = 0,
    on_progress: Optional[Callable[[int, int, int, Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Correction-only stack:
      input:  multi-page TIFF (each page is a 2D sinogram)
      output: multi-page TIFF (each page is corrected sinogram)

    Runs correction in parallel (ProcessPoolExecutor) and writes pages in the
    original selection order.
    """
    tifffile = _import_tifffile()

    in_path = Path(input_tiff)
    if not in_path.exists():
        raise FileNotFoundError(f"Input stack not found: {in_path}")

    out_path = Path(output_sino_tiff)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use overwrite=True): {out_path}")

    # Page count
    with tifffile.TiffFile(str(in_path)) as tf:
        n_pages = len(tf.pages)

    # Reuse existing selection logic by building a temporary StackSpec-like object
    tmp = StackSpec(
        input_tiff=input_tiff,
        output_recon_tiff="__unused__.tif",
        output_sino_tiff=output_sino_tiff,
        start=start,
        end=end,
        step=step,
        indices=indices,
        overwrite=overwrite,
        center_mode="each",  # irrelevant here
    )
    selected = _select_indices(n_pages, tmp)
    total = len(selected)
    if total == 0:
        logger.warning("No pages selected from stack.")
        return {"total_pages": n_pages, "selected_pages": 0, "processed": 0, "failures": []}

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    failures: List[Dict[str, Any]] = []
    done = 0

    # Safer for Windows multiprocessing: pass plain dict
    params_payload = params.to_dict() if hasattr(params, "to_dict") else params

    with tifffile.TiffWriter(str(out_path), bigtiff=True) as tw, ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_correct_stack_page_worker, str(in_path), page_idx, params_payload): page_idx
            for page_idx in selected
        }

        pending: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}
        next_i = 0  # index into selected list

        for fut in as_completed(futures):
            page_idx = futures[fut]
            if should_cancel and should_cancel():
                logger.warning("Stack correction cancelled by user.")
                break

            try:
                pidx, sino_corr, meta = fut.result()
                pending[pidx] = (sino_corr, meta)
            except Exception as e:
                failures.append({"page": page_idx, "error": repr(e)})
                done += 1
                if on_progress:
                    on_progress(done, total, page_idx, {"error": repr(e)})
                continue

            # Flush in correct order whenever the next page is ready
            while next_i < len(selected) and selected[next_i] in pending:
                p = selected[next_i]
                sino_corr, meta = pending.pop(p)

                tw.write(
                    sino_corr,  # already float32
                    contiguous=True,
                    metadata={"axes": "YX"},
                )

                done += 1
                if on_progress:
                    on_progress(done, total, p, meta)

                next_i += 1

    return {
        "total_pages": n_pages,
        "selected_pages": total,
        "processed": done,
        "failures": failures,
        "output_sino_tiff": str(out_path),
        "workers": workers,
    }


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False