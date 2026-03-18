from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any
import re
import logging

from .engine import Params, process_file, correct_file


logger = logging.getLogger(__name__)


# ---------------------------
# Batch selection params
# ---------------------------

@dataclass
class BatchSpec:
    input_dir: str
    output_dir: str

    # File discovery
    extensions: Tuple[str, ...] = (".tif", ".tiff")
    recursive: bool = False
    glob_pattern: Optional[str] = None  # e.g. "*.tif" (if None -> use extensions)

    # Filtering
    include_regex: Optional[str] = None
    exclude_regex: Optional[str] = None
    contains: Optional[str] = None
    not_contains: Optional[str] = None

    # Selection by index (after filtering & sorting)
    start: Optional[int] = None   # inclusive
    end: Optional[int] = None     # exclusive
    step: int = 1
    indices: Optional[List[int]] = None  # explicit indices list (overrides start/end if provided)

    # Output behavior
    recon_suffix: str = "_recon"
    sino_suffix: str = "_sino_corr"
    save_sino: bool = False
    overwrite: bool = False
    center_mode: str = "once"  # "once" | "each"



# ---------------------------
# Helpers: ordering, listing, filtering, selecting
# ---------------------------

def _natural_key(p: Path):
    """
    Natural-ish sort key:
    'img2.tif' < 'img10.tif'
    """
    parts = re.split(r"(\d+)", p.name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_image_files(spec: BatchSpec) -> List[Path]:
    in_dir = Path(spec.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {in_dir}")

    if spec.glob_pattern:
        pattern = spec.glob_pattern
        it = in_dir.rglob(pattern) if spec.recursive else in_dir.glob(pattern)
        files = [p for p in it if p.is_file()]
    else:
        exts = {e.lower() for e in spec.extensions}
        it = in_dir.rglob("*") if spec.recursive else in_dir.glob("*")
        files = [p for p in it if p.is_file() and p.suffix.lower() in exts]

    files.sort(key=_natural_key)
    return files


def filter_files(files: Sequence[Path], spec: BatchSpec) -> List[Path]:
    out = list(files)

    if spec.contains:
        out = [p for p in out if spec.contains in p.name]

    if spec.not_contains:
        out = [p for p in out if spec.not_contains not in p.name]

    if spec.include_regex:
        rx = re.compile(spec.include_regex)
        out = [p for p in out if rx.search(p.name)]

    if spec.exclude_regex:
        rx = re.compile(spec.exclude_regex)
        out = [p for p in out if not rx.search(p.name)]

    return out


def select_files(files: Sequence[Path], spec: BatchSpec) -> List[Path]:
    if spec.indices is not None and len(spec.indices) > 0:
        selected = []
        for i in spec.indices:
            if i < 0 or i >= len(files):
                raise IndexError(f"Index {i} out of range for {len(files)} files")
            selected.append(files[i])
        return selected

    # Slice selection
    start = spec.start
    end = spec.end
    step = spec.step if spec.step else 1
    return list(files[slice(start, end, step)])


def build_file_list(spec: BatchSpec) -> List[Path]:
    files = list_image_files(spec)
    files = filter_files(files, spec)
    files = select_files(files, spec)
    return files


def _make_output_paths(in_path: Path, spec: BatchSpec) -> Tuple[Path, Optional[Path]]:
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve extension if it’s tif/tiff; else default to .tif
    ext = in_path.suffix if in_path.suffix.lower() in (".tif", ".tiff") else ".tif"
    stem = in_path.stem

    recon_out = out_dir / f"{stem}{spec.recon_suffix}{ext}"
    sino_out = out_dir / f"{stem}{spec.sino_suffix}{ext}" if spec.save_sino else None
    return recon_out, sino_out


# ---------------------------
# Batch processing
# ---------------------------

def process_batch(
    file_list: Sequence[Path],
    params: Params,
    spec: BatchSpec,
    on_progress: Optional[Callable[[int, int, Path, Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Process a list of sinogram files.

    on_progress(done, total, current_path, meta) is called after each file.
    should_cancel() -> bool can be used by GUI/Fiji to cancel.

    Returns a summary dict with successes/failures/skips.
    """
    if spec.center_mode not in ("once", "each"):
        raise ValueError(f"Invalid center_mode: {spec.center_mode}")
    total = len(file_list)
    done = 0
    successes = []
    failures = []
    skipped = []
    shared_center = None  # computed from first successful file if center_mode=="once"


    for in_path in file_list:
        if should_cancel and should_cancel():
            logger.warning("Batch cancelled by user.")
            break

        recon_out, sino_out = _make_output_paths(in_path, spec)

        # Skip if output exists and overwrite is False
        if not spec.overwrite and recon_out.exists():
            skipped.append(str(in_path))
            done += 1
            if on_progress:
                on_progress(done, total, in_path, {"skipped": True, "reason": "recon exists"})
            continue

        try:
            # Decide whether to reuse center
            center_override = None
            if params.center is None and spec.center_mode == "once" and shared_center is not None:
                center_override = shared_center

            meta = process_file(
                input_path=str(in_path),
                recon_output_path=str(recon_out),
                p=params,
                sino_output_path=str(sino_out) if sino_out else None,
                center_override=center_override,
            )

            successes.append(str(in_path))
            # Capture center from the first successful reconstruction, reuse for rest
            if params.center is None and spec.center_mode == "once" and shared_center is None:
                shared_center = float(meta["center_used"])
                logger.info("Batch: using shared center=%g for subsequent files.", shared_center)

            done += 1
            if on_progress:
                on_progress(done, total, in_path, meta)

        except Exception as e:
            failures.append({"file": str(in_path), "error": repr(e)})
            done += 1
            if on_progress:
                on_progress(done, total, in_path, {"error": repr(e)})

    return {
        "total": total,
        "processed": done,
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "output_dir": str(Path(spec.output_dir)),
    }

def correct_batch(
    file_list: Sequence[Path],
    params: Params,
    spec: BatchSpec,
    on_progress: Optional[Callable[[int, int, Path, Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Correction-only batch: writes corrected sinograms (no reconstruction).
    Uses spec.sino_suffix for output naming.
    """
    total = len(file_list)
    done = 0
    successes = []
    failures = []
    skipped = []

    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in file_list:
        if should_cancel and should_cancel():
            logger.warning("Batch correction cancelled by user.")
            break

        ext = in_path.suffix if in_path.suffix.lower() in (".tif", ".tiff") else ".tif"
        stem = in_path.stem
        sino_out = out_dir / f"{stem}{spec.sino_suffix}{ext}"

        if not spec.overwrite and sino_out.exists():
            skipped.append(str(in_path))
            done += 1
            if on_progress:
                on_progress(done, total, in_path, {"skipped": True, "reason": "sino exists"})
            continue

        try:
            meta = correct_file(
                input_path=str(in_path),
                sino_output_path=str(sino_out),
                p=params,
            )
            successes.append(str(in_path))
            done += 1
            if on_progress:
                on_progress(done, total, in_path, meta)

        except Exception as e:
            failures.append({"file": str(in_path), "error": repr(e)})
            done += 1
            if on_progress:
                on_progress(done, total, in_path, {"error": repr(e)})

    return {
        "total": total,
        "processed": done,
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "output_dir": str(out_dir),
    }


