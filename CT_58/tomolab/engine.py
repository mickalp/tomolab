#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core engine for sinogram stripe/ring handling + FBP reconstruction using Algotom.

This module is intentionally GUI- and CLI-agnostic.
CLI (ringremoval/cli.py) should be a thin wrapper around this engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np


logger = logging.getLogger(__name__)


def _import_algotom_removal():
    import algotom.prep.removal as rem
    return rem

def _import_algotom_calc():
    import algotom.prep.calculation as calc
    return calc

def _import_algotom_rec():
    import algotom.rec.reconstruction as rec
    return rec

def _import_algotom_io():
    import algotom.io.loadersaver as losa
    return losa

# ---------------------------
# Parameters (dataclass)
# ---------------------------

@dataclass
class Params:
    # Input handling
    transpose: bool = False  # if stored as (detectors, projections)

    # Domain handling
    mode: str = "auto"  # "auto" | "intensity" | "log"

    # Correction selection
    apply_correction: bool = True
    correction: str = "auto"  # "auto" | "algotom" | "repair"

    # Algotom remove_all_stripe params
    snr: float = 3.0
    la_size: int = 51
    sm_size: int = 21
    dim: int = 1  # 1 for vertical stripes
    
    # --- Algotom: filtering-based stripes ---
    filt_sigma: int = 3
    filt_size: int = 21
    filt_dim: int = 1
    filt_sort: bool = True
    
    # --- Algotom: sorting-based stripes ---
    sort_size: int = 21
    sort_dim: int = 1
    
    # --- Algotom: wavelet-FFT stripes ---
    wfft_level: int = 5
    wfft_size: int = 1
    wfft_wavelet_name: str = "db9"
    wfft_window_name: str = "gaussian"   # "gaussian" or "butter"
    wfft_sort: bool = False
    
    # --- Algotom: dead / large stripe specific ---
    dead_snr: float = 3.0
    dead_size: int = 51
    dead_residual: bool = True
    
    large_snr: float = 3.0
    large_size: int = 51
    large_drop_ratio: float = 0.1
    large_norm: bool = True


    # Repair params (log-domain safe)
    repair_thresh: float = 3.0
    repair_max_cols: int = 1000

    # Geometry / recon
    center: Optional[float] = None
    angles_start: float = 0.0
    angles_end: float = 360.0
    filter_name: str = "hann"
    gpu: bool = False

    # Overrides for apply_log (normally inferred from mode)
    no_log: bool = False
    force_log: bool = False

    # Numeric stability
    eps: float = 1e-6


# ---------------------------
# Utilities (moved from script)
# ---------------------------

def detect_domain_auto(sino: np.ndarray) -> str:
    """
    Heuristic domain detection:
      - "intensity" sinograms (I or I/I0) are typically in (0, ~1] and non-negative.
      - "log" (line-integral) sinograms can be >> 1 and may include zeros/negatives.

    Returns: "intensity" or "log"
    """
    s = np.asarray(sino, dtype=np.float32)
    finite = np.isfinite(s)
    if not finite.any():
        return "log"

    sf = s[finite]
    mn = float(sf.min())
    mx = float(sf.max())
    mean = float(sf.mean())

    # Strong indicators:
    if mx > 5.0:
        return "log"
    if mn < 0.0 and mx > 1.5:
        return "log"
    if mx <= 1.5 and mn >= 0.0 and mean <= 1.0:
        return "intensity"

    # Fallback:
    frac_0_1 = float(np.mean((sf >= 0.0) & (sf <= 1.0)))
    return "intensity" if frac_0_1 > 0.9 else "log"


def clean_sinogram_for_recon(
    sino_in: np.ndarray,
    sino_fallback: Optional[np.ndarray] = None,
    apply_log: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    1) Replace NaN/Inf with values from fallback (or median).
    2) If apply_log=True, enforce strictly positive values (log requires >0).

    Intended mainly for intensity-domain data, but safe for either.
    """
    s = np.asarray(sino_in, dtype=np.float32).copy()

    bad = ~np.isfinite(s)
    if bad.any():
        if sino_fallback is not None and sino_fallback.shape == s.shape:
            s[bad] = np.asarray(sino_fallback, dtype=np.float32)[bad]
        else:
            fill = np.nanmedian(s)
            if not np.isfinite(fill):
                fill = eps
            s[bad] = fill
        logger.warning("Found %d NaN/Inf values; replaced them.", int(bad.sum()))

    if apply_log:
        nonpos = s <= 0
        if nonpos.any():
            pos = s[s > 0]
            floor = float(pos.min()) if pos.size else eps
            floor = max(floor, eps)
            s[nonpos] = floor
            logger.warning("Found %d values <= 0; clipped to %g before log.", int(nonpos.sum()), floor)

    return s


def repair_bad_columns_logdomain(
    sino: np.ndarray,
    thresh: float = 3.0,
    max_cols: int = 1000,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Safe stripe handling for log-domain sinograms.
    Detect outlier detector columns via robust statistics (median + MAD)
    and replace each bad column by interpolation of nearest good neighbors.
    """
    s = np.asarray(sino, dtype=np.float32).copy()
    nproj, ndet = s.shape

    col_profile = np.median(s, axis=0)
    med = np.median(col_profile)
    mad = np.median(np.abs(col_profile - med))
    scale = 1.4826 * mad + eps
    z = np.abs(col_profile - med) / scale
    


    bad_cols = np.where(z > thresh)[0]
    if bad_cols.size == 0:
        logger.info("repair: no outlier columns detected.")
        
    # --DEBBUGING--
    #     logger.warning(
    # "repair debug: finite_sino=%d/%d, finite_col_profile=%d/%d, z: nan=%d min=%s max=%s mad=%s scale=%s",
    # int(np.isfinite(s).sum()), s.size,
    # int(np.isfinite(col_profile).sum()), col_profile.size,
    # int(np.isnan(z).sum()),
    # str(np.nanmin(z) if np.isfinite(z).any() else None),
    # str(np.nanmax(z) if np.isfinite(z).any() else None),
    # str(mad), str(scale),
# )
        return s

    if bad_cols.size > max_cols:
        logger.warning("repair: flagged %d columns (> %d); skipping repair.", int(bad_cols.size), int(max_cols))
        return s

    good_mask = np.ones(ndet, dtype=bool)
    good_mask[bad_cols] = False
    good_idx = np.where(good_mask)[0]

    if good_idx.size == 0:
        logger.warning("repair: no good columns available; skipping repair.")
        return s

    for c in bad_cols:
        left = good_idx[good_idx < c]
        right = good_idx[good_idx > c]

        if left.size and right.size:
            l = left[-1]
            r = right[0]
            w = (c - l) / (r - l)
            s[:, c] = (1.0 - w) * s[:, l] + w * s[:, r]
        elif left.size:
            l = left[-1]
            s[:, c] = s[:, l]
        else:
            r = right[0]
            s[:, c] = s[:, r]

    logger.info("repair: repaired %d outlier columns (thresh=%g).", int(bad_cols.size), float(thresh))
    return s


# ---------------------------
# Engine orchestration
# ---------------------------

def decide_domain_and_log(sino: np.ndarray, p: Params) -> Tuple[str, bool]:
    """Return (domain, apply_log), honoring p.mode and p.force_log/p.no_log."""
    if p.force_log and p.no_log:
        raise ValueError("You can't set both force_log and no_log.")

    if p.mode == "auto":
        domain = detect_domain_auto(sino)
    else:
        if p.mode not in ("intensity", "log"):
            raise ValueError(f"Invalid mode: {p.mode}")
        domain = p.mode

    if p.force_log:
        apply_log = True
    elif p.no_log:
        apply_log = False
    else:
        apply_log = (domain == "intensity")

    return domain, apply_log


def choose_correction_method(domain: str, p: Params) -> str:
    """Return 'none' | 'algotom' | 'repair' based on params + domain."""
    if not p.apply_correction:
        return "none"

    if p.correction == "auto":
        return "algotom" if domain == "intensity" else "repair"

    allowed = (
        "algotom",          # remove_all_stripe
        "repair",           # your log-domain repair
        "filtering",        # remove_stripe_based_filtering
        "sorting",          # remove_stripe_based_sorting
        "wavelet_fft",      # remove_stripe_based_wavelet_fft
        "dead",             # remove_dead_stripe
        "large",            # remove_large_stripe
    )
    if p.correction not in allowed:
        raise ValueError(f"Invalid correction: {p.correction}")
        
    return p.correction


def correct_sinogram(sino: np.ndarray, p: Params, domain: str, apply_log: bool) -> Tuple[np.ndarray, str]:
    """Apply selected correction, return (sino_corr, method)."""
    method = choose_correction_method(domain, p)
    logger.info("domain=%s apply_log=%s correction=%s", domain, apply_log, method)
    
    

    if method == "none":
        sino_corr = np.asarray(sino, dtype=np.float32)

    elif method == "algotom":
        with np.errstate(divide="ignore", invalid="ignore"):
            rem = _import_algotom_removal()
            sino_corr = rem.remove_all_stripe(
                sino,
                p.snr,
                p.la_size,
                p.sm_size,
                dim=p.dim,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )

    elif method == "repair":
        sino_corr = repair_bad_columns_logdomain(
            sino,
            thresh=p.repair_thresh,
            max_cols=p.repair_max_cols,
            eps=p.eps,
        )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )
        
    elif method == "filtering":
        
        with np.errstate(divide="ignore", invalid="ignore"):
            rem = _import_algotom_removal()
            sino_corr = rem.remove_stripe_based_filtering(
                sino,
                sigma=p.filt_sigma,
                size=p.filt_size,
                dim=p.filt_dim,
                sort=p.filt_sort,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )

    elif method == "sorting":
        with np.errstate(divide="ignore", invalid="ignore"):
            rem = _import_algotom_removal()
            sino_corr = rem.remove_stripe_based_sorting(
                sino,
                size=p.sort_size,
                dim=p.sort_dim,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )
    
    elif method == "wavelet_fft":
        with np.errstate(divide="ignore", invalid="ignore"):
            rem = _import_algotom_removal()
            sino_corr = rem.remove_stripe_based_wavelet_fft(
                sino,
                level=p.wfft_level,
                size=p.wfft_size,
                wavelet_name=p.wfft_wavelet_name,
                window_name=p.wfft_window_name,
                sort=p.wfft_sort,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )
    
    elif method == "dead":
        rem = _import_algotom_removal()
        with np.errstate(divide="ignore", invalid="ignore"):
            sino_corr = rem.remove_dead_stripe(
                sino,
                snr=p.dead_snr,
                size=p.dead_size,
                residual=p.dead_residual,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )
    
    elif method == "large":
        rem = _import_algotom_removal()
        with np.errstate(divide="ignore", invalid="ignore"):
            sino_corr = rem.remove_large_stripe(
                sino,
                snr=p.large_snr,
                size=p.large_size,
                drop_ratio=p.large_drop_ratio,
                norm=p.large_norm,
            )
        sino_corr = clean_sinogram_for_recon(
            sino_corr,
            sino_fallback=sino,
            apply_log=apply_log,
            eps=p.eps,
        )
    
    
        

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return sino_corr, method


def reconstruct_fbp(
    sino_corr: np.ndarray,
    sino_for_center: np.ndarray,
    p: Params,
    apply_log: bool,
    center_override: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """FBP reconstruction + returns (recon, center_used)."""
    nproj, ndet = sino_corr.shape

    if center_override is not None:
        center = float(center_override)
    elif p.center is not None:
        center = float(p.center)
    else:
        calc = _import_algotom_calc()
        center = float(calc.find_center_vo(sino_for_center))

    angles_deg = np.linspace(p.angles_start, p.angles_end, nproj, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    
    rec = _import_algotom_rec()
    recon = rec.fbp_reconstruction(
        sino_corr,
        center=center,
        angles=angles_rad,
        apply_log=apply_log,
        filter_name=p.filter_name,
        gpu=p.gpu,
    )

    return recon.astype(np.float32), center


def process_sinogram_array(
    sino: np.ndarray,
    p: Params,
    center_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process a 2D sinogram array and return:
      (sino_corr, recon, meta)
    """
    if sino.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {sino.shape}")

    domain, apply_log = decide_domain_and_log(sino, p)

    sino_corr, method = correct_sinogram(sino, p, domain=domain, apply_log=apply_log)

    recon, center_used = reconstruct_fbp(
        sino_corr=sino_corr,
        sino_for_center=sino,
        p=p,
        apply_log=apply_log,
        center_override=center_override,
    )

    meta = {
        "domain": domain,
        "apply_log": apply_log,
        "correction": method,
        "center_used": center_used,
        "shape_sino": tuple(sino.shape),
        "shape_recon": tuple(recon.shape),
    }
    return sino_corr, recon, meta

def correct_sinogram_array(
    sino: np.ndarray,
    p: Params,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply only stripe/ring correction to a 2D sinogram array.
    Returns (sino_corr, meta). No reconstruction is performed.
    """
    if sino.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {sino.shape}")

    domain, apply_log = decide_domain_and_log(sino, p)
    sino_corr, method = correct_sinogram(sino, p, domain=domain, apply_log=apply_log)

    meta = {
        "domain": domain,
        "apply_log": apply_log,
        "correction": method,
        "shape_sino": tuple(sino.shape),
        "shape_sino_corr": tuple(sino_corr.shape),
    }
    return sino_corr, meta

def correct_file(
    input_path: str,
    sino_output_path: str,
    p: Params,
) -> Dict[str, Any]:
    """
    Load a sinogram image from disk, apply correction only, save corrected sinogram.
    Returns meta dict.
    """
    losa = _import_algotom_io()
    sino = losa.load_image(input_path)
    if sino.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram image, got shape {sino.shape}")

    if p.transpose:
        sino = sino.T

    sino_corr, meta = correct_sinogram_array(sino, p)
    losa.save_image(sino_output_path, sino_corr.astype(np.float32))
    logger.info("Saved corrected sinogram to %s", sino_output_path)
    return meta


def process_file(
    input_path: str,
    recon_output_path: str,
    p: Params,
    sino_output_path: Optional[str] = None,
    center_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load a sinogram image from disk, process it, and save outputs.
    Returns meta dict.
    """
    losa = _import_algotom_io()
    sino = losa.load_image(input_path)
    if sino.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram image, got shape {sino.shape}")

    if p.transpose:
        sino = sino.T

    sino_corr, recon, meta = process_sinogram_array(sino, p, center_override=center_override)


    if sino_output_path:
        losa.save_image(sino_output_path, sino_corr.astype(np.float32))
        logger.info("Saved processed sinogram to %s", sino_output_path)

    losa.save_image(recon_output_path, recon.astype(np.float32))
    logger.info("Reconstructed slice saved to %s", recon_output_path)

    return meta

