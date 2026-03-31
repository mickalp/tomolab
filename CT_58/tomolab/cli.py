#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import logging


from .engine import Params, process_file, correct_file
from .batch import BatchSpec, build_file_list, process_batch, correct_batch
from .stack import StackSpec, process_tiff_stack, correct_tiff_stack
from .projections import ProjectionsToSinogramsSpec, build_sinogram_stack_from_projection_dir
from .projections import sinograms_to_projection_files




def add_algorithm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--transpose", action="store_true",
                   help="Transpose sinogram if stored as (detectors, projections).")

    p.add_argument("--mode", choices=["auto", "intensity", "log"], default="auto",
                   help="Sinogram domain: auto/intensity/log. (default: auto)")

    p.add_argument("--no-correction", dest="apply_correction", action="store_false", default=True,
                   help="Disable stripe correction (ON by default).")
    p.add_argument(
        "--correction",
        choices=["auto", "algotom", "repair", "filtering", "sorting", "wavelet_fft", "dead", "large"],
        default="algotom",
        help="Stripe/ring correction method. "
             "algotom=remove_all_stripe; filtering/sorting/wavelet_fft/dead/large use Algotom variants; "
             "repair=robust column repair."
    )


    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--la-size", type=int, default=51)
    p.add_argument("--sm-size", type=int, default=21)
    p.add_argument("--dim", type=int, default=1)

    p.add_argument("--repair-thresh", type=float, default=5.0)
    p.add_argument("--repair-max-cols", type=int, default=1000)
    


    p.add_argument("--center", type=float, default=None)
    p.add_argument("--angles-start", type=float, default=0.0)
    p.add_argument("--angles-end", type=float, default=360.0)
    p.add_argument("--filter", dest="filter_name", default="hann")
    p.add_argument("--gpu", action="store_true")

    p.add_argument("--no-log", action="store_true")
    p.add_argument("--force-log", action="store_true")
    
    # --- filtering ---
    p.add_argument("--filt-sigma", type=int, default=3)
    p.add_argument("--filt-size", type=int, default=21)
    p.add_argument("--filt-dim", type=int, default=1)
    p.add_argument("--filt-sort", action="store_true", default=True)
    p.add_argument("--no-filt-sort", dest="filt_sort", action="store_false")
    
    # --- sorting ---
    p.add_argument("--sort-size", type=int, default=21)
    p.add_argument("--sort-dim", type=int, default=1)
    
    # --- wavelet_fft ---
    p.add_argument("--wfft-level", type=int, default=5)
    p.add_argument("--wfft-size", type=int, default=1)
    p.add_argument("--wfft-wavelet-name", default="db9")
    p.add_argument("--wfft-window-name", choices=["gaussian", "butter"], default="gaussian")
    p.add_argument("--wfft-sort", action="store_true", default=False)
    
    # --- dead / large ---
    p.add_argument("--dead-snr", type=float, default=3.0)
    p.add_argument("--dead-size", type=int, default=51)
    p.add_argument("--dead-residual", action="store_true", default=True)
    p.add_argument("--no-dead-residual", dest="dead_residual", action="store_false")
    
    p.add_argument("--large-snr", type=float, default=3.0)
    p.add_argument("--large-size", type=int, default=51)
    p.add_argument("--large-drop-ratio", type=float, default=0.1)
    p.add_argument("--large-norm", action="store_true", default=True)
    p.add_argument("--no-large-norm", dest="large_norm", action="store_false")



def args_to_params(args: argparse.Namespace) -> Params:
    return Params(
        transpose=args.transpose,
        mode=args.mode,
        apply_correction=args.apply_correction,
        correction=args.correction,
        snr=args.snr,
        la_size=args.la_size,
        sm_size=args.sm_size,
        dim=args.dim,
        repair_thresh=args.repair_thresh,
        repair_max_cols=args.repair_max_cols,
        center=args.center,
        angles_start=args.angles_start,
        angles_end=args.angles_end,
        filter_name=args.filter_name,
        gpu=args.gpu,
        no_log=args.no_log,
        force_log=args.force_log,
        filt_sigma=args.filt_sigma,
        filt_size=args.filt_size,
        filt_dim=args.filt_dim,
        filt_sort=args.filt_sort,
        
        sort_size=args.sort_size,
        sort_dim=args.sort_dim,
        
        wfft_level=args.wfft_level,
        wfft_size=args.wfft_size,
        wfft_wavelet_name=args.wfft_wavelet_name,
        wfft_window_name=args.wfft_window_name,
        wfft_sort=args.wfft_sort,
        
        dead_snr=args.dead_snr,
        dead_size=args.dead_size,
        dead_residual=args.dead_residual,
        
        large_snr=args.large_snr,
        large_size=args.large_size,
        large_drop_ratio=args.large_drop_ratio,
        large_norm=args.large_norm

    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ring/stripe handling + FBP reconstruction (Algotom).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # ---- make sinogram stack from projections dir ----
    p_mk = sub.add_parser(
        "make-sino-stack",
        help="Build sinograms from a directory of projection TIFFs (default: multi-page stack).",
    )
    p_mk.add_argument("projections_dir", help="Directory with projection TIFFs (e.g. tomo_0001.tif ...)")
    p_mk.add_argument(
        "output",
        help=("Output path. Default writes a multi-page TIFF stack. "
              "If --separate is set, this is an output directory for per-sinogram TIFFs."),
    )
    p_mk.add_argument("--glob", dest="glob_pattern", default="tomo_*.tif",
                      help='Glob pattern for projections (default: "tomo_*.tif")')
    p_mk.add_argument(
    "--temp-dir",
    default=None,
    help="Directory for temporary memmap file (recommended on Windows, e.g. F:\\temp\\_tmp)",
)
    p_mk.add_argument("--recursive", action="store_true", help="Search subdirectories")
    p_mk.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_mk.add_argument("--dtype", default="float32", help='Output dtype (default: "float32")')
    p_mk.add_argument("--no-bigtiff", dest="bigtiff", action="store_false", default=True,
                      help="Disable BigTIFF when writing stack output")
    
    p_mk.add_argument("--separate", action="store_true",
                      help="Write sinograms as separate TIFF files instead of a multi-page stack.")
    p_mk.add_argument("--template", default="sino_{index:04d}.tif",
                      help='Filename template for --separate (default: "sino_{index:04d}.tif")')
    
    p_mk.add_argument("--verbose", action="store_true", help="Enable verbose logging")



    # ---- single ----
    p_single = sub.add_parser("single", help="Process a single sinogram file")
    p_single.add_argument("input", help="Input sinogram image (e.g. .tif)")
    p_single.add_argument("recon_output", help="Output reconstructed image (e.g. .tif)")
    p_single.add_argument("--sino-output", default=None,
                          help="Optional: save the corrected/processed sinogram to this path.")
    p_single.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    add_algorithm_args(p_single)
    p_csingle = sub.add_parser("correct-single", help="Correct a single sinogram (no reconstruction)")
    p_csingle.add_argument("input", help="Input sinogram image (e.g. .tif)")
    p_csingle.add_argument("sino_output", help="Output corrected sinogram image (e.g. .tif)")
    p_csingle.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    add_algorithm_args(p_csingle)
    
    # --- corrected sinograms to corrected projections ---
    
    p_sp = sub.add_parser(
        "sino-to-proj",
        help="Convert corrected sinograms -> corrected projection TIFF files (no reconstruction).",
    )
    p_sp.add_argument("input", help="Input sinograms: a stack TIFF (mode=stack) OR a folder (mode=files)")
    p_sp.add_argument("output_dir", help="Output directory for corrected projection TIFF files")
    p_sp.add_argument("--mode", choices=["stack", "files"], default="stack",
                      help="Input mode: stack TIFF (default) or directory of sinogram files")
    p_sp.add_argument("--template", default="tomo_{index:04d}.tif",
                      help='Output filename template (default: "tomo_{index:04d}.tif")')
    p_sp.add_argument("--overwrite", action="store_true", help="Overwrite outputs if exist")
    p_sp.add_argument("--temp-dir", default=None,
                      help="Directory for temporary memmap file (recommended on Windows)")


    # ---- batch ----
    p_batch = sub.add_parser("batch", help="Process a directory of sinogram files")
    p_batch.add_argument("input_dir", help="Input directory containing sinogram images")
    p_batch.add_argument("output_dir", help="Output directory for results")

    p_batch.add_argument("--recursive", action="store_true", help="Search subdirectories")
    p_batch.add_argument("--glob", dest="glob_pattern", default=None,
                         help='Glob pattern (e.g. "*.tif"). If set, overrides extensions.')
    p_batch.add_argument("--ext", nargs="*", default=[".tif", ".tiff"],
                         help='Extensions to include if --glob is not used (default: .tif .tiff)')
    p_cbatch = sub.add_parser("correct-batch", help="Correct a directory of sinograms (no reconstruction)")
    
    # --- srguments for sinograms ---
    p_cbatch.add_argument("input_dir", help="Input directory containing sinogram images")
    p_cbatch.add_argument("output_dir", help="Output directory for corrected sinograms")
    
    p_cbatch.add_argument("--recursive", action="store_true", help="Search subdirectories")
    p_cbatch.add_argument("--glob", dest="glob_pattern", default=None,
                          help='Glob pattern (e.g. "*.tif"). If set, overrides extensions.')
    p_cbatch.add_argument("--ext", nargs="*", default=[".tif", ".tiff"],
                          help='Extensions to include if --glob is not used (default: .tif .tiff)')
    
    p_cbatch.add_argument("--include-regex", default=None)
    p_cbatch.add_argument("--exclude-regex", default=None)
    p_cbatch.add_argument("--contains", default=None)
    p_cbatch.add_argument("--not-contains", default=None)
    
    p_cbatch.add_argument("--start", type=int, default=None)
    p_cbatch.add_argument("--end", type=int, default=None)
    p_cbatch.add_argument("--step", type=int, default=1)
    p_cbatch.add_argument("--indices", default=None)
    
    p_cbatch.add_argument("--overwrite", action="store_true")
    p_cbatch.add_argument("--sino-suffix", default="_sino_corr")
    p_cbatch.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    add_algorithm_args(p_cbatch)
    
    # --- end of argumentf for sinograms ---
    

    # NEW: center handling in batch
    p_batch.add_argument("--center-mode", choices=["once", "each"], default="once",
                         help="Center handling for batch: once=compute once and reuse; each=compute per file")

    p_batch.add_argument("--include-regex", default=None)
    p_batch.add_argument("--exclude-regex", default=None)
    p_batch.add_argument("--contains", default=None)
    p_batch.add_argument("--not-contains", default=None)

    p_batch.add_argument("--start", type=int, default=None, help="Slice start index (inclusive)")
    p_batch.add_argument("--end", type=int, default=None, help="Slice end index (exclusive)")
    p_batch.add_argument("--step", type=int, default=1, help="Slice step (default: 1)")
    p_batch.add_argument("--indices", default=None,
                         help="Comma-separated explicit indices (overrides start/end), e.g. 0,5,7")

    p_batch.add_argument("--save-sino", action="store_true", help="Also save corrected sinograms")
    p_batch.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_batch.add_argument("--recon-suffix", default="_recon")
    p_batch.add_argument("--sino-suffix", default="_sino_corr")

    p_batch.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    add_algorithm_args(p_batch)

    # ---- stack ----
    p_stack = sub.add_parser("stack", help="Process a multi-page TIFF stack (one sinogram per page)")
    p_stack.add_argument("input_tiff", help="Input multi-page TIFF (sinogram per page)")
    p_stack.add_argument("output_recon_tiff", help="Output multi-page TIFF (recon per page)")

    p_stack.add_argument("--output-sino-tiff", default=None,
                         help="Optional output multi-page TIFF for corrected sinograms")

    p_stack.add_argument("--start", type=int, default=None, help="First page index (inclusive)")
    p_stack.add_argument("--end", type=int, default=None, help="Last page index (exclusive)")
    p_stack.add_argument("--step", type=int, default=1, help="Step (default: 1)")
    p_stack.add_argument("--indices", default=None,
                         help="Comma-separated explicit page indices, e.g. 0,5,7")

    # NEW: center handling in stack
    p_stack.add_argument("--center-mode", choices=["once", "each"], default="once",
                         help="Center handling for stack: once=compute once and reuse; each=compute per page")

    p_stack.add_argument("--overwrite", action="store_true", help="Overwrite existing output TIFF(s)")
    p_stack.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    add_algorithm_args(p_stack)
    
    # ---arguments for sinograms stack ---
    
    p_cstack = sub.add_parser("correct-stack", help="Correct a multi-page TIFF stack (no reconstruction)")
    p_cstack.add_argument("input_tiff", help="Input multi-page TIFF (sinogram per page)")
    p_cstack.add_argument("output_sino_tiff", help="Output multi-page TIFF (corrected sinogram per page)")
    
    p_cstack.add_argument("--start", type=int, default=None)
    p_cstack.add_argument("--end", type=int, default=None)
    p_cstack.add_argument("--step", type=int, default=1)
    p_cstack.add_argument("--indices", default=None)
    
    p_cstack.add_argument("--overwrite", action="store_true")
    p_cstack.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p_cstack.add_argument("--workers", type=int, default=12,
                      help="Number of processes for correction (0=auto)")
    
    add_algorithm_args(p_cstack)


    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    log = logging.getLogger(__name__)

    if args.cmd == "single":
        p = args_to_params(args)
        meta = process_file(
            input_path=args.input,
            recon_output_path=args.recon_output,
            p=p,
            sino_output_path=args.sino_output,
        )
        log.info("Done. domain=%s apply_log=%s correction=%s center=%g",
                 meta["domain"], meta["apply_log"], meta["correction"], meta["center_used"])
        return

    if args.cmd == "sino-to-proj":
        meta = sinograms_to_projection_files(
            input_mode=args.mode,
            input_path=args.input,
            output_dir=args.output_dir,
            projection_template=args.template,
            overwrite=args.overwrite,
            temp_dir=args.temp_dir,
        )
        log.info("Done: wrote %d projections to %s", meta["n_projections"], meta["output_dir"])
        return
    
    
    if args.cmd == "make-sino-stack":
        output_mode = "files" if args.separate else "stack"
        spec = ProjectionsToSinogramsSpec(
            projections_dir=args.projections_dir,
            output_sinogram_stack_tiff=args.output,  # used for stack, or as fallback dir in files mode
            output_mode=output_mode,
            output_sinogram_dir=(args.output if args.separate else None),
            filename_template=args.template,
            glob_pattern=args.glob_pattern,
            recursive=args.recursive,
            dtype=args.dtype,
            overwrite=args.overwrite,
            bigtiff=args.bigtiff,
            temp_dir=args.temp_dir,
        )
        meta = build_sinogram_stack_from_projection_dir(spec)
        log.info("Done: mode=%s output=%s pages=%d page_shape=%s",
                 meta["output_mode"], meta["output"], meta["sinogram_pages"], meta["sinogram_shape_per_page"])
        return



    if args.cmd == "batch":
        p = args_to_params(args)

        indices = None
        if args.indices:
            indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]

        spec = BatchSpec(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            extensions=tuple(args.ext),
            recursive=args.recursive,
            glob_pattern=args.glob_pattern,

            center_mode=args.center_mode,  # IMPORTANT for speed up

            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            contains=args.contains,
            not_contains=args.not_contains,
            start=args.start,
            end=args.end,
            step=args.step,
            indices=indices,

            save_sino=args.save_sino,
            overwrite=args.overwrite,
            recon_suffix=args.recon_suffix,
            sino_suffix=args.sino_suffix,
        )

        files = build_file_list(spec)
        log.info("Selected %d files.", len(files))
        if len(files) == 0:
            log.warning("No files selected; nothing to do.")
            return

        def progress(done, total, path, meta):
            if meta.get("skipped"):
                log.info("[%d/%d] SKIP %s (%s)", done, total, path.name, meta.get("reason"))
            elif "error" in meta:
                log.error("[%d/%d] FAIL %s (%s)", done, total, path.name, meta["error"])
            else:
                log.info("[%d/%d] OK   %s (center=%s)", done, total, path.name, meta.get("center_used"))

        summary = process_batch(files, params=p, spec=spec, on_progress=progress)
        log.info("Batch finished. ok=%d skipped=%d fail=%d out=%s",
                 len(summary["successes"]), len(summary["skipped"]), len(summary["failures"]), summary["output_dir"])
        return

    if args.cmd == "stack":
        p = args_to_params(args)

        indices = None
        if args.indices:
            indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]

        spec = StackSpec(
            input_tiff=args.input_tiff,
            output_recon_tiff=args.output_recon_tiff,
            output_sino_tiff=args.output_sino_tiff,

            center_mode=args.center_mode,  # IMPORTANT

            start=args.start,
            end=args.end,
            step=args.step,
            indices=indices,
            overwrite=args.overwrite,
        )

        def progress(done, total, page_idx, meta):
            if "error" in meta:
                log.error("[%d/%d] FAIL page=%d (%s)", done, total, page_idx, meta["error"])
            else:
                log.info("[%d/%d] OK   page=%d (center=%s)", done, total, page_idx, meta.get("center_used"))

        summary = process_tiff_stack(spec=spec, params=p, on_progress=progress)
        log.info("Stack finished. selected=%d processed=%d fail=%d out=%s",
                 summary["selected_pages"], summary["processed"], len(summary["failures"]), summary["output_recon_tiff"])
        return
    
    if args.cmd == "correct-single":
        p = args_to_params(args)
        meta = correct_file(
            input_path=args.input,
            sino_output_path=args.sino_output,
            p=p,
        )
        log.info("Correct-only done. domain=%s apply_log=%s correction=%s",
                 meta["domain"], meta["apply_log"], meta["correction"])
        return

    if args.cmd == "correct-batch":
        p = args_to_params(args)

        indices = None
        if args.indices:
            indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]

        spec = BatchSpec(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            extensions=tuple(args.ext),
            recursive=args.recursive,
            glob_pattern=args.glob_pattern,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            contains=args.contains,
            not_contains=args.not_contains,
            start=args.start,
            end=args.end,
            step=args.step,
            indices=indices,
            overwrite=args.overwrite,
            sino_suffix=args.sino_suffix,
        )

        files = build_file_list(spec)
        log.info("Selected %d files.", len(files))
        if len(files) == 0:
            log.warning("No files selected; nothing to do.")
            return

        def progress(done, total, path, meta):
            if meta.get("skipped"):
                log.info("[%d/%d] SKIP %s (%s)", done, total, path.name, meta.get("reason"))
            elif "error" in meta:
                log.error("[%d/%d] FAIL %s (%s)", done, total, path.name, meta["error"])
            else:
                log.info("[%d/%d] OK   %s", done, total, path.name)

        summary = correct_batch(files, params=p, spec=spec, on_progress=progress)
        log.info("Correction batch finished. ok=%d skipped=%d fail=%d out=%s",
                 len(summary["successes"]), len(summary["skipped"]), len(summary["failures"]), summary["output_dir"])
        return


    if args.cmd == "correct-stack":
        p = args_to_params(args)

        indices = None
        if args.indices:
            indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]

        def progress(done, total, page_idx, meta):
            if "error" in meta:
                log.error("[%d/%d] FAIL page=%d (%s)", done, total, page_idx, meta["error"])
            else:
                log.info("[%d/%d] OK   page=%d", done, total, page_idx)

        summary = correct_tiff_stack(
            input_tiff=args.input_tiff,
            output_sino_tiff=args.output_sino_tiff,
            params=p,
            start=args.start,
            end=args.end,
            step=args.step,
            indices=indices,
            overwrite=args.overwrite,
            workers=args.workers,
            on_progress=progress,
        )
        log.info("Stack correction finished. selected=%d processed=%d fail=%d out=%s",
                 summary["selected_pages"], summary["processed"], len(summary["failures"]), summary["output_sino_tiff"])
        return



if __name__ == "__main__":
    main()
