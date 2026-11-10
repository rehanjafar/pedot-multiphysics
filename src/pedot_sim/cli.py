"""
Command-line interface for the PEDOT multiphysics simulation.

Entry-point (installed via pyproject.toml console_scripts)::

    pedot-sim run-single  --config PATH --run-id ID  [--campaign-id CID]
    pedot-sim run-batch   --config PATH --campaign-id CID [--max-workers N]
    pedot-sim verify      --config PATH --campaign-id CID
    pedot-sim analyze     --campaign-id CID [--output-base PATH]
    pedot-sim make-report --campaign-id CID [--output-base PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("pedot_sim")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbosity: int = 0) -> None:
    """Configure root and package loggers for console output."""
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO

    fmt = "%(asctime)s [%(levelname)-5.5s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> "SimConfig":
    """Load a SimConfig from a JSON file (or YAML if pyyaml is available)."""
    from pedot_sim.config import SimConfig, load_config

    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_config(path)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            print(
                "ERROR: PyYAML is required to load .yaml config files.  "
                "Install with:  pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        # Reconstruct from nested dict
        from pedot_sim.config import (
            BatchConfig,
            GeometryConfig,
            NumericalConfig,
            NucleationConfig,
            OutputConfig,
            PhysicsConfig,
            WaveformConfig,
        )
        return SimConfig(
            geometry=GeometryConfig(**data.get("geometry", {})),
            waveform=WaveformConfig(**data.get("waveform", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
            numerical=NumericalConfig(**data.get("numerical", {})),
            nucleation=NucleationConfig(**data.get("nucleation", {})),
            output=OutputConfig(**data.get("output", {})),
            batch=BatchConfig(**data.get("batch", {})),
        )
    else:
        print(f"ERROR: Unsupported config file format: {suffix}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_run_single(args: argparse.Namespace) -> int:
    """Execute a single simulation run."""
    from pedot_sim.runner_single import run_single

    config = _load_config(args.config)
    output_base = Path(args.output_base) if args.output_base else Path("data")

    print(f"Starting single run: {args.run_id}")
    print(f"  Campaign : {args.campaign_id}")
    print(f"  Config   : {args.config}")
    print(f"  Output   : {output_base.resolve()}")
    print()

    t0 = time.perf_counter()
    metrics = run_single(
        config=config,
        run_id=args.run_id,
        output_base=output_base,
        campaign_id=args.campaign_id,
    )
    elapsed = time.perf_counter() - t0

    status = metrics.get("status", "completed")
    if status == "failed":
        print(f"\nRun FAILED after {elapsed:.1f}s: {metrics.get('error', 'unknown')}")
        return 1

    print(f"\nRun completed in {elapsed:.1f}s")
    print(f"  Mean thickness  : {metrics.get('mean_h', 0):.4e} m")
    print(f"  Max thickness   : {metrics.get('max_h', 0):.4e} m")
    print(f"  Asymmetry idx   : {metrics.get('AI', 0):.4f}")
    print(f"  Uniformity idx  : {metrics.get('UI', 0):.4f}")
    print(f"  Depletion depth : {metrics.get('depletion_depth', 0):.4e} m")
    print(f"  Timescale ratio : {metrics.get('timescale_ratio', 0):.2f}")
    return 0


def _cmd_run_batch(args: argparse.Namespace) -> int:
    """Execute a batch campaign."""
    from pedot_sim.runner_batch import run_batch

    config = _load_config(args.config)
    output_base = Path(args.output_base) if args.output_base else Path("data")
    sweep = getattr(args, "sweep", "factorial")

    print(f"Starting batch campaign: {args.campaign_id}")
    print(f"  Sweep      : {sweep}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Config     : {args.config}")
    print(f"  Output     : {output_base.resolve()}")
    print()

    t0 = time.perf_counter()
    df = run_batch(
        config=config,
        campaign_id=args.campaign_id,
        output_base=output_base,
        max_workers=args.max_workers,
        sweep=sweep,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nBatch completed in {elapsed:.1f}s")
    print(f"  Total runs aggregated: {len(df)}")
    if len(df) > 0:
        if "AI" in df.columns:
            print(f"  Mean AI : {df['AI'].astype(float).mean():.4f}")
        if "UI" in df.columns:
            print(f"  Mean UI : {df['UI'].astype(float).mean():.4f}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Run numerical verification (mesh + timestep refinement) and sanity checks."""
    from pedot_sim.runner_batch import aggregate_results

    config = _load_config(args.config)
    output_base = Path(args.output_base) if args.output_base else Path("data")

    print(f"Verifying campaign: {args.campaign_id}")
    print()

    # Run numerical verification suite
    try:
        from pedot_sim.verification import run_full_verification
        print("Running mesh & timestep refinement studies...")
        run_full_verification(config, args.campaign_id, output_base)
        print("Numerical verification complete.\n")
    except Exception as e:
        print(f"Numerical verification failed: {e}\n")
        return 1

    df = aggregate_results(args.campaign_id, output_base)

    if len(df) == 0:
        print("WARNING: No completed runs found.")
        return 1

    # ---- Sanity checks ---------------------------------------------------
    issues: List[str] = []

    key_cols = ["AI", "UI", "mean_h", "depletion_depth"]
    for col in key_cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                issues.append(f"{col} has {n_nan} NaN values")

    if "mean_h" in df.columns:
        n_neg = (df["mean_h"].astype(float) < 0).sum()
        if n_neg > 0:
            issues.append(f"{n_neg} runs have negative mean thickness")

    if "AI" in df.columns:
        n_neg_ai = (df["AI"].astype(float) < 0).sum()
        if n_neg_ai > 0:
            issues.append(f"{n_neg_ai} runs have negative asymmetry index")

    if "UI" in df.columns:
        vals = df["UI"].astype(float)
        n_bad_ui = (vals < -0.01).sum()
        if n_bad_ui > 0:
            issues.append(f"{n_bad_ui} runs have negative uniformity index")

    # Check run directory count
    raw_dir = output_base / "simulated_raw" / args.campaign_id
    n_dirs = 0
    if raw_dir.exists():
        n_dirs = sum(1 for d in raw_dir.iterdir() if d.is_dir())

    print(f"  Run directories found : {n_dirs}")
    print(f"  Completed runs        : {len(df)}")
    print()

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("All verification checks passed.")
        return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run full statistical analysis + optimization for a campaign."""
    from pedot_sim.runner_batch import aggregate_results

    output_base = Path(args.output_base)

    print(f"Analyzing campaign: {args.campaign_id}")

    df = aggregate_results(args.campaign_id, output_base)

    if len(df) == 0:
        print("No completed runs found.")
        return 1

    print(f"\nAggregated {len(df)} runs")

    # Run full statistical analysis
    try:
        from pedot_sim.analysis_stats import run_full_analysis
        print("Running ANOVA + regression analysis...")
        run_full_analysis(args.campaign_id, output_base)
        print("Statistical analysis complete.")
    except Exception as e:
        print(f"Statistical analysis failed: {e}")
        return 1

    # Run optimization analysis
    try:
        from pedot_sim.analysis_opt import run_optimization_analysis
        print("Running optimization / Pareto analysis...")
        run_optimization_analysis(args.campaign_id, output_base)
        print("Optimization analysis complete.")
    except Exception as e:
        print(f"Optimization analysis failed: {e}")
        return 1

    results_path = output_base / "simulated_agg" / args.campaign_id / "results.csv"
    print(f"\nResults saved to: {results_path}")
    return 0


def _cmd_make_report(args: argparse.Namespace) -> int:
    """Generate a comprehensive HTML (+PDF) report for a campaign."""
    output_base = Path(args.output_base)
    agg_dir = output_base / "simulated_agg" / args.campaign_id
    results_path = agg_dir / "results.csv"

    if not results_path.exists():
        print(f"ERROR: No results.csv found at {results_path}")
        print("Run 'pedot-sim analyze' first.")
        return 1

    print(f"Generating report for campaign: {args.campaign_id}")

    # Use the full reporting module
    try:
        from pedot_sim.reporting import ReportGenerator
        generator = ReportGenerator(args.campaign_id, output_base)
        html_path = generator.generate_full_report()
        print(f"\nHTML report: {html_path}")
        pdf_path = generator.report_dir / "final_report.pdf"
        if pdf_path.exists():
            print(f"PDF report:  {pdf_path}")
        return 0
    except Exception as e:
        logger.warning("Full report generation failed, falling back to minimal: %s", e)

    # Fallback to minimal report
    import pandas as pd
    df = pd.read_csv(results_path)

    try:
        import jinja2
    except ImportError:
        print("ERROR: jinja2 is required for report generation.", file=sys.stderr)
        return 1

    template = jinja2.Environment(autoescape=True).from_string(_MINIMAL_REPORT_TEMPLATE)

    summary_html = df.describe().to_html() if len(df) > 0 else "<p>No data</p>"

    html = template.render(
        campaign_id=args.campaign_id,
        n_runs=len(df),
        summary=summary_html,
        columns=list(df.columns),
        data=df.to_dict(orient="records"),
    )

    from pedot_sim.utils_io import ensure_dir
    fallback_dir = ensure_dir(output_base / "reports" / args.campaign_id)
    report_path = fallback_dir / f"report_{args.campaign_id}.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to: {report_path}")
    return 0


_MINIMAL_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>PEDOT Simulation Report - {{ campaign_id }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               Roboto, Helvetica, Arial, sans-serif;
               max-width: 1100px; margin: 2em auto; padding: 0 1em;
               color: #222; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0;
                font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
        th { background: #f5f5f5; font-weight: 600; }
        h1 { color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 0.3em; }
        h2 { color: #555; }
        .meta { color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>PEDOT Simulation Report: {{ campaign_id }}</h1>
    <p class="meta">Total runs: {{ n_runs }}</p>
    <h2>Summary Statistics</h2>
    {{ summary | safe }}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pedot-sim",
        description="PEDOT multiphysics electropolymerization simulation CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run-single ------------------------------------------------------
    p_single = subparsers.add_parser(
        "run-single",
        help="Execute a single simulation run",
    )
    p_single.add_argument(
        "--config", type=Path, required=True,
        help="Path to simulation config file (JSON or YAML)",
    )
    p_single.add_argument(
        "--run-id", type=str, required=True,
        help="Unique identifier for this run",
    )
    p_single.add_argument(
        "--campaign-id", type=str, default="default",
        help="Campaign grouping identifier (default: 'default')",
    )
    p_single.add_argument(
        "--output-base", type=str, default=None,
        help="Root output directory (default: data)",
    )

    # ---- run-batch -------------------------------------------------------
    p_batch = subparsers.add_parser(
        "run-batch",
        help="Execute a batch campaign of simulation runs",
    )
    p_batch.add_argument(
        "--config", type=Path, required=True,
        help="Path to simulation config file (JSON or YAML)",
    )
    p_batch.add_argument(
        "--campaign-id", type=str, required=True,
        help="Campaign identifier",
    )
    p_batch.add_argument(
        "--max-workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    p_batch.add_argument(
        "--sweep", type=str, default="factorial",
        choices=["factorial", "dense"],
        help="Sweep type (default: factorial)",
    )
    p_batch.add_argument(
        "--output-base", type=str, default=None,
        help="Root output directory (default: data)",
    )

    # ---- verify ----------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Run verification checks on a completed campaign",
    )
    p_verify.add_argument(
        "--config", type=Path, required=True,
        help="Path to simulation config file",
    )
    p_verify.add_argument(
        "--campaign-id", type=str, required=True,
        help="Campaign identifier to verify",
    )
    p_verify.add_argument(
        "--output-base", type=str, default=None,
        help="Root output directory (default: data)",
    )

    # ---- analyze ---------------------------------------------------------
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Re-aggregate results for a campaign",
    )
    p_analyze.add_argument(
        "--campaign-id", type=str, required=True,
        help="Campaign identifier",
    )
    p_analyze.add_argument(
        "--output-base", type=str, default="data",
        help="Root output directory (default: data)",
    )

    # ---- make-report -----------------------------------------------------
    p_report = subparsers.add_parser(
        "make-report",
        help="Generate an HTML report for a campaign",
    )
    p_report.add_argument(
        "--campaign-id", type=str, required=True,
        help="Campaign identifier",
    )
    p_report.add_argument(
        "--output-base", type=str, default="data",
        help="Root output directory (default: data)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_COMMAND_DISPATCH = {
    "run-single": _cmd_run_single,
    "run-batch": _cmd_run_batch,
    "verify": _cmd_verify,
    "analyze": _cmd_analyze,
    "make-report": _cmd_make_report,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.  Installed as ``pedot-sim`` console script."""
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    verbosity = [a for a in argv if a in ("-v", "-vv", "--verbose")]
    argv = verbosity + [a for a in argv if a not in ("-v", "-vv", "--verbose")]
    args = parser.parse_args(argv)

    if hasattr(args, "campaign_id"):
        from .utils_io import validate_id
        try:
            validate_id(args.campaign_id)
        except ValueError as exc:
            parser.error(str(exc))
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = _COMMAND_DISPATCH.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.debug("Unhandled exception", exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
