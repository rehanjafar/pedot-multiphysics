"""
Automated report generation (HTML + PDF).

Primary: Jinja2-based HTML report with MathJax for equations.
Optional: PDF via weasyprint or wkhtmltopdf.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils_io import ensure_dir, load_json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate a thesis-like report for a simulation campaign.

    Parameters
    ----------
    campaign_id : str
    output_base : Path
        Project root directory.
    """

    def __init__(self, campaign_id: str, output_base: Path) -> None:
        from .utils_io import validate_id
        validate_id(campaign_id)
        self.campaign_id = campaign_id
        self.output_base = Path(output_base)
        self.campaign_agg = self.output_base / "simulated_agg" / campaign_id
        self.report_dir = ensure_dir(self.output_base / "reports" / campaign_id)
        self.fig_dir = ensure_dir(self.report_dir / "figures")
        self.appendix_dir = ensure_dir(self.report_dir / "appendix")

    def collect_figures(self) -> list[dict]:
        """Gather all campaign-level figures."""
        figures = []
        if self.fig_dir.exists():
            for f in sorted(self.fig_dir.iterdir()):
                if f.suffix not in (".png", ".svg"):
                    continue
                rel = f.relative_to(self.report_dir)
                figures.append({"path": str(rel), "caption": f.stem.replace("_", " ").title()})
        return figures

    def collect_tables(self) -> dict:
        """Gather ANOVA, regression, and results tables."""
        tables = {}
        # Results summary
        results_path = self.campaign_agg / "results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            tables["results_summary"] = df.describe().to_html(classes="table")
            tables["results_head"] = df.head(20).to_html(classes="table", index=False)
            tables["n_runs"] = len(df)
        # ANOVA
        for dv in ["AI", "UI"]:
            anova_path = self.campaign_agg / f"{dv}_anova_table.csv"
            if anova_path.exists():
                tables[f"{dv}_anova"] = pd.read_csv(anova_path).to_html(classes="table", index=False)
        # Best conditions
        best_path = self.campaign_agg / "best_conditions.csv"
        if best_path.exists():
            tables["best_conditions"] = pd.read_csv(best_path).to_html(classes="table", index=False)
        # Descriptive stats
        desc_path = self.campaign_agg / "descriptive_stats.csv"
        if desc_path.exists():
            tables["descriptive_stats"] = pd.read_csv(desc_path).to_html(classes="table", index=False)

        from markupsafe import Markup
        return {key: Markup(value) if isinstance(value, str) else value for key, value in tables.items()}

    def render_html(self) -> Path:
        """Render the final HTML report using Jinja2."""
        try:
            from jinja2 import Environment, FileSystemLoader, BaseLoader
        except ImportError:
            logger.error("Jinja2 not installed. Cannot generate HTML report.")
            return self._write_minimal_html()

        figures = self.collect_figures()
        tables = self.collect_tables()

        raw = self.output_base / "simulated_raw" / self.campaign_id
        configs = [load_json(f) for f in sorted(raw.glob("*/params.json"))]
        context = {"run_configs": configs,
            "title": "Multiphysics Simulation of PEDOT Electropolymerization",
            "subtitle": "Waveform-Controlled Electropolymerization for Anisotropic Conductive Polymer Films",
            "author": "Rehan Jafar",
            "date": datetime.now().strftime("%B %d, %Y"),
            "campaign_id": self.campaign_id,
            "figures": figures,
            "tables": tables,
            "n_runs": tables.get("n_runs", "N/A"),
        }

        # Try loading template from file
        template_path = Path(__file__).parent / "templates" / "final_report.html.j2"
        if template_path.exists():
            env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=True)
            template = env.get_template(template_path.name)
        else:
            # Use built-in template
            env = Environment(loader=BaseLoader(), autoescape=True)
            template = env.from_string(_BUILTIN_HTML_TEMPLATE)

        html_content = template.render(**context)

        output_path = self.report_dir / "final_report.html"
        with open(output_path, "w") as f:
            f.write(html_content)

        # Copy figures to report directory (they should already be there)
        logger.info("HTML report saved to %s", output_path)
        return output_path

    def render_pdf(self, html_path: Optional[Path] = None) -> Optional[Path]:
        """Try to render PDF from HTML."""
        if html_path is None:
            html_path = self.report_dir / "final_report.html"
        if not html_path.exists():
            logger.warning("HTML report not found for PDF conversion")
            return None

        pdf_path = self.report_dir / "final_report.pdf"

        # Try weasyprint
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            logger.info("PDF report saved to %s", pdf_path)
            return pdf_path
        except ImportError:
            pass

        # Try wkhtmltopdf
        if shutil.which("wkhtmltopdf"):
            import subprocess
            try:
                subprocess.run(
                    ["wkhtmltopdf", str(html_path), str(pdf_path)],
                    check=True, capture_output=True,
                )
                logger.info("PDF report saved to %s (via wkhtmltopdf)", pdf_path)
                return pdf_path
            except subprocess.CalledProcessError as e:
                logger.warning("wkhtmltopdf failed: %s", e)

        logger.warning("PDF generation not available (install weasyprint or wkhtmltopdf)")
        return None

    def generate_full_report(self) -> Path:
        """Generate complete report (HTML + optional PDF)."""
        html_path = self.render_html()
        self.render_pdf(html_path)

        # Save config dump to appendix
        self._save_appendix()

        return html_path

    def _save_appendix(self) -> None:
        """Save appendix materials."""
        # Run manifest
        manifest_path = self.campaign_agg / "manifest.json"
        if manifest_path.exists():
            shutil.copy2(manifest_path, self.appendix_dir / "run_manifest.json")

        # Failed runs
        failed_path = self.campaign_agg / "failed_runs.csv"
        if failed_path.exists():
            shutil.copy2(failed_path, self.appendix_dir / "failed_runs.csv")

    def _write_minimal_html(self) -> Path:
        """Write a minimal HTML report without Jinja2."""
        output_path = self.report_dir / "final_report.html"
        figures = self.collect_figures()
        fig_html = "\n".join(
            f'<figure><img src="{f["path"]}" style="max-width:100%"><figcaption>{f["caption"]}</figcaption></figure>'
            for f in figures
        )
        html = f"""<!DOCTYPE html>
<html><head><title>PEDOT Simulation Report - {self.campaign_id}</title>
<style>body{{font-family:serif;max-width:900px;margin:auto;padding:20px}}
figure{{margin:20px 0;text-align:center}}
figcaption{{font-style:italic;margin-top:5px}}</style></head>
<body><h1>PEDOT Multiphysics Simulation Report</h1>
<p>Campaign: {self.campaign_id}</p>
<p>Generated: {datetime.now().isoformat()}</p>
<h2>Figures</h2>{fig_html}
<p><em>Install Jinja2 for a complete report.</em></p></body></html>"""
        with open(output_path, "w") as f:
            f.write(html)
        return output_path


# ---------------------------------------------------------------------------
# Built-in HTML template (used when reports/templates/final_report.html.j2
# does not exist)
# ---------------------------------------------------------------------------

_BUILTIN_HTML_TEMPLATE = '<!doctype html><html lang="en"><meta charset="utf-8"><title>{{ title }}</title>\n<body><h1>{{ title }}</h1><p>Campaign: {{ campaign_id }} — {{ date }}</p>\n<p>Research prototype. Results are uncalibrated and are not experimental validation.</p>\n<h2>Methods and assumptions</h2><p>Masked theta diffusion with local implicit reaction splitting (first order overall), insulated inactive faces and fixed feeder reservoirs. Migration uses conservative upwind face fluxes. Film growth uses consumed monomer at one effective exposed face per electrode cell.</p>\n<p>Stage A is an imposed diffusion/reaction surrogate. Stages B/C add phenomenological field enhancement. Stage D balances a phenomenological signed BV proxy with anodic-only deposition, under a prescribed solution field. No self-consistent electrolyte/electrode current coupling, calibrated counter-reaction, evolving geometry, convection or experimental calibration is included. Molar flux per depth is not measured current.</p>\n<h2>Actual run configurations</h2>{% for config in run_configs %}<pre>{{ config | tojson(indent=2) }}</pre>{% endfor %}\n<h2>Tables</h2>{% for key, table in tables.items() %}<h3>{{ key }}</h3>{{ table }}{% endfor %}\n<h2>Figures</h2>{% for fig in figures %}<figure><img src="{{ fig.path }}" alt="{{ fig.caption }}"><figcaption>{{ fig.caption }}</figcaption></figure>{% endfor %}</body></html>'
