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
            for f in sorted(self.fig_dir.glob("*.png")):
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

        return tables

    def render_html(self) -> Path:
        """Render the final HTML report using Jinja2."""
        try:
            from jinja2 import Environment, FileSystemLoader, BaseLoader
        except ImportError:
            logger.error("Jinja2 not installed. Cannot generate HTML report.")
            return self._write_minimal_html()

        figures = self.collect_figures()
        tables = self.collect_tables()

        context = {
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
        template_path = self.output_base / "reports" / "templates" / "final_report.html.j2"
        if template_path.exists():
            env = Environment(loader=FileSystemLoader(str(template_path.parent)))
            template = env.get_template(template_path.name)
        else:
            # Use built-in template
            env = Environment(loader=BaseLoader())
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

_BUILTIN_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body { font-family: 'Georgia', serif; max-width: 1000px; margin: auto; padding: 30px; line-height: 1.6; color: #222; }
h1 { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #444; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 40px; }
h3 { color: #555; }
.subtitle { text-align: center; font-style: italic; color: #666; }
.meta { text-align: center; color: #888; margin-bottom: 30px; }
figure { margin: 20px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #ddd; padding: 5px; }
figcaption { font-style: italic; color: #666; margin-top: 5px; }
table.table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }
table.table th, table.table td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
table.table th { background: #f5f5f5; }
.equation { text-align: center; margin: 15px 0; }
.toc { background: #f9f9f9; padding: 15px; border: 1px solid #ddd; margin: 20px 0; }
.toc a { text-decoration: none; color: #336; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p class="subtitle">{{ subtitle }}</p>
<p class="meta">{{ author }} &mdash; {{ date }} &mdash; Campaign: {{ campaign_id }}</p>

<div class="toc">
<h3>Table of Contents</h3>
<ol>
<li><a href="#abstract">Abstract</a></li>
<li><a href="#rq">Research Question &amp; Hypothesis</a></li>
<li><a href="#equations">Governing Equations</a></li>
<li><a href="#geometry">Geometry &amp; Boundary Conditions</a></li>
<li><a href="#waveforms">Waveforms &amp; Parameter Matrix</a></li>
<li><a href="#numerical">Numerical Methods</a></li>
<li><a href="#verification">Verification Results</a></li>
<li><a href="#results-run">Results: Per-Run Examples</a></li>
<li><a href="#results-campaign">Results: Campaign Statistics</a></li>
<li><a href="#optimization">Optimization &amp; Pareto Front</a></li>
<li><a href="#limitations">Limitations &amp; Assumptions</a></li>
<li><a href="#future">Future Work</a></li>
<li><a href="#appendix">Appendix</a></li>
</ol>
</div>

<h2 id="abstract">1. Abstract</h2>
<p>This report presents results from a multiphysics simulation campaign studying
waveform-controlled PEDOT electropolymerization in a two-compartment salt-bridge
bipolar electrochemical system. The simulation couples diffusion-reaction transport,
electric field distribution, Nernst-Planck migration, and Butler-Volmer electrode
kinetics to predict how frequency, diffusion coefficient, and cell geometry
influence film morphology metrics including Anisotropy Index (AI) and
Uniformity Index (UI).</p>

<h2 id="rq">2. Research Question &amp; Hypothesis</h2>
<p><strong>Research Question:</strong> How do variations in waveform parameters, including
frequency, amplitude, and DC offset, influence the field-diffusion dynamics such as
electrolyte composition and ion diffusion in a modelled two-compartment salt-bridge
electropolymerization system and predict tendencies toward uniform or anisotropic
PEDOT growth for application in wearable biosensors?</p>
<p><strong>Hypothesis:</strong> Uniformity (low AI) is achieved when &tau;<sub>diff</sub> &lt;&lt; &tau;<sub>rxn</sub>
(high frequency, low viscosity). Anisotropy (high AI) emerges when &tau;<sub>diff</sub> &gt; &tau;<sub>rxn</sub>
(low frequency, high viscosity, strong DC offset).</p>

<h2 id="equations">3. Governing Equations</h2>

<h3>Stage A: Diffusion-Reaction</h3>
<div class="equation">
\[ \frac{\partial C}{\partial t} = D \nabla^2 C - R \]
</div>

<h3>Stage B: Laplace Equation for Electric Field</h3>
<div class="equation">
\[ \nabla^2 V = 0, \quad \mathbf{E} = -\nabla V \]
</div>
<p>Field-enhanced reaction: \( R = k_{rxn} \cdot C \cdot (1 + \beta |\mathbf{E}|) \)</p>

<h3>Stage C: Nernst-Planck Transport</h3>
<div class="equation">
\[ \frac{\partial C}{\partial t} = \nabla \cdot \left( D \nabla C + \frac{zFD}{RT} C \nabla V \right) - R \]
</div>

<h3>Stage D: Butler-Volmer Kinetics</h3>
<div class="equation">
\[ j = j_0 \frac{C}{C_{ref}} \left[ \exp\left(\frac{\alpha n F \eta}{RT}\right) - \exp\left(-\frac{(1-\alpha)nF\eta}{RT}\right) \right] \]
</div>
<p>Film growth: \( \frac{dh}{dt} = \frac{|j| M}{n F \rho} \)</p>

<h2 id="geometry">4. Geometry &amp; Boundary Conditions</h2>
<p>Two-compartment domain with central salt bridge. Left feeder at \(+V(t)/2\),
right feeder at \(-V(t)/2\). Floating bipolar electrode (BPE) in the bridge center.</p>

<h2 id="waveforms">5. Waveforms &amp; Parameter Matrix</h2>
<p>Applied potential: \( V(t) = V_{DC} + V_{AC} \sin(2\pi f t) \)</p>
<p>Factorial design: 3 frequencies &times; 3 D levels &times; 2 geometries &times; 3 replicates = {{ n_runs }} runs.</p>

<h2 id="numerical">6. Numerical Methods</h2>
<p>Finite-difference discretization on a structured 2D grid. Laplace equation solved
via SOR iteration. Transport via semi-implicit Euler with Crank-Nicolson option.
Migration treated via operator splitting.</p>

<h2 id="verification">7. Verification Results</h2>
{% for fig in figures if 'verification' in fig.path %}
<figure><img src="{{ fig.path }}"><figcaption>{{ fig.caption }}</figcaption></figure>
{% endfor %}

<h2 id="results-run">8. Results: Per-Run Examples</h2>
{% for fig in figures if 'timeseries' in fig.path or 'thickness' in fig.path or 'concentration' in fig.path %}
<figure><img src="{{ fig.path }}"><figcaption>{{ fig.caption }}</figcaption></figure>
{% endfor %}

<h2 id="results-campaign">9. Results: Campaign Statistics</h2>
{% if tables.results_summary %}
<h3>Summary Statistics</h3>
{{ tables.results_summary }}
{% endif %}
{% if tables.AI_anova %}
<h3>ANOVA: Anisotropy Index</h3>
{{ tables.AI_anova }}
{% endif %}
{% if tables.UI_anova %}
<h3>ANOVA: Uniformity Index</h3>
{{ tables.UI_anova }}
{% endif %}
{% if tables.descriptive_stats %}
<h3>Descriptive Statistics by Condition</h3>
{{ tables.descriptive_stats }}
{% endif %}
{% for fig in figures if 'ai_vs' in fig.path or 'ui_vs' in fig.path or 'heatmap' in fig.path or 'response' in fig.path or 'main_effects' in fig.path or 'sensitivity' in fig.path or 'scatter' in fig.path %}
<figure><img src="{{ fig.path }}"><figcaption>{{ fig.caption }}</figcaption></figure>
{% endfor %}

<h2 id="optimization">10. Optimization &amp; Pareto Front</h2>
{% if tables.best_conditions %}
<h3>Best Conditions</h3>
{{ tables.best_conditions }}
{% endif %}
{% for fig in figures if 'pareto' in fig.path %}
<figure><img src="{{ fig.path }}"><figcaption>{{ fig.caption }}</figcaption></figure>
{% endfor %}

<h2 id="limitations">11. Limitations &amp; Assumptions</h2>
<ul>
<li>2D domain approximation (no out-of-plane variation)</li>
<li>Simplified single-step EDOT oxidation kinetics</li>
<li>Constant electrolyte conductivity (no concentration-dependent conductivity)</li>
<li>No convective transport</li>
<li>Stochastic nucleation modeled as Poisson process (simplified from Subramanian &amp; Martin observations)</li>
<li>Film growth does not feed back into domain geometry</li>
</ul>

<h2 id="future">12. Future Work</h2>
<ul>
<li>3D domain extension for full morphology prediction</li>
<li>Multi-step reaction mechanism with oligomer intermediates</li>
<li>Concentration-dependent conductivity for more realistic migration</li>
<li>Experimental validation with CV/CA data</li>
<li>Machine learning surrogate models for rapid parameter exploration</li>
</ul>

<h2 id="appendix">13. Appendix</h2>
{% if tables.results_head %}
<h3>Run Results (First 20 Rows)</h3>
{{ tables.results_head }}
{% endif %}

<hr>
<p style="text-align:center;color:#999;font-size:0.9em;">
Generated automatically by pedot-multiphysics v0.1.0<br>
{{ date }}
</p>

</body>
</html>
"""
