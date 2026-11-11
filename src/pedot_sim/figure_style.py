"""Shared, exportable scientific figure style and offline galleries."""

from __future__ import annotations

import html
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_io import save_json

BLUE = "#2463A6"
ORANGE = "#D0782A"
TEAL = "#008577"
INK = "#17324D"
COLORS = [BLUE, ORANGE, TEAL, "#8565A8", "#C25264", "#607D8B", "#9A7600"]


@contextmanager
def scientific_style():
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": "#BBC8D4",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=COLORS),
            "grid.color": "#DCE5EC",
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    ):
        yield


def save_figure(fig, directory, name, question, caption, origin):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, -0.015, origin, fontsize=8, color="#466078", ha="left")
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return {
        "name": name,
        "png": f"{name}.png",
        "svg": f"{name}.svg",
        "question": question,
        "caption": caption,
        "origin": origin,
    }


def save_interactive(fig, directory, name, title, subtitle, scene=None):
    """Write offline HTML with a local JS bundle, plus inspectable plot data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial", "color": INK},
        title={
            "text": html.escape(title) + "<br><sup>" + html.escape(subtitle) + "</sup>",
            "x": 0.04,
            "font": {"size": 19},
        },
        margin={"l": 35, "r": 35, "t": 100, "b": 35},
        height=680,
    )
    if scene:
        fig.update_layout(scene=scene)
    fig.write_html(
        directory / f"{name}.html",
        include_plotlyjs="directory",
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 3},
        },
    )
    (directory / f"{name}.plotly.json").write_text(fig.to_json(), encoding="utf-8")
    return f"{name}.html"


def write_gallery(directory, title, subtitle, figures, notes, downloads=()):
    """Captions and caveats travel with all static/interactive representations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    esc = html.escape
    cards = []
    for figure in figures:
        interactive = figure.get("interactive")
        links = f'<a href="{esc(figure["png"], quote=True)}">PNG</a> <a href="{esc(figure["svg"], quote=True)}">SVG</a>'
        if interactive:
            links += f' <a href="{esc(interactive, quote=True)}">Explore in 3D</a>'
        cards.append(
            '<article><span class="badge">'
            + esc(figure["origin"])
            + "</span><h2>"
            + esc(figure["question"])
            + '</h2><a href="'
            + esc(figure["png"], quote=True)
            + '"><img loading="lazy" src="'
            + esc(figure["png"], quote=True)
            + '" alt="'
            + esc(figure["question"], quote=True)
            + '"></a><p>'
            + esc(figure["caption"])
            + "</p><nav>"
            + links
            + "</nav></article>"
        )
    assets = "".join(
        '<li><a href="' + esc(path, quote=True) + '">' + esc(label) + "</a></li>"
        for label, path in downloads
    )
    document = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>"""
        + esc(title)
        + """</title>
<style>body{margin:0;background:#F1F5F8;color:#17324D;font:16px/1.6 system-ui,sans-serif}
header,main,footer{max-width:1240px;margin:auto;padding:36px 28px}header{padding-bottom:20px}
h1{font-size:38px;line-height:1.18;letter-spacing:-1px;margin:8px 0 16px}h2{font-size:21px;line-height:1.35}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#008577}.lead{max-width:920px;color:#466078}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}
article{background:white;border:1px solid #DCE5EC;border-radius:14px;padding:25px;box-shadow:0 7px 22px #17324d08}
article img{width:100%;height:auto}article p{font-size:14px;color:#466078}.badge{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#008577}
a{color:#2463A6;text-decoration:none}a:hover{text-decoration:underline}nav{display:flex;gap:20px;font-size:13px;font-weight:600}
.notes{border-left:4px solid #D0782A;background:white;padding:14px 24px;margin-bottom:28px}.notes li{margin:6px 0}
footer{font-size:13px;color:#466078}@media(max-width:600px){h1{font-size:29px}header,main,footer{padding:22px 16px}}
</style></head><body><header><div class="eyebrow">ECOVOLT · RESEARCH EVIDENCE</div><h1>"""
        + esc(title)
        + """</h1>
<p class="lead">"""
        + esc(subtitle)
        + """</p></header><main><section class="notes"><strong>How to read these results</strong><ul>"""
        + "".join("<li>" + esc(n) + "</li>" for n in notes)
        + """</ul></section><section class="grid">"""
        + "".join(cards)
        + """</section>
<h2>Data and provenance</h2><ul>"""
        + assets
        + """</ul></main><footer>Keep the HTML and accompanying files together for offline use. Raw numerical values and metadata remain authoritative; visual interpolation does not create new observations.</footer></body></html>"""
    )
    (directory / "index.html").write_text(document, encoding="utf-8")
    save_json(
        {
            "title": title,
            "subtitle": subtitle,
            "figures": figures,
            "notes": notes,
            "downloads": [{"label": a, "path": b} for a, b in downloads],
        },
        directory / "figure_manifest.json",
    )
    return directory / "index.html"
