"""
dashboard.py
------------
Framework-neutral dashboard prototype: every function returns a Plotly Figure, so the same code can be
dropped into Streamlit (`st.plotly_chart(fig)`), Dash (`dcc.Graph(figure=fig)`) or exported to one
standalone HTML page with `figures_to_html`.

Inputs are the DataFrames produced by `reporting.build_monthly_report` (plus optional forecast and
country tables). Nothing here fetches data from the internet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise ImportError("plotly is required for the dashboard: pip install plotly") from exc


def fig_volume(volume: pd.DataFrame) -> "go.Figure":
    """Postings per complete day by month; partial months are drawn hatched and labelled."""
    v = volume.copy()
    v["label"] = v["month"] + " (" + v["status"] + ", " + v["complete_days"].astype(str) + " complete days)"
    fig = px.bar(v, x="label", y="postings_per_day", text_auto=".0f",
                 title="Postings per complete day, by month (partial months are provisional)",
                 labels={"label": "Month (coverage)", "postings_per_day": "Postings per complete day"})
    fig.update_traces(marker_pattern_shape=["/" if s != "complete" else "" for s in v["status"]])
    return fig


def fig_category_share(categories: pd.DataFrame, top_n: int = 12) -> "go.Figure":
    order = categories.groupby("category")["postings"].sum().nlargest(top_n).index
    d = categories[categories["category"].isin(order)]
    return px.bar(d, x="share_pct", y="category", color="month", barmode="group", orientation="h",
                  title="Category share of postings (derived from title keywords)",
                  labels={"share_pct": "Share of postings (%)", "category": ""},
                  category_orders={"category": list(order)})


def fig_comparison(comp: pd.DataFrame, label_col: str, title: str) -> "go.Figure":
    """Horizontal bars of % change latest vs previous month (provisional if either month is partial)."""
    d = comp.dropna(subset=["pct_change"]).sort_values("pct_change")
    fig = px.bar(d, x="pct_change", y=label_col, orientation="h", title=title,
                 labels={"pct_change": "% change in postings per day", label_col: ""})
    fig.add_vline(x=0, line_width=1)
    return fig


def fig_country_map(country_hourly: pd.DataFrame) -> "go.Figure":
    """Choropleth of median offered hourly rate (needs columns country, country_iso3, median_rate, n)."""
    return px.choropleth(country_hourly, locations="country_iso3", color="median_rate",
                         hover_name="country", hover_data={"n": True, "country_iso3": False},
                         color_continuous_scale="Viridis",
                         title="Median offered hourly rate (USD) by client country - hourly postings with a stated range",
                         labels={"median_rate": "USD / hour"})


def fig_forecast(history: pd.Series, forecast: pd.DataFrame) -> "go.Figure":
    """history: daily counts (index=date). forecast: columns baseline, lower, upper (index=date)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", name="80% empirical interval"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["baseline"], mode="lines+markers", name="Forecast"))
    fig.update_layout(title="Daily postings: actual and forecast", xaxis_title="Date", yaxis_title="Postings per day")
    return fig


def build_figures(report: Dict[str, pd.DataFrame], country_hourly: Optional[pd.DataFrame] = None,
                  history: Optional[pd.Series] = None, forecast: Optional[pd.DataFrame] = None) -> Dict[str, "go.Figure"]:
    figs = {"volume": fig_volume(report["volume"]), "category_share": fig_category_share(report["categories"])}
    if "comparison_categories" in report:
        figs["category_change"] = fig_comparison(report["comparison_categories"], "category",
                                                 "Category demand: latest vs previous month")
        figs["country_change"] = fig_comparison(report["comparison_countries"], "country",
                                                "Country demand: latest vs previous month")
    if country_hourly is not None and len(country_hourly):
        figs["hourly_map"] = fig_country_map(country_hourly)
    if history is not None and forecast is not None:
        figs["forecast"] = fig_forecast(history, forecast)
    return figs


def figures_to_html(figs: Dict[str, "go.Figure"], path: str | Path, title: str = "Job market dashboard (prototype)",
                    note: str = "", inline_js: bool = False) -> Path:
    """Write all figures into ONE standalone HTML file (plotly.js from CDN unless inline_js=True)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head>"
             f"<body style='font-family:sans-serif;max-width:1100px;margin:auto'><h1>{title}</h1><p>{note}</p>"]
    for i, fig in enumerate(figs.values()):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if inline_js else "cdn") if i == 0 else False))
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
