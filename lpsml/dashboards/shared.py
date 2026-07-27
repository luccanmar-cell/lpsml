from __future__ import annotations

"""Shared visual and data helpers for the Streamlit dashboards."""

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


LPS_RED = "#A51E36"
LPS_RED_DARK = "#751426"
LPS_GOLD = "#E7AA2F"
LPS_SLATE = "#263746"
LPS_MUTED = "#66727D"
LPS_SURFACE = "#FFFFFF"
LPS_BACKGROUND = "#F4F5F6"
LPS_BORDER = "#D6DCE0"


def configure_dashboard(page_title: str) -> None:
    """Apply a restrained LPS-inspired visual system to a Streamlit page."""
    st.set_page_config(page_title=page_title, layout="wide")
    st.markdown(
        f"""
        <style>
        :root {{
            --lps-red: {LPS_RED};
            --lps-red-dark: {LPS_RED_DARK};
            --lps-gold: {LPS_GOLD};
            --lps-slate: {LPS_SLATE};
            --lps-muted: {LPS_MUTED};
            --lps-surface: {LPS_SURFACE};
            --lps-background: {LPS_BACKGROUND};
        }}
        .stApp {{
            background: var(--lps-background);
            color: var(--lps-slate);
        }}
        [data-testid="stAppViewContainer"] > .main {{
            background: var(--lps-background);
        }}
        .block-container {{
            max-width: 1500px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }}
        [data-testid="stSidebar"] {{
            background: #E9EDF0;
            border-right: 1px solid {LPS_BORDER};
        }}
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: var(--lps-red-dark);
        }}
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p {{
            color: var(--lps-slate);
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
            color: var(--lps-muted) !important;
        }}
        a {{
            color: var(--lps-red-dark);
        }}
        .lps-hero {{
            background: linear-gradient(120deg, var(--lps-red-dark), var(--lps-red));
            border-bottom: 5px solid var(--lps-gold);
            border-radius: 12px;
            color: white;
            margin-bottom: 1.4rem;
            padding: 1.5rem 1.8rem 1.35rem;
            box-shadow: 0 10px 28px rgba(57, 30, 35, 0.14);
        }}
        .lps-hero .eyebrow {{
            color: #F9D887;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }}
        .lps-hero h1 {{
            color: white;
            font-size: 2rem;
            margin: 0;
            padding: 0;
        }}
        .lps-hero p {{
            color: rgba(255, 255, 255, 0.86);
            margin: 0.45rem 0 0;
        }}
        [data-testid="stMetric"] {{
            background: var(--lps-surface);
            border: 1px solid {LPS_BORDER};
            border-left: 4px solid var(--lps-red);
            border-radius: 10px;
            min-height: 112px;
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 16px rgba(38, 55, 70, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--lps-muted);
            font-weight: 600;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--lps-slate);
        }}
        [data-testid="stMetricDelta"] {{
            font-weight: 600;
        }}
        div[data-testid="stDataFrame"] {{
            background: white;
            border: 1px solid {LPS_BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stExpander"] {{
            background: var(--lps-surface);
            border-color: {LPS_BORDER};
        }}
        [data-testid="stAlert"] {{
            color: var(--lps-slate);
            border: 1px solid {LPS_BORDER};
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: var(--lps-surface);
            border-color: var(--lps-red);
            color: var(--lps-red-dark);
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: var(--lps-red);
            border-color: var(--lps-red);
            color: white;
        }}
        code {{
            color: var(--lps-slate);
        }}
        h2, h3 {{
            color: var(--lps-slate);
            letter-spacing: -0.015em;
        }}
        hr {{
            border-color: {LPS_BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    """Render the common dashboard masthead."""
    st.markdown(
        f"""
        <div class="lps-hero">
            <div class="eyebrow">LPS &middot; Portfolio analytics</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure: go.Figure, height: int) -> go.Figure:
    """Apply the common Plotly theme."""
    figure.update_layout(
        colorway=[LPS_RED, LPS_GOLD, LPS_SLATE, "#698C7B", "#C65F45"],
        paper_bgcolor=LPS_SURFACE,
        plot_bgcolor=LPS_SURFACE,
        font={"family": "Inter, Segoe UI, sans-serif", "color": LPS_SLATE},
        title={"font": {"size": 19, "color": LPS_SLATE}, "x": 0.01},
        margin={"l": 45, "r": 25, "t": 65, "b": 45},
        height=height,
        hoverlabel={"bgcolor": "white", "font_color": LPS_SLATE},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    figure.update_xaxes(gridcolor="#E8EBED", zerolinecolor="#BFC6CB")
    figure.update_yaxes(gridcolor="#E8EBED", zerolinecolor="#BFC6CB")
    return figure


def histogram_bins(values: np.ndarray) -> int:
    """Choose stable histogram detail from the data distribution."""
    if len(values) < 2 or np.all(values == values[0]):
        return 15
    suggested = len(np.histogram_bin_edges(values, bins="fd")) - 1
    return max(15, min(60, suggested))


def central_bounds(
    values: np.ndarray,
    percentile: float,
    symmetric: bool,
) -> tuple[float, float, np.ndarray]:
    """Return display bounds and a mask without dropping tail observations."""
    if symmetric:
        cutoff = float(np.percentile(np.abs(values), percentile))
        lower, upper = -cutoff, cutoff
        mask = np.abs(values) <= cutoff
    else:
        lower = float(values.min())
        upper = float(np.percentile(values, percentile))
        mask = values <= upper
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    return lower, upper, mask


def add_bounded_reference_line(
    figure: go.Figure,
    value: float,
    lower_bound: float,
    upper_bound: float,
    label: str,
    color: str,
    annotation_position: str,
) -> None:
    """Draw a reference line without expanding the selected central view."""
    plotted_value = float(np.clip(value, lower_bound, upper_bound))
    suffix = " (outside view)" if plotted_value != value else ""
    figure.add_vline(
        x=plotted_value,
        line_dash="dash",
        line_color=color,
        annotation_text=f"{label}: {value:,.2f}{suffix}",
        annotation_position=annotation_position,
    )


def histogram_figure(
    values: np.ndarray,
    displayed_values: np.ndarray,
    bins: int,
    lower_bound: float,
    upper_bound: float,
    title: str,
    axis_title: str,
    show_zero: bool,
) -> go.Figure:
    """Build a branded histogram with mean and median references."""
    figure = go.Figure(
        go.Histogram(
            x=displayed_values,
            nbinsx=bins,
            marker={
                "color": LPS_RED,
                "line": {"color": "white", "width": 1},
            },
            opacity=0.9,
            hovertemplate=f"{axis_title}: %{{x:,.2f}}<br>Policies: %{{y:,}}<extra></extra>",
        )
    )
    add_bounded_reference_line(
        figure,
        float(values.mean()),
        lower_bound,
        upper_bound,
        "Mean",
        LPS_GOLD,
        "top right",
    )
    add_bounded_reference_line(
        figure,
        float(np.median(values)),
        lower_bound,
        upper_bound,
        "Median",
        LPS_SLATE,
        "top left",
    )
    if show_zero:
        figure.add_vline(x=0, line_color="#89949D", line_width=1)
    figure.update_layout(
        title=title,
        bargap=0.05,
        xaxis_title=axis_title,
        yaxis_title="Policies",
    )
    figure.update_xaxes(range=[lower_bound, upper_bound])
    return style_figure(figure, 500)


def cdf_figure(
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    title: str,
    axis_title: str,
    use_log_x: bool,
) -> go.Figure:
    """Build an empirical CDF with the 90th and 95th percentiles."""
    sorted_values = np.sort(values)
    probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    if use_log_x:
        positive = sorted_values > 0
        sorted_values = sorted_values[positive]
        probability = probability[positive]
        use_log_x = bool(len(sorted_values))

    quantile_90, quantile_95 = np.quantile(values, [0.90, 0.95])
    figure = go.Figure(
        go.Scatter(
            x=sorted_values,
            y=probability,
            mode="lines",
            line={"color": LPS_RED, "width": 3},
            name="Empirical CDF",
            hovertemplate=(
                f"{axis_title}: %{{x:,.2f}}<br>"
                "Cumulative share: %{y:.1%}<extra></extra>"
            ),
        )
    )
    for quantile, label, color, position in (
        (quantile_90, "P90", LPS_GOLD, "top left"),
        (quantile_95, "P95", LPS_RED_DARK, "top right"),
    ):
        if not use_log_x or quantile > 0:
            figure.add_vline(
                x=float(quantile),
                line_dash="dash",
                line_color=color,
                annotation_text=f"{label}: {quantile:,.2f}",
                annotation_position=position,
            )
    figure.update_layout(
        title=title,
        xaxis_title=axis_title,
        yaxis_title="Cumulative share",
        yaxis={"range": [0, 1.01], "tickformat": ".0%"},
    )
    if use_log_x:
        figure.update_xaxes(type="log")
    else:
        figure.update_xaxes(range=[lower_bound, upper_bound])
    return style_figure(figure, 450)


def business_table_columns(frame: pd.DataFrame) -> list[str]:
    """Keep readable source features and premium fields, excluding encodings."""
    identity = [column for column in ("NroPoliza",) if column in frame.columns]
    technical = {"Absolute Percent Error", "Dataset Split"}

    def is_encoded(column: str) -> bool:
        return (
            column.startswith("Accesorios_")
            or column.startswith("Cobertura_")
            or column.endswith("Encoded")
        )

    def is_premium(column: str) -> bool:
        return column == "Prediction" or column.startswith("Prima")

    raw_features = [
        column
        for column in frame.columns
        if column not in identity
        and column not in technical
        and not is_encoded(column)
        and not is_premium(column)
        and not column.startswith("_")
        and not column.endswith("_Counterfactual")
    ]
    features: list[str] = []
    for column in raw_features:
        features.append(column)
        counterfactual = f"{column}_Counterfactual"
        if counterfactual in frame.columns and not is_premium(counterfactual):
            features.append(counterfactual)
    premiums = [
        column
        for column in frame.columns
        if is_premium(column) and not column.startswith("_")
    ]
    return list(dict.fromkeys([*identity, *features, *premiums]))


def table_row_options(row_count: int) -> list[int]:
    """Offer useful table sizes without values larger than the selection."""
    options = [value for value in (10, 25, 50, 100) if value <= row_count]
    if not options or options[-1] != row_count and row_count < 100:
        options.append(row_count)
    return sorted(set(options))


def finite_array(values: Iterable[float]) -> np.ndarray:
    """Return finite floating-point values."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]
