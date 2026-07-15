from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from metrics_utils import compute_regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def load_scored_dataset(dataset_path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the parquet until its modification time changes."""
    del modified_time_ns
    return pd.read_parquet(dataset_path)


def target_options(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    options = {"Final Prima": ("Prima", "Prediction")}
    suffix = " Prediction"
    for prediction_column in df.columns:
        if prediction_column == "Prediction" or not prediction_column.endswith(suffix):
            continue
        actual_column = prediction_column.removesuffix(suffix)
        if actual_column in df.columns:
            options[actual_column] = (actual_column, prediction_column)
    return options


def histogram_bins(values: np.ndarray) -> int:
    if len(values) < 2 or np.all(values == values[0]):
        return 15
    suggested = len(np.histogram_bin_edges(values, bins="fd")) - 1
    return max(15, min(50, suggested))


def add_bounded_reference_line(
    figure: go.Figure,
    value: float,
    lower_bound: float,
    upper_bound: float,
    label: str,
    color: str,
    annotation_position: str,
) -> None:
    """Draw a full-distribution reference without expanding the central view."""
    plotted_value = float(np.clip(value, lower_bound, upper_bound))
    suffix = " (outside view)" if plotted_value != value else ""
    figure.add_vline(
        x=plotted_value,
        line_dash="dash",
        line_color=color,
        annotation_text=f"{label}: {value:.2f}{suffix}",
        annotation_position=annotation_position,
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Scored parquet was not found: {dataset_path}")

    st.set_page_config(page_title="Prima error dashboard", layout="wide")
    st.title("Prima prediction error dashboard")
    st.caption(str(dataset_path))

    df = load_scored_dataset(str(dataset_path), dataset_path.stat().st_mtime_ns)
    query = st.text_area(
        "Filter query",
        value='`Dataset Split` == "test"',
        help=(
            "Uses trusted Pandas query syntax, for example: "
            "CP == 7000 and ModoFacturacion in [1, 2]"
        ),
    )
    with st.expander("Query examples"):
        st.code('CP == 7000 and ModoFacturacion in [1, 2]')
        st.code('`Dataset Split` == "test" and CP >= 7000 and CP < 7100')
        st.warning("The query box is intended for trusted local input.")

    try:
        filtered = df.query(query, engine="python") if query.strip() else df
    except Exception as error:
        st.error(f"Invalid filter query: {error}")
        st.stop()
    if filtered.empty:
        st.warning("The query returned no rows.")
        st.stop()

    options = target_options(filtered)
    controls = st.columns(2)
    selected_target = controls[0].selectbox("Prediction target", list(options))
    error_mode = controls[1].radio(
        "Error mode",
        ["Absolute error", "Absolute percent error", "Signed error"],
        horizontal=True,
    )
    actual_column, prediction_column = options[selected_target]
    actual = filtered[actual_column].to_numpy(dtype=float)
    predicted = filtered[prediction_column].to_numpy(dtype=float)
    signed_errors = predicted - actual

    if error_mode == "Absolute error":
        errors = np.abs(signed_errors)
        axis_title = "Absolute error"
    elif error_mode == "Absolute percent error":
        denominator = np.abs(actual)
        valid = denominator > np.finfo(float).eps
        excluded = int((~valid).sum())
        actual = actual[valid]
        predicted = predicted[valid]
        errors = np.abs(predicted - actual) / denominator[valid] * 100
        axis_title = "Absolute percentage error (%)"
        if excluded:
            st.info(f"Excluded {excluded} rows with a zero target from percentage error.")
    else:
        errors = signed_errors
        axis_title = "Signed error (prediction - actual)"

    finite = np.isfinite(errors)
    errors = errors[finite]
    actual = actual[finite]
    predicted = predicted[finite]
    if not len(errors):
        st.warning("No finite errors are available for this selection.")
        st.stop()

    metrics = compute_regression_metrics(actual, predicted)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Rows", f"{len(errors):,}")
    metric_columns[1].metric("MAE", f"{metrics['mae']:,.2f}")
    mape = metrics["mape_percent"]
    metric_columns[2].metric("MAPE", f"{mape:,.2f}%" if mape is not None else "Undefined")

    tail_controls = st.columns(2)
    display_percentile = tail_controls[0].slider(
        "Central distribution shown (%)",
        min_value=90.0,
        max_value=100.0,
        value=99.5,
        step=0.1,
        help="Rows outside this central range remain included in metrics and tail summaries.",
    )
    if error_mode == "Signed error":
        magnitude_cutoff = float(np.percentile(np.abs(errors), display_percentile))
        display_mask = np.abs(errors) <= magnitude_cutoff
        display_lower, display_upper = -magnitude_cutoff, magnitude_cutoff
        negative_tail = int((errors < display_lower).sum())
        positive_tail = int((errors > display_upper).sum())
        tail_description = (
            f"{negative_tail + positive_tail:,} rows outside the central view "
            f"({negative_tail:,} negative, {positive_tail:,} positive)."
        )
        summary_errors = np.abs(errors)
        summary_prefix = "|error|"
    else:
        display_upper = float(np.percentile(errors, display_percentile))
        display_lower = float(np.min(errors))
        display_mask = errors <= display_upper
        tail_count = int((~display_mask).sum())
        tail_description = (
            f"{tail_count:,} rows above the {display_percentile:.1f}th-percentile "
            "cutoff."
        )
        summary_errors = errors
        summary_prefix = "Error"

    if np.isclose(display_lower, display_upper):
        padding = max(abs(display_lower) * 0.05, 1.0)
        display_lower -= padding
        display_upper += padding

    displayed_errors = errors[display_mask]
    bin_count = tail_controls[1].slider(
        "Histogram bins",
        min_value=10,
        max_value=80,
        value=histogram_bins(displayed_errors),
    )
    quantile_90_summary, quantile_95_summary, quantile_99_summary = np.quantile(
        summary_errors,
        [0.90, 0.95, 0.99],
    )
    tail_metrics = st.columns(5)
    tail_metrics[0].metric(f"{summary_prefix} P90", f"{quantile_90_summary:,.2f}")
    tail_metrics[1].metric(f"{summary_prefix} P95", f"{quantile_95_summary:,.2f}")
    tail_metrics[2].metric(f"{summary_prefix} P99", f"{quantile_99_summary:,.2f}")
    tail_metrics[3].metric(f"Maximum {summary_prefix.lower()}", f"{summary_errors.max():,.2f}")
    tail_metrics[4].metric("Rows outside view", f"{int((~display_mask).sum()):,}")
    st.caption(tail_description)

    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    figure = px.histogram(
        pd.DataFrame({"error": displayed_errors}),
        x="error",
        nbins=bin_count,
        title=f"{selected_target}: {error_mode} central distribution",
        labels={"error": axis_title},
    )
    add_bounded_reference_line(
        figure,
        mean_error,
        display_lower,
        display_upper,
        "Mean",
        "darkred",
        "top right",
    )
    add_bounded_reference_line(
        figure,
        median_error,
        display_lower,
        display_upper,
        "Median",
        "black",
        "top left",
    )
    if error_mode == "Signed error":
        figure.add_vline(x=0, line_color="gray", line_width=1)
    figure.update_traces(
        marker_color="steelblue",
        marker_line_color="white",
        marker_line_width=1,
        opacity=0.9,
    )
    figure.update_layout(bargap=0.06, yaxis_title="Count", height=520)
    st.plotly_chart(figure, width="stretch")

    sorted_errors = np.sort(errors)
    cumulative_probability = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    use_log_cdf = False
    if error_mode != "Signed error":
        use_log_cdf = st.checkbox(
            "Use logarithmic CDF x-axis",
            value=True,
            help="Shows the complete positive tail without compressing it into a linear axis.",
        )
    cdf_x = sorted_errors
    cdf_y = cumulative_probability
    if use_log_cdf:
        positive = cdf_x > 0
        if positive.any():
            cdf_x = cdf_x[positive]
            cdf_y = cdf_y[positive]
        else:
            use_log_cdf = False
    quantile_90, quantile_95 = np.quantile(errors, [0.90, 0.95])
    cdf_figure = go.Figure(
        go.Scatter(
            x=cdf_x,
            y=cdf_y,
            mode="lines",
            line={"color": "steelblue", "width": 2},
            name="Empirical CDF",
        )
    )
    if not use_log_cdf or quantile_90 > 0:
        cdf_figure.add_vline(
            x=float(quantile_90),
            line_dash="dash",
            line_color="darkorange",
            annotation_text=f"90%: {quantile_90:.2f}",
            annotation_position="top left",
        )
    if not use_log_cdf or quantile_95 > 0:
        cdf_figure.add_vline(
            x=float(quantile_95),
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"95%: {quantile_95:.2f}",
            annotation_position="top right",
        )
    cdf_view = "full log view" if use_log_cdf else "central linear view"
    cdf_figure.update_layout(
        title=f"{selected_target}: empirical error CDF ({cdf_view})",
        xaxis_title=axis_title,
        yaxis_title="Cumulative probability",
        yaxis={"range": [0, 1.01], "tickformat": ".0%"},
        height=460,
    )
    if use_log_cdf:
        cdf_figure.update_xaxes(type="log")
    else:
        cdf_figure.update_xaxes(range=[display_lower, display_upper])
    st.plotly_chart(cdf_figure, width="stretch")


if __name__ == "__main__":
    main()
