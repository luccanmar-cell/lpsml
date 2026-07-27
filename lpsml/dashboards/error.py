from __future__ import annotations

"""Interactive model-error dashboard for scored premium datasets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lpsml.dashboards.shared import (
    business_table_columns,
    cdf_figure,
    central_bounds,
    configure_dashboard,
    histogram_bins,
    histogram_figure,
    render_header,
    table_row_options,
)
from lpsml.reporting.metrics import compute_regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def load_scored_dataset(dataset_path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the Parquet until its modification time changes."""
    del modified_time_ns
    return pd.read_parquet(dataset_path)


def target_options(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Map readable targets to their actual and predicted columns."""
    options = {"Final Prima": ("Prima", "Prediction")}
    suffix = " Prediction"
    for prediction_column in frame.columns:
        if prediction_column == "Prediction" or not prediction_column.endswith(suffix):
            continue
        actual_column = prediction_column.removesuffix(suffix)
        if actual_column in frame.columns:
            options[actual_column] = (actual_column, prediction_column)
    return options


def error_analysis(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    error_mode: str,
) -> tuple[pd.DataFrame, str, str, int]:
    """Create an aligned row-level error measure for plotting and tables."""
    analysis = frame.copy()
    actual = pd.to_numeric(analysis[actual_column], errors="coerce")
    predicted = pd.to_numeric(analysis[prediction_column], errors="coerce")
    signed = predicted - actual
    excluded = 0

    if error_mode == "Absolute error":
        metric = signed.abs()
        axis_title = "Absolute error"
        summary_prefix = "Error"
    elif error_mode == "Absolute percent error":
        valid = actual.abs() > np.finfo(float).eps
        excluded = int((~valid).sum())
        metric = signed.abs().div(actual.abs()).mul(100).where(valid)
        axis_title = "Absolute percentage error (%)"
        summary_prefix = "Error"
    else:
        metric = signed
        axis_title = "Signed error (prediction − actual)"
        summary_prefix = "|error|"

    analysis["_Actual"] = actual
    analysis["_Predicted"] = predicted
    analysis["_Metric"] = metric
    finite = np.isfinite(
        analysis[["_Actual", "_Predicted", "_Metric"]].to_numpy(dtype=float)
    ).all(axis=1)
    return analysis.loc[finite].copy(), axis_title, summary_prefix, excluded


def render_furthest_table(
    analysis: pd.DataFrame,
    metric_label: str,
    selected_target: str,
) -> None:
    """Display and export the policies furthest from the selected mean."""
    st.divider()
    heading, control = st.columns([4, 1])
    heading.subheader("Policies furthest from the mean")
    options = table_row_options(len(analysis))
    row_count = control.selectbox(
        "Policies shown",
        options,
        index=min(1, len(options) - 1),
        key="error_table_rows",
    )

    mean_value = float(analysis["_Metric"].mean())
    ranked = analysis.assign(
        _deviation=(analysis["_Metric"] - mean_value).abs()
    ).nlargest(row_count, "_deviation")
    columns = business_table_columns(ranked)
    table = ranked[columns].copy()
    table[metric_label] = ranked["_Metric"]
    table["Deviation from mean"] = ranked["_deviation"]
    st.caption(
        f"Largest absolute distances from the mean {metric_label.lower()} "
        f"for {selected_target}."
    )
    st.dataframe(table, width="stretch", hide_index=True, height=430)
    st.download_button(
        "Download displayed policies",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="policies_furthest_from_mean.csv",
        mime="text/csv",
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Scored Parquet was not found: {dataset_path}")

    configure_dashboard("LPS · Model error analysis")
    render_header(
        "Model error analysis",
        "Inspect prediction quality, distribution tails, and the policies "
        "furthest from typical model behavior.",
    )
    frame = load_scored_dataset(str(dataset_path), dataset_path.stat().st_mtime_ns)

    st.sidebar.header("Analysis controls")
    st.sidebar.caption(str(dataset_path))
    query = st.sidebar.text_area(
        "Filter query",
        value='`Dataset Split` == "test"',
        help="Uses trusted Pandas query syntax.",
    )
    with st.sidebar.expander("Query examples"):
        st.code('CP == 7000 and ModoFacturacion in [1, 2]')
        st.code('`Dataset Split` == "test" and CP >= 7000 and CP < 7100')
        st.caption("The query box is intended for trusted local input.")

    try:
        filtered = frame.query(query, engine="python") if query.strip() else frame
    except Exception as error:
        st.error(f"Invalid filter query: {error}")
        st.stop()
    if filtered.empty:
        st.warning("The query returned no rows.")
        st.stop()

    options = target_options(filtered)
    selected_target = st.sidebar.selectbox("Prediction target", list(options))
    error_mode = st.sidebar.selectbox(
        "Error measure",
        ["Absolute error", "Absolute percent error", "Signed error"],
    )
    display_percentile = st.sidebar.slider(
        "Central distribution shown (%)",
        min_value=90.0,
        max_value=100.0,
        value=99.5,
        step=0.1,
        help="Tail rows remain included in metrics, the CDF, and the policy table.",
    )
    use_log_cdf = (
        st.sidebar.checkbox("Logarithmic CDF x-axis", value=True)
        if error_mode != "Signed error"
        else False
    )

    actual_column, prediction_column = options[selected_target]
    analysis, axis_title, summary_prefix, excluded = error_analysis(
        filtered,
        actual_column,
        prediction_column,
        error_mode,
    )
    if analysis.empty:
        st.warning("No finite errors are available for this selection.")
        st.stop()
    if excluded:
        st.info(f"Excluded {excluded:,} rows with a zero target from percentage error.")

    values = analysis["_Metric"].to_numpy(dtype=float)
    metrics = compute_regression_metrics(
        analysis["_Actual"],
        analysis["_Predicted"],
    )
    cards = st.columns(4)
    cards[0].metric("Policies", f"{len(analysis):,}")
    cards[1].metric("MAE", f"{metrics['mae']:,.2f}")
    mape = metrics["mape_percent"]
    cards[2].metric("MAPE", f"{mape:,.2f}%" if mape is not None else "Undefined")
    cards[3].metric(f"Mean {axis_title.lower()}", f"{values.mean():,.2f}")

    symmetric = error_mode == "Signed error"
    lower, upper, display_mask = central_bounds(
        values,
        display_percentile,
        symmetric=symmetric,
    )
    displayed = values[display_mask]
    default_bins = histogram_bins(displayed)
    bins = st.sidebar.slider(
        "Histogram bins",
        min_value=10,
        max_value=80,
        value=default_bins,
    )

    summary_values = np.abs(values) if symmetric else values
    quantiles = np.quantile(summary_values, [0.90, 0.95, 0.99])
    tails = st.columns(5)
    for column, label, value in zip(tails[:3], ("P90", "P95", "P99"), quantiles):
        column.metric(f"{summary_prefix} {label}", f"{value:,.2f}")
    tails[3].metric(f"Maximum {summary_prefix.lower()}", f"{summary_values.max():,.2f}")
    tails[4].metric("Policies outside view", f"{int((~display_mask).sum()):,}")

    st.subheader("Error distribution")
    st.plotly_chart(
        histogram_figure(
            values,
            displayed,
            bins,
            lower,
            upper,
            title=f"{selected_target} · {error_mode}",
            axis_title=axis_title,
            show_zero=symmetric,
        ),
        width="stretch",
    )
    st.caption(
        f"The histogram shows the central {display_percentile:.1f}% view. "
        "All finite policies remain included elsewhere."
    )

    st.subheader("Cumulative distribution")
    st.plotly_chart(
        cdf_figure(
            values,
            lower,
            upper,
            title=f"{selected_target} · Empirical error CDF",
            axis_title=axis_title,
            use_log_x=use_log_cdf,
        ),
        width="stretch",
    )
    st.caption(
        "P90 and P95 indicate the error values below which 90% and 95% of "
        "the selected policies fall."
    )
    render_furthest_table(analysis, axis_title, selected_target)


if __name__ == "__main__":
    main()
