from __future__ import annotations

"""Interactive portfolio-impact dashboard for counterfactual scenarios."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lpsml.dashboards.shared import (
    LPS_RED,
    LPS_SLATE,
    business_table_columns,
    cdf_figure,
    central_bounds,
    configure_dashboard,
    histogram_bins,
    histogram_figure,
    render_header,
    style_figure,
    table_row_options,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def load_counterfactual_dataset(
    dataset_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """Cache the Parquet until its modification time changes."""
    del modified_time_ns
    return pd.read_parquet(dataset_path)


def target_options(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Map readable targets to baseline and counterfactual columns."""
    suffix = "_Counterfactual"
    options: dict[str, tuple[str, str]] = {}
    for counterfactual_column in frame.columns:
        if not counterfactual_column.startswith("Prima") or not counterfactual_column.endswith(
            suffix
        ):
            continue
        target = counterfactual_column.removesuffix(suffix)
        baseline_column = f"{target}_Baseline"
        if baseline_column in frame.columns:
            label = "Final Prima" if target == "Prima" else target
            options[label] = (baseline_column, counterfactual_column)
    if "Final Prima" in options:
        return {"Final Prima": options.pop("Final Prima"), **options}
    return options


def filter_portfolio(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply simple business-facing portfolio filters from the sidebar."""
    filtered = frame
    with st.sidebar.expander("Portfolio filters", expanded=True):
        if "Pol6TTaCod" in frame.columns:
            tariffs = sorted(frame["Pol6TTaCod"].dropna().astype(str).unique())
            selected_tariffs = st.multiselect(
                "Tariff types",
                tariffs,
                help="Leave empty to include every tariff.",
            )
            if selected_tariffs:
                filtered = filtered[
                    filtered["Pol6TTaCod"].astype(str).isin(selected_tariffs)
                ]

        if "CoberturaLabel" in frame.columns:
            coverages = sorted(frame["CoberturaLabel"].dropna().astype(str).unique())
            selected_coverages = st.multiselect(
                "Coverages",
                coverages,
                help="Leave empty to include every coverage.",
            )
            if selected_coverages:
                filtered = filtered[
                    filtered["CoberturaLabel"].astype(str).isin(selected_coverages)
                ]

        if "antig" in frame.columns:
            ages = pd.to_numeric(frame["antig"], errors="coerce").dropna()
            if not ages.empty:
                minimum, maximum = int(ages.min()), int(ages.max())
                selected_age = st.slider(
                    "Vehicle age",
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum),
                )
                numeric_age = pd.to_numeric(filtered["antig"], errors="coerce")
                filtered = filtered[numeric_age.between(*selected_age)]
    return filtered


def build_change_analysis(
    frame: pd.DataFrame,
    baseline_column: str,
    counterfactual_column: str,
) -> pd.DataFrame:
    """Add aligned monetary and percentage scenario changes."""
    analysis = frame.copy()
    baseline = pd.to_numeric(analysis[baseline_column], errors="coerce")
    counterfactual = pd.to_numeric(
        analysis[counterfactual_column],
        errors="coerce",
    )
    change = counterfactual - baseline
    valid_percentage = baseline.abs() > np.finfo(float).eps
    analysis["_Baseline"] = baseline
    analysis["_Counterfactual"] = counterfactual
    analysis["_Change"] = change
    analysis["_PercentChange"] = change.div(baseline).mul(100).where(valid_percentage)
    finite = np.isfinite(
        analysis[["_Baseline", "_Counterfactual", "_Change"]].to_numpy(dtype=float)
    ).all(axis=1)
    return analysis.loc[finite].copy()


def render_furthest_table(
    analysis: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    selected_target: str,
) -> None:
    """Display and export the policies furthest from the mean scenario change."""
    st.divider()
    heading, control = st.columns([4, 1])
    heading.subheader("Policies furthest from the mean")
    options = table_row_options(len(analysis))
    row_count = control.selectbox(
        "Policies shown",
        options,
        index=min(1, len(options) - 1),
        key="counterfactual_table_rows",
    )

    mean_value = float(analysis[metric_column].mean())
    ranked = analysis.assign(
        _deviation=(analysis[metric_column] - mean_value).abs()
    ).nlargest(row_count, "_deviation")
    columns = business_table_columns(ranked)
    table = ranked[columns].copy()
    table["Signed change"] = ranked["_Change"]
    table["Percentage change (%)"] = ranked["_PercentChange"]
    table[f"Deviation from mean {metric_label.lower()}"] = ranked["_deviation"]
    st.caption(
        f"Largest absolute distances from the mean {metric_label.lower()} "
        f"for {selected_target}."
    )
    st.dataframe(table, width="stretch", hide_index=True, height=430)
    st.download_button(
        "Download displayed policies",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="counterfactual_policies_furthest_from_mean.csv",
        mime="text/csv",
    )


def render_component_contributions(frame: pd.DataFrame) -> None:
    """Show the mean signed counterfactual change for each premium component."""
    component_changes: list[tuple[str, float]] = []
    for component in (
        "PrimaRC",
        "PrimaCasco",
        "PrimaClausulaAjuste",
        "PrimaAccesorio",
    ):
        baseline = f"{component}_Baseline"
        counterfactual = f"{component}_Counterfactual"
        if baseline in frame.columns and counterfactual in frame.columns:
            change = pd.to_numeric(frame[counterfactual], errors="coerce") - pd.to_numeric(
                frame[baseline],
                errors="coerce",
            )
            component_changes.append((component, float(change.mean())))
    if not component_changes:
        return

    labels, values = zip(*component_changes)
    colors = [LPS_RED if value >= 0 else LPS_SLATE for value in values]
    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker_color=colors,
            text=[f"{value:,.2f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>Mean change: %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="#89949D", line_width=1)
    figure.update_layout(
        title="Mean signed change by premium component",
        xaxis_title="Premium component",
        yaxis_title="Mean signed change",
    )
    st.plotly_chart(style_figure(figure, 430), width="stretch")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Counterfactual Parquet was not found: {dataset_path}")

    configure_dashboard("LPS · Counterfactual impact")
    render_header(
        "Counterfactual portfolio impact",
        "Measure average premium changes, distribution tails, and the policies "
        "furthest from the portfolio response.",
    )
    frame = load_counterfactual_dataset(
        str(dataset_path),
        dataset_path.stat().st_mtime_ns,
    )
    options = target_options(frame)
    if not options:
        st.error(
            "The dataset does not contain matching *_Baseline and "
            "*_Counterfactual premium columns. Regenerate it with "
            "counterfactual_inference.py."
        )
        st.stop()

    st.sidebar.header("Scenario controls")
    st.sidebar.caption(str(dataset_path))
    selected_target = st.sidebar.selectbox("Premium target", list(options))
    measure = st.sidebar.selectbox(
        "Distribution measure",
        ["Monetary change", "Percentage change"],
    )
    display_percentile = st.sidebar.slider(
        "Central distribution shown (%)",
        min_value=90.0,
        max_value=100.0,
        value=99.5,
        step=0.1,
        help="Tail policies remain included in metrics, the CDF, and the policy table.",
    )
    filtered = filter_portfolio(frame)
    if filtered.empty:
        st.warning("The selected filters returned no policies.")
        st.stop()

    baseline_column, counterfactual_column = options[selected_target]
    analysis = build_change_analysis(
        filtered,
        baseline_column,
        counterfactual_column,
    )
    if analysis.empty:
        st.warning("No finite counterfactual changes are available.")
        st.stop()

    metric_column = "_Change" if measure == "Monetary change" else "_PercentChange"
    metric_label = "signed change" if measure == "Monetary change" else "change (%)"
    distribution = analysis[np.isfinite(analysis[metric_column])].copy()
    excluded = len(analysis) - len(distribution)
    if distribution.empty:
        st.warning("No finite values are available for the selected measure.")
        st.stop()
    if excluded:
        st.info(
            f"Excluded {excluded:,} policies with a zero baseline from percentage change."
        )

    baseline_total = float(analysis["_Baseline"].sum())
    counterfactual_total = float(analysis["_Counterfactual"].sum())
    portfolio_change = counterfactual_total - baseline_total
    portfolio_percent = (
        portfolio_change / baseline_total * 100
        if abs(baseline_total) > np.finfo(float).eps
        else np.nan
    )
    percent_values = analysis["_PercentChange"].replace([np.inf, -np.inf], np.nan)
    cards = st.columns(4)
    cards[0].metric("Affected policies", f"{len(analysis):,}")
    cards[1].metric("Baseline portfolio", f"{baseline_total:,.2f}")
    cards[2].metric("Counterfactual portfolio", f"{counterfactual_total:,.2f}")
    cards[3].metric(
        "Portfolio change",
        f"{portfolio_change:,.2f}",
        delta=f"{portfolio_percent:,.2f}%" if np.isfinite(portfolio_percent) else None,
    )
    secondary = st.columns(4)
    secondary[0].metric("Mean signed change", f"{analysis['_Change'].mean():,.2f}")
    secondary[1].metric(
        "Mean policy change",
        f"{percent_values.mean():,.2f}%",
    )
    secondary[2].metric(
        f"Std. dev. of {metric_label}",
        f"{distribution[metric_column].std(ddof=0):,.2f}",
    )
    secondary[3].metric(
        "Policies increased",
        f"{(analysis['_Change'] > 0).mean():.1%}",
    )

    values = distribution[metric_column].to_numpy(dtype=float)
    lower, upper, display_mask = central_bounds(
        values,
        display_percentile,
        symmetric=True,
    )
    displayed = values[display_mask]
    bins = st.sidebar.slider(
        "Histogram bins",
        min_value=10,
        max_value=80,
        value=histogram_bins(displayed),
    )

    st.subheader("Change distribution")
    st.plotly_chart(
        histogram_figure(
            values,
            displayed,
            bins,
            lower,
            upper,
            title=f"{selected_target} · {measure}",
            axis_title=metric_label.capitalize(),
            show_zero=True,
        ),
        width="stretch",
    )
    st.caption(
        f"The histogram shows the central {display_percentile:.1f}% view; "
        f"{int((~display_mask).sum()):,} tail policies remain included elsewhere."
    )

    st.subheader("Cumulative distribution")
    st.plotly_chart(
        cdf_figure(
            values,
            lower,
            upper,
            title=f"{selected_target} · Empirical change CDF",
            axis_title=metric_label.capitalize(),
            use_log_x=False,
        ),
        width="stretch",
    )
    st.caption(
        "P90 and P95 indicate the signed changes below which 90% and 95% of "
        "the selected policies fall."
    )
    render_furthest_table(
        distribution,
        metric_column,
        metric_label,
        selected_target,
    )
    st.divider()
    st.subheader("Component contribution")
    render_component_contributions(analysis)


if __name__ == "__main__":
    main()
