"""Reusable plotting for the manuscript result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sga.config import MODEL_DISPLAY_NAMES

#: Classifier order used in every manuscript comparison panel.
MODEL_ORDER = ["dnn", "catboost", "rf", "lr", "svc", "stacking"]

#: Training strategies compared in Figure 3.
STRATEGY_LABELS = {
    "baseline": "Country-specific baseline",
    "unified": "Unified (pooled) model",
    "cross_domain": "Cross-domain (trained on the other cohort)",
}

#: Bar colours for Figure 3, fixed to the manuscript caption (baseline orange,
#: unified blue, cross-domain green). Fixed rather than sampled from a colormap, so
#: a missing arm cannot recolour the other two.
STRATEGY_COLORS = {
    "baseline": "tab:orange",
    "unified": "tab:blue",
    "cross_domain": "tab:green",
}

#: Cohort key -> display name.
COHORT_LABELS = {"total": "Pooled", "malaysia": "Malaysia", "india": "India"}

#: Gestational-week window reported in Figure 5. Spans the whole third trimester
#: admitted by ``cleaning.CONTINUOUS_FEATURE_LOGICAL_RANGE["ga"]`` (196-300 days),
#: so the "extremes of gestational age" the Results discuss are actually plotted.
GA_WEEK_RANGE = (28, 43)

#: Maternal-age bin edges (4-year intervals) reported in Figure 5. Spans the whole
#: admitted maternal-age range (13-55 years), so the youngest and oldest intervals
#: the Results discuss are not silently dropped by ``pd.cut``.
AGE_BIN_STEP = 4
AGE_BIN_EDGES = np.arange(12, 57, AGE_BIN_STEP)

_SAMPLE_BAR_COLOR = "lightgray"


def apply_manuscript_style():
    """Set the serif typeface used throughout the manuscript figures."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def _save(fig, save_path, dpi=300):
    """Write ``fig`` to ``save_path``, creating parent directories."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")
    return save_path


def _age_bin_labels(bin_edges=AGE_BIN_EDGES):
    """Return the ``"20-23"``-style labels for a set of age bin edges."""
    edges = np.asarray(bin_edges)
    steps = np.diff(edges)
    return [f"{start}-{start + step - 1}" for start, step in zip(edges[:-1], steps)]


# ── Figure 3: AUROC comparison across classifiers and strategies ─────────────


def plot_auroc_comparison(records, save_path, cohort=None, ylim=(0.4, 1.0)):
    """Grouped AUROC bars with 95% CI error bars (manuscript Figure 3)."""
    df = pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("No AUROC records to plot; check the results directories.")

    apply_manuscript_style()
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    strategies = [s for s in STRATEGY_LABELS if s in set(df["strategy"])]
    palette = [STRATEGY_COLORS[s] for s in strategies]

    positions = np.arange(len(models), dtype=float)
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, strategy in enumerate(strategies):
        offset = (index - (len(strategies) - 1) / 2) * width
        heights, lower, upper = [], [], []
        for model in models:
            row = df[(df["model"] == model) & (df["strategy"] == strategy)]
            if row.empty:
                heights.append(np.nan)
                lower.append(0.0)
                upper.append(0.0)
                continue
            value = float(row["auroc"].iloc[0])
            heights.append(value)
            lower.append(max(value - float(row["ci_low"].iloc[0]), 0.0))
            upper.append(max(float(row["ci_high"].iloc[0]) - value, 0.0))

        ax.bar(
            positions + offset,
            heights,
            width=width * 0.92,
            color=palette[index],
            edgecolor="black",
            linewidth=0.6,
            label=STRATEGY_LABELS[strategy],
            yerr=[lower, upper],
            capsize=4,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models],
                       rotation=20, ha="right", fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_ylabel("AUROC (95% CI)", fontsize=12)
    ax.set_xlabel("Classifier", fontsize=12)
    title = "External-test AUROC by training strategy"
    if cohort:
        title = f"{title} - {COHORT_LABELS.get(cohort, cohort.capitalize())}"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, save_path)


# ── Figure 5: subgroup AUROC with sample-count bars ──────────────────────────


def filter_by_country(dfs, country):
    """Keep only the rows of one cohort in each prediction frame."""
    filtered = []
    for df in dfs:
        if "country" not in df.columns:
            raise KeyError(
                "Each prediction frame needs a 'country' column to split by "
                "subgroup; use the external-test CSVs from 05b0_subgroup_inference."
            )
        filtered.append(
            df[df["country"].astype(str).str.lower() == country.lower()].copy()
        )
    return filtered


def _gestational_weeks(df, week_range):
    """Add a rounded ``ga_weeks`` column and clip to the reported window."""
    out = df.copy()
    out["ga_weeks"] = (out["ga"] // 7) + ((out["ga"] % 7) >= 4)
    low, high = week_range
    return out[(out["ga_weeks"] >= low) & (out["ga_weeks"] <= high)]


def _reference_frame(binned, what):
    """The frame whose row counts the grey sample bars report.

    All Figure 5 arms are re-scored on the identical external-test rows, so the bars
    describe one arm, not their sum. A length disagreement means the arms were scored
    on different rows and the panel would be comparing unlike things.
    """
    lengths = {len(df) for df in binned}
    if len(lengths) > 1:
        print(
            f"  WARNING: the {what} panel's arms cover different row counts "
            f"({sorted(lengths)}); the sample bars follow the first arm. Re-run "
            "05b0_subgroup_inference for every arm with the same settings."
        )
    return binned[0]


def _subgroup_auroc(subset):
    """AUROC of one subgroup, or NaN when it holds a single outcome class."""
    if subset["Actual"].nunique() < 2:
        return np.nan
    return roc_auc_score(subset["Actual"], subset["predicted_probability"])


def _finish_subgroup_axes(ax1, ax2, total_samples, xlabel, title):
    """Apply the shared two-axis styling of the Figure 5 panels."""
    ax1.set_ylabel("AUROC", color="black", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=12)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)

    ax2.set_ylabel("Number of Samples", color="gray", fontsize=12)
    ax2.set_ylim(0, max(total_samples) * 1.2 if any(total_samples) else 10)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=12)

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.tick_params(axis="x", labelsize=12)

    legend = ax1.legend(loc="upper right", frameon=True, fontsize=12)
    legend.set_zorder(100)
    legend.get_frame().set_zorder(100)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_auroc_by_gestational_week(
    dfs, labels, save_path, colors=None, title="AUROC per Gestational Week",
    week_range=GA_WEEK_RANGE,
):
    """AUROC per gestational week with grey sample-count bars (Figure 5, left)."""
    apply_manuscript_style()
    colors = colors if colors is not None else plt.get_cmap("Set1")(
        np.linspace(0, 1, len(dfs))
    )
    binned = [_gestational_weeks(df, week_range) for df in dfs]

    all_weeks = sorted({week for df in binned for week in df["ga_weeks"].unique()})
    # Every arm is scored on the SAME external-test rows, so counting across all of
    # them would report n_arms x the true "number of samples tested" of the caption.
    counted = _reference_frame(binned, "gestational week")
    total_samples = [int((counted["ga_weeks"] == week).sum()) for week in all_weeks]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Bars first so the AUROC lines stay legible on top of them.
    ax2 = ax1.twinx()
    ax2.bar(all_weeks, total_samples, color=_SAMPLE_BAR_COLOR,
            label="Total Sample Count", alpha=0.6, zorder=1)

    records = []
    for df, label, color in zip(binned, labels, colors):
        overall = _subgroup_auroc(df)
        print(f"  {label}: overall AUROC {overall:.4f} on {len(df)} rows")
        scores = [_subgroup_auroc(df[df["ga_weeks"] == week]) for week in all_weeks]
        for week, score, n in zip(all_weeks, scores, total_samples):
            records.append({
                "stratum_type": "gestational_week", "arm": label,
                "stratum": int(week), "auroc": score, "n_samples": n,
                "overall_auroc": overall,
            })
        valid = [(w, s) for w, s in zip(all_weeks, scores) if not np.isnan(s)]
        if valid:
            x, y = zip(*valid)
            ax1.plot(x, y, color=color, marker="o", linestyle="-", label=label, zorder=99)
        ax1.axhline(y=overall, color=color, linestyle="--", alpha=0.7, zorder=99)

    ax1.set_xlim(week_range[0] - 1, week_range[1] + 1)
    _finish_subgroup_axes(ax1, ax2, total_samples, "Gestational Week", title)
    fig.tight_layout()
    _save(fig, save_path)
    return pd.DataFrame(records)


def plot_auroc_by_maternal_age(
    dfs, labels, save_path, colors=None,
    title="AUROC per Maternal Age Interval", bin_edges=AGE_BIN_EDGES,
):
    """AUROC per maternal-age interval with sample-count bars (Figure 5, right)."""
    apply_manuscript_style()
    colors = colors if colors is not None else plt.get_cmap("Set1")(
        np.linspace(0, 1, len(dfs))
    )
    bin_labels = _age_bin_labels(bin_edges)

    binned = []
    for df in dfs:
        out = df.copy()
        out["age_bin"] = pd.cut(
            out["m_age"], bins=bin_edges, labels=bin_labels, right=False
        )
        binned.append(out)

    counted = _reference_frame(binned, "maternal-age")
    total_samples = [int((counted["age_bin"] == name).sum()) for name in bin_labels]
    # The edges span the whole admitted maternal-age range so no mother is dropped,
    # but a cohort rarely fills the outermost bins. Trim only the empty bins at the
    # ends, which keeps every occupied interval -- including the youngest and oldest
    # the Results discuss -- without leaving dead space on the axis.
    occupied = [i for i, n in enumerate(total_samples) if n > 0]
    if occupied:
        first, last = occupied[0], occupied[-1]
        bin_labels = list(bin_labels[first:last + 1])
        total_samples = total_samples[first:last + 1]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax2.bar(bin_labels, total_samples, color=_SAMPLE_BAR_COLOR,
            label="Total Sample Count", alpha=0.6, zorder=1)

    records = []
    for df, label, color in zip(binned, labels, colors):
        overall = _subgroup_auroc(df)
        print(f"  {label}: overall AUROC {overall:.4f} on {len(df)} rows")
        scores = [_subgroup_auroc(df[df["age_bin"] == name]) for name in bin_labels]
        for name, score, n in zip(bin_labels, scores, total_samples):
            records.append({
                "stratum_type": "maternal_age", "arm": label,
                "stratum": name, "auroc": score, "n_samples": n,
                "overall_auroc": overall,
            })
        # Plot at numeric positions so the line spans undefined bins and stays aligned
        # with the categorical bars at 0..n-1.
        valid = [(i, s) for i, s in enumerate(scores) if not np.isnan(s)]
        if valid:
            x, y = zip(*valid)
            ax1.plot(x, y, color=color, marker="o", linestyle="-", label=label, zorder=99)
        ax1.axhline(y=overall, color=color, linestyle="--", alpha=0.7, zorder=99)

    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels)
    _finish_subgroup_axes(ax1, ax2, total_samples, "Maternal Age (years)", title)
    fig.tight_layout()
    _save(fig, save_path)
    return pd.DataFrame(records)


# ── Figure 7: training-set size ablation ─────────────────────────────────────


# ── Figure 7: AUROC against training-set size ────────────────────────────────
#
# Colour and marker follow the MODEL, never the plotting order, so adding or
# dropping one classifier cannot repaint the others and two renderings of the
# figure remain comparable.

#: ``(key, label, colour, marker)`` in the display order of Figure 7's legend.
#: The palette is Okabe-Ito, which stays distinguishable in the common forms of
#: colour-vision deficiency and in greyscale print.
FIGURE7_MODELS = [
    ("catboost", "CatBoost", "#0072B2", "o"),
    ("rf", "Random Forest", "#E69F00", "s"),
    ("lr", "Logistic Regression", "#009E73", "D"),
    ("svc", "SVC", "#D55E00", "^"),
    ("stacking", "Stacking", "#CC79A7", "v"),
    ("dnn", "Neural Network", "#56B4E9", "P"),
]

#: Fixed axis limits, so panels from different runs are directly comparable.
FIGURE7_AUROC_YLIM = (0.60, 0.85)
FIGURE7_PCT_YLIM = (-2.0, 14.0)
#: The India cohort dips to ~0.556, so the by-cohort panel keeps its own range.
FIGURE7_BY_COHORT_YLIM = (0.50, 0.88)

FIGURE7_XLABEL = "Training sample size (k)"
FIGURE7_BASELINE_SIZE = 1000

_INK = "#1a1a1a"
_INK_MUTED = "#6b6b6b"
_GRID = "#e4e4e2"
_SURFACE = "#ffffff"

FIGURE7_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#c9c9c6",
    "axes.linewidth": 0.8,
    "axes.labelcolor": _INK,
    "xtick.color": _INK_MUTED,
    "ytick.color": _INK_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
}


def size_to_int(label):
    """``'original_18688'`` -> 18688 ; ``'12000'`` -> 12000."""
    text = str(label)
    return int(text.split("_")[-1]) if text.startswith("original") else int(float(text))


def _size_tick_label(value):
    """18000 -> ``'18'`` ; 18688 -> ``'18.7'`` (only the odd full-pool point)."""
    thousands = value / 1000.0
    if abs(thousands - round(thousands)) < 1e-9:
        return f"{thousands:.0f}"
    return f"{thousands:.1f}"


def _figure7_series(summary, split, model_key, column):
    """``(sizes, values)`` for one classifier on one evaluation split."""
    subset = summary[(summary["eval_split"] == split) & (summary["model"] == model_key)]
    return subset["n"].to_numpy(), subset[column].to_numpy()


def _figure7_style_axes(ax, sizes, ylabel, title, tick_every=1, ylim=None):
    """Shared axis furniture for every Figure 7 panel."""
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=_INK, pad=8)
    ax.set_xlabel(FIGURE7_XLABEL, fontsize=9, color=_INK)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK)
    ax.grid(axis="y", color=_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(sizes)
    ax.set_xticklabels(
        [
            _size_tick_label(value) if index % tick_every == 0 else ""
            for index, value in enumerate(sizes)
        ]
    )
    pad = (sizes[-1] - sizes[0]) * 0.025
    ax.set_xlim(sizes[0] - pad, sizes[-1] + pad)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _figure7_line(ax, sizes, values, colour, marker, label):
    """One classifier's line, styled identically across every panel."""
    ax.plot(
        sizes,
        values,
        color=colour,
        linewidth=2.0,
        marker=marker,
        markersize=4.2,
        markeredgecolor=_SURFACE,
        markeredgewidth=0.7,
        label=label,
        zorder=3,
        clip_on=True,
    )


def load_size_summary(csv_path, metric="roc_auc", max_size=None):
    """Read the size-sweep summary CSV and add an integer ``n`` column.

    Expects the file ``R1.experiment_R1_2_data_scaling_inference`` writes,
    ``per_country_eval_all_summary_by_country.csv``, with ``model``, ``size``,
    ``eval_split`` and ``<metric>_mean`` / ``<metric>_std`` columns.
    """
    summary = pd.read_csv(csv_path)
    required = {"model", "size", "eval_split", f"{metric}_mean"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing the column(s) {sorted(missing)}")

    summary["n"] = summary["size"].map(size_to_int)
    if max_size is not None:
        summary = summary[summary["n"] <= max_size]
    return summary.sort_values("n").reset_index(drop=True)


def draw_figure7_auroc(
    ax, summary, split, title, metric="roc_auc", show_sd=False,
    tick_every=1, ylim=FIGURE7_AUROC_YLIM, models=None,
):
    """Panel (a): AUROC against training-set size."""
    models = FIGURE7_MODELS if models is None else models
    sizes = sorted(summary[summary["eval_split"] == split]["n"].unique())
    for key, label, colour, marker in models:
        n, values = _figure7_series(summary, split, key, f"{metric}_mean")
        if len(n) == 0:
            continue
        if show_sd and f"{metric}_std" in summary.columns:
            _, sd = _figure7_series(summary, split, key, f"{metric}_std")
            ax.fill_between(
                n, values - sd, values + sd, color=colour, alpha=0.09,
                linewidth=0, zorder=1,
            )
        _figure7_line(ax, n, values, colour, marker, label)
    _figure7_style_axes(ax, sizes, "AUROC", title, tick_every, ylim)


def draw_figure7_percentage_change(
    ax, summary, split, title, metric="roc_auc",
    baseline_size=FIGURE7_BASELINE_SIZE, tick_every=1, ylim=FIGURE7_PCT_YLIM,
    models=None,
):
    """Panel (b): percentage AUROC difference against the baseline-size model."""
    models = FIGURE7_MODELS if models is None else models
    sizes = sorted(summary[summary["eval_split"] == split]["n"].unique())
    for key, label, colour, marker in models:
        n, values = _figure7_series(summary, split, key, f"{metric}_mean")
        if len(n) == 0:
            continue
        at_baseline = np.where(n == baseline_size)[0]
        if len(at_baseline) == 0:
            print(f"  [skip] {key}: no result at size {baseline_size}")
            continue
        reference = values[at_baseline[0]]
        _figure7_line(
            ax, n, (values - reference) / reference * 100.0, colour, marker, label
        )
    ax.axhline(0, color="#b0b0ad", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    _figure7_style_axes(
        ax, sizes, "Percentage difference in AUROC", title, tick_every, ylim
    )


def _figure7_legend(fig, ax, ncol=6, y=-0.035):
    """One shared legend below the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, y), handlelength=1.8, columnspacing=1.6, labelcolor=_INK,
    )


def _save_figure7(fig, stem, dpi=400):
    """Write both the PNG and the PDF of one Figure 7 panel set."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    written = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        print(f"  wrote {path}")
    plt.close(fig)
    return written


def plot_figure7_combined(
    summary, stem, split="total", metric="roc_auc", show_sd=False,
    baseline_size=FIGURE7_BASELINE_SIZE,
):
    """Manuscript Figure 7: both panels side by side under one legend."""
    with plt.rc_context(FIGURE7_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
        draw_figure7_auroc(
            axes[0], summary, split, "(a) AUROC across training-set sizes",
            metric=metric, show_sd=show_sd,
        )
        draw_figure7_percentage_change(
            axes[1], summary, split,
            f"(b) Percentage difference in AUROC vs {baseline_size:,}-sample model",
            metric=metric, baseline_size=baseline_size,
        )
        _figure7_legend(fig, axes[0], ncol=6)
        return _save_figure7(fig, stem)


def plot_figure7_auroc_only(
    summary, stem, split="total", metric="roc_auc", show_sd=False
):
    """Panel (a) on its own, with its own legend."""
    with plt.rc_context(FIGURE7_RC):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
        draw_figure7_auroc(
            ax, summary, split, "AUROC across training-set sizes",
            metric=metric, show_sd=show_sd,
        )
        _figure7_legend(fig, ax, ncol=3, y=-0.06)
        return _save_figure7(fig, stem)


def plot_figure7_percentage_only(
    summary, stem, split="total", metric="roc_auc",
    baseline_size=FIGURE7_BASELINE_SIZE,
):
    """Panel (b) on its own, with its own legend."""
    with plt.rc_context(FIGURE7_RC):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
        draw_figure7_percentage_change(
            ax, summary, split,
            f"% difference in AUROC vs {baseline_size:,}-sample model",
            metric=metric, baseline_size=baseline_size,
        )
        _figure7_legend(fig, ax, ncol=3, y=-0.06)
        return _save_figure7(fig, stem)


def plot_figure7_by_cohort(summary, stem, metric="roc_auc", show_sd=False):
    """Supplementary three-panel view: total, Malaysia and India."""
    panels = [
        ("total", "(a) Total cohort"),
        ("malaysia", "(b) Malaysia cohort"),
        ("india", "(c) India cohort"),
    ]
    with plt.rc_context(FIGURE7_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9))
        for ax, (split, title) in zip(axes, panels):
            draw_figure7_auroc(
                ax, summary, split, title, metric=metric, show_sd=show_sd,
                tick_every=2, ylim=FIGURE7_BY_COHORT_YLIM,
            )
        _figure7_legend(fig, axes[0], ncol=6)
        return _save_figure7(fig, stem)


def figure7_plotted_values(
    summary, metric="roc_auc", baseline_size=FIGURE7_BASELINE_SIZE, models=None
):
    """Exactly what the panels draw, as a table - the figure's companion CSV."""
    models = FIGURE7_MODELS if models is None else models
    rows = []
    for split in ("total", "malaysia", "india"):
        for key, label, _, _ in models:
            n, values = _figure7_series(summary, split, key, f"{metric}_mean")
            if len(n) == 0:
                continue
            at_baseline = np.where(n == baseline_size)[0]
            if len(at_baseline) == 0:
                continue
            reference = values[at_baseline[0]]
            for size, value in zip(n, values):
                rows.append(
                    {
                        "cohort": split,
                        "model": label,
                        "training_size": int(size),
                        "auroc": round(float(value), 4),
                        f"pct_diff_vs_{baseline_size}": round(
                            float((value - reference) / reference * 100), 2
                        ),
                    }
                )
    return pd.DataFrame(rows)


def warn_if_figure7_clipped(
    summary, split="total", metric="roc_auc",
    baseline_size=FIGURE7_BASELINE_SIZE, models=None,
):
    """Report any series the fixed y-limits would crop, rather than cropping silently."""
    models = FIGURE7_MODELS if models is None else models
    messages = []
    for key, label, _, _ in models:
        n, values = _figure7_series(summary, split, key, f"{metric}_mean")
        if len(n) == 0:
            continue
        if values.min() < FIGURE7_AUROC_YLIM[0] or values.max() > FIGURE7_AUROC_YLIM[1]:
            messages.append(
                f"  AUROC {label}: {values.min():.4f}-{values.max():.4f} "
                f"outside {FIGURE7_AUROC_YLIM}"
            )
        at_baseline = np.where(n == baseline_size)[0]
        if len(at_baseline) == 0:
            continue
        reference = values[at_baseline[0]]
        percentage = (values - reference) / reference * 100.0
        if percentage.min() < FIGURE7_PCT_YLIM[0] or percentage.max() > FIGURE7_PCT_YLIM[1]:
            messages.append(
                f"  %diff {label}: {percentage.min():.2f}-{percentage.max():.2f} "
                f"outside {FIGURE7_PCT_YLIM}"
            )
    if messages:
        print("WARNING: fixed y-limits crop the following series:")
        print("\n".join(messages))
    return messages


# ── Cohort-level fairness across operating points ────────────────────────────


def plot_fairness_vs_threshold(sweep, save_path):
    """Per-cohort sensitivity and the signed Equal Opportunity Difference.

    Panel (b) carries the Newcombe interval band and a zero line, so a reader can
    see at which cut-offs the gap is distinguishable from zero at all - which is
    the point the Cohort-Level Fairness section makes.
    """
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(
        sweep["threshold"], sweep["tpr_malaysia"], color="#1F4E79",
        marker="o", markersize=3.5, label="Malaysia",
    )
    ax.plot(
        sweep["threshold"], sweep["tpr_india"], color="#C00000",
        marker="s", markersize=3.5, label="India",
    )
    ax.set_xlabel("Decision threshold (calibrated SGA risk)")
    ax.set_ylabel("True-positive rate")
    ax.set_title("(a) Sensitivity by cohort")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        sweep["threshold"], sweep["equal_opportunity_diff"], color="#1a1a1a",
        marker="o", markersize=3.5,
    )
    ax.fill_between(
        sweep["threshold"],
        sweep["equal_opportunity_diff_ci_low"],
        sweep["equal_opportunity_diff_ci_high"],
        color="#1a1a1a",
        alpha=0.12,
    )
    ax.axhline(0, color="#b0b0ad", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlabel("Decision threshold (calibrated SGA risk)")
    ax.set_ylabel("Equal Opportunity Difference (Malaysia - India)")
    ax.set_title("(b) Disparity vs operating point")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, save_path)


def plot_threshold_sweep(sweep, save_path):
    """Sensitivity and specificity against the cut-off, with 95% CI bands.

    ``sweep`` is the long-form table appendix Table S4 is built from: one row per
    ``(split, threshold)`` with ``sensitivity``/``specificity`` and their bounds.
    """
    apply_manuscript_style()
    splits = list(dict.fromkeys(sweep["split" if "split" in sweep else "cohort"]))
    key = "split" if "split" in sweep else "cohort"
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False)

    for ax, split in zip(axes[0], splits):
        subset = sweep[sweep[key] == split]
        ax.plot(
            subset["threshold"], subset["sensitivity"], color="#C00000",
            label="Sensitivity",
        )
        ax.fill_between(
            subset["threshold"], subset["sensitivity_ci_low"],
            subset["sensitivity_ci_high"], color="#C00000", alpha=0.15,
        )
        ax.plot(
            subset["threshold"], subset["specificity"], color="#1F4E79",
            label="Specificity",
        )
        ax.fill_between(
            subset["threshold"], subset["specificity_ci_low"],
            subset["specificity_ci_high"], color="#1F4E79", alpha=0.15,
        )
        ax.set_title(COHORT_LABELS.get(str(split), str(split)))
        ax.set_xlabel("Decision threshold (calibrated SGA risk)")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, save_path)
