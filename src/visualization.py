"""
visualization.py — Plotting helpers for the ACR'25 reproduction study.

Functions
---------
plot_by_class        : Overlay histogram of a feature split by class.
plot_categorical_dist: Bar charts (count + proportion) for categorical features.
plot_binary_dist     : Bar charts for binary features.
plot_continuous_dist : Histograms for the top-N highest-variance continuous features.
plot_spearman_corr   : Spearman correlation heatmap + high-correlation pair report.
plot_importance      : Horizontal bar chart of feature importances or coefficients.
iqr_outlier_count    : Count IQR-based outliers in a Series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def iqr_outlier_count(s: pd.Series) -> int:
    """Return the number of values outside 1.5 × IQR fences."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def plot_by_class(ax, df: pd.DataFrame, col: str, target: str,
                  class_map: dict, transform=None, bins: int = 80) -> None:
    """
    Overlay density histograms of `col` for each class in `class_map`.

    Parameters
    ----------
    ax        : Matplotlib Axes object.
    df        : DataFrame containing `col` and `target`.
    col       : Feature column name.
    target    : Binary target column name.
    class_map : {class_value: (label, color)} mapping.
    transform : Optional callable applied to each class slice before plotting.
    bins      : Number of histogram bins.
    """
    for cls, (lbl, color) in class_map.items():
        vals = df[df[target] == cls][col]
        if transform is not None:
            vals = transform(vals)
        ax.hist(vals, bins=bins, alpha=0.6, label=lbl, color=color, density=True)
    ax.legend()


def plot_categorical_dist(df: pd.DataFrame, cat_cols: list, target: str,
                          class_labels: list, title_prefix: str,
                          top_n: int = None) -> None:
    """
    Two-row bar chart panel: raw count (top) and class-normalised proportion (bottom)
    for each categorical feature.

    Parameters
    ----------
    df            : Training-set DataFrame.
    cat_cols      : List of categorical column names.
    target        : Binary target column name.
    class_labels  : [negative_label, positive_label] strings.
    title_prefix  : Dataset name used in the figure title.
    top_n         : If set, show only the N most frequent categories per feature.
    """
    if not cat_cols:
        print(f"{title_prefix}: no categorical features to plot.")
        return

    ncols = len(cat_cols)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9))
    if ncols == 1:
        axes = [[axes[0]], [axes[1]]]

    for j, feat in enumerate(cat_cols):
        if top_n is not None and df[feat].nunique() > top_n:
            top_cats = df[feat].value_counts().nlargest(top_n).index
            df_feat  = df[df[feat].isin(top_cats)]
            suffix   = f" (top {top_n})"
        else:
            df_feat = df
            suffix  = ""

        ct_raw = pd.crosstab(df_feat[feat], df_feat[target])
        ct_raw.columns = class_labels
        ct_raw.plot(kind="bar", ax=axes[0][j],
                    color=["steelblue", "tomato"], edgecolor="white",
                    legend=(j == 0))
        axes[0][j].set_title(f"{feat}{suffix} — raw count", fontsize=9)
        axes[0][j].set_xlabel("")
        axes[0][j].tick_params(axis="x", rotation=45)

        ct_norm = pd.crosstab(df_feat[feat], df_feat[target], normalize="index")
        ct_norm.columns = class_labels
        ct_norm.plot(kind="bar", ax=axes[1][j],
                     color=["steelblue", "tomato"], edgecolor="white", legend=False)
        axes[1][j].set_title(f"{feat}{suffix} — class proportion", fontsize=9)
        axes[1][j].set_xlabel("")
        axes[1][j].tick_params(axis="x", rotation=45)

    plt.suptitle(f"{title_prefix} — Categorical Features", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_binary_dist(df: pd.DataFrame, binary_cols: list, target: str,
                     class_labels: list, title_prefix: str,
                     ncols_per_row: int = 5) -> None:
    """
    Count + proportion bar charts for all binary features in one figure.

    Parameters
    ----------
    df            : Training-set DataFrame.
    binary_cols   : List of binary (0/1) column names.
    target        : Binary target column name.
    class_labels  : [negative_label, positive_label] strings.
    title_prefix  : Dataset name used in the figure title.
    ncols_per_row : Maximum features per row of subplots.
    """
    if not binary_cols:
        print(f"{title_prefix}: no binary features to plot.")
        return

    n = len(binary_cols)
    ncols = min(ncols_per_row, n)
    n_feat_rows  = (n + ncols - 1) // ncols
    total_rows   = 2 * n_feat_rows

    fig, axes = plt.subplots(total_rows, ncols,
                              figsize=(4 * ncols, 3.5 * total_rows))
    axes = np.array(axes).reshape(total_rows, ncols)

    for i, feat in enumerate(binary_cols):
        feat_row = i // ncols
        j        = i  % ncols
        for plot_row, (normalise, ylabel) in enumerate([(False, "Count"), (True, "Proportion")]):
            ax = axes[feat_row * 2 + plot_row, j]
            ct = pd.crosstab(df[feat], df[target],
                             normalize="index" if normalise else False)
            ct.columns = class_labels
            ct.plot(kind="bar", ax=ax,
                    color=["steelblue", "tomato"], edgecolor="white",
                    legend=(i == 0 and plot_row == 0))
            ax.set_title(f"{feat} — {'proportion' if normalise else 'count'}", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=0)

    for i in range(n, n_feat_rows * ncols):
        for plot_row in range(2):
            axes[(i // ncols) * 2 + plot_row, i % ncols].set_visible(False)

    plt.suptitle(f"{title_prefix} — Binary Features", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_continuous_dist(df: pd.DataFrame, cont_cols: list, target: str,
                         class_labels: list, title_prefix: str,
                         top_n: int = 6, ncols_per_row: int = 3) -> None:
    """
    Density histograms (log-Y) for the top-N highest-variance continuous features.

    Parameters
    ----------
    df            : Training-set DataFrame.
    cont_cols     : List of continuous column names.
    target        : Binary target column name.
    class_labels  : [negative_label, positive_label] strings.
    title_prefix  : Dataset name used in the figure title.
    top_n         : Number of features to plot (selected by variance).
    ncols_per_row : Subplot columns per row.
    """
    if not cont_cols:
        print(f"{title_prefix}: no continuous features to plot.")
        return

    top   = df[cont_cols].var().nlargest(top_n).index.tolist()
    nrows = (len(top) + ncols_per_row - 1) // ncols_per_row
    ncols = min(ncols_per_row, len(top))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    colors = ["steelblue", "tomato"]

    for i, feat in enumerate(top):
        ax = axes[i]
        for cls, (lbl, col) in enumerate(zip(class_labels, colors)):
            vals = df[df[target] == cls][feat]
            ax.hist(vals.to_numpy(), bins=50, alpha=0.6,
                    label=lbl, color=col, density=True)
        ax.set_yscale("log")
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel(feat, fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"{title_prefix} — Top {top_n} Continuous Features by Variance (density-normalised)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    print(f"Top-{top_n} by variance: {top}")


def plot_spearman_corr(df: pd.DataFrame, cont_cols: list, title: str,
                       threshold: float = 0.85, figsize: tuple = (9, 9)) -> None:
    """
    Spearman correlation heatmap (lower triangle) + console report of high-correlation pairs.

    Parameters
    ----------
    df        : Training-set DataFrame.
    cont_cols : Continuous column names to include.
    title     : Figure title.
    threshold : |r_s| above which pairs are flagged as highly correlated.
    figsize   : Figure size.
    """
    corr_matrix = df[cont_cols].corr(method="spearman")

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.2,
                linecolor="white", annot=False)
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()

    high_corr = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_matrix.iloc[i, j]
            if abs(v) > threshold:
                high_corr.append((cols[i], cols[j], round(v, 3)))

    high_corr.sort(key=lambda x: -abs(x[2]))
    print(f"Pairs with |r_s| > {threshold}: {len(high_corr)} found")
    for a, b, r in high_corr:
        print(f"  {a} ↔ {b}: r_s = {r}")
    print(f"\nInsight: {len(high_corr)} highly correlated pairs indicate redundancy.")


def plot_importance(pipeline, feature_names: list, title: str, top_n: int = 15) -> None:
    """
    Horizontal bar chart of the top-N feature importances or absolute coefficients.

    Works with tree-based models (feature_importances_) and linear models (coef_).

    Parameters
    ----------
    pipeline      : Fitted sklearn Pipeline whose last step is the classifier.
    feature_names : Ordered list of feature names (after preprocessing).
    title         : Plot title.
    top_n         : Number of top features to display.
    """
    model = pipeline[-1]

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    else:
        print(f"No importance attribute found for {title}")
        return

    top = imp.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Importance Score (Absolute Weight / Gain)")
    plt.tight_layout()
    plt.show()
