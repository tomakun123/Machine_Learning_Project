"""
Exploratory Data Analysis for Model_Dataset.csv

Creates 4 figures in Figures/:
  1. feature_distributions.png  --> histogram + density curve for each of the 13 features, plus class balance of the target
  2. features_by_target.png     --> density of each feature split by Target (next day Up vs Down)
  3. up_rate_by_decile.png      --> % of next days that were Up, for each decile (10% bucket) of each feature
  4. correlation_matrix.png     --> correlation between all features (and the target)
Also prints each feature's correlation with the target to the terminal.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

### Setup ###
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # .../Exploratory & Unsupervised Analysis
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                            # .../Machine_Learning_Project
DATA_PATH = os.path.join(PROJECT_DIR, "Datasets", "Model_Dataset.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")

data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
features = [c for c in data.columns if c != "Target"]
os.makedirs(FIG_DIR, exist_ok=True)

UP = "#2a78d6"       # Target = 1 (next day closes higher)
DOWN = "#eb6834"     # Target = 0 (next day closes lower)
NEUTRAL = "#52514e" 
INK = "#0b0b0b"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": NEUTRAL,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

LABELS = {
    "1D_Return": "AIQ 1-day return",
    "5D_Return": "AIQ 5-day return",
    "20D_Return": "AIQ 20-day return",
    "20D_Volatility": "AIQ 20-day volatility",
    "Dist_50D_MA": "Distance from 50-day MA",
    "Up_Day_Streak": "Up/down-day streak",
    "Overnight_Gap": "Overnight gap",
    "RSI": "RSI (14-day)",
    "Rel_Volume": "Relative volume",
    "QQQ_1D_Return": "QQQ 1-day return",
    "VIX_Level": "VIX level",
    "VIX_1D_Change": "VIX 1-day change",
    "SMH_1D_Return": "SMH 1-day return",
}

def x_limits(series):
    """Zoom the x-axis to the 0.5th-99.5th percentile so a few extreme days (e.g. the COVID crash) don't squash the plot."""
    lo, hi = series.quantile([0.005, 0.995])
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


### Figure 1: Feature distributions ###
fig, axes = plt.subplots(4, 4, figsize=(16, 13))
axes = axes.flatten()

for ax, col in zip(axes, features):
    s = data[col]
    if col == "Up_Day_Streak":
        counts = s.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=UP, width=0.8)
    else:
        lo, hi = x_limits(s)
        sns.histplot(s[(s >= lo) & (s <= hi)], bins=40, stat="density", color=UP, alpha=0.35,
                     edgecolor="#fcfcfb", linewidth=0.5, ax=ax)
        sns.kdeplot(s, color=UP, linewidth=2, ax=ax, clip=(lo, hi))
        ax.set_xlim(lo, hi)
    ax.axvline(s.mean(), color=NEUTRAL, linestyle="--", linewidth=1)
    ax.set_title(LABELS[col])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(0.98, 0.95, f"mean {s.mean():.3g}\nstd {s.std():.3g}\nskew {s.skew():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=NEUTRAL)

ax = axes[len(features)]
counts = data["Target"].value_counts().sort_index()
bars = ax.bar(["Down (0)", "Up (1)"], counts.values, color=[DOWN, UP], width=0.6)
for bar, n in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n}\n({n / counts.sum():.1%})",
            ha="center", va="bottom", fontsize=8, color=INK)
ax.set_ylim(0, counts.max() * 1.25)
ax.set_title("Target class balance")
for ax in axes[len(features) + 1:]:
    ax.set_visible(False)

fig.suptitle("Distribution of each feature (dashed line = mean; x-axis trimmed to 0.5th–99.5th percentile)",
             fontsize=14, color=INK, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "feature_distributions.png"), dpi=150)
plt.close(fig)


### Figure 2: Feature distributions split by target ###
fig, axes = plt.subplots(4, 4, figsize=(16, 13))
axes = axes.flatten()
for ax, col in zip(axes, features):
    s = data[col]
    lo, hi = x_limits(s)
    for target, color, name in [(0, DOWN, "Next day Down"), (1, UP, "Next day Up")]:
        subset = s[data["Target"] == target]
        if col == "Up_Day_Streak":
            share = subset.value_counts(normalize=True).sort_index()
            offset = -0.2 if target == 0 else 0.2
            ax.bar(share.index + offset, share.values, width=0.4, color=color, label=name)
        else:
            sns.kdeplot(subset, color=color, linewidth=2, ax=ax, label=name, clip=(lo, hi))
            ax.axvline(subset.mean(), color=color, linestyle="--", linewidth=1)
    if col != "Up_Day_Streak":
        ax.set_xlim(lo, hi)

    t, p = stats.ttest_ind(s[data["Target"] == 1], s[data["Target"] == 0], equal_var=False)
    ax.text(0.98, 0.95, f"t-test p = {p:.3f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=NEUTRAL)
    ax.set_title(LABELS[col])
    ax.set_xlabel("")
    ax.set_ylabel("")

legend_ax = axes[len(features)]
legend_ax.set_visible(True)
legend_ax.axis("off")
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc="center", fontsize=11, frameon=False,
                 title="Target", title_fontsize=11)
for ax in axes[len(features) + 1:]:
    ax.set_visible(False)

fig.suptitle("Feature distributions by target: today's features before an Up day vs a Down day (dashed = class mean)",
             fontsize=14, color=INK, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "features_by_target.png"), dpi=150)
plt.close(fig)


### Figure 3: % of Up days within each decile of each feature ###
base_rate = data["Target"].mean()
fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharey=True)
axes = axes.flatten()
for ax, col in zip(axes, features):
    decile = pd.qcut(data[col].rank(method="first"), 10, labels=range(1, 11))
    up_rate = data.groupby(decile, observed=True)["Target"].mean()
    ax.bar(up_rate.index.astype(int), up_rate.values, color=UP, width=0.75)
    ax.axhline(base_rate, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 0.75)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(LABELS[col])
    ax.set_xlabel("Decile (1 = lowest values, 10 = highest)")
for ax in axes[len(features):]:
    ax.set_visible(False)
fig.suptitle(f"% of next days that closed Up, by feature decile (dashed = overall base rate {base_rate:.1%})",
             fontsize=14, color=INK, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "up_rate_by_decile.png"), dpi=150)
plt.close(fig)


### Figure 4: Correlation matrix ###
corr = data[features + ["Target"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide the duplicate upper triangle
diverging = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2f"])
tick_labels = [LABELS.get(c, "Target (next day Up)") for c in corr.columns]

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, mask=mask, cmap=diverging, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=1.5, linecolor="#fcfcfb", square=True,
            xticklabels=tick_labels, yticklabels=tick_labels,
            cbar_kws={"shrink": 0.7, "label": "Pearson correlation"}, ax=ax)
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("Correlation matrix of features and target (blue = negative, red = positive)",
             fontsize=14, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"), dpi=150)
plt.close(fig)


### Summary: correlation of each feature with the target ###
summary = pd.DataFrame({
    "corr_with_target": corr["Target"].drop("Target"),
    "t_test_p_value": [stats.ttest_ind(data.loc[data["Target"] == 1, c], data.loc[data["Target"] == 0, c],
                                       equal_var=False).pvalue for c in features],
}).sort_values("corr_with_target", key=abs, ascending=False)
print("Correlation of each feature with the Target (sorted by strength):")
print(summary.round(4).to_string())

# Pairs of features that are strongly correlated with each other (possible redundancy)
pairs = corr.loc[features, features].where(~np.tril(np.ones((len(features),) * 2, dtype=bool))).stack()
print("\nFeature pairs with |correlation| > 0.5:")
print(pairs[pairs.abs() > 0.5].sort_values(key=abs, ascending=False).round(3).to_string())
print(f"\nSaved 4 figures to {FIG_DIR}")
