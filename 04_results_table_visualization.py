import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("results/results.csv")
OUT_DIR = Path("results/summary_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_PATH)

# Helper columns
df["model_dim"] = df["model"] + df["dims"].astype(str)
df["feature_norm"] = df["rep"] + " + " + df["norm"]

# Order
feature_order = [
    "T + none", "T + zscore",
    "FOURIER + none", "FOURIER + zscore",
    "WAVELET_STATS + none", "WAVELET_STATS + zscore"
]
model_order = ["ANN16", "SNN16", "ANN32", "SNN32"]

# Accuracy table (heatmap)
acc_table = df.pivot(
    index="feature_norm",
    columns="model_dim",
    values="test_acc"
).loc[feature_order, model_order]

fig = plt.figure(figsize=(10, 4.8))
ax = plt.gca()
im = ax.imshow(acc_table.values, aspect="auto")

ax.set_xticks(range(len(model_order)))
ax.set_yticks(range(len(feature_order)))
ax.set_xticklabels(model_order)
ax.set_yticklabels(feature_order)

for i in range(acc_table.shape[0]):
    for j in range(acc_table.shape[1]):
        val = acc_table.iloc[i, j]
        ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center")

ax.set_title("Test Accuracy (%)")
ax.set_xlabel("Model")
ax.set_ylabel("Feature Representation")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.savefig(OUT_DIR / "accuracy_table.png", dpi=200)
plt.close(fig)

# SNN efficiency (avg spikes)
snn_df = df[df["model"] == "SNN"]

spike_table = snn_df.pivot(
    index="feature_norm",
    columns="dims",
    values="avg_spikes"
).loc[feature_order]

fig = plt.figure(figsize=(8, 4.8))
ax = plt.gca()
im = ax.imshow(spike_table.values, aspect="auto")

ax.set_xticks(range(len(spike_table.columns)))
ax.set_yticks(range(len(feature_order)))
ax.set_xticklabels([f"SNN{d}" for d in spike_table.columns])
ax.set_yticklabels(feature_order)

for i in range(spike_table.shape[0]):
    for j in range(spike_table.shape[1]):
        val = spike_table.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center")

ax.set_title("SNN Average Spike Count")
ax.set_xlabel("Model")
ax.set_ylabel("Feature Representation")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.savefig(OUT_DIR / "snn_spike_table.png", dpi=200)
plt.close(fig)

# ANN vs SNN accuracy gap
gap_rows = []
for rep in df["rep"].unique():
    for norm in df["norm"].unique():
        for d in [16, 32]:
            ann = df[(df.rep == rep) & (df.norm == norm) & (df.dims == d) & (df.model == "ANN")]
            snn = df[(df.rep == rep) & (df.norm == norm) & (df.dims == d) & (df.model == "SNN")]
            if len(ann) and len(snn):
                gap_rows.append({
                    "feature_norm": f"{rep} + {norm}",
                    "dims": d,
                    "acc_gap": ann.test_acc.values[0] - snn.test_acc.values[0]
                })

gap_df = pd.DataFrame(gap_rows)

fig = plt.figure(figsize=(10, 4))
ax = plt.gca()

for d in [16, 32]:
    sub = gap_df[gap_df.dims == d]
    ax.plot(sub["feature_norm"], sub["acc_gap"] * 100, marker="o", label=f"{d} dims")

ax.axhline(0, linestyle="--")
ax.set_ylabel("Accuracy Gap (ANN − SNN) [%]")
ax.set_title("Accuracy Difference Between ANN and SNN")
ax.set_xticklabels(sub["feature_norm"], rotation=30, ha="right")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig.savefig(OUT_DIR / "ann_snn_accuracy_gap.png", dpi=200)
plt.close(fig)

print("Summary plots saved to:", OUT_DIR)
