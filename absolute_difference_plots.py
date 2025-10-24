import pandas as pd
import matplotlib.pyplot as plt
import os

data_file = "simulation_results.csv"
outdir = os.path.join("plots", "comparisons")
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(data_file)

# Rename columns to match your plotting conventions
df = df.rename(columns={
    "R0_target": "R0",
    "final_EBMA": "final_DSMA",
    "peak_EBMA": "peak_DSMA",
    "tpeak_EBMA": "tpeak_DSMA",
    "final_Static": "final_CM",
    "peak_Static": "peak_CM",
    "tpeak_Static": "tpeak_CM"
})

# Focus on power-law distributions
df = df[df["distribution"].str.contains("Power law")].copy()

styles = {
    2.0: {"color": "#9e9ac8", "linestyle": ":"},
    2.5: {"color": "#756bb1", "linestyle": "--"},
    3.0: {"color": "#54278f", "linestyle": "-"},
}

R0_xlim = (0.5, 3.0)
dI_ylim = (-0.1, 0.1)
dt_ylim = (-40, 10)
dR_ylim = (-0.1, 0.1)

# === Compute differences ===
df["dImax"] = df["peak_CM"] - df["peak_DSMA"]
df["dRinf"] = df["final_CM"] - df["final_DSMA"]
df["dtpeak"] = df["tpeak_CM"] - df["tpeak_DSMA"]

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex="col")

# Δ Imax
ax = axes[0]
for alpha, style in styles.items():
    d = df[df["distribution"].str.contains(f"α={alpha}")]
    ax.plot(d["R0"], d["dImax"], **style, label=f"$\\alpha={alpha}$")
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlim(R0_xlim)
ax.set_ylim(dI_ylim)
ax.set_ylabel(r"$\Delta I_{\max}$")
ax.set_title("Peak prevalence (CM − DSMA)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Δ tpeak
ax = axes[1]
for alpha, style in styles.items():
    d = df[df["distribution"].str.contains(f"α={alpha}")]
    ax.plot(d["R0"], d["dtpeak"], **style)
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlim(R0_xlim)
ax.set_ylim(dt_ylim)
ax.set_ylabel(r"$\Delta t_{\max}$")
ax.set_title("Peak timing (CM − DSMA)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Δ R∞
ax = axes[2]
for alpha, style in styles.items():
    d = df[df["distribution"].str.contains(f"α={alpha}")]
    ax.plot(d["R0"], d["dRinf"], **style)
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlim(R0_xlim)
ax.set_ylim(dR_ylim)
ax.set_ylabel(r"$\Delta R(\infty)$")
ax.set_title("Final size (CM − DSMA)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Shared x-labels
for ax in axes:
    ax.set_xlabel(r"Target $R_0^\star$")
    ax.tick_params(axis="x", which="both", labelbottom=True)

# Legend
fig.legend(
    [plt.Line2D([0], [0], color=style["color"], linestyle=style["linestyle"])
     for alpha, style in styles.items()],
    [f"Power law $\\alpha={alpha}$" for alpha in styles.keys()],
    loc="upper center", ncol=3
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save
outfile = os.path.join(outdir, "summary_differences")
fig.savefig(outfile + ".png", dpi=300)
fig.savefig(outfile + ".pdf")
plt.close(fig)

print(f"Saved figure to {outfile}.png and .pdf")
