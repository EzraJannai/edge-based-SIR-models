import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the trajectories file from the new solver
df = pd.read_csv("trajectories.csv")

outdir = os.path.join("plots", "trajectories")
os.makedirs(outdir, exist_ok=True)

# Match solver naming: DSMA = EBMA (fleeting), CM = Static (static EBCM)
rename_map = {"DSMA": "EBMA", "CM": "Static"}
df["model"] = df["model"].replace(rename_map)

# Also rename R0_target to R0 for plotting convenience
if "R0_target" in df.columns:
    df = df.rename(columns={"R0_target": "R0"})

colors = {"EBMA": "#ca005d", "Static": "#007bc7"}
R0_values = sorted(df["R0"].unique())

fig, axes = plt.subplots(2, len(R0_values), figsize=(15, 8), sharex=True)
for axrow in axes:
    for ax in np.atleast_1d(axrow):
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

time_xlim = (0, 70)
I_ylim = (0, 0.08)
kI_ylim = (0, 6)
R_ylim = (0, 0.4)

legend_handles, legend_labels = [], []

for j, R0 in enumerate(R0_values):
    df_R0 = df.query("R0 == @R0")

    axI = axes[0, j]
    axk = axI.twinx()

    for model in ["EBMA", "Static"]:
        d = df_R0.query("model == @model").copy()
        if d.empty:
            continue

        cutoff = 1e-7
        d.loc[d["I"] < cutoff, "mean_kI"] = np.nan
        d["mean_kI"] = d["mean_kI"].ffill()

        h1, = axI.plot(d["time"], d["I"], color=colors[model],
                       linestyle="-", label=f"{model} I(t)")
        h2, = axk.plot(d["time"], d["mean_kI"], color=colors[model],
                       linestyle=":", label=f"{model} ⟨k⟩₍ᵢ₎")

        if j == 0:
            legend_handles.extend([h1, h2])
            legend_labels.extend([h1.get_label(), h2.get_label()])

    axI.set_xlim(time_xlim)
    axI.set_ylim(I_ylim)
    axk.set_ylim(kI_ylim)
    axI.set_title(f"$R_0 = {R0}$")
    if j == 0:
        axI.set_ylabel("Prevalence I(t)")
        axk.set_ylabel("Mean infected degree")

    axR = axes[1, j]
    for model in ["EBMA", "Static"]:
        d = df_R0.query("model == @model")
        hR, = axR.plot(d["time"], d["R"], color=colors[model],
                       linestyle="-", label=f"{model} R(t)")
        if j == 0:
            legend_handles.append(hR)
            legend_labels.append(hR.get_label())

    axR.set_xlim(time_xlim)
    axR.set_ylim(R_ylim)
    if j == 0:
        axR.set_ylabel("Recovered R(t)")

for ax in axes[1, :]:
    ax.set_xlabel("Time (1/γ units)")

fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4)
plt.tight_layout(rect=[0, 0, 1, 0.92])

outfile = os.path.join(outdir, "trajectories")
fig.savefig(outfile + ".png", dpi=300)
fig.savefig(outfile + ".pdf")
plt.close(fig)

print(f"Saved plots to {outfile}.png and .pdf")
