import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example data
data = {
    "Model": ["AlphaNet", "LEFTNet", "LEFTNet-df", "EquiformerV2"],
    "GSM": [828, 852, 895, 888],
    "Intended": [650, 661, 55, 3],
}

df = pd.DataFrame(data)

# Melt to long format for seaborn
df_long = df.melt(
    id_vars="Model",
    value_vars=["GSM", "Intended"],
    var_name="Category",
    value_name="Reactions",
)

# Define colors
palette = {"GSM": "skyblue", "Intended": "lightcoral"}

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    data=df_long, x="Model", y="Reactions", hue="Category", palette=palette, alpha=0.3
)

# Add scatter points + annotations
for i, row in df.iterrows():
    # GSM
    ax.scatter(i, row["GSM"], color="skyblue", s=100, edgecolor="k", zorder=3)
    ax.text(i, row["GSM"] + 10, str(row["GSM"]), ha="center", va="bottom", fontsize=10)

    # Intended
    ax.scatter(i, row["Intended"], color="lightcoral", s=60, edgecolor="k", zorder=3)
    ax.text(
        i,
        row["Intended"] + 10,
        str(row["Intended"]),
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Styling
ax.set_ylabel("Number of Reactions")
ax.set_xlabel("")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
sns.despine()

plt.legend(title="", loc="upper center", frameon=False)
plt.tight_layout()
fname = "results/plots/reactbench.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()


####################################################################################################

sns.set_theme(
    # context: dict[str, Any] | Literal['paper', 'notebook', 'talk', 'poster'] = "notebook"
    style="whitegrid",
    palette="pastel",
)

# data
df = pd.DataFrame(
    {
        "Model": ["AlphaNet", "LEFTNet", "LEFTNet-df", "EquiformerV2"],
        "GSM Success(E-F)": [828, 852, 895, 888],
        "Intended(E-F)": [650, 661, 55, 3],
    }
)
# sort s.t. smallest is first
df = df.sort_values(by="GSM Success(E-F)", ascending=True)
long = df.melt(id_vars="Model", var_name="Series", value_name="y")


# params
palette = {"GSM Success(E-F)": "tab:blue", "Intended(E-F)": "tab:red"}
ms = 14  # scatter markersize (points)
lw = ms / 2  # line width chosen to look equal to ball diameter
alpha_stem = 0.25

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 5))

# x positions by category
xpos = np.arange(len(df["Model"]))

# draw stems and balls, no dodge (same x for both series)
for i, s in enumerate(long["Series"].unique()):
    col = palette[s]
    yvals = df[s].values
    # stems
    ax.vlines(
        x=xpos,
        ymin=0,
        ymax=yvals,
        colors=col,
        linewidth=lw * (1 + i / 2),
        alpha=alpha_stem,
    )
    # balls
    ax.scatter(
        xpos,
        yvals,
        s=ms * 2 * np.pi * (1 + i / 2),
        color=col,
        # edgecolor="black",
        edgecolor=col,
        linewidth=0.0,  # 0.5,
        zorder=3,
        label=s,
    )
    # # labels
    # for x, y in zip(xpos, yvals):
    #     ax.text(x, y + 10, f"{y}", ha="center", va="bottom", fontsize=11)

# axes cosmetics
ax.set_xticks(xpos, df["Model"])
ax.set_ylabel("Number of Reactions")
ax.set_ylim(0, 1000)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
# remove x axis grid
ax.grid(False, axis="x")
sns.despine()
ax.legend(loc="upper center", frameon=False, fontsize=12, title="")

plt.tight_layout()
fname = "results/plots/reactbench_stem.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()
