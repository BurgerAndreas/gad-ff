import pandas as pd
import os
import wandb

import seaborn as sns
import matplotlib.pyplot as plt

from gadff.colours import COLOUR_LIST, METHOD_TO_COLOUR

api = wandb.Api()

# for .csv files
OUT_DIR = "results/"
os.makedirs(OUT_DIR, exist_ok=True)
OUTFILE = os.path.join(OUT_DIR, "reactbench.csv")

# for plots
PLOTS_DIR = "results_reactbench/plots/reactbench"
os.makedirs(PLOTS_DIR, exist_ok=True)


# Project is specified by <entity/project-name>
runs = api.runs("andreas-burger/reactbench")

summary_list, config_list, name_list = [], [], []
for run in runs:
    if "final" not in run.tags:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv(OUTFILE)
print(f"Saved csv to {OUTFILE}")

"""
Make a grouped bar plot
where each group is a metric
and each bar within a group is a model

colour by name of the method
name is based on hessian_method and calc fields

Metrics:
gsm_success
Number of successful GSM calculations that lead to an initial guess

converged_ts
local TS search (RS-P-RFO) converged
Convergence is defined as reaching all the following criteria within 50 steps: maximum force of $4.5e^{-4}$, RMS force of $3.0e^{-4}$, maximum step of $1.8e^{-3}$, RMS step of $1.2e^{-3}$ in atomic units (Hartree, Bohr), default in Gaussian.

ts_success
if is transition state according to frequency analysis

convert_ts
converged and ts_success
not the same as ts_success

irc_success (ignore)
IRC found two different geometries, both different from the initial transition state
(transition state, reactant, product)
This counts how many structures have energies different from the minimum.
Since one structure will always be 0 (the lowest energy):
At least 3 distinct energy levels were found (minimum + 2 others)

intended_count
converged to initial reactant and product
"""

rename_metrics = {
    "gsm_success": "GSM Success",
    "converged_ts": "RFO Converged",
    "ts_success": "TS Success",
    "convert_ts": "RFO Converged and TS Success",
    "irc_success": "IRC Success",  # ignore
    "intended_count": "IRC Intended",
}

# Try to use the eval export if present; otherwise derive from runs_df
try:
    df = pd.read_csv("results/eval_reactbench_wandb_export.csv", quotechar='"')
    df["Metric"] = df["Metric"].map(rename_metrics)
except Exception:
    records = []
    for _, row in runs_df.iterrows():
        cfg = row.get("config", {}) or {}
        summ = row.get("summary", {}) or {}
        base_method = str(cfg.get("hessian_method", "unknown"))
        calc = cfg.get("calc")
        calc_str = None if calc is None else str(calc)
        if calc_str and calc_str.lower() not in ["none", "nan", "na", ""]:
            method_label = f"{base_method}-{calc_str}"
        else:
            method_label = base_method
        for metric_key, metric_label in rename_metrics.items():
            value = summ.get(metric_key)
            if value is None:
                continue
            records.append(
                {
                    "Metric": metric_label,
                    "Value": value,
                    "Method": method_label,
                }
            )
    df = pd.DataFrame.from_records(records)

sns.set_theme(style="whitegrid", palette="pastel")

# data
allowed_metrics = [
    "GSM Success",
    "RFO Converged",
    "TS Success",
    "RFO Converged and TS Success",
    "IRC Intended",
]
df = df[df["Metric"].isin(allowed_metrics)]

# Build palette mapping from method labels to consistent colours
methods = list(pd.unique(df["Method"]))
palette = {}
colour_iter = iter(COLOUR_LIST)
for m in methods:
    colour = METHOD_TO_COLOUR.get(m)
    if colour is None:
        try:
            colour = next(colour_iter)
        except StopIteration:
            # fallback to seaborn palette cycling
            colour = sns.color_palette("pastel", len(methods)).as_hex()[
                methods.index(m) % len(methods)
            ]
    palette[m] = colour

order = allowed_metrics
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=df,
    x="Metric",
    y="Value",
    hue="Method",
    order=order,
    palette=palette,
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Count")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

outfile = os.path.join(PLOTS_DIR, "reactbench.png")
plt.savefig(outfile, dpi=300)
print(f"Saved plot to {outfile}")

# Second plot: difference to the previous stage (exclude GSM Success)
prev_map = {
    "RFO Converged": "GSM Success",
    "TS Success": "RFO Converged",
    "RFO Converged and TS Success": "TS Success",
    "IRC Intended": "RFO Converged and TS Success",
}

wide = df.pivot_table(
    index="Method", columns="Metric", values="Value", aggfunc="first"
).fillna(0)

delta_records = []
for method, row in wide.iterrows():
    for current_metric in order[1:]:  # skip GSM Success
        prev_metric = prev_map[current_metric]
        delta_value = row[prev_metric] - row[current_metric]
        delta_records.append(
            {
                "Metric": current_metric,
                "Value": float(delta_value),
                "Method": method,
            }
        )

delta_df = pd.DataFrame.from_records(delta_records)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=delta_df,
    x="Metric",
    y="Value",
    hue="Method",
    order=order[1:],
    palette=palette,
    ax=ax2,
)
ax2.set_xlabel("")
ax2.set_ylabel("Count difference vs previous stage")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
ax2.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

outfile2 = os.path.join(PLOTS_DIR, "reactbench_diff.png")
plt.savefig(outfile2, dpi=300)
print(f"Saved plot to {outfile2}")
