import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""
Make a grouped bar plot
where each group is a metric
and each bar within a group is a model

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
Since one structure will always be 0 (the lowest energy), having ≥2 non-zero values means:
At least 3 distinct energy levels were found (minimum + 2 others)

intended_count
converged to initial reactant and product
"""

rename_metrics = {
    "gsm_success": "GSM Success",
    "converged_ts": "RFO Converged",
    "ts_success": "TS Success",
    "convert_ts": "RFO Converged and TS Success",
    "irc_success": "IRC Success", # ignore
    "intended_count": "IRC Intended",
}

df = pd.read_csv("results/eval_reactbench_wandb_export.csv", quotechar='"')
df["Metric"] = df["Metric"].map(rename_metrics)

sns.set_theme(
    style="whitegrid",
    palette="pastel"
)

# data
df = df[df["Metric"].isin(["GSM Success", "RFO Converged", "TS Success", "RFO Converged and TS Success", "IRC Intended"])]
df = df.sort_values(by="Metric", ascending=True)