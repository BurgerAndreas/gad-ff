import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS = {
    "HIP": "results_evalhorm/hesspred_v2_RGD1_predict_metrics.csv",
    "AD": "results_evalhorm/eqv2_RGD1_autograd_metrics.csv",
    # "AD (E-F)": "results_evalhorm/eqv2_orig_RGD1_autograd_metrics.csv",
    # "HIP": "../hip/results_evalhorm/hip_v2_RGD1_predict_metrics.csv",
}

METRICS = [
    ("hessian_mae", "Hessian MAE"),
    ("eigvec1_cos_eckart", "Eigvec 1 Cosine Similarity"),
    ("eigval_mae_eckart", "Eigval MAE (Eckart)"),
    ("eigval1_mae_eckart", "Eigval 1 MAE (Eckart)"),
]

if __name__ == "__main__":
    plot_dir = "plots/eval_horm"
    os.makedirs(plot_dir, exist_ok=True)

    dfs = {}
    for label, path in RESULTS.items():
        dfs[label] = pd.read_csv(path)
        print(f"Loaded {path} ({len(dfs[label])} samples)")

    sns.set_theme(style="whitegrid", context="poster")

    for col, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(8, 8))

        for label, df in dfs.items():
            if col not in df.columns:
                continue
            grouped = df.groupby("natoms")[col].agg(["mean", "std", "count"]).reset_index()
            grouped["se"] = grouped["std"] / grouped["count"] ** 0.5
            line = ax.plot(
                grouped["natoms"],
                grouped["mean"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=label,
            )
            color = line[0].get_color()
            ax.fill_between(
                grouped["natoms"],
                grouped["mean"] - grouped["se"],
                grouped["mean"] + grouped["se"],
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Number of Atoms")
        ax.set_ylabel(ylabel)
        legend = ax.legend()
        legend.set_title("Model")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(1.0)
        plt.tight_layout()

        plot_path = f"{plot_dir}/{col}_compare_rgd1.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved {plot_path}")
        plt.show()
