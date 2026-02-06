import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS = {
    "HIP": "results_evalhorm/hesspred_v2_RGD1_predict_metrics.csv",
    "AD": "results_evalhorm/eqv2_RGD1_autograd_metrics.csv",
    "AD (E-F)": "results_evalhorm/eqv2_orig_RGD1_autograd_metrics.csv",
    # "HIP": "../hip/results_evalhorm/hip_v2_RGD1_predict_metrics.csv",
}

METRICS = [
    ("hessian_mae", r"Hessian MAE [eV/$\AA^2$]"),
    ("hessian_mre", "Hessian MRE"),
    ("eigvec1_cos_eckart", r"CosSim $\mathbf{v}_1$"),
    ("eigval_mae_eckart", r"$\lambda$ MAE [eV/$\AA^2$]"),
    ("eigval_mre_eckart", r"$\lambda$ MRE"),
    ("eigval1_mae_eckart", r"$\lambda_1$ MAE [eV/$\AA^2$]"),
    ("eigval1_mre_eckart", r"$\lambda_1$ MRE"),
    ("eigvec_overlap_error", r"$\lVert\lvert Q_{\text{model}} Q_{\text{true}}^T \rvert - I \rVert_F$"),
]

HESSIAN_METHOD_TO_COLOUR = {
    "HIP": "#ae5a41",
    "AD": "#295c7e",
    "AD (E-F)": "#5a5255",
}

if __name__ == "__main__":
    plot_dir = "plots/eval_horm"
    os.makedirs(plot_dir, exist_ok=True)

    dfs = {}
    for label, path in RESULTS.items():
        dfs[label] = pd.read_csv(path)
        print(f"Loaded {path} ({len(dfs[label])} samples)")

    sns.set_theme(style="whitegrid", context="poster")
    
    for include_e_f in [True, False]:
        if not include_e_f:
            dfs = {label: df for label, df in dfs.items() if label != "AD (E-F)"}

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
                    color=HESSIAN_METHOD_TO_COLOUR[label],
                )
                color = line[0].get_color()
                ax.fill_between(
                    grouped["natoms"],
                    grouped["mean"] - grouped["se"],
                    grouped["mean"] + grouped["se"],
                    alpha=0.2,
                    color=color,
                )

            ax.set_xlim(4.5, 32.5)
            ax.set_xlabel("Number of Atoms")
            ax.set_ylabel(ylabel)
            legend = ax.legend()
            legend.set_title("Model")
            legend.get_frame().set_edgecolor("none")
            legend.get_frame().set_alpha(1.0)
            plt.tight_layout()

            plot_path = f"{plot_dir}/{col}_compare_rgd1{'_e-f' if include_e_f else ''}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved {plot_path}")
            plt.show()
