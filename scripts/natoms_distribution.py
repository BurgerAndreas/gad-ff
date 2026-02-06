import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import fix_dataset_path

DATASETS = {
    "ts1x-val": "ts1x-val.lmdb",
    "RGD1": "RGD1.lmdb",
}


def get_natoms(dataset_name, lmdb_name, max_samples=None):
    out_path = f"results_evalhorm/{dataset_name}_natoms.csv"
    if os.path.exists(out_path):
        print(f"Loaded {out_path}")
        return pd.read_csv(out_path)
    dataset = LmdbDataset(fix_dataset_path(lmdb_name))
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    natoms_list = []
    for i in tqdm(range(n), desc=dataset_name):
        data = dataset[i]
        natoms_list.append(data.pos.shape[0])
    df = pd.DataFrame({"natoms": natoms_list})
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} samples)")
    return df


if __name__ == "__main__":
    os.makedirs("results_evalhorm", exist_ok=True)
    os.makedirs("plots/eval_horm", exist_ok=True)

    dfs = {}
    for name, lmdb_name in DATASETS.items():
        dfs[name] = get_natoms(name, lmdb_name, max_samples=None)

    # Print counts per natoms
    for name, df in dfs.items():
        print(f"\n--- {name} ({len(df)} total samples) ---")
        counts = df["natoms"].value_counts().sort_index()
        for n, c in counts.items():
            print(f"  natoms={n}: {c}")

    # Combined histogram
    legend_names = {"ts1x-val": "T1x", "RGD1": "RGD1"}
    combined = pd.concat(
        [df.assign(dataset=legend_names.get(name, name)) for name, df in dfs.items()],
        ignore_index=True,
    )

    sns.set_theme(style="whitegrid", context="poster")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(
        data=combined,
        x="natoms",
        hue="dataset",
        stat="percent",
        common_norm=False,
        multiple="dodge",
        discrete=True,
        shrink=0.8,
        ax=ax,
    )
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Percentage of Samples")
    legend = ax.get_legend()
    legend.set_title("Dataset")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    plt.tight_layout()

    plot_path = "plots/eval_horm/natoms_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {plot_path}")
    plt.show()
