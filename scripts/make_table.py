"""Read CSV results from results_evalhorm/ and print a LaTeX table or bar plot."""

import pandas as pd
import sys
import os
import glob
import argparse

RESULTS_DIR = "results_evalhorm"

# Columns to extract and their display format
# (csv_col, header, format_str, lower_is_better)
COLUMNS = [
    ("hessian_mae", "Hessian", "{:.3f}", True),
    ("eigval_mae_eckart", "Eigenvalues", "{:.3f}", True),
    ("eigvec1_cos_eckart", r"CosSim $\bm{v}_1$", "{:.3f}", False),
    ("eigval1_mae_eckart", r"$\lambda_1$", "{:.3f}", True),
    ("time", "Time", "{:.1f}", True),
]

# Map canonical column names to possible aliases in older CSV formats
COLUMN_ALIASES = {
    "hessian_mae": ["hessian_mae", "hessian_error"],
    "eigval_mae_eckart": ["eigval_mae_eckart"],
    "eigvec1_cos_eckart": ["eigvec1_cos_eckart"],
    "eigval1_mae_eckart": ["eigval1_mae_eckart"],
    "time": ["time"],
}


def _parse_col(series):
    """Parse a column, handling tensor(...) strings from older CSVs."""
    import re
    def _parse_val(x):
        if isinstance(x, (int, float)):
            return x
        s = str(x).strip()
        m = re.search(r"tensor\(([-\d.eE+]+)", s)
        if m:
            return float(m.group(1))
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")
    return series.map(_parse_val)


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = _parse_col(df[col])
    row = {}
    for csv_col, _, _, _ in COLUMNS:
        aliases = COLUMN_ALIASES.get(csv_col, [csv_col])
        found = False
        for alias in aliases:
            if alias in df.columns:
                row[csv_col] = df[alias].mean()
                found = True
                break
        if not found:
            print(f"Warning: column {csv_col} not found in {csv_path}", file=sys.stderr)
            row[csv_col] = float("nan")
    return row


def make_table(csv_files, labels, groups=None, bold_best=False):
    """
    csv_files: list of paths
    labels: list of display names (same length)
    groups: list of (group_name, count) tuples, e.g. [("AD", 5), ("Predicted", 2)]
            If None, no grouping.
    bold_best: if True, bold the best value in each column
    """
    rows = []
    orig_flags = []
    for f in csv_files:
        rows.append(load_results(f))
        orig_flags.append("_orig" in os.path.basename(f))

    # Find best per column if needed
    best = {}
    if bold_best:
        for csv_col, _, _, lower_better in COLUMNS:
            vals = [r[csv_col] for r in rows]
            if lower_better:
                best[csv_col] = min(vals)
            else:
                best[csv_col] = max(vals)

    # Header
    header_top = r"\multirow{2}{*}{Hessian} & \multirow{2}{*}{Model} & \multirow{2}{*}{Hessian Data}"
    header_bot = " & & "
    units = {
        "hessian_mae": r"eV/\AA$^2$",
        "eigval_mae_eckart": r"eV/\AA$^2$",
        "eigvec1_cos_eckart": "unitless",
        "eigval1_mae_eckart": r"eV/\AA$^2$",
        "time": "ms",
    }
    arrows = {True: r"$\downarrow$", False: r"$\uparrow$"}

    for csv_col, name, _, lower in COLUMNS:
        header_top += f" & {name} {arrows[lower]}"
        header_bot += f" & {units[csv_col]}"
    header_top += r" \\"
    header_bot += r" \\"

    ncols = 3 + len(COLUMNS)
    lines = []
    lines.append(r"\begin{tabular}{" + "c" * ncols + "}")
    lines.append(r"\hline")
    lines.append(header_top)
    lines.append(header_bot)
    lines.append(r"\hline")

    # Rows
    idx = 0
    if groups:
        for gname, gcount in groups:
            lines.append(r"\multirow{" + str(gcount) + r"}{*}{" + gname + "}")
            for j in range(gcount):
                row = rows[idx]
                label = labels[idx]
                orig = "" if orig_flags[idx] else r"\checkmark"
                cells = f" & {label} & {orig}"
                for csv_col, _, fmt, _ in COLUMNS:
                    val = fmt.format(row[csv_col])
                    if bold_best and row[csv_col] == best[csv_col]:
                        val = r"\textbf{ " + val + r" }"
                    cells += f" & {val}"
                cells += r" \\"
                lines.append(cells)
                idx += 1
            lines.append(r"\hline")
    else:
        for i, (row, label) in enumerate(zip(rows, labels)):
            orig = "" if orig_flags[i] else r"\checkmark"
            cells = f" & {label} & {orig}"
            for csv_col, _, fmt, _ in COLUMNS:
                val = fmt.format(row[csv_col])
                if bold_best and row[csv_col] == best[csv_col]:
                    val = r"\textbf{ " + val + r" }"
                cells += f" & {val}"
            cells += r" \\"
            lines.append(cells)
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _model_key(filename):
    """Extract model key from filename for colour mapping."""
    base = os.path.basename(filename).replace("_metrics.csv", "")
    # Strip suffixes to get the model family
    for suffix in ["_orig", "_ts1x-val", "_autograd", "_predict"]:
        base = base.replace(suffix, "")
    # Map known prefixes
    if base.startswith("alpha"):
        return "alphanet"
    if base.startswith("left-df"):
        return "leftnet-df"
    if base.startswith("left"):
        return "leftnet"
    if base.startswith("eqv2") or base.startswith("eq_l1"):
        return "eqv2"
    if "hesspred" in base or base.startswith("hip"):
        return "hesspred"
    return None


PLOT_DIR = "plots/eval_horm"


def make_plot(csv_files, labels, output=None):
    """Bar plot of hessian MAE using seaborn poster style."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from gadff.colours import METHOD_TO_COLOUR, HESSIAN_METHOD_TO_COLOUR

    sns.set_context("poster")
    sns.set_style("whitegrid")

    # Load data
    values = []
    colours = []
    for f, label in zip(csv_files, labels):
        row = load_results(f)
        values.append(row["hessian_mae"])
        if "_orig" in os.path.basename(f):
            colours.append(HESSIAN_METHOD_TO_COLOUR["E-F"])
        elif "predict" in f:
            colours.append(HESSIAN_METHOD_TO_COLOUR["predict"])
        else:
            colours.append(HESSIAN_METHOD_TO_COLOUR["autograd"])

    # Hard-coded display order, grouped in pairs
    LABEL_ORDER = [
        "AlphaNet (E-F)", "AlphaNet",
        "LEFTNet-DF (E-F)", "LEFTNet-DF",
        "LEFTNet (E-F)", "LEFTNet",
        "EquiformerV2 (E-F)", "EquiformerV2",
        "HIP-EquiformerV2", "HIP-EquiformerV2*",
    ]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    order = [label_to_idx[l] for l in LABEL_ORDER if l in label_to_idx]
    values = [values[i] for i in order]
    colours = [colours[i] for i in order]
    labels = [labels[i] for i in order]

    # X positions with gaps between pairs
    positions = []
    for i in range(len(labels)):
        positions.append(i + (i // 2) * 0.5)

    label_fontsize = sns.plotting_context()["font.size"] * 0.65
    os.makedirs(PLOT_DIR, exist_ok=True)
    for log_scale in [False, True]:
        fig, ax = plt.subplots(figsize=(10, 9))
        bars = ax.bar(positions, values, color=colours, edgecolor="none")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=label_fontsize)
        ax.set_ylabel("Hessian MAE [eV/\u00c5$^2$]")
        if log_scale:
            ax.set_yscale("log")
        # Bar value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=label_fontsize * 0.8, fontweight="bold")
        # Legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor=HESSIAN_METHOD_TO_COLOUR["E-F"], label="AD (E-F)"),
            Patch(facecolor=HESSIAN_METHOD_TO_COLOUR["autograd"], label="AD"),
            Patch(facecolor=HESSIAN_METHOD_TO_COLOUR["predict"], label="HIP"),
        ]
        ax.legend(handles=legend_items, fontsize=label_fontsize+2, loc="upper right", edgecolor="none")
        sns.despine(left=True)
        fig.tight_layout()

        suffix = "_log" if log_scale else ""
        out = output or os.path.join(PLOT_DIR, f"hessian_mae_bar{suffix}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csvs",
        nargs="*",
        help="CSV files to include. If empty, uses a default set.",
    )
    parser.add_argument(
        "--labels", "-l", nargs="*", help="Display labels for each CSV"
    )
    args = parser.parse_args()

    MODEL_DISPLAY_NAMES = {
        "alpha": "AlphaNet",
        "left": "LEFTNet",
        "left-df": "LEFTNet-DF",
        "eqv2": "EquiformerV2",
        "eq_l1_mae": "HIP-EqV2 (MAE)",
        "eq_l1_mse": "HIP-EqV2 (MSE)",
        "eq_l1_luca8mae": "HIP-EqV2 (L8-MAE)",
        "hip_v2": "HIP-EqV2",
        "hesspred_v2": "HessPred-V2",
    }

    def _auto_label(f):
        base = os.path.basename(f).replace("_metrics.csv", "")
        is_orig = "_orig" in base
        is_predict = "predict" in base
        if is_predict:               
            print("HIP", base)                                  
            clean = "HIP-EquiformerV2"                                  
        else:                                                          
            clean = base 
        # Strip suffixes to find model key
        for suf in ["_orig", "_ts1x-val", "_autograd", "_predict"]:
            clean = clean.replace(suf, "")
        # Long hesspred names -> short key
        if "hesspred" in clean and clean not in MODEL_DISPLAY_NAMES:
            clean = "hesspred_v2"
        name = MODEL_DISPLAY_NAMES.get(clean, clean)
        if is_orig:
            name += " (E-F)"
        return name

    if args.csvs:
        csv_files = args.csvs
        labels = args.labels or [_auto_label(f) for f in csv_files]
    else:
        # Auto-discover all ts1x-val CSVs
        csv_files = sorted(
            f
            for f in glob.glob(os.path.join(RESULTS_DIR, "*ts1x*val*_metrics.csv"))
        )
        if not csv_files:
            print("No ts1x-val CSV files found in", RESULTS_DIR, file=sys.stderr)
            sys.exit(1)
        labels = args.labels or [_auto_label(f) for f in csv_files]

    # Group: orig (E-F) first, then hessian data models
    orig_files, orig_labels = [], []
    hess_files, hess_labels = [], []
    for f, l in zip(csv_files, labels):
        if "predict" in f:
            # only keep "hesspred_v2" and "hip_v2"
            if "hesspred_v2" in f or "hip_v2" in f:
                if "hip_v2" in f:
                    l = "HIP-EquiformerV2*"
            else:
                continue
        if "_orig" in os.path.basename(f):
            orig_files.append(f)
            orig_labels.append(l)
        else:
            hess_files.append(f)
            hess_labels.append(l)

    ordered_files = orig_files + hess_files
    ordered_labels = orig_labels + hess_labels
    groups = []
    if orig_files:
        groups.append(("E-F Only", len(orig_files)))
    if hess_files:
        groups.append(("Hessian Data", len(hess_files)))

    
    make_plot(ordered_files, ordered_labels)
    print(make_table(ordered_files, ordered_labels, groups=groups or None, bold_best=True))
