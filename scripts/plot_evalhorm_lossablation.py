# if hesspred in the name, rename to EquiformerV2
# hessian_method predict -> learned

# metrics:
# time_incltransform -> time
# eigval1_mae -> lambda1
# eigvec1_cos -> CosSim

import csv
from pathlib import Path
import math


CSV_PATH = Path("/ssd/Code/gad-ff/results/wandb_export_loss_ablation.csv")


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _fmt_time_ms(value: float) -> str:
    return _fmt(value, 1)


def _bold(s: str, do_bold: bool) -> str:
    return f"\\textbf{{ {s} }}" if do_bold else s


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Normalize names where needed
    for r in rows:
        name = (r.get("Name") or "").lower()
        if "hesspred" in name:
            r["model_name"] = "EquiformerV2"
    return rows


def select_by_method(rows, method: str):
    return [r for r in rows if (r.get("hessian_method") or "").lower() == method]


def pick_metrics(r):
    return {
        "name": r.get("Name", ""),
        "model": r.get("model_name", ""),
        "hessian": _to_float(r.get("hessian_mae", "nan")),
        "eigvals": _to_float(r.get("eigval_mae", "nan")),
        "cos1": _to_float(r.get("eigvec1_cos", "nan")),
        "lambda1": _to_float(r.get("eigval1_mae", "nan")),
        "time_ms": _to_float(r.get("time", "nan")),
    }


def _best_index(values, maximize: bool = False):
    best_idx = None
    best_val = None
    for idx, v in enumerate(values):
        if isinstance(v, float) and math.isnan(v):
            continue
        if best_idx is None:
            best_idx = idx
            best_val = v
            continue
        if maximize:
            if v > best_val:
                best_idx = idx
                best_val = v
        else:
            if v < best_val:
                best_idx = idx
                best_val = v
    return best_idx


def main() -> None:
    rows = load_rows(CSV_PATH)

    predict_rows = [
        r for r in select_by_method(rows, "predict") if (r.get("model_name") or "") == "EquiformerV2"
    ]
    metrics_list = [pick_metrics(r) for r in predict_rows]

    # Compute best indices per metric
    h_best = _best_index([m["hessian"] for m in metrics_list], maximize=False)
    eig_best = _best_index([m["eigvals"] for m in metrics_list], maximize=False)
    cos_best = _best_index([m["cos1"] for m in metrics_list], maximize=True)
    lam_best = _best_index([m["lambda1"] for m in metrics_list], maximize=False)
    t_best = _best_index([m["time_ms"] for m in metrics_list], maximize=False)

    # Begin LaTeX table
    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\hline")
    lines.append(
        "\\multirow{2}{*}{Loss} & \\multirow{2}{*}{Model} & Hessian $\\downarrow$  & Eigenvalues $\\downarrow$ & CosSim $\\evec_1$ $\\uparrow$ & $\\eval_1$ $\\downarrow$ & Time $\\downarrow$ \\\\"
    )
    lines.append(" & & eV/\\AA$^2$ & eV/\\AA$^2$ & unitless & eV/\\AA$^2$ & ms \\")
    lines.append("\\hline")

    # Prediction rows (EquiformerV2 variants)
    for idx, m in enumerate(metrics_list):
        h_str = _bold(_fmt(m["hessian"]), (h_best is not None and idx == h_best))
        e_str = _bold(_fmt(m["eigvals"]), (eig_best is not None and idx == eig_best))
        c_str = _bold(_fmt(m["cos1"]), (cos_best is not None and idx == cos_best))
        l_str = _bold(_fmt(m["lambda1"]), (lam_best is not None and idx == lam_best))
        t_str = _bold(_fmt_time_ms(m["time_ms"]), (t_best is not None and idx == t_best))
        lines.append(
            f"{m['name']} & {m['model']} & {h_str} & {e_str} & {c_str} & {l_str} & {t_str} \\\\"
        )

    lines.append("")
    lines.append("\\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
