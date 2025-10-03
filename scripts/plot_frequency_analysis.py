import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

###################################################################
# V1
###################################################################

_true = "true_is_ts"
_pred = "model_is_ts"
_correct = "is_ts_agree"
name_to_name = {
    "hesspred": "HIP (ours)",
    "eqv2": "EquiformerV2",
    "left-df": "LeftNet-DF",
    "left": "LeftNet",
    "alpha": "AlphaNet",
}
hessian_method_to_name = {
    "autograd": "AD",
    "predict": "Learned",
}
csv_path = "results/eval_horm_wandb_export.csv"
df0 = pd.read_csv(csv_path, quotechar='"')
df = df0[["Name", "hessian_method", _pred, _true, _correct]].copy()
# rename
df["Name"] = df["Name"].map(name_to_name)
df["hessian_method"] = df["hessian_method"].map(hessian_method_to_name)


print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Confusion counts (TP, FP, FN, TN) derived from rates and sample size
N = df0["max_samples"].astype(int)
tp = 0.5 * (df[_pred] + df[_true] - 1 + df[_correct]) * N
conf = pd.DataFrame({"Name": df["Name"], "TP": tp.round().astype(int)})
conf["FP"] = (df[_pred] * N - conf["TP"]).round().astype(int)
conf["FN"] = (df[_true] * N - conf["TP"]).round().astype(int)
conf["TN"] = N - (conf["TP"] + conf["FP"] + conf["FN"])
conf["N"] = N
conf[["TP", "FP", "FN", "TN"]] = conf[["TP", "FP", "FN", "TN"]].clip(lower=0)

print()
print("Confusion counts (per model):")
print(conf[["Name", "TP", "FP", "FN", "TN", "N"]].to_string(index=False))

# Confusion rates (conditional percentages)
conf_rates = conf.copy()
conf_rates["P"] = conf_rates["TP"] + conf_rates["FN"]
conf_rates["Nneg"] = conf_rates["TN"] + conf_rates["FP"]
conf_rates["PP"] = conf_rates["TP"] + conf_rates["FP"]
conf_rates["PN"] = conf_rates["TN"] + conf_rates["FN"]
conf_rates["TPR"] = conf_rates["TP"] / conf_rates["P"].replace(0, pd.NA)
conf_rates["FNR"] = conf_rates["FN"] / conf_rates["P"].replace(0, pd.NA)
conf_rates["TNR"] = conf_rates["TN"] / conf_rates["Nneg"].replace(0, pd.NA)
conf_rates["FPR"] = conf_rates["FP"] / conf_rates["Nneg"].replace(0, pd.NA)
conf_rates["Precision"] = conf_rates["TP"] / conf_rates["PP"].replace(0, pd.NA)
conf_rates["NPV"] = conf_rates["TN"] / conf_rates["PN"].replace(0, pd.NA)
conf_rates["Accuracy"] = (conf_rates["TP"] + conf_rates["TN"]) / conf_rates[
    "N"
].replace(0, pd.NA)


def fmt_pct(x: float) -> str:
    # return f"{x*100:.1f}%"
    return f"{round(x * 100)}%"


print()
print("Confusion rates (per model):")
print(
    conf_rates[
        ["Name", "TPR", "FPR", "TNR", "FNR", "Precision", "NPV", "Accuracy"]
    ].to_string(
        index=False,
        formatters={
            k: fmt_pct
            for k in ["TPR", "FPR", "TNR", "FNR", "Precision", "NPV", "Accuracy"]
        },
    )
)

# LaTeX table for confusion rates
print()
# conf_frac_tbl = conf.copy()
# for c in ["TP", "FP", "FN", "TN"]:
# 	conf_frac_tbl[c] = conf_frac_tbl[c] / conf_frac_tbl["N"]

table_df = pd.DataFrame(
    {
        "Hessian": df["hessian_method"],
        "Name": df["Name"],
        "TPR": conf_rates["TPR"],
        "FPR": conf_rates["FPR"],
        "TNR": conf_rates["TNR"],
        "FNR": conf_rates["FNR"],
        "Precision": conf_rates["Precision"],
        "Accuracy": conf_rates["Accuracy"],
    }
)
# Order: AD first, then learned; then by Name for stability
table_df = table_df.sort_values(["Hessian", "Name"]).reset_index(drop=True)
# Append rates: Precision and Accuracy
# prec_acc = conf_rates[["Name", "Precision", "Accuracy"]].copy()
# prec_acc["Precision"] = prec_acc["Precision"].fillna(0.0)
# prec_acc["Accuracy"] = prec_acc["Accuracy"].fillna(0.0)
# table_df = table_df.merge(prec_acc, on="Name", how="left")

best_tp_idx = int(table_df["TPR"].idxmax())
best_fp_idx = int(table_df["FPR"].idxmin())
best_fn_idx = int(table_df["FNR"].idxmin())
best_tn_idx = int(table_df["TNR"].idxmax())
best_prec_idx = int(table_df["Precision"].idxmax())
best_acc_idx = int(table_df["Accuracy"].idxmax())


def fmt_pct_int(x: float) -> str:
    return f"{round(x * 100)}\\%"


lines = []
lines.append("\\begin{tabular}{llrrrrrr}")
lines.append("\\hline")
lines.append(
    r"Hessian & Name & TPR $\uparrow$ & FPR $\downarrow$ & FNR $\downarrow$ & TNR $\uparrow$ & Precision $\uparrow$ & Accuracy $\uparrow$ \\"
)
lines.append("\\hline")
for i, row in table_df.iterrows():
    tp = fmt_pct_int(row["TPR"])
    fp = fmt_pct_int(row["FPR"])
    fn = fmt_pct_int(row["FNR"])
    tn = fmt_pct_int(row["TNR"])
    prec = fmt_pct_int(row["Precision"])
    acc = fmt_pct_int(row["Accuracy"])
    if i == best_tp_idx:
        tp = f"\\textbf{{{tp}}}"
    if i == best_fp_idx:
        fp = f"\\textbf{{{fp}}}"
    if i == best_fn_idx:
        fn = f"\\textbf{{{fn}}}"
    if i == best_tn_idx:
        tn = f"\\textbf{{{tn}}}"
    if i == best_prec_idx:
        prec = f"\\textbf{{{prec}}}"
    if i == best_acc_idx:
        acc = f"\\textbf{{{acc}}}"
    lines.append(
        f"{row['Hessian']} & {row['Name']}  & {tp} & {fp}  & {fn}  & {tn} & {prec} & {acc} \\\\"
    )
lines.append("\\hline")
lines.append("\\end{tabular}")
print("\n".join(lines))


###################################################################
# V2
###################################################################

# neg_num_agree
"Name", "Accuracy"
# "MSE","0.863"
# "MAE","0.911"
"AlphaNet", "0.707"
"LeftNet-DF", "0.778"
"LeftNet", "0.823"
"EquiformerV2", "0.748"
"HIP (ours)", "0.919"

# LaTeX table (V2): Hessian, Name, Accuracy only
v2_rows = [
    {"Hessian": "AD", "Model": "AlphaNet", "Accuracy": 0.707},
    {"Hessian": "AD", "Model": "LeftNet-DF", "Accuracy": 0.778},
    {"Hessian": "AD", "Model": "LeftNet", "Accuracy": 0.823},
    {"Hessian": "AD", "Model": "EquiformerV2", "Accuracy": 0.748},
    {"Hessian": "Predicted", "Model": "HIP (ours)", "Accuracy": 0.919},
]

v2_df = pd.DataFrame(v2_rows)
best_acc_idx_v2 = int(v2_df["Accuracy"].idxmax())

v2_lines = []
v2_lines.append("\\begin{tabular}{llr}")
v2_lines.append("\\hline")
v2_lines.append(r"Hessian & Model & Accuracy $\uparrow$ \\")
v2_lines.append("\\hline")
for i, row in v2_df.iterrows():
    acc = fmt_pct_int(row["Accuracy"])
    if i == best_acc_idx_v2:
        acc = f"\\textbf{{{acc}}}"
    v2_lines.append(f"{row['Hessian']} & {row['Model']} & {acc} \\\\")
v2_lines.append("\\hline")
v2_lines.append("\\end{tabular}")
print()
print("\n".join(v2_lines))

###################################################################
# V3
###################################################################

# LaTeX table (V2): Hessian, Name, Accuracy only
v2_rows = [
    {"Hessian": "MSE", "Model": "HIP", "Accuracy": 0.863},
    {"Hessian": "MAE", "Model": "HIP", "Accuracy": 0.911},
    {"Hessian": "MAE+Sub", "Model": "HIP (ours)", "Accuracy": 0.919},
]

v2_df = pd.DataFrame(v2_rows)
best_acc_idx_v2 = int(v2_df["Accuracy"].idxmax())

v2_lines = []
v2_lines.append("\\begin{tabular}{lc}")
v2_lines.append("\\hline")
v2_lines.append(r"Loss & Accuracy $\uparrow$ \\")
v2_lines.append("\\hline")
for i, row in v2_df.iterrows():
    acc = fmt_pct_int(row["Accuracy"])
    if i == best_acc_idx_v2:
        acc = f"\\textbf{{{acc}}}"
    v2_lines.append(f"{row['Hessian']} & {acc} \\\\")
v2_lines.append("\\hline")
v2_lines.append("\\end{tabular}")
print()
print("\n".join(v2_lines))
