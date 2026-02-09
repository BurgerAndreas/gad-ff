import pandas as pd
import wandb

api = wandb.Api()

r"""
Goal: print a table like this:
\multirow{2}{*}{Hessian} & \multirow{2}{*}{Model} & Hessian $\downarrow$  & Eigenvalues $\downarrow$ & CosSim $\evec_1$ $\uparrow$ & $\eval_1$ $\downarrow$ & Time $\downarrow$ \\
 & & eV/\AA$^2$ & eV/\AA$^2$ & unitless & eV/\AA$^2$ & ms \\
\hline
\multirow{4}{*}{AD} 
 & AlphaNet & 0.259 & 0.148 & 0.415 & 0.040 & 767.0 \\
 & LEFTNet-CF & 0.226 & 0.130 & 0.244 & 0.015 & 1110.7 \\
 & LEFTNet-DF & 0.304 & 0.142 & 0.290 & 0.013 & 341.3 \\
 & EquiformerV2 & 0.133 & 0.056 & 0.092 & 0.003 & 633.0 \\
 & EquiformerV2 (E-F) & 0.243 & 0.111 & 1.224 & 0.106 & 633.0 \\
\hline
\multirow{1}{*}{Predicted } & HIP-EquiformerV2 & \textbf{ 0.030 } & \textbf{ 0.063 } & \textbf{ 0.982 } & \textbf{ 0.031 } & \textbf{ 38.5 } \\
\multirow{1}{*}{Predicted } & HIP-EquiformerV2 (end-to-end) & \textbf{ 0.030 } & \textbf{ 0.063 } & \textbf{ 0.982 } & \textbf{ 0.031 } & \textbf{ 38.5 } \\
    
Relevant columns:
hessian_mae, eigval_mae, eigvec1_cos_eckart, eigval1_mae_eckart, time

config:
hessian_method=autograd, predict
model.name
checkpoint: _orig means (E-F), hesspred_v2 means HIP-EquiformerV2, hip_v2 means HIP-EquiformerV2 (end-to-end)
"""

# Project is specified by <entity/project-name>
runs = api.runs("andreas-burger/horm")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # only keep dataset=RGD1.lmdb
    if run.config["dataset"] != "RGD1.lmdb":
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

runs_df.to_csv("rgd1.csv")
