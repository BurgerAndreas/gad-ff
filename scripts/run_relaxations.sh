#!/bin/bash

SCRIPT="uv run scripts/second_order_relaxation_pysiyphus.py"

# t1x_val_reactant_hessian_100 (no noise)
$SCRIPT --xyz ../Datastore/t1x/t1x_val_reactant_hessian_100.h5 --max_samples 30 --coord redund --thresh gau
$SCRIPT --xyz ../Datastore/t1x/t1x_val_reactant_hessian_100.h5 --max_samples 80 --coord redund --thresh gau

# noise 0 (no displacement)
$SCRIPT --xyz t1x_0 --max_samples 30 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 80 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 30 --coord cart --thresh gau

# noise 0.03
$SCRIPT --xyz t1x_003 --max_samples 30 --coord redund --thresh gau
$SCRIPT --xyz t1x_0.03 --max_samples 80 --coord redund --thresh gau

# noise 0.035
$SCRIPT --xyz t1x_0.035 --max_samples 80 --coord redund --thresh gau

# noise 0.05
$SCRIPT --xyz t1x_005 --max_samples 30 --coord redund --thresh gau
$SCRIPT --xyz t1x_0.05 --max_samples 80 --coord redund --thresh gau

# tight2 + noise 0 (various sample sizes)
$SCRIPT --xyz t1x_0 --max_samples 1 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 2 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 10 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 20 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 30 --coord redund --thresh gau
$SCRIPT --xyz t1x_0 --max_samples 80 --coord redund --thresh gau


uv run scripts/second_order_relaxation_pysiyphus.py --xyz ../Datastore/t1x/t1x_val_reactant_hessian_100_noiserms0.03.h5 --max_samples 80 --coord redund --thresh gau  