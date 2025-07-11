# GAD (Gentlest Ascent Dynamics) Results Plotting

This directory contains a comprehensive plotting script for analyzing GAD transition state search results across different eigen methods.

## Usage

```bash
python playground/plot_gad_results.py
```

## What it does

The script automatically:

1. **Scans** the `playground/logs_gad/` directory for all `results_*.json` files
2. **Loads** RMSD final values for each eigen method and test scenario
3. **Parses** experiment metadata from scenario names (dt, steps, test type)
4. **Generates** comprehensive visualizations and comparative analysis

## Input Data Structure

GAD results are stored as JSON files with the pattern `results_{eigen_method}.json`:
- **File examples**: `results_qr.json`, `results_svd.json`, `results_ase.json`
- **Content**: `{"scenario_name": rmsd_final_value, ...}`
- **Scenarios**: Different test conditions (starting points, time steps, iterations)

## Output Files

### 1. RMSD Analysis (`gad_rmsd_analysis.png`)
- **Performance Heatmap**: Color-coded matrix showing RMSD for each method×scenario combination
- **Grouped Bar Chart**: RMSD performance by eigen method across all scenarios
- **Performance Statistics**: Mean/min/max RMSD with error bars for each method
- **Scenario Difficulty**: Average RMSD by test scenario (identifies hardest/easiest tests)

### 2. Method Comparison (`gad_method_comparison.png`)
- **Method Ranking Heatmap**: Rank (1=best) for each method on each scenario
- **Average Ranking**: Overall method performance ranking across all scenarios
- **Consistency Analysis**: Standard deviation of RMSD (lower = more consistent)
- **Key Scenario Comparison**: Side-by-side performance on critical test cases

### 3. Summary Tables
- **`gad_results_summary.csv`**: Complete matrix of RMSD values with statistics
- **`gad_method_summary.csv`**: Method performance summary (mean, std, min, max)

## Test Scenarios Explained

The GAD experiments test different aspects of the algorithm:

### Convergence Tests
- **`ts_from_ts`**: Fixed point test (should stay at transition state)
- **`ts_from_perturbed_ts`**: Recovery from perturbed transition state

### Starting Point Tests  
- **`ts_from_r_*`**: Starting from reactant with different parameters
- **`ts_from_r_p_*`**: Starting from reactant-product interpolation

### Parameter Variations
- **`dt0.01` vs `dt0.1`**: Different time step sizes
- **`s100`, `s1000`, `s10000`**: Different number of integration steps

## Eigen Methods Compared

The script compares different eigenvalue decomposition methods:
- **`qr`**: QR decomposition
- **`svd`**: Singular Value Decomposition  
- **`svdforce`**: SVD with force projections
- **`inertia`**: Inertia tensor-based method
- **`ase`**: ASE library implementation
- **`geo`**: Geometric library implementation
- **`eckartsvd`**: Eckart frame with SVD

## Key Insights

The analysis reveals:

### Performance Ranking
- Which eigen methods consistently perform best/worst
- Method reliability across different test scenarios
- Scenario-specific method advantages

### Robustness Analysis
- How sensitive each method is to different starting conditions
- Consistency of performance across parameter variations
- Identification of failure modes

## Adding New Results

To include new GAD results:
1. Ensure new `results_{method}.json` files are in `playground/logs_gad/`
2. Follow the standard format: `{"scenario": rmsd_value, ...}`
3. Run the script - it automatically detects all result files

## Requirements

```python
pandas
matplotlib
seaborn
numpy
``` 