# Sella TS Results Plotting

This directory contains a comprehensive plotting script for analyzing Sella transition state (TS) search results.

## Usage

```bash
python playground/plot_sella_results.py
```

## What it does

The script automatically:

1. **Scans** the `playground/plots/` directory for all `sella_ts_*` result folders
2. **Loads** `summary.json` files from each experiment
3. **Parses** experiment metadata from directory names (starting point, coordinates, hessian method)
4. **Generates** comprehensive visualizations and analysis

## Output Files

### 1. RMSD Analysis (`rmsd_analysis.png`)
- **Initial vs Final RMSD**: Scatter plot showing how initial RMSD relates to final RMSD
- **RMSD Improvement by Starting Point**: Bar chart comparing different starting geometries
- **RMSD Comparison**: Side-by-side comparison of initial vs final RMSD for each experiment
- **RMSD vs Time**: Shows relationship between computation time and RMSD improvement

### 2. Timing Analysis (`timing_analysis.png`)
- **Total Optimization Time**: Bar chart of total computation time per experiment
- **Hessian Computation Times**: Comparison of autodiff, predict, and finite difference timings
- **Time vs Steps**: Relationship between number of optimization steps and total time
- **Time per Step**: Average computation time per optimization step

### 3. Steps Analysis (`steps_analysis.png`)
- **Steps by Experiment**: Bar chart showing number of optimization steps per experiment
- **Steps vs RMSD Improvement**: Scatter plot correlating number of steps with improvement
- **Steps by Starting Point**: Comparison of steps needed for different starting geometries and coordinate systems
- **Optimization Efficiency**: RMSD improvement per step (efficiency metric)

### 4. Summary Table (`sella_results_summary.csv`)
Organized CSV file with all key metrics:
- Experiment name and metadata
- RMSD metrics (initial, final, improvement)
- Timing metrics (total time, steps, time per step)
- Sorted by RMSD improvement (best results first)

## Key Findings

Based on the current results:

### Best Performing Methods
1. **Linear R-TS interpolation**: Best RMSD improvement (0.287 Å)
2. **Starting from reactant**: Good RMSD improvement (0.281 Å) 

### Performance Issues
- **Linear R-P interpolation**: Poor performance (-0.067 Å, worse than starting point)
- **Internal coordinates**: Fastest (17.8s) but poor accuracy (-0.133 Å)

### Timing Insights
- Most experiments take ~95-98 seconds with 2000 steps
- Internal coordinates much faster but less accurate
- Time per step fairly consistent (~0.05s) except for internal coordinates

## Adding New Results

The script automatically detects new `sella_ts_*` directories. Just ensure each contains a `summary.json` file with the standard metrics.

## Requirements

```python
pandas
matplotlib
seaborn
numpy
``` 