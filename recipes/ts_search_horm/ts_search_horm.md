# Transition State Search Workflow for HORM

Described in this paper:
Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models
https://chemrxiv.org/engage/chemrxiv/article-details/68270569927d1c2e66165ad8


## Harnessing TS Workflow Summary

* Input reactant and product geometries
	* perform MLIP-level geometry optimization on both endpoints using the chosen MLIP
* construct minimum energy path via growing string method (GSM)
	* use pygsm with ASE calculator calling the MLIP for energies and forces
	* set nine nodes, climbing‐image enabled, translation‐rotation‐internal coordinates
	* run until string convergence
* extract highest‐energy node as TS initial guess
* refine TS via Hessian‐based restricted‐step rational‐function‐optimization (RS‑I‑RFO)
	* employ pysisyphus with trust radius = 0.2 Å, Gaussian default convergence thresholds, max steps = 50
	* compute Hessians from MLIP‐force derivatives at every step (or from DFT every three steps if using DFT)
	* iterate until saddle‐point convergence
* validate optimized TS via intrinsic reaction coordinate (IRC) calculation
	* integrate in mass‐weighted coordinates with EulerPC integrator in pysisyphus
	* optimize IRC endpoints to stationary points
	* compare endpoint adjacency matrices to input reactant and product using TAFFI
	* if both sides match, label TS “intended”; otherwise label “unintended”


## Harnessing TS Workflow in detail


1. MLIP-level geometry optimization is performed on the input reactant and product 
2. growing string method (GSM) [72] calculation is initiated using the optimized geometries to construct a MEP
	1. energies and gradients evaluated by the corresponding MLIP 
3. Hessian based TS optimization determined by the RS-I-RFO
	1. https://link.springer.com/content/pdf/10.1007/s002140050387.pdf
		1. On the automatic restricted-step rational-function-optimization method
	2. The highest-energy node, ideally near the saddle point on the potential energy surface (PES), serves as the TS initial guess 
	3. refined using Hessian-based restricted step rational-function-optimization (RS-I-RFO) optimization
	4. During each optimization step, the Hessians, computed from the derivative of MLIP forces, guide the TS search
4. intrinsic reaction coordinate (IRC) calculation confirms that the identified TS corresponds to the expected reaction pathway by comparing the IRC endpoints with the input reactant and product 
	1. If they match on both sides, the TSis called “intended”, indicating that the TS search has successfully located a TS along the expected reaction pathway. Otherwise, it is termed “unintended”, suggesting that the identified TS does not correspond to the anticipated reaction mechanism
	2. RMSD cutoff = ?

Two statistics:
- the number of reactions with a converged GSM pathway (denoted as GSM Success), and the number of reactions with the intended transition state (denoted as intended reactions). 
	- A lower GSM success rate indicates that the MLIP fails to provide reliable predictions at certain points on the PES. Besides, a higher intended rate reflects the MLIPs ability to accurately reproduce the DFT PES, including capturing the PES curvatures such that the TS could be accurately predicted.
- The GSM-obtained TSs (i.e., the highest energy nodes) are classified as “intended” or “unintended” based on their outcome after TS optimization and intrinsic reaction coordinate (IRC) calculation
	- If the TS successfully converges to an intended TS after both steps, it is labeled as “intended”. If it fails at either step or does not lead to the intended TS, it is labeled as “unintended”. 
	- RMSD between the GSMobtained and DFT-optimized TSs exhibit distinct distributions for the “intended” and “unintended” categories, emphasizing the importance of providing an accurate initial TS guess in the TS search workflow.
Third statistic:
- energy barriers benchmarked against DFT-calculated values 
	- All MLIPs were trained using the same level of theory (ω97X/6-31G*)
	- In general, the absolute barrier errors for optimized TSs are much lower than those for GSM-derived barriers
		- (1) the enhanced quality of the optimized TSs, and (2) the TS optimization helps exclude unintended TSs. Such improvement emphasizes the importance of TS optimization and verification after GSM calculations.


The GSM is performed using the pygsm package, [82] where various MLIPs are incorporated into the ASE [83] calculator for energy and force evaluations. Default convergence parameters, including nine nodes, the climbing image setting, and the translation-rotation-internal coordinate system, are applied consistently across all calculations.

NEB method [84] was also tested as an alternative approach to construct the MEP. However, since NEB calculations heavily rely on an initial guess, there is a higher risk of calculating high-energy regions. Consequently, the convergence rate for NEB was found to be much lower than that of GSM when using pre-trained MLIPs and was thus discarded.

The RS-I-RFO optimization is carried out using the pysisyphus package, [85] with a trust radius set to 0.2Å, the convergence threshold set to the default Gaussian parameters, and the maximum number of optimization steps limited to 50. If the calculation is performed using MLIPs, Hessians are recalculated at every step. If DFT is used for the optimization, Hessians are recalculated every three steps. IRC performed to verify the optimized TS is also carried out using the pysisyphus package. IRCs are integrated in mass-weighted coordinates with the default “EulerPC” integrator. The resulting endpoints of the IRC integration are further optimized to stationary points. The comparison of IRC endpoints and input reactants and products is based on the adjacency matrix computed by the TAFFI package. [86]

Dataset. The Transition1x dataset consists of 10,073 reactions that are originally sampled by Grambow et al. [75]. These reactions are re-computed by CI-NEB at the ωB97X/631G* level, resulting in 9.6 million geometries. The same data split of reactions used in the React-OT study was applied, [31] with 9,000 and 1,073 reactions allocated to the training and validation set, respectively. After performing TS optimization and IRC calculations at the same level, 113 reactions were identified as unintended and subsequently removed from the validation set for the TS search tasks (leaving 960 validation?).

The original KHP decomposition network was established using YARP. [65] While the reaction exploration adhered to a “breaking two bonds and forming two bonds” rule, the IRC calculations uncovered unintended reactions that involve more complex chemistry, such as “breaking three bonds and forming three bonds.” Initial exploration was performed at the M052X-D3/def2-SVP level of theory. To ensure consistency with the Transition1x dataset, these 131 reactions were re-calculated at ωB97X/6-31G* level. To maximize the identification of intended TSs and increase the probability of finding the most stable TS, a conformational sampling strategy was applied. [88] This strategy sampled both the reactant and product sides, generating up to ten reaction conformations per reaction. The subsequent calculations followed the regular YARP recipe.

## Software

### pyGSM
- https://github.com/deepprinciple/pyGSM
- this custom fork added cell support of ASE https://github.com/deepprinciple/pyGSM/blob/a96d6d08c5cebc1d5982a8262bb8513ebe58a551/pyGSM/level_of_theories/ase.py

Todo: how to use it via the ASE calculator?
- we use it via the ASE calculator?
- see pyGSM/examples/ase_api_example.py

### pysisyphus

pysisyphus is a software-suite for the exploration of potential energy surfaces in ground- and excited states. It implements several methods to search for stationary points (minima and first order saddle points) and the calculation of minimum energy paths by means of IRC and Chain of States methods like Nudged Elastic Band and Growing String. Furthermore it provides tools to easily analyze & modify geometries (aligning, translating, interpolating, ...) and to visualize the calculation results/progress.
- https://github.com/deepprinciple/pysisyphus
- https://pysisyphus.readthedocs.io/en/dev/
- https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26390
- This custom fork added:
    - support for dft_c3
    - MLFF support https://github.com/deepprinciple/pysisyphus/blob/master/pysisyphus/calculators/MLFF.py and https://github.com/deepprinciple/pysisyphus/blob/master/pysisyphus/run.py
    - PySCF salvation models https://github.com/deepprinciple/pysisyphus/blob/master/pysisyphus/calculators/PySCF.py

Todo: how to run it?

```bash
# Run calculations (Minima optimization, TS search, IRC, NEB, GS, ...)
pysis
# Plotting of path-energies, optimization progress, IRC progress, etc ...
pysisplot
# Manipulate a .trj file or multiple .xyz files
pysistrj
```

