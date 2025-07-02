# RGD1
https://www.nature.com/articles/s41597-023-02043-z

## TL;DR

After filtering we get 140_673 triplets of (transition state, reactant, product), where each reactant-product pair appears only once.

- train split: 112538 reactions, 8249716 total atoms
- val split: 14067 reactions, 1031949 total atoms
- test split: 14068 reactions, 1031144 total atoms

## Train/val/test split

Use the same validation strategy as OA-ReactDiff and React-OT:
Train on all of RGD1, validate on part of Transition1x:
partitioned Transition1x randomly, with 9,000 reactions used in training and validation, 
and the remaining 1,073 reactions as set-aside test set

It is not guaranteed that all molecules in test reactions are unseen by a trained model 
due to the overlapping structures in different reactions. 
However, each elementary reaction is unique, so atmost either the reactant or product, 
but not both, is seen in the training set.

## RGD1 dataset info
We attempted ~760k reactions at the xtb level, 
but after filtering unintended transition states 
(i.e., those classified to not correspond to the reactant and product that were used in the double-ended search) 
we were left with ~176k transition states. 
Most of the RGD-1 data at the DFT level is on the filtered ~176k intended TSs.
React-OT pretrained on the full set of intended and unintended at the xtb level (~760k).

The dataset is composed of 176,992 organic reactions possessing: 
at least one validated TS, activation energy, heat of reaction, reactant and product geometries, frequencies, and atom-mapping. 
For 33,032 reactions, more than one TS was discovered by conformational sampling.
Note: here 'reaction' refers to a reaction-product pair, not a transition state.
I.e. the number of reactions is smaller than the number of transition states.

## RGD1 file structure
- RGD1_CHNO.h5 contains the geometry information of the transition states. 176_992 entries.
- RGD1CHNO_AMsmiles.csv contains atom-mapped SMILES, activation energies, and enthalpies of formation for each reaction. Only 176_898 entries?
- RandP_smiles.txt is a dictionary to map the reactant and product smiles appear in RGD1_CHNO.h5 to a molecule index (molX).
- RGD1_RPs.h5 provides xtb and DFT optimized geometries of each individual reactant/product molecules. 123_088 entries, because some reactants and products are identical. But 836 are missing?
- 3D ML models can be trained by combining RGD1_RPs.h5, RGD1_CHNO.h5, and RandP_smiles.txt
- DFT_reaction_info.csv is supplied to reproduce figures in the article. Ignore.

In the RGD1_CHNO.h5 file: 
Property                             | Key     | Units
-------------------------------------|---------|--------
Reactant smiles                      | Rsmiles | -
Product smiles                       | Psmiles | -
Reactant single point energy         | R_E     | Hartree
Reactant enthalpy                    | R_H     | Hartree
Reactant Gibbs free energy           | R_F     | Hartree
Product single point energy          | P_E     | Hartree
Product enthalpy                     | P_H     | Hartree
Product Gibbs free energy            | P_F     | Hartree
Transition state single point energy | TS_E    | Hartree
Transition state enthalpy            | TS_H    | Hartree
Transition state Gibbs free energy   | TS_F    | Hartree
Reactant geometry                    | RG      | Å
Product geometry                     | PG      | Å
Transition state geometry            | TSG     | Å

In the .csv file:
Column   | Description
---------|--------------------------------------------------
Rind     | Reaction index
Rsmiles  | Atom-mapped smiles of reactant(s)
Psmiles  | Atom-mapped smiles of product(s)
DE_F     | Activation energy of the forward reaction
DE_B     | Activation energy of the backward reaction
DG_F     | Free energy of activation of the forward reaction
DG_B     | Free energy of activation of the backward reaction
DH       | Enthalpy of reaction (forward reaction)

The Rind refers to the reaction index with a format of MR_XXX_X

The RGD1-xTB-760k dataset that React-OT used is here: 
https://transfer.rcac.purdue.edu/file-manager?origin_id=1cc8429c-f64c-11ed-9bb7-c9bb788c490e&path=%2F