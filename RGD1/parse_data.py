import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# not used:
# DFT_reaction_info.csv is supplied to reproduce figures in the article;

# convert number to symbol
rdg1_num2element = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# https://www.nature.com/articles/s41597-023-02043-z/tables/2
rdg1_hdf5_keys = {
    "Rsmiles": {"description": "Reactant smiles", "unit": ""},
    "Psmiles": {"description": "Product smiles", "unit": ""},
    "R_E": {"description": "Reactant single point energy", "unit": "Hartree"},
    "R_H": {"description": "Reactant enthalpy", "unit": "Hartree"},
    "R_F": {"description": "Reactant Gibbs free energy", "unit": "Hartree"},
    "P_E": {"description": "Product single point energy", "unit": "Hartree"},
    "P_H": {"description": "Product enthalpy", "unit": "Hartree"},
    "P_F": {"description": "Product Gibbs free energy", "unit": "Hartree"},
    "TS_E": {"description": "Transition state single point energy", "unit": "Hartree"},
    "TS_H": {"description": "Transition state enthalpy", "unit": "Hartree"},
    "TS_F": {"description": "Transition state Gibbs free energy", "unit": "Hartree"},
    "RG": {"description": "Reactant geometry", "unit": "Angstrom"},
    "PG": {"description": "Product geometry", "unit": "Angstrom"},
    "TSG": {"description": "Transition state geometry", "unit": "Angstrom"},
}

# https://www.nature.com/articles/s41597-023-02043-z/tables/1
rdg1_csv_columns = {
    "reaction": {"description": "Reaction index", "unit": ""},
    "reactant": {"description": "Atom-mapped smiles of reactant(s)", "unit": ""},
    "product": {"description": "Atom-mapped smiles of product(s)", "unit": ""},
    "DE_F": {
        "description": "Activation energy of the forward reaction",
        "unit": "kcal/mol",
    },
    "DE_B": {
        "description": "Activation energy of the backward reaction",
        "unit": "kcal/mol",
    },
    "DG_F": {
        "description": "Free energy of activation of the forward reaction",
        "unit": "kcal/mol",
    },
    "DG_B": {
        "description": "Free energy of activation of the backward reaction",
        "unit": "kcal/mol",
    },
    "DH": {
        "description": "Enthalpy of reaction (forward reaction)",
        "unit": "kcal/mol",
    },
}

if __name__ == "__main__":
    DIR_RGD1 = "Data/RGD1"

    # load in h5 files
    # RGD1_CHNO.h5 contains the geometry information and energies;
    rxns = h5py.File(f"{DIR_RGD1}/RGD1_CHNO.h5", "r")
    # RGD1_RPs.h5 provides xtb and DFT optimized geometries of each individual reactant/product molecules.
    RPs = h5py.File(f"{DIR_RGD1}/RGD1_RPs.h5", "r")

    # load in molecule dictionary
    # RandP_smiles.txt is a dictionary to map the reactant and product smiles
    # that appear in RGD1_CHNO.h5 to a molecule index (molX);
    lines = open(f"{DIR_RGD1}/RandP_smiles.txt", "r", encoding="utf-8").readlines()
    RP_dict = dict()
    for lc, line in enumerate(lines):
        if lc == 0:
            continue
        RP_dict[line.split()[0]] = line.split()[1]

    # load extra info
    # RGD1CHNO_AMsmiles.csv contains atom-mapped SMILES, activation energies, and enthalpies of formation for each reaction;
    rxns_extra = pd.read_csv(f"{DIR_RGD1}/RGD1CHNO_AMsmiles.csv")
    print(f"Loaded {len(rxns_extra)} reactions from {DIR_RGD1}/RGD1CHNO_AMsmiles.csv")
    print(rxns_extra.head())
    print(rxns_extra.columns)

    print(f"Parsing {len(rxns)} reactions")

    missing_mols, missing_rxns = [], []
    for Rind, Rxn in tqdm(rxns.items()):
        # print(f"Paring Reaction {Rind}")

        # get atom-mapped smiles
        rxn_row = rxns_extra[rxns_extra["reaction"] == Rind]
        if len(rxn_row) == 1:
            AtomMapped_Rsmiles = rxn_row["reactant"].iloc[
                0
            ]  # atom-mapped reactant smiles
            AtomMapped_Psmiles = rxn_row["product"].iloc[
                0
            ]  # atom-mapped product smiles
        elif len(rxn_row) > 1:
            tqdm.write(f"Warning: Reaction {Rind} has multiple rows in rxns_extra")
            continue
        else:
            tqdm.write(f"Warning: Reaction {Rind} not found in rxns_extra")
            continue

        # parse single point energy (SPE)
        R_E, P_E, TS_E = (
            np.array(Rxn.get("R_E")),
            np.array(Rxn.get("P_E")),
            np.array(Rxn.get("TS_E")),
        )

        # parse enthalpy
        R_H, P_H, TS_H = (
            np.array(Rxn.get("R_H")),
            np.array(Rxn.get("P_H")),
            np.array(Rxn.get("TS_H")),
        )

        # parse Gibbs free energy
        R_F, P_F, TS_F = (
            np.array(Rxn.get("R_F")),
            np.array(Rxn.get("P_F")),
            np.array(Rxn.get("TS_F")),
        )

        # parse smiles
        Rsmiles, Psmiles = Rxn.get("Rsmiles")[()].decode("utf-8"), Rxn.get("Psmiles")[
            ()
        ].decode("utf-8")

        print(f"Rsmiles: {Rsmiles}, AtomMapped_Rsmiles: {AtomMapped_Rsmiles}")
        print(f"Psmiles: {Psmiles}, AtomMapped_Psmiles: {AtomMapped_Psmiles}")

        # parse elements
        elements = [rdg1_num2element[Ei] for Ei in np.array(Rxn.get("elements"))]

        # parse geometries
        TS_G = np.array(Rxn.get("TSG"))
        R_G = np.array(Rxn.get("RG"))
        P_G = np.array(Rxn.get("PG"))
        # Note: RP and PG are unoptimized

        # load in seperated reactant/product molecules
        Rmols = [RP_dict[i] for i in Rsmiles.split(".")]
        Pmols = [RP_dict[i] for i in Psmiles.split(".")]
        for mol in Rmols + Pmols:
            # obtain the corresponding molecule
            try:
                molecule = RPs[mol]
                # obtain DFT level energy and geometry
                DFT_G = np.array(molecule.get("DFTG"))
                DFT_SPE = np.array(molecule.get("DFT_SPE"))
                # note: other keys are '_id', 'elements', 'xTBG', 'xTB_SPE'
                ID = molecule.get("_id")[()].decode("utf-8")
                elements = molecule.get("elements")[()]
                xTBG = np.array(molecule.get("xTBG"))
                xTB_SPE = np.array(molecule.get("xTB_SPE"))
            except:
                tqdm.write(f"In rxn {Rind}, molecule {mol} info is missing")
                missing_mols.append(mol)
                missing_rxns.append(Rind)

        break

    print(
        f"{len(missing_mols)} / {len(rxns)} reactions are missing -> {len(RP_dict)} reactions in total"
    )
