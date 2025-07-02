import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from align_ordered_mols import find_rigid_alignment, get_rmsd

"""

"""

raw_data_dir = "rgd1_raw"

# convert number to symbol
NUM2ELEMENT_RGD1 = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

num_rxns_multi_ts = 33_032
num_ts = 176_992
# num_rxns_single_ts = 176_992 - num_rxns_multi_ts # 143_960

# load in h5 files
RXN_ind2geometry = h5py.File(f"{raw_data_dir}/RGD1_CHNO.h5", "r")
RP_molidx2geometry = h5py.File(f"{raw_data_dir}/RGD1_RPs.h5", "r")

# load CSV file with atom-mapped SMILES and energies
csv_data = pd.read_csv(f"{raw_data_dir}/RGD1CHNO_AMsmiles.csv")
print(f"\nLoaded CSV data with {len(csv_data)} reactions")
print(f"CSV columns: {list(csv_data.columns)}")

num_rxns = len(RXN_ind2geometry)

# load in molecule dictionary
lines = open(f"{raw_data_dir}/RandP_smiles.txt", "r", encoding="utf-8").readlines()
RP_smiles2molidx = dict()
for lc, line in enumerate(lines):
    if lc == 0:
        continue
    RP_smiles2molidx[line.split()[0]] = line.split()[1]

print(f"\nLoaded {num_rxns} reactions")
print(f"First 5 reactions:")
for i, (k, v) in enumerate(RXN_ind2geometry.items()):
    print(k, v)
    if i > 5:
        break
print(f"keys of first sample: {list(RXN_ind2geometry['MR_100001_2'].keys())}")

print(f"\nLoaded {len(RP_molidx2geometry)} reactants and products geometries")
print(f"First 5 reactants and products geometries:")
for i, (k, v) in enumerate(RP_molidx2geometry.items()):
    print(k, v)
    if i > 5:
        break
print(f"keys of first sample: {list(RP_molidx2geometry['mol2'].keys())}")

print(f"\nLoaded {len(RP_smiles2molidx)} reactants and products smiles")
print("First 5 reactants and products smiles:")
for i, (k, v) in enumerate(RP_smiles2molidx.items()):
    print(k, v)
    if i > 5:
        break

# missing_mols = []
# missing_rxns = []
# for i_rxn, (Rind, Rxn) in tqdm(enumerate(RXN_ind2geometry.items())):

#     #print(f"Paring Reaction {Rind}")

#     # parse SPE
#     R_E,P_E,TS_E = np.array(Rxn.get('R_E')),np.array(Rxn.get('P_E')),np.array(Rxn.get('TS_E'))

#     # parse enthalpy
#     R_H,P_H,TS_H = np.array(Rxn.get('R_H')),np.array(Rxn.get('P_H')),np.array(Rxn.get('TS_H'))

#     # parse Gibbs free energy
#     R_F,P_F,TS_F = np.array(Rxn.get('R_F')),np.array(Rxn.get('P_F')),np.array(Rxn.get('TS_F'))

#     # parse smiles
#     Rsmiles = Rxn.get('Rsmiles')[()].decode('utf-8')
#     Psmiles = Rxn.get('Psmiles')[()].decode('utf-8')

#     # parse elements
#     elements = [NUM2ELEMENT_RGD1[Ei] for Ei in np.array(Rxn.get('elements'))]

#     # parse geometries
#     TS_G = np.array(Rxn.get('TSG'))
#     R_G = np.array(Rxn.get('RG'))
#     P_G = np.array(Rxn.get('PG'))
#     # Note: RP and PG are unoptimized

#     # load in seperated reactant/product molecules
#     Rmols = [RP_smiles2molidx[i] for i in Rsmiles.split('.')]
#     Pmols = [RP_smiles2molidx[i] for i in Rsmiles.split('.')]
#     for mol in Rmols+Pmols:
#         # obtain the corresponding molecule
#         try:
#             molecule = RP_molidx2geometry[mol]
#             # obtain DFT level energy and geometry
#             DFT_G = np.array(molecule.get('DFTG'))
#             DFT_SPE = np.array(molecule.get('DFT_SPE'))
#             # note: other keys are '_id', 'elements', 'xTBG', 'xTB_SPE'
#         except:
#             # print(f"In rxn {Rind}, molecule {mol} info is missing")
#             missing_mols.append(mol)
#             missing_rxns.append(Rind)
# # Missing molecule info in 836 reactions
# print(f"Missing molecule info in {len(missing_mols)} reactions")
num_missing_mols = 836  # len(missing_mols)

#####################################################################################

# Create list for single transition state reactions
print("\nIdentifying reactions with single transition states")
# idea:
# The reaction IDs follow a pattern like "MR_XXXXXX_Y" where Y is the transition state index.

# First, group reactions by their base ID (everything before the last underscore)
reaction_groups = {}
for Rind in RXN_ind2geometry.keys():
    base_id = "_".join(Rind.split("_")[:-1])  # Remove the last part after underscore
    if base_id not in reaction_groups:
        reaction_groups[base_id] = []
    reaction_groups[base_id].append(Rind)
print(f"Found {len(reaction_groups)} reaction groups")

# Identify single TS reactions
single_ts_reaction_ids = []
multi_ts_reaction_ids = []
for base_id, reaction_list in reaction_groups.items():
    if len(reaction_list) == 1:
        single_ts_reaction_ids.extend(reaction_list)
    else:
        multi_ts_reaction_ids.extend(reaction_list)

print(f"Found {len(single_ts_reaction_ids)} reactions with single transition states")
print(
    f"Found {len(multi_ts_reaction_ids)} total transition states from reactions with multiple TSs"
)
print(
    f"Found {len([base_id for base_id, reaction_list in reaction_groups.items() if len(reaction_list) > 1])} reactions with multiple transition states"
)

# Create mapping from reaction ID to CSV data for faster lookup
csv_data_dict = csv_data.set_index("reaction").to_dict("index")
print(f"Created CSV lookup dictionary with {len(csv_data_dict)} entries")

# this excludes all reactions with multiple transition states
# instead we can include one transition state per reaction
first_ts_per_reaction = []
for base_id, reaction_list in reaction_groups.items():
    # Sort and take the first (lowest index)
    first_ts_per_reaction.append(sorted(reaction_list)[0])
print(f"Found {len(first_ts_per_reaction)} reactions with single transition states")

#####################################################################################
# Process reactions and validate atom ordering consistency in one loop
print("\n" + "=" * 80)
print("PROCESSING REACTIONS AND VALIDATING ATOM ORDERING")
print("=" * 80)

processed_reactions = 0
sample_reactions = []
rmsd_stats = {"R_P_before": [], "R_P_after": [], "R_TS_before": [], "R_TS_after": []}

for Rind in tqdm(first_ts_per_reaction, desc="Processing and validating reactions"):
    try:
        # Only process first 100 samples for testing
        if processed_reactions >= 100:
            break

        Rxn = RXN_ind2geometry[Rind]
        processed_reactions += 1

        # Load geometries and elements directly from HDF5
        R_geom = np.array(Rxn.get("RG"))
        P_geom = np.array(Rxn.get("PG"))
        TS_geom = np.array(Rxn.get("TSG"))

        # element ordering is the same across R, P, and TS
        elements = [NUM2ELEMENT_RGD1[Ei] for Ei in np.array(Rxn.get("elements"))]

        # Compute RMSD before alignment
        rmsd_R_P_before = get_rmsd(R_geom, P_geom)
        rmsd_R_TS_before = get_rmsd(R_geom, TS_geom)

        # Align product and TS to reactant
        R_P, t_P = find_rigid_alignment(P_geom, R_geom)  # Align P to R
        R_TS, t_TS = find_rigid_alignment(TS_geom, R_geom)  # Align TS to R

        # Apply alignment transformations
        P_geom_aligned = (R_P.dot(P_geom.T)).T + t_P
        TS_geom_aligned = (R_TS.dot(TS_geom.T)).T + t_TS

        # Compute RMSD after alignment
        rmsd_R_P_after = get_rmsd(R_geom, P_geom_aligned)
        rmsd_R_TS_after = get_rmsd(R_geom, TS_geom_aligned)

        # Collect RMSD statistics
        rmsd_stats["R_P_before"].append(rmsd_R_P_before)
        rmsd_stats["R_P_after"].append(rmsd_R_P_after)
        rmsd_stats["R_TS_before"].append(rmsd_R_TS_before)
        rmsd_stats["R_TS_after"].append(rmsd_R_TS_after)

        # Store first few reactions for sample display
        if len(sample_reactions) < 3:
            # Create reaction data dictionary for sample display
            reaction_data = {
                "reaction_id": Rind,
                "R_E": np.array(Rxn.get("R_E")),
                "P_E": np.array(Rxn.get("P_E")),
                "TS_E": np.array(Rxn.get("TS_E")),
                "Rsmiles": Rxn.get("Rsmiles")[()].decode("utf-8"),
                "Psmiles": Rxn.get("Psmiles")[()].decode("utf-8"),
                "elements": elements,
                "TS_geometry": TS_geom,
                "R_geometry": R_geom,
                "P_geometry": P_geom,
                "P_geometry_aligned": P_geom_aligned,
                "TS_geometry_aligned": TS_geom_aligned,
                "rmsd_R_P_before": rmsd_R_P_before,
                "rmsd_R_TS_before": rmsd_R_TS_before,
                "rmsd_R_P_after": rmsd_R_P_after,
                "rmsd_R_TS_after": rmsd_R_TS_after,
            }

            # Add CSV data if available
            if Rind in csv_data_dict:
                csv_row = csv_data_dict[Rind]
                reaction_data.update(
                    {
                        "atom_mapped_reactant_smiles": csv_row["reactant"],
                        "atom_mapped_product_smiles": csv_row["product"],
                        "activation_energy_forward": csv_row["DE_F"],
                        "activation_energy_backward": csv_row["DE_B"],
                        "enthalpy_change": csv_row["DH"],
                    }
                )

            sample_reactions.append(reaction_data)

    except Exception as e:
        print(f"Error processing reaction {Rind}: {e}")
        continue


# Additional detailed check for first few sample reactions
print(f"\nDETAILED CHECK FOR FIRST {len(sample_reactions)} SAMPLE REACTIONS:")
for i, reaction in enumerate(sample_reactions):
    reaction_id = reaction["reaction_id"]

    R_geom = reaction["R_geometry"]
    P_geom = reaction["P_geometry"]
    TS_geom = reaction["TS_geometry"]
    P_geom_aligned = reaction["P_geometry_aligned"]
    TS_geom_aligned = reaction["TS_geometry_aligned"]
    elements = reaction["elements"]

    print(f"\nReaction {reaction_id}:")
    print(f"  Number of atoms: {len(elements)}")
    print(f"  Elements: {' '.join(elements)}")
    print(f"  R geometry shape: {R_geom.shape}")
    print(f"  P geometry shape: {P_geom.shape}")
    print(f"  TS geometry shape: {TS_geom.shape}")

    # Display RMSD results
    print(f"\n  RMSD Analysis:")
    print(f"    R-P RMSD before alignment: {reaction['rmsd_R_P_before']:.4f} Å")
    print(f"    R-P RMSD after alignment:  {reaction['rmsd_R_P_after']:.4f} Å")
    print(f"    R-TS RMSD before alignment: {reaction['rmsd_R_TS_before']:.4f} Å")
    print(f"    R-TS RMSD after alignment:  {reaction['rmsd_R_TS_after']:.4f} Å")
    print(
        f"    Improvement R-P: {reaction['rmsd_R_P_before'] - reaction['rmsd_R_P_after']:.4f} Å"
    )
    print(
        f"    Improvement R-TS: {reaction['rmsd_R_TS_before'] - reaction['rmsd_R_TS_after']:.4f} Å"
    )

    # # Show first few atoms and their positions (original and aligned)
    # print(f"\n  First 3 atoms - Original vs Aligned positions:")
    # for j in range(min(3, len(elements))):
    #     print(f"    Atom {j} ({elements[j]}):")
    #     print(f"      R:  {R_geom[j]}")
    #     print(f"      P:  {P_geom[j]} -> {P_geom_aligned[j]} (aligned)")
    #     print(f"      TS: {TS_geom[j]} -> {TS_geom_aligned[j]} (aligned)")
    #     r_to_p_dist_orig = np.linalg.norm(P_geom[j] - R_geom[j])
    #     r_to_p_dist_aligned = np.linalg.norm(P_geom_aligned[j] - R_geom[j])
    #     r_to_ts_dist_orig = np.linalg.norm(TS_geom[j] - R_geom[j])
    #     r_to_ts_dist_aligned = np.linalg.norm(TS_geom_aligned[j] - R_geom[j])
    #     print(f"      R->P distance: {r_to_p_dist_orig:.3f} -> {r_to_p_dist_aligned:.3f} Å")
    #     print(f"      R->TS distance: {r_to_ts_dist_orig:.3f} -> {r_to_ts_dist_aligned:.3f} Å")

# Summary statistics for RMSD across all processed reactions
if rmsd_stats["R_P_before"]:
    print(f"\nOVERALL RMSD STATISTICS ({len(rmsd_stats['R_P_before'])} reactions):")
    print("=" * 60)

    r_p_before_mean = np.mean(rmsd_stats["R_P_before"])
    r_p_after_mean = np.mean(rmsd_stats["R_P_after"])
    r_ts_before_mean = np.mean(rmsd_stats["R_TS_before"])
    r_ts_after_mean = np.mean(rmsd_stats["R_TS_after"])

    r_p_improvement = r_p_before_mean - r_p_after_mean
    r_ts_improvement = r_ts_before_mean - r_ts_after_mean

    print(f"R-P RMSD:")
    print(
        f"  Before alignment: {r_p_before_mean:.4f} ± {np.std(rmsd_stats['R_P_before']):.4f} Å"
    )
    print(
        f"  After alignment:  {r_p_after_mean:.4f} ± {np.std(rmsd_stats['R_P_after']):.4f} Å"
    )
    print(f"  Average improvement: {r_p_improvement:.4f} Å")
    print(f"  Improvement ratio: {r_p_improvement/r_p_before_mean*100:.1f}%")

    print(f"\nR-TS RMSD:")
    print(
        f"  Before alignment: {r_ts_before_mean:.4f} ± {np.std(rmsd_stats['R_TS_before']):.4f} Å"
    )
    print(
        f"  After alignment:  {r_ts_after_mean:.4f} ± {np.std(rmsd_stats['R_TS_after']):.4f} Å"
    )
    print(f"  Average improvement: {r_ts_improvement:.4f} Å")
    print(f"  Improvement ratio: {r_ts_improvement/r_ts_before_mean*100:.1f}%")

    print(f"\nRMSD Range Analysis:")
    print(
        f"  R-P before: min={np.min(rmsd_stats['R_P_before']):.4f}, max={np.max(rmsd_stats['R_P_before']):.4f}"
    )
    print(
        f"  R-P after:  min={np.min(rmsd_stats['R_P_after']):.4f}, max={np.max(rmsd_stats['R_P_after']):.4f}"
    )
    print(
        f"  R-TS before: min={np.min(rmsd_stats['R_TS_before']):.4f}, max={np.max(rmsd_stats['R_TS_before']):.4f}"
    )
    print(
        f"  R-TS after:  min={np.min(rmsd_stats['R_TS_after']):.4f}, max={np.max(rmsd_stats['R_TS_after']):.4f}"
    )

print("=" * 80)
