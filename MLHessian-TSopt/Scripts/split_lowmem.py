from transition1x import Dataloader as t1xloader
import ani1x.dataloader as ani1xloader
    
from ase import Atoms, units
import numpy as np
import os
import shutil
import json
from tqdm import tqdm

N_tot, A_max = 9644740, 23


# DIR_ROOT = '/ssd/Code/gad-ff'
dir_this_file = os.path.dirname(os.path.abspath(__file__))
DIR_ROOT = os.path.dirname(os.path.dirname(dir_this_file))

# there should by a file {DIR_ANI1x}/data/ani1x-release.h5
DIR_ANI1x = f'{DIR_ROOT}/Data/ANI1x'
# there should by a file {DIR_T1x}/data/transition1x.h5
DIR_T1x = f'{DIR_ROOT}/Data/Transition1x'

# # first cleanup the files from last run
# # remove {DIR_ANI1x}/aug_data
# if os.path.exists(f'{DIR_ANI1x}/aug_data'):
#     shutil.rmtree(f'{DIR_ANI1x}/aug_data')
# # remove {DIR_T1x}/augmented_data
# if os.path.exists(f'{DIR_T1x}/reaction_data'):
#     shutil.rmtree(f'{DIR_T1x}/reaction_data')
# if os.path.exists(f'{DIR_T1x}/augmented_data'):
#     shutil.rmtree(f'{DIR_T1x}/augmented_data')
# if os.path.exists(f'{DIR_T1x}/composition_data'):
#     shutil.rmtree(f'{DIR_T1x}/composition_data')
    
    
# print(f"Files in {DIR_ANI1x}/data:")
# print(os.listdir(f'{DIR_ANI1x}/data'))
# print(f"Files in {DIR_T1x}/data:")
# print(os.listdir(f'{DIR_T1x}/data'))


do_process_t1x_reaction = False
do_process_t1x_composition = False
do_process_ani1x = False
do_process_ani1x_aug = False
do_combine_t1x_ani1x = False
do_create_composition_splits = False
do_create_aug_conformation_splits = True


def process_t1x_reaction():
    print('\n# Processing Transition1x reactions')
    dataloader = t1xloader(f'{DIR_T1x}/data/transition1x.h5')

    current_rxn = None
    RXN, R, Z, E_abs, E, F = [], [], [], [], [], []

    os.makedirs(f'{DIR_T1x}/reaction_data', exist_ok=True)

    for molecule in tqdm(dataloader, desc='Processing Transition1x'):
        last_rxn = current_rxn
        current_rxn = molecule['rxn']
        if current_rxn != last_rxn:
            np.savez_compressed(f'{DIR_T1x}/reaction_data/{last_rxn}.npz',
                R=np.array(R),
                Z=np.array(Z),
                E=np.array(E).reshape((-1, 1)) * units.eV / (units.kcal/units.mol),
                F=np.array(F) * units.eV / (units.kcal/units.mol))
            RXN, R, Z, E_abs, E, F = [], [], [], [], [], []
        RXN.append(current_rxn)
        R.append(molecule['positions'])
        Z.append(molecule['atomic_numbers'])
        E_abs.append(molecule['wB97x_6-31G(d).energy'])
        E.append(molecule['wB97x_6-31G(d).atomization_energy'])
        F.append(molecule['wB97x_6-31G(d).forces'])

    np.savez_compressed(f'{DIR_T1x}/reaction_data/{current_rxn}.npz',
        R=np.array(R),
        Z=np.array(Z),
        E=np.array(E).reshape((-1, 1)) * units.eV / (units.kcal/units.mol),
        F=np.array(F) * units.eV / (units.kcal/units.mol))
    print('done!')
    return RXN, R, Z, E_abs, E, F

if do_process_t1x_reaction:
    process_t1x_reaction()


def count_t1x_reaction():
    N_tot, A_max = 0, 0
    for file in tqdm(os.listdir(f'{DIR_T1x}/reaction_data')):
        if file.endswith('.npz'):
            data = np.load(f'{DIR_T1x}/reaction_data/{file}')
        else:
            continue
        if len(data['Z'].shape) < 2:
            print(f"Shape of {file} is faulty: {data['Z'].shape}")
            continue
        N_tot += data['Z'].shape[0]
        A_max = max(A_max, data['Z'].shape[1])
    return N_tot, A_max

N_tot, A_max = count_t1x_reaction()
print('Total number of samples (T1x reaction):', N_tot)
print('Maximum number of atoms (T1x reaction):', A_max)



def process_t1x_composition():
    print('\n# Processing Transition1x compositions')
    dataloader = t1xloader(f'{DIR_T1x}/data/transition1x.h5')

    current_formula = None
    RXN, R, Z, E_abs, E, F = [], [], [], [], [], []

    os.makedirs(f'{DIR_T1x}/composition_data', exist_ok=True)

    for molecule in tqdm(dataloader):
        last_formula = current_formula
        current_formula = molecule['formula']
        if current_formula != last_formula:
            np.savez_compressed(f'{DIR_T1x}/composition_data/{last_formula}.npz', 
                R=np.array(R),
                Z=np.array(Z),
                E=np.array(E).reshape((-1, 1)) * units.eV / (units.kcal/units.mol), 
                F=np.array(F) * units.eV / (units.kcal/units.mol)
            )
            RXN, R, Z, E_abs, E, F = [], [], [], [], [], []
        
        R.append(molecule['positions'])
        Z.append(molecule['atomic_numbers'])
        E.append(molecule['wB97x_6-31G(d).atomization_energy'])
        F.append(molecule['wB97x_6-31G(d).forces'])

    np.savez_compressed(f'{DIR_T1x}/composition_data/{current_formula}.npz', 
        R=np.array(R),
        Z=np.array(Z),
        E=np.array(E).reshape((-1, 1)) * units.eV / (units.kcal/units.mol), 
        F=np.array(F) * units.eV / (units.kcal/units.mol)
    )
    print('done!')
    return RXN, R, Z, E_abs, E, F

if do_process_t1x_composition:
    process_t1x_composition()


def count_t1x_composition():
    N_tot, A_max = 0, 0
    for file in tqdm(os.listdir(f'{DIR_T1x}/composition_data')):
        if file.endswith('.npz'):
            data = np.load(f'{DIR_T1x}/composition_data/{file}')
        else:
            continue
        if len(data['Z'].shape) < 2:
            print(f"Shape of {file} is faulty: {data['Z'].shape}")
            continue
        N_tot += data['Z'].shape[0]
        A_max = max(A_max, data['Z'].shape[1])
    return N_tot, A_max

N_tot, A_max = count_t1x_composition()
print('Total number of samples (T1x composition):', N_tot)
print('Maximum number of atoms (T1x composition):', A_max)



def process_ani1x_data():
    print('\n# Processing ANI1x')
    self_energy = {'H':-0.500607632585, 'C':-37.8302333826, 'N':-54.5680045287, 'O':-75.0362229210}

    for molecule in tqdm(ani1xloader.iter_data_buckets(f'{DIR_ANI1x}/data/ani1x-release.h5', keys=['wb97x_dz.energy','wb97x_dz.forces'])):
        formula = Atoms(numbers=molecule['atomic_numbers']).symbols
        np.savez_compressed(f'{DIR_ANI1x}/data/{formula}.npz',
            R=np.array(molecule['coordinates']),
            Z=np.tile(molecule['atomic_numbers'], (molecule['coordinates'].shape[0], 1)),
            E=(np.array(molecule['wb97x_dz.energy']).reshape((-1, 1)) - sum([self_energy[atom]*len(formula.search(atom)) for atom in ('H', 'C', 'N', 'O')]))  * units.Ha / (units.kcal/units.mol),
            F=np.array(molecule['wb97x_dz.forces']) * units.Ha / (units.kcal/units.mol))
    print('done!')

if do_process_ani1x:
    process_ani1x_data()



def count_ani1x():
    print('\n# Counting ANI1x')
    N_tot, A_max = 0, 0
    for file in tqdm(os.listdir(f'{DIR_ANI1x}/data')):
        if file.endswith('.npz'):
            data = np.load(f'{DIR_ANI1x}/data/{file}')
        else:
            continue
        if len(data['Z'].shape) < 2:
            print(f"Shape of {file} is faulty: {data['Z'].shape}")
            continue
        N_tot += data['Z'].shape[0]
        A_max = max(A_max, data['Z'].shape[1])
    return N_tot, A_max
    
N_tot, A_max = count_ani1x()
print('Total number of samples (ANI1x):', N_tot)
print('Maximum number of atoms (ANI1x):', A_max)



def process_ani1x_aug_data():
    print('\n# Processing ANI1x augmented data')
    count = {'CH': 0, 'NH': 0, 'OH': 0, 'CC': 0, 'CN': 0, 'CO': 0, 'NN': 0, 'NO': 0, 'OO': 0}

    N_tot = 0
    if not os.path.exists(f'{DIR_ANI1x}/aug_data'):
        os.makedirs(f'{DIR_ANI1x}/aug_data')
    for file in tqdm(os.listdir(f'{DIR_ANI1x}/data')):
        if file.endswith('.npz'):
            data = np.load(f'{DIR_ANI1x}/data/{file}')
        else:
            continue

        dist_matrix = np.linalg.norm(data['R'][:, :, None, :] - data['R'][:, None, :, :], axis=-1)
        dist_matrix[:, range(data['R'].shape[1]), range(data['R'].shape[1])] = np.nan

        compressed_CH = (dist_matrix < 1.095) & (data['Z'][:, :, None] == 1) & (data['Z'][:, None, :] == 6)
        compressed_NH = (dist_matrix < 1.018) & (data['Z'][:, :, None] == 1) & (data['Z'][:, None, :] == 7)
        compressed_OH = (dist_matrix < 0.961) & (data['Z'][:, :, None] == 1) & (data['Z'][:, None, :] == 8)
        compressed_CC = (dist_matrix < 1.202) & (data['Z'][:, :, None] == 6) & (data['Z'][:, None, :] == 6)
        compressed_CN = (dist_matrix < 1.156) & (data['Z'][:, :, None] == 6) & (data['Z'][:, None, :] == 7)
        compressed_CO = (dist_matrix < 1.204) & (data['Z'][:, :, None] == 6) & (data['Z'][:, None, :] == 8)
        compressed_NN = (dist_matrix < 1.099) & (data['Z'][:, :, None] == 7) & (data['Z'][:, None, :] == 7)
        compressed_NO = (dist_matrix < 1.219) & (data['Z'][:, :, None] == 7) & (data['Z'][:, None, :] == 8)
        compressed_OO = (dist_matrix < 1.469) & (data['Z'][:, :, None] == 8) & (data['Z'][:, None, :] == 8)
        compressed = compressed_CH | compressed_NH | compressed_OH | compressed_CC | compressed_CN | compressed_CO | compressed_NN | compressed_NO | compressed_OO
        
        if np.any(compressed):
            count['CH'] += np.count_nonzero(np.any(compressed_CH, axis=(1, 2)))
            count['NH'] += np.count_nonzero(np.any(compressed_NH, axis=(1, 2)))
            count['OH'] += np.count_nonzero(np.any(compressed_OH, axis=(1, 2)))
            count['CC'] += np.count_nonzero(np.any(compressed_CC, axis=(1, 2)))
            count['CN'] += np.count_nonzero(np.any(compressed_CN, axis=(1, 2)))
            count['CO'] += np.count_nonzero(np.any(compressed_CO, axis=(1, 2)))
            count['NN'] += np.count_nonzero(np.any(compressed_NN, axis=(1, 2)))
            count['NO'] += np.count_nonzero(np.any(compressed_NO, axis=(1, 2)))
            count['OO'] += np.count_nonzero(np.any(compressed_OO, axis=(1, 2)))

            np.savez_compressed(f'{DIR_ANI1x}/aug_data/{file}', 
                R=data['R'][np.any(compressed, axis=(1, 2))],
                Z=data['Z'][np.any(compressed, axis=(1, 2))],
                E=data['E'][np.any(compressed, axis=(1, 2))],
                F=data['F'][np.any(compressed, axis=(1, 2))])
            N_tot += np.count_nonzero(np.any(compressed, axis=(1, 2)))
    return N_tot, count

if do_process_ani1x_aug:
    N_tot, count = process_ani1x_aug_data()
    print("Total number of samples (ANI1x augmented):", N_tot)
    print("Count of augmented samples (ANI1x augmented):", count)



def combine_t1x_ani1x():
    print('\n# Combining Transition1x and ANI1x augmented data')
    os.makedirs(f'{DIR_T1x}/augmented_data', exist_ok=True)

    for file in tqdm(set(os.listdir(f'{DIR_T1x}/composition_data') + os.listdir(f'{DIR_ANI1x}/aug_data'))):
        if file.endswith('.npz'):
            pass
        else:
            continue
        
        match (file in os.listdir(f'{DIR_T1x}/composition_data')), (file in os.listdir(f'{DIR_ANI1x}/aug_data')):
            case (True, True):
                data1 = np.load(f'{DIR_T1x}/composition_data/{file}')
                data2 = np.load(f'{DIR_ANI1x}/data/{file}')
                np.savez_compressed(f'{DIR_T1x}/augmented_data/{file}',
                    R=np.concatenate((data1['R'], data2['R'])),
                    Z=np.concatenate((data1['Z'], data2['Z'])),
                    E=np.concatenate((data1['E'], data2['E'])),
                    F=np.concatenate((data1['F'], data2['F']))
                )
            case (True, False):
                data1 = np.load(f'{DIR_T1x}/composition_data/{file}')
                shutil.copyfile(f'{DIR_T1x}/composition_data/{file}', f'{DIR_T1x}/augmented_data/{file}')
            case (False, True):
                data2 = np.load(f'{DIR_ANI1x}/aug_data/{file}')
                shutil.copyfile(f'{DIR_ANI1x}/aug_data/{file}', f'{DIR_T1x}/augmented_data/{file}')
    print('done!')

if do_combine_t1x_ani1x:
    combine_t1x_ani1x()



def create_splits():
    print('\n# Creating composition splits')
    set_test = ["C2H4N4O", "C3H7NO2", "C4H5NO", "C3H7N3", "C2H4N2", "C4H10O", "C5H7NO", "C5H11N", "C3H4N2O", "C4H2", "C2H5N3O", "C5H9N", "C3H5NO", "C6H14O", "C3H8O2", "C2H4O2", "C7H12"]
    set_val = [["C5H8O", "C3H4N4", "C2H3N3O2", "C2H3N5", "CH3N5", "C4H6N2", "C3H6O3", "C2H4N2O2", "C3H2O3", "CN2O3", "C3H5N", "C7H8", "C4H5N3", "C5H4O", "C2HNO3", "C5H8N2", "C2H3N3O"],
            ["C4HNO", "C3H2N2", "C3HNO2", "C3H6N2O", "CH2N4O", "C5H2O", "C2H2O2", "C5H12O", "C5H10O2", "C6H13N", "C4H7N3", "C3H2N2O", "C3H3N3", "C4H10O2", "C5H10N2", "CH4N2O", "C3H6O"],
            ["C3HN3O", "C4N2", "C4H3NO2", "C3H5N3O", "C5H6N2", "C3H4O2", "C2H5NO", "C3H8N2O2", "C6H8O", "C4H7N", "C4H8O2", "C2HNO", "C3N2O", "C2H2N4", "C5H11NO", "C3H8O", "CHN3O2", "C3H6N2"],
            ["C6H10O", "C4H3N", "C6H12O", "C5H4", "C3H8N2O", "C4H5N", "C2N2", "C6H10", "C4H2O2", "C2H4N4", "C6H14", "C6H11N", "C2H2N2O", "C4H6O2", "C2H6N2O", "C5H5NO", "C6H6"]]
    set_train = [['C6H13N', 'C2H4N4', 'C2H6O2', 'C2H6N2O', 'C3H6N2O', 'C4H2N2O', 'C4H6', 'C3H6N2', 'C5H5N', 'C3H2N4', 'C3N2O', 'C5H3NO', 'C6H6', 'C3H4N2', 'C4H3NO2', 'C4H7NO', 'C2H2N4', 'C2H5NO2', 'C3H5NO2', 'C4H3NO', '.DS_S', 'C3H4O3', 'C3H4O2', 'C3H5NO3', 'C2H6O', 'C5H5NO', 'C2H4N2O', 'C3H2N2', 'C6H12O', 'C3H2O', 'C3H6O2', 'C3H8N2O', 'C2H2N2O2', 'C2H2O2', 'C3H8N2O2', 'C5H12', 'C4H9NO2', 'C3HNO2', 'C2N2', 'C5H4', 'C5H10N2', 'C4H5NO2', 'C4H8O3', 'C4H8O2', 'C7H16', 'C3H5N3', 'C4H4N2O', 'C4H8N2O', 'C7H14', 'C2H3N3', 'C4H7N', 'C5H6', 'C6H6O', 'C5H10', 'C3H4O', 'C5H12O', 'C4N2', 'C3H4', 'C4H6N2O', 'C5H12O2', 'C3H3N3', 'C7H10', 'C2H2N2O', 'C3H2N2O', 'C4H6O', 'C6H7N', 'C5H2O', 'C3H3NO3', 'C4H7NO2', 'C4H10O3', 'C3H8O', 'C4H8N2', 'C5H8O2', 'C4H10O2', 'C5H10O2', 'CHN3O2', 'C3H2N2O2', 'C2H3NO2', 'C3H3NO2', 'C3H7NO', 'C6H8O', 'C4H9N', 'C5H4O2', 'C2H3NO', 'C4H4N2', 'CH2N4O', 'C6H4O', 'C4H2O2', 'C4H5N', 'C3H5N3O', 'C3HN', 'C3H8', 'CH4N2O', 'C4HNO', 'C4H10N2O', 'C3H7N', 'C5H6O2', 'C2H3N', 'C6H5N', 'C3H7N3O', 'C4H4O', 'C5H6N2', 'C4H6O3', 'C4H6O2', 'C5H10O', 'C3H6O', 'C2H5NO', 'C5H8', 'C6H9N', 'C4H4O2', 'C4H8O', 'C3H6N2O2', 'C3H3NO', 'C5H4N2', 'C4H4O3', 'C5H7N', 'C2HNO', 'C4H3N', 'C4H10', 'CHN3O', 'C5H9NO', 'C6H11N', 'C6H14', 'C5H11NO', 'C3H3N3O', 'C4H7N3', 'C3H4N2O2', 'C3H8O3', 'C4H8', 'C6H12', 'C5H6O', 'C3HN3O', 'C6H10', 'C4H3N3', 'C6H8', 'C6H10O', 'C4H9NO', 'C2H4O'],
                ['C3H2O3', 'C2H4N4', 'C2H6O2', 'C2H6N2O', 'C4H2N2O', 'C4H6', 'C3H6N2', 'C5H5N', 'C3H2N4', 'C3N2O', 'C5H3NO', 'C6H6', 'C3H4N2', 'C4H3NO2', 'C4H7NO', 'C2H2N4', 'C2H5NO2', 'C3H5NO2', 'C4H3NO', '.DS_S', 'C3H4O3', 'C3H4O2', 'C3H5NO3', 'C5H8O', 'C2H6O', 'C5H5NO', 'C2H4N2O', 'C6H12O', 'C3H2O', 'C3H6O2', 'C3H4N4', 'C3H8N2O', 'C2H2N2O2', 'C3H6O3', 'C3H8N2O2', 'C5H4O', 'C5H12', 'C4H9NO2', 'C2N2', 'C5H4', 'C5H8N2', 'C4H5NO2', 'C4H8O3', 'C4H8O2', 'C7H16', 'C3H5N3', 'C4H4N2O', 'C3H5N', 'C4H8N2O', 'C7H14', 'C2H3N3', 'C4H7N', 'C5H6', 'C2H3N3O2', 'C6H6O', 'C5H10', 'C3H4O', 'C4N2', 'C3H4', 'C4H6N2O', 'C5H12O2', 'C7H10', 'C2H2N2O', 'C4H6O', 'C6H7N', 'CN2O3', 'C3H3NO3', 'C4H7NO2', 'C4H10O3', 'C3H8O', 'C4H8N2', 'C2H3N5', 'C5H8O2', 'C4H10O2', 'C5H10O2', 'CH3N5', 'CHN3O2', 'C3H2N2O2', 'C2HNO3', 'C2H3NO2', 'C3H3NO2', 'C3H7NO', 'C6H8O', 'C4H9N', 'C5H4O2', 'C2H3NO', 'C4H4N2', 'C6H4O', 'C4H2O2', 'C4H5N', 'C3H5N3O', 'C3HN', 'C3H8', 'C4H6N2', 'C2H4N2O2', 'C4H10N2O', 'C3H7N', 'C5H6O2', 'C2H3N', 'C6H5N', 'C3H7N3O', 'C4H4O', 'C5H6N2', 'C4H6O3', 'C4H6O2', 'C5H10O', 'C2H5NO', 'C7H8', 'C5H8', 'C6H9N', 'C4H4O2', 'C4H8O', 'C3H6N2O2', 'C3H3NO', 'C5H4N2', 'C4H4O3', 'C5H7N', 'C2HNO', 'C4H10', 'CHN3O', 'C5H9NO', 'C6H11N', 'C6H14', 'C3H3N3O', 'C2H3N3O', 'C4H7N3', 'C4H5N3', 'C3H4N2O2', 'C3H8O3', 'C4H8', 'C6H12', 'C5H6O', 'C6H10', 'C4H3N3', 'C6H8', 'C6H10O', 'C4H9NO', 'C2H4O'],
                ['C3H2O3', 'C6H13N', 'C2H4N4', 'C2H6O2', 'C2H6N2O', 'C3H6N2O', 'C4H2N2O', 'C4H6', 'C5H5N', 'C3H2N4', 'C5H3NO', 'C6H6', 'C3H4N2', 'C4H7NO', 'C2H5NO2', 'C3H5NO2', 'C4H3NO', '.DS_S', 'C3H4O3', 'C3H5NO3', 'C5H8O', 'C2H6O', 'C5H5NO', 'C2H4N2O', 'C3H2N2', 'C6H12O', 'C3H2O', 'C3H6O2', 'C3H4N4', 'C3H8N2O', 'C2H2N2O2', 'C3H6O3', 'C2H2O2', 'C5H4O', 'C5H12', 'C4H9NO2', 'C3HNO2', 'C2N2', 'C5H4', 'C5H10N2', 'C5H8N2', 'C4H5NO2', 'C4H8O3', 'C7H16', 'C3H5N3', 'C4H4N2O', 'C3H5N', 'C4H8N2O', 'C7H14', 'C2H3N3', 'C5H6', 'C2H3N3O2', 'C6H6O', 'C5H10', 'C3H4O', 'C5H12O', 'C3H4', 'C4H6N2O', 'C5H12O2', 'C3H3N3', 'C7H10', 'C2H2N2O', 'C3H2N2O', 'C4H6O', 'C6H7N', 'CN2O3', 'C5H2O', 'C3H3NO3', 'C4H7NO2', 'C4H10O3', 'C4H8N2', 'C2H3N5', 'C5H8O2', 'C4H10O2', 'C5H10O2', 'CH3N5', 'C3H2N2O2', 'C2HNO3', 'C2H3NO2', 'C3H3NO2', 'C3H7NO', 'C4H9N', 'C5H4O2', 'C2H3NO', 'C4H4N2', 'CH2N4O', 'C6H4O', 'C4H2O2', 'C4H5N', 'C3HN', 'C3H8', 'C4H6N2', 'C2H4N2O2', 'CH4N2O', 'C4HNO', 'C4H10N2O', 'C3H7N', 'C5H6O2', 'C2H3N', 'C6H5N', 'C3H7N3O', 'C4H4O', 'C4H6O3', 'C4H6O2', 'C5H10O', 'C3H6O', 'C7H8', 'C5H8', 'C6H9N', 'C4H4O2', 'C4H8O', 'C3H6N2O2', 'C3H3NO', 'C5H4N2', 'C4H4O3', 'C5H7N', 'C2HNO', 'C4H10', 'CHN3O', 'C5H9NO', 'C6H11N', 'C6H14', 'C3H3N3O', 'C2H3N3O', 'C4H7N3', 'C4H5N3', 'C3H4N2O2', 'C3H8O3', 'C4H8', 'C6H12', 'C5H6O', 'C6H10', 'C4H3N3', 'C6H8', 'C6H10O', 'C4H9NO', 'C2H4O'],
                ['C3H2O3', 'C6H13N', 'C2H6O2', 'C3H6N2O', 'C4H2N2O', 'C4H6', 'C3H6N2', 'C5H5N', 'C3H2N4', 'C3N2O', 'C5H3NO', 'C3H4N2', 'C4H3NO2', 'C4H7NO', 'C2H2N4', 'C2H5NO2', 'C3H5NO2', 'C4H3NO', '.DS_S', 'C3H4O3', 'C3H4O2', 'C3H5NO3', 'C5H8O', 'C2H6O', 'C2H4N2O', 'C3H2N2', 'C3H2O', 'C3H6O2', 'C3H4N4', 'C2H2N2O2', 'C3H6O3', 'C2H2O2', 'C3H8N2O2', 'C5H4O', 'C5H12', 'C4H9NO2', 'C3HNO2', 'C5H10N2', 'C5H8N2', 'C4H5NO2', 'C4H8O3', 'C4H8O2', 'C7H16', 'C3H5N3', 'C4H4N2O', 'C3H5N', 'C4H8N2O', 'C7H14', 'C2H3N3', 'C4H7N', 'C5H6', 'C2H3N3O2', 'C6H6O', 'C5H10', 'C3H4O', 'C5H12O', 'C4N2', 'C3H4', 'C4H6N2O', 'C5H12O2', 'C3H3N3', 'C7H10', 'C3H2N2O', 'C4H6O', 'C6H7N', 'CN2O3', 'C5H2O', 'C3H3NO3', 'C4H7NO2', 'C4H10O3', 'C3H8O', 'C4H8N2', 'C2H3N5', 'C5H8O2', 'C4H10O2', 'C5H10O2', 'CH3N5', 'CHN3O2', 'C3H2N2O2', 'C2HNO3', 'C2H3NO2', 'C3H3NO2', 'C3H7NO', 'C6H8O', 'C4H9N', 'C5H4O2', 'C2H3NO', 'C4H4N2', 'CH2N4O', 'C6H4O', 'C3H5N3O', 'C3HN', 'C3H8', 'C4H6N2', 'C2H4N2O2', 'CH4N2O', 'C4HNO', 'C4H10N2O', 'C3H7N', 'C5H6O2', 'C2H3N', 'C6H5N', 'C3H7N3O', 'C4H4O', 'C5H6N2', 'C4H6O3', 'C5H10O', 'C3H6O', 'C2H5NO', 'C7H8', 'C5H8', 'C6H9N', 'C4H4O2', 'C4H8O', 'C3H6N2O2', 'C3H3NO', 'C5H4N2', 'C4H4O3', 'C5H7N', 'C2HNO', 'C4H10', 'CHN3O', 'C5H9NO', 'C6H11N', 'C6H14', 'C3H3N3O', 'C2H3N3O', 'C4H7N3', 'C4H5N3', 'C3H4N2O2', 'C3H8O3', 'C4H8', 'C6H12', 'C5H6O', 'C3HN3O', 'C4H3N3', 'C6H8', 'C4H9NO', 'C2H4O']]

    os.makedirs(f'{DIR_T1x}/splits/', exist_ok=True)
    for num_crossval in range(4):
        output_dir = f'{DIR_T1x}/splits/composition_split_5{num_crossval}aug'
        os.makedirs(output_dir, exist_ok=True)

        # First pass: determine sizes and classify files
        N = {'train': 0, 'val': 0, 'test': 0, 'extra': 0}
        file_lists = {'train': [], 'val': [], 'test': [], 'extra': []}
        
        for file in tqdm(os.listdir(f'{DIR_T1x}/augmented_data'), desc=f"CV {num_crossval} Scan"):
            if not file.endswith('.npz'):
                continue

            formula = file[:-4]
            if formula in set_test:
                set_split = 'test'
            elif formula in set_val[num_crossval]:
                set_split = 'val'
            elif formula in set_train[num_crossval]:
                set_split = 'train'
            else:
                set_split = 'extra'

            data = np.load(f'{DIR_T1x}/augmented_data/{file}')
            if len(data['R'].shape) < 2:
                print(f"Shape of {file} is faulty: {data['R'].shape}")
                continue
            
            N[set_split] += data['R'].shape[0]
            file_lists[set_split].append(file)

        # Create memory-mapped files
        mmap_files = {}
        for split in ['train', 'val', 'test', 'extra']:
            if N[split] == 0: continue
            mmap_files[split] = {
                'R': np.memmap(f'{output_dir}/{split}_R.mmap', dtype=np.float32, mode='w+', shape=(N[split], A_max, 3)),
                'Z': np.memmap(f'{output_dir}/{split}_Z.mmap', dtype=np.int64, mode='w+', shape=(N[split], A_max)),
                'E': np.memmap(f'{output_dir}/{split}_E.mmap', dtype=np.float64, mode='w+', shape=(N[split], 1)),
                'F': np.memmap(f'{output_dir}/{split}_F.mmap', dtype=np.float64, mode='w+', shape=(N[split], A_max, 3)),
            }

        # Second pass: fill memory-mapped arrays
        current_idx = {'train': 0, 'val': 0, 'test': 0, 'extra': 0}
        for split in ['train', 'val', 'test', 'extra']:
            if N[split] == 0: continue
            for file in tqdm(file_lists[split], desc=f"CV {num_crossval} Fill {split}"):
                data = np.load(f'{DIR_T1x}/augmented_data/{file}')
                n_samples = data['R'].shape[0]
                A = data['R'].shape[1]
                
                start = current_idx[split]
                end = start + n_samples
                
                mmap_files[split]['R'][start:end] = np.pad(data['R'], pad_width=((0,0),(0,A_max-A),(0,0)))
                mmap_files[split]['Z'][start:end] = np.pad(data['Z'], pad_width=((0,0),(0,A_max-A)))
                mmap_files[split]['E'][start:end] = data['E']
                mmap_files[split]['F'][start:end] = np.pad(data['F'], pad_width=((0,0),(0,A_max-A),(0,0)))
                
                current_idx[split] = end
        
        # Save splits to npz
        for set_split in ['train', 'val', 'test']:
            if N[set_split] > 0:
                np.savez_compressed(f'{output_dir}/{set_split}_data.npz',
                    R=mmap_files[set_split]['R'],
                    Z=mmap_files[set_split]['Z'],
                    E=mmap_files[set_split]['E'],
                    F=mmap_files[set_split]['F'])
                print(set_split, 'samples:', N[set_split])

        # Handle 'extra' data (train + extra)
        N_extra_total = N['train'] + N['extra']
        if N_extra_total > 0:
            extra_R = np.memmap(f'{output_dir}/extra_R.mmap', dtype=np.float32, mode='w+', shape=(N_extra_total, A_max, 3))
            extra_Z = np.memmap(f'{output_dir}/extra_Z.mmap', dtype=np.int64, mode='w+', shape=(N_extra_total, A_max))
            extra_E = np.memmap(f'{output_dir}/extra_E.mmap', dtype=np.float64, mode='w+', shape=(N_extra_total, 1))
            extra_F = np.memmap(f'{output_dir}/extra_F.mmap', dtype=np.float64, mode='w+', shape=(N_extra_total, A_max, 3))

            if N['train'] > 0:
                extra_R[:N['train']] = mmap_files['train']['R'][:]
                extra_Z[:N['train']] = mmap_files['train']['Z'][:]
                extra_E[:N['train']] = mmap_files['train']['E'][:]
                extra_F[:N['train']] = mmap_files['train']['F'][:]
            
            if N['extra'] > 0:
                extra_R[N['train']:] = mmap_files['extra']['R'][:]
                extra_Z[N['train']:] = mmap_files['extra']['Z'][:]
                extra_E[N['train']:] = mmap_files['extra']['E'][:]
                extra_F[N['train']:] = mmap_files['extra']['F'][:]

            np.savez_compressed(f'{output_dir}/extra_data.npz',
                R=extra_R, Z=extra_Z, E=extra_E, F=extra_F)
            print('extra samples:', N_extra_total)
        
        # Cleanup
        for split in mmap_files:
            for key in mmap_files[split]:
                mmap_files[split][key]._mmap.close()
                if os.path.exists(f'{output_dir}/{split}_{key}.mmap'):
                    os.remove(f'{output_dir}/{split}_{key}.mmap')
        
        if N_extra_total > 0:
            if 'extra_R' in locals() and extra_R._mmap: extra_R._mmap.close()
            if os.path.exists(f'{output_dir}/extra_R.mmap'): os.remove(f'{output_dir}/extra_R.mmap')
            if 'extra_Z' in locals() and extra_Z._mmap: extra_Z._mmap.close()
            if os.path.exists(f'{output_dir}/extra_Z.mmap'): os.remove(f'{output_dir}/extra_Z.mmap')
            if 'extra_E' in locals() and extra_E._mmap: extra_E._mmap.close()
            if os.path.exists(f'{output_dir}/extra_E.mmap'): os.remove(f'{output_dir}/extra_E.mmap')
            if 'extra_F' in locals() and extra_F._mmap: extra_F._mmap.close()
            if os.path.exists(f'{output_dir}/extra_F.mmap'): os.remove(f'{output_dir}/extra_F.mmap')

        print(f"Complete composition split {num_crossval}")

    print('done!')

if do_create_composition_splits:
    create_splits()



def create_crossval_splits():
    print('\n# Creating augmented conformation splits')
    np.random.seed(0)
    
    # First pass: count total samples and create memory-mapped arrays
    N_tot = 0
    for file in tqdm(os.listdir(f'{DIR_T1x}/augmented_data'), desc="Counting samples"):
        if (file.endswith('.npz')) and (file in os.listdir(f'{DIR_T1x}/composition_data')):
            data = np.load(f'{DIR_T1x}/augmented_data/{file}')
            if len(data['R'].shape) >= 2:
                N_tot += data['R'].shape[0]

    # Create memory-mapped arrays for the full dataset
    mmap_dir = f'{DIR_T1x}/augmented_splits/temp_mmap'
    os.makedirs(mmap_dir, exist_ok=True)
    
    mmap_files = {
        'R': np.memmap(f'{mmap_dir}/R.mmap', dtype=np.float32, mode='w+', shape=(N_tot, A_max, 3)),
        'Z': np.memmap(f'{mmap_dir}/Z.mmap', dtype=np.int64, mode='w+', shape=(N_tot, A_max)),
        'E': np.memmap(f'{mmap_dir}/E.mmap', dtype=np.float64, mode='w+', shape=(N_tot, 1)),
        'F': np.memmap(f'{mmap_dir}/F.mmap', dtype=np.float64, mode='w+', shape=(N_tot, A_max, 3))
    }

    # Second pass: fill memory-mapped arrays
    current_idx = 0
    for file in tqdm(os.listdir(f'{DIR_T1x}/augmented_data'), desc="Loading data"):
        if (file.endswith('.npz')) and (file in os.listdir(f'{DIR_T1x}/composition_data')):
            data = np.load(f'{DIR_T1x}/augmented_data/{file}')
            if len(data['R'].shape) < 2:
                print(f"Shape of {file} is faulty: {data['R'].shape}")
                continue

            n_samples = data['R'].shape[0]
            A = data['R'].shape[1]
            
            start = current_idx
            end = start + n_samples
            
            mmap_files['R'][start:end] = np.pad(data['R'], pad_width=((0,0),(0,A_max-A),(0,0)))
            mmap_files['Z'][start:end] = np.pad(data['Z'], pad_width=((0,0),(0,A_max-A)))
            mmap_files['E'][start:end] = data['E']
            mmap_files['F'][start:end] = np.pad(data['F'], pad_width=((0,0),(0,A_max-A),(0,0)))
            
            current_idx = end

    # Create cross-validation splits
    for num_crossval in range(4):
        N = 0
        split = np.random.choice(['train', 'val', 'test'], N_tot, p=[0.8, 0.1, 0.1])
        
        output_dir = f'{DIR_T1x}/augmented_splits/conformation_split_{num_crossval}aug1M'
        os.makedirs(output_dir, exist_ok=True)
        
        for set_split in ['train', 'val', 'test']:
            # Create memory-mapped arrays for the split
            split_mmap = {
                'R': np.memmap(f'{mmap_dir}/{set_split}_R.mmap', dtype=np.float32, mode='w+', 
                              shape=(np.sum(split==set_split), A_max, 3)),
                'Z': np.memmap(f'{mmap_dir}/{set_split}_Z.mmap', dtype=np.int64, mode='w+', 
                              shape=(np.sum(split==set_split), A_max)),
                'E': np.memmap(f'{mmap_dir}/{set_split}_E.mmap', dtype=np.float64, mode='w+', 
                              shape=(np.sum(split==set_split), 1)),
                'F': np.memmap(f'{mmap_dir}/{set_split}_F.mmap', dtype=np.float64, mode='w+', 
                              shape=(np.sum(split==set_split), A_max, 3))
            }
            
            # Copy data to split arrays
            split_idx = np.where(split==set_split)[0]
            for key in ['R', 'Z', 'E', 'F']:
                split_mmap[key][:] = mmap_files[key][split_idx]
            
            # Save to npz
            np.savez_compressed(f'{output_dir}/{set_split}_data.npz',
                R=split_mmap['R'],
                Z=split_mmap['Z'],
                E=split_mmap['E'],
                F=split_mmap['F'])
            
            print(set_split, 'samples:', np.sum(split==set_split))
            N += np.sum(split==set_split)
            
            # Cleanup split memory maps
            for key in split_mmap:
                split_mmap[key]._mmap.close()
                if os.path.exists(f'{mmap_dir}/{set_split}_{key}.mmap'):
                    os.remove(f'{mmap_dir}/{set_split}_{key}.mmap')
        
        print('total:', N)
        print(f"Complete augmented conformation split {num_crossval}")

    # Cleanup main memory maps
    for key in mmap_files:
        mmap_files[key]._mmap.close()
        if os.path.exists(f'{mmap_dir}/{key}.mmap'):
            os.remove(f'{mmap_dir}/{key}.mmap')
    
    # Remove temporary directory
    if os.path.exists(mmap_dir):
        shutil.rmtree(mmap_dir)
    
    print('done!')

if do_create_aug_conformation_splits:
    create_crossval_splits()
