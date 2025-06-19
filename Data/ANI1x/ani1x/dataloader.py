# from
# https://github.com/aiqm/ANI1x_datasets/blob/master/dataloader.py

import h5py
import numpy as np
import os


def iter_data_buckets(h5filename, keys=["wb97x_dz.energy"]):
    """Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard("atomic_numbers")
    keys.discard("coordinates")
    with h5py.File(h5filename, "r") as f:
        for grp in f.values():
            Nc = grp["coordinates"].shape[0]
            mask = np.ones(Nc, dtype=np.bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d["atomic_numbers"] = grp["atomic_numbers"][()]
            d["coordinates"] = grp["coordinates"][()][mask]
            yield d


if __name__ == "__main__":
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the ANI-1x data set
    path_to_h5file = f"{this_file_dir}/data/ani1x-release.h5"

    # List of keys to point to requested data
    data_keys = [
        "wb97x_dz.energy",
        "wb97x_dz.forces",
    ]  # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
    # data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
    # data_keys = ['ccsd(t)_cbs.energy'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
    # data_keys = ['wb97x_dz.dipoles'] # A subset of this data was used for training the ACA charge model (https://doi.org/10.1021/acs.jpclett.8b01939)

    # Example for extracting DFT/DZ energies and forces
    for data in iter_data_buckets(path_to_h5file, keys=data_keys):
        X = data["coordinates"]
        Z = data["atomic_numbers"]
        E = data["wb97x_dz.energy"]
        F = data["wb97x_dz.forces"]
