from collections import namedtuple
import traceback
import torch
from typing import Optional
from torch import Tensor
from torch_geometric.data import Data as TGData
import torchani
import numpy as np
import os
import ase.io
from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG, AU2EV
from pysisyphus.xyzloader import make_xyz_str

# Andreas
try:
    from nets.prediction_utils import compute_extra_props
except Exception as e:
    print(f"Error importing compute_extra_props: {e}")
    traceback.print_exc()
    compute_extra_props = None

try:
    from ocpmodels.common.relaxation.ase_utils import ase_atoms_to_torch_geometric
except Exception as e:
    print(f"Error importing ase_atoms_to_torch_geometric: {e}")
    traceback.print_exc()
    ase_atoms_to_torch_geometric = None


OptResult = namedtuple("OptResult", "opt_geom opt_log")

# from alphanet.calculator import mols_to_batch


def onehot_convert(atomic_numbers):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return np.array(onehot)


def alphanet_mols_to_batch(molecules):
    """
    Function used to transfer a list of ase mols into leftnet input entry format
    """
    natoms, batch, charge = [], [], []
    for count, mol in enumerate(molecules):
        atomic_numbers = mol.get_atomic_numbers()
        coordinates = mol.get_positions()
        natoms.append(len(atomic_numbers))
        batch += [count for i in atomic_numbers]
        charge += [i for i in atomic_numbers]
        if count == 0:
            pos = coordinates
            one_hot = onehot_convert(atomic_numbers)
        else:
            pos = np.vstack([pos, coordinates])
            one_hot = np.vstack([one_hot, onehot_convert(atomic_numbers)])
    # compile as data
    data = TGData(
        natoms=torch.tensor(np.array(natoms), dtype=torch.int64),
        pos=torch.tensor(pos, dtype=torch.float32).requires_grad_(True),
        one_hot=torch.tensor(one_hot, dtype=torch.int64),
        z=torch.tensor(np.array(charge), dtype=torch.int32),
        batch=torch.tensor(np.array(batch), dtype=torch.int64),
        ae=torch.tensor(np.array(natoms), dtype=torch.int64),
    )
    return data


# from torchani.utils import _get_derivatives_not_none
def _get_derivatives_not_none(
    x: Tensor,
    y: Tensor,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
) -> Tensor:
    ret = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    assert ret is not None
    return ret


# from torchani.utils import hessian
def ani_compute_hessian(
    coordinates: Tensor,
    energies: Optional[Tensor] = None,
    forces: Optional[Tensor] = None,
) -> Tensor:
    """Compute analytical hessian from the energy graph or force graph.

    Arguments:
        coordinates (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`
        energies (:class:`torch.Tensor`): Tensor of shape `(molecules,)`, if specified,
            then `forces` must be `None`. This energies must be computed from
            `coordinates` in a graph.
        forces (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`, if specified,
            then `energies` must be `None`. This forces must be computed from
            `coordinates` in a graph.

    Returns:
        :class:`torch.Tensor`: Tensor of shape `(molecules, 3A, 3A)` where A is the number of
        atoms in each molecule
    """
    if energies is None and forces is None:
        raise ValueError("Energies or forces must be specified")
    if energies is not None and forces is not None:
        raise ValueError("Energies or forces can not be specified at the same time")
    if forces is None:
        assert energies is not None
        forces = -_get_derivatives_not_none(coordinates, energies, create_graph=True)
    flattened_force = forces.flatten(start_dim=1)
    force_components = flattened_force.unbind(dim=1)
    return -torch.stack(
        [
            _get_derivatives_not_none(coordinates, f, retain_graph=True).flatten(
                start_dim=1
            )
            for f in force_components
        ],
        dim=1,
    )


def compute_hessian(coords, energy, forces=None):
    # compute force if not given (first-order derivative)
    if forces is None:
        forces = -_get_derivatives_not_none(coords, energy, create_graph=True)
    # get number of element (n_atoms * 3)
    n_comp = forces.view(-1).shape[0]
    # Initialize hessian
    hess = []
    for f in forces.view(-1):
        # compute second-order derivative for each element
        hess_row = _get_derivatives_not_none(coords, -f, retain_graph=True)
        hess.append(hess_row)
    # stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


class MLFF(Calculator):
    conf_key = "mlff"

    def __init__(
        self,
        method="ani",
        model_kwargs={},
        device="cpu",
        ckpt_path=None,
        **kwargs,
    ):
        """MLFF calculator.

        Wrapper for running energy, gradient and Hessian calculations by
        different MLFF.

        Parameters
        ----------
        method: str
            select a MLFF from dpa-2, ani-1xnr, mace-off23, leftnet

        mem : int
            Mememory per core in MB.
        quiet : bool, optional
            Suppress creation of log files.
        """
        super().__init__(**kwargs)

        self.method = method
        valid_method = (
            "ani",
            "mace",
            "dpa2",
            "orb",
            "left",
            "chg",
            "alpha",
            "left-d",
            "orb-d",
            "matter",
            "equiformerv2",
        )
        assert self.method in valid_method, (
            f"Invalid method argument. Allowed arguments are: {', '.join(valid_method)}!"
        )

        # load model
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if self.method == "equiformerv2":
            # Andreas: DeepPrinciple yet unpublished code
            # from equiformerv2.calculator import EquiformerV2Calculator
            # self.model = EquiformerV2Calculator(
            #     weight="/root/.local/mlff/equiformerv2/ts1x-tuned_epoch2199.ckpt",
            #     device="cpu",
            # )

            from gadff.equiformer_ase_calculator import EquiformerASECalculator

            if ckpt_path is not None:
                model_kwargs["checkpoint_path"] = ckpt_path

            self.model = EquiformerASECalculator(
                device=self.device,
                **model_kwargs,
            )
            self.model.device = self.device
            self.model.potential.to(self.device)

        elif self.method == "alpha":
            from horm_alphanet.calculator import AlphaNetCalculator

            if ckpt_path is None:
                ckpt_path = "/root/.local/mlff/alphanet/ts1x-tuned.ckpt"

            self.model = AlphaNetCalculator(weight=ckpt_path, device=self.device)
        elif self.method == "ani":
            # use a fine-tuned model
            # from torchani.calculator import ANICalculator
            # self.model = ANICalculator('/root/.local/mlff/ani/ts1x-tuned_epoch439.ckpt').model
            # use a pretrained model
            self.model = (
                torchani.models.ANI1x(periodic_table_index=True)
                .to(self.device)
                .double()
            )
        elif self.method == "chg":
            from chgnet.model.dynamics import CHGNetCalculator, chgnet_finetuned

            # use a fine-tuned model
            model = chgnet_finetuned(device="cpu")
            self.model = CHGNetCalculator(model, device=self.device)
            # use a pretrained model
            # self.model = CHGNetCalculator(device=self.device)
        elif self.method == "dpa2":
            from deepmd.infer import DeepEval as DeepPot

            if ckpt_path is None:
                ckpt_path = "/root/.local/mlff/dpa2/ts1x-tuned_epoch1000.pt"

            # deepeval = DeepPot("/root/.local/mlff/dpa2/dpa2-26head.pt", head='Domains_Drug')
            deepeval = DeepPot(ckpt_path, head="Domains_Drug")
            self.model = deepeval.deep_eval.dp.to(self.device)
        elif self.method[:4] == "left":
            from oa_reactdiff.trainer.calculator import LeftNetCalculator

            if "-d" in self.method:
                if ckpt_path is None:
                    ckpt_path = "/root/.local/mlff/leftnet/ts1x-tuned_df_epoch799.ckpt"
                self.model = LeftNetCalculator(
                    ckpt_path,
                    device=self.device,
                    use_autograd=False,
                )
            else:
                if ckpt_path is None:
                    ckpt_path = "/root/.local/mlff/leftnet/ts1x-tuned_epoch999.ckpt"
                self.model = LeftNetCalculator(
                    ckpt_path,
                    device=self.device,
                    use_autograd=True,
                )
        elif self.method == "mace":
            from mace.calculators import mace_off, mace_off_finetuned

            # choose to use fine-tuned/pretrained model
            if torch.cuda.is_available():
                # calc = mace_off(model="medium", default_dtype="float64", device='cuda')
                calc = mace_off_finetuned(device="cuda")
            else:
                # calc = mace_off(model="medium", default_dtype="float64", device='cpu')
                calc = mace_off_finetuned(device="cpu")
            self.model = calc
        elif self.method[:3] == "orb":
            self.device = "cpu"
            from orb_models.forcefield import pretrained

            # use pretrained model
            # orbff = pretrained.orb_v2()
            # use fine-tuned model
            orbff = pretrained.orb_v2_finetuned(device=self.device)
            self.model = orbff
        elif self.method == "matter":
            from mattersim.forcefield import MatterSimCalculator

            if ckpt_path is None:
                ckpt_path = "MatterSim-v1.0.0-5M.pth"

            calc = MatterSimCalculator(load_path=ckpt_path, device=self.device)
            self.model = calc
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def prepare_mol(self, atoms, coords):
        coords = coords * BOHR2ANG
        string = make_xyz_str(atoms, coords.reshape((-1, 3)))
        with open("mlff.xyz", "w") as f:
            f.write(string)
        mol = ase.io.read("mlff.xyz")
        os.remove("mlff.xyz")
        return mol

    def store_and_track(self, results, func, atoms, coords):
        prepare_kwargs = {}  # Andreas: was missing in original code?
        if self.track:
            self.store_overlap_data(atoms, coords)
            if self.track_root():
                # Redo the calculation with the updated root
                results = func(atoms, coords, **prepare_kwargs)
        return results

    def get_energy(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == "ani":
            species = (
                torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long)
                .to(self.device)
                .unsqueeze(0)
            )
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute energy
            energy = self.model((species, coordinates)).energies
            energy = energy.item()
        elif self.method == "dpa2":
            species = (
                torch.tensor(
                    [i - 1 for i in molecule.get_atomic_numbers()], dtype=torch.long
                )
                .to(self.device)
                .unsqueeze(0)
            )
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute energy
            output = self.model(coordinates, species)
            energy = output[0]["energy"].item() / AU2EV
        elif self.method[:3] == "orb":
            from orb_models.forcefield.atomic_system import (
                SystemConfig,
                ase_atoms_to_atom_graphs,
            )

            batch = ase_atoms_to_atom_graphs(
                molecule,
                system_config=SystemConfig(radius=10.0, max_num_neighbors=20),
                brute_force_knn=None,
            ).to(self.device)
            out = self.model.predict(batch)
            energy = out["graph_pred"].item() / AU2EV
        else:  # work for MACE, LEFTNET, CHGNET, ALPHANET
            # set box for chgnet
            if self.method == "chg":
                molecule.cell = [100, 100, 100]
            molecule.calc = self.model
            # compute energy
            energy = molecule.get_potential_energy() / AU2EV

        results = {
            "energy": energy,
        }
        return results

    def get_forces(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == "ani":
            species = (
                torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long)
                .to(self.device)
                .unsqueeze(0)
            )
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute force
            energy = self.model((species, coordinates)).energies
            forces = (
                -_get_derivatives_not_none(coordinates, energy).detach().cpu().numpy()
                * BOHR2ANG
            )
            energy = energy.item()
        elif self.method == "dpa2":
            species = (
                torch.tensor(
                    [i - 1 for i in molecule.get_atomic_numbers()], dtype=torch.long
                )
                .to(self.device)
                .unsqueeze(0)
            )
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute energy
            output = self.model(coordinates, species)
            energy = output[0]["energy"].item() / AU2EV
            forces = output[0]["force"].detach().numpy() / AU2EV * BOHR2ANG
        elif self.method[:3] == "orb":
            from orb_models.forcefield.atomic_system import (
                SystemConfig,
                ase_atoms_to_atom_graphs,
            )

            batch = ase_atoms_to_atom_graphs(
                molecule,
                system_config=SystemConfig(radius=10.0, max_num_neighbors=20),
                brute_force_knn=None,
            ).to(self.device)
            out = self.model.predict(batch)
            energy = out["graph_pred"] / AU2EV
            if "-d" in self.method:
                forces = out["node_pred"].detach().numpy() / AU2EV * BOHR2ANG
            else:
                forces = (
                    -_get_derivatives_not_none(batch.positions, energy)
                    .detach()
                    .cpu()
                    .numpy()
                    * BOHR2ANG
                )
            energy = energy.item()
        else:  # work for MACE, LEFTNET, CHGNET, and ALPHANET
            # set box for chgnet
            if self.method == "chg":
                molecule.cell = [100, 100, 100]
            molecule.calc = self.model
            # compute energy
            energy = molecule.get_potential_energy() / AU2EV
            forces = molecule.get_forces() / AU2EV * BOHR2ANG

        results = {
            "energy": energy,
            "forces": forces.flatten(),
        }

        return results

    def get_hessian(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == "alpha":
            from horm_alphanet.calculator import mols_to_batch

            data = mols_to_batch([molecule]).to(self.device)
            energy, forces = self.model.model.forward(data)
            hessian = (
                compute_hessian(data.pos, energy, forces).detach().cpu().numpy()
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item() / AU2EV
        elif self.method == "ani":
            species = (
                torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long)
                .to(self.device)
                .unsqueeze(0)
            )
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute hessian
            energy = self.model((species, coordinates)).energies
            hessian = (
                compute_hessian(coordinates, energy).detach().cpu().numpy()
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item()
        elif self.method == "chg":
            from pymatgen.io.ase import AseAtomsAdaptor

            molecule.cell = [100, 100, 100]
            structure = AseAtomsAdaptor.get_structure(molecule)
            model = self.model.model
            model.is_intensive = False
            model.composition_model.is_intensive = False
            graph = model.graph_converter(structure)
            model_prediction = model.forward([graph.to(self.device)], task="ef")
            energy = model_prediction["e"][0]
            forces = model_prediction["f"][0]
            pos = model_prediction["bg"].atom_positions[0]
            hessian = (
                compute_hessian(pos, energy, forces).detach().cpu().numpy()
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item() / AU2EV
        elif self.method == "dpa2":
            species = torch.tensor(
                [i - 1 for i in molecule.get_atomic_numbers()],
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            coordinates = (
                torch.from_numpy(molecule.get_positions())
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            # compute energy
            output = self.model(coordinates, species)
            energy = output[0]["energy"].item() / AU2EV
            forces = output[0]["force"] / AU2EV
            # use autograd to compute hessian
            hessian = (
                compute_hessian(coordinates, energy, forces).detach().cpu().numpy()
                * BOHR2ANG
                * BOHR2ANG
            )
        elif self.method[:4] == "left":
            from oa_reactdiff.trainer.calculator import mols_to_batch

            data = mols_to_batch([molecule]).to(self.device)
            # compute energy and force
            if "-d" in self.method:
                energy, forces = self.model.model.forward(data)
            else:
                energy, forces = self.model.model.forward_autograd(data)
            # use autograd of force to compute hessian
            hessian = (
                compute_hessian(data.pos, energy, forces).detach().cpu().numpy()
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item() / AU2EV
        elif self.method == "mace":
            molecule.calc = self.model
            # compute hessian
            hessian = (
                self.model.get_hessian(atoms=molecule).reshape(
                    molecule.get_number_of_atoms() * 3,
                    molecule.get_number_of_atoms() * 3,
                )
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = molecule.get_potential_energy() / AU2EV
        elif self.method[:3] == "orb":
            from orb_models.forcefield.atomic_system import (
                SystemConfig,
                ase_atoms_to_atom_graphs,
            )

            batch = ase_atoms_to_atom_graphs(
                molecule,
                system_config=SystemConfig(radius=10.0, max_num_neighbors=20),
                brute_force_knn=None,
            ).to(self.device)
            # compute energy
            out = self.model.predict(batch)
            energy = out["graph_pred"] / AU2EV
            # compute force
            if "-d" in self.method:
                forces = out["node_pred"] / AU2EV
                hessian = (
                    compute_hessian(batch.positions, energy, forces)
                    .detach()
                    .cpu()
                    .numpy()
                    * BOHR2ANG
                    * BOHR2ANG
                )
            else:
                hessian = (
                    compute_hessian(batch.positions, energy).detach().cpu().numpy()
                    * BOHR2ANG
                    * BOHR2ANG
                )
            # compute hessian
            energy = energy.item()
        elif self.method == "matter":
            from mattersim.datasets.utils.build import build_dataloader
            from mattersim.forcefield.potential import batch_to_dict

            # prepare input dict
            model = self.model.potential
            args_dict = {"batch_size": 1, "only_inference": 1}
            dataloader = build_dataloader(
                [molecule], model_type=model.model_name, **args_dict
            )
            graph = [graph for graph in dataloader][0]
            inp = batch_to_dict(graph)
            out = model.forward(inp, include_forces=True, include_stresses=False)
            energy = out["total_energy"][0] / AU2EV
            forces = out["forces"] / AU2EV
            hessian = (
                compute_hessian(inp["atom_pos"], energy, forces).detach().cpu().numpy()
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item()
        elif self.method == "equiformerv2":
            # Andreas: DeepPrinciple yet unpublished code
            # from equiformerv2.calculator import mols_to_batch
            # data = mols_to_batch([molecule])

            # Not sure which is better: mols_to_batch or ase_atoms_to_torch_geometric ?
            # data = ase_atoms_to_torch_geometric(molecule)
            data = alphanet_mols_to_batch([molecule])

            energy, forces, out = self.model.potential.forward(
                data, eigen=False, hessian=False
            )
            forces = forces.reshape(-1)
            hessian = (
                compute_hessian(data.pos, energy, forces).detach().cpu().numpy()
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item() / AU2EV

        results = {
            "energy": energy,
            "forces": forces.detach().cpu().numpy(),
            "hessian": hessian,
        }

        return results

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"MLFF({self.method})"
