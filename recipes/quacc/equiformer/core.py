"""Core recipes for the Equiformer code."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from ase.vibrations.data import VibrationsData
from monty.dev import requires

from quacc import get_settings, job
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize, VibSummarize
from quacc.utils.dicts import recursive_dict_merge

from sella import Sella

from gadff.equiformer_ase_calculator import EquiformerASECalculator

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quacc.types import (
        Filenames,
        OptParams,
        OptSchema,
        RunSchema,
        SourceDirectory,
        VibThermoSchema,
    )


@job
def static_job(
    atoms: Atoms,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    calc: EquiformerASECalculator | None = None,
    **calc_kwargs,
) -> RunSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Equiformer calculator.

    Returns
    -------
    RunSchema
        Dictionary of results, specified in [quacc.schemas.ase.Summarize.run][].
        See the type-hint for the data structure.
    """
    calc_defaults = {}
    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)

    if calc is None:
        calc = EquiformerASECalculator(**calc_flags)
    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        additional_fields={"name": "Equiformer Static"} | (additional_fields or {})
    ).run(final_atoms, atoms)


@job
def relax_job(
    atoms: Atoms,
    opt_params: OptParams | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    calc: EquiformerASECalculator | None = None,
    **calc_kwargs,
) -> OptSchema:
    """
    Relax a structure.

    Parameters
    ----------
    atoms
        Atoms object
    opt_params
        Dictionary of custom kwargs for the optimization process. For a list
        of available keys, refer to [quacc.runners.ase.Runner.run_opt][].
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Dictionary of custom kwargs for the Equiformer calculator.

    Returns
    -------
    OptSchema
        Dictionary of results, specified in [quacc.schemas.ase.Summarize.opt][].
        See the type-hint for the data structure.
    """
    calc_defaults = {}
    opt_defaults = {"optimizer": Sella} 

    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)

    if calc is None:
        calc = EquiformerASECalculator(**calc_flags)
    dyn = Runner(atoms, calc, copy_files=copy_files).run_opt(**opt_flags)

    return _add_hessian(
        Summarize(
            additional_fields={"name": "Equiformer Relax"} | (additional_fields or {})
        ).opt(dyn)
    )


@job
def freq_job(
    atoms: Atoms,
    temperature: float = 298.15,
    pressure: float = 1.0,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    calc: EquiformerASECalculator | None = None,
    **calc_kwargs,
) -> VibThermoSchema:
    """
    Perform a frequency calculation using the given atoms object.

    Parameters
    ----------
    atoms
        The atoms object representing the system.
    temperature
        The temperature for the thermodynamic analysis.
    pressure
        The pressure for the thermodynamic analysis.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Equiformer calculator.

    Returns
    -------
    VibThermoSchema
        Dictionary of results. See the type-hint for the data structure.
    """
    calc_defaults = {}
    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)

    if calc is None:
        calc = EquiformerASECalculator(**calc_flags)
    
    # Calculate with hessian to get frequencies
    calc.calculate(atoms, properties=["energy", "forces", "hessian"])
    final_atoms = atoms.copy()
    final_atoms.calc = calc

    summary = Summarize(
        additional_fields={"name": "Equiformer Frequency"} | (additional_fields or {})
    ).run(final_atoms, atoms)

    vib = VibrationsData(final_atoms, summary["results"]["hessian"])
    return VibSummarize(
        vib,
        directory=summary["dir_name"],
        additional_fields={"name": "ASE Vibrations and Thermo Analysis"},
    ).vib_and_thermo(
        "ideal_gas",
        energy=summary["results"]["energy"],
        temperature=temperature,
        pressure=pressure,
    )


def _add_hessian(summary: dict[str, Any], **calc_kwargs) -> dict[str, Any]:
    """
    Calculate and add Hessian to the summary.

    This function takes a summary dictionary containing information about a
    molecular trajectory and calculates the Hessian using the Equiformer
    machine learning calculator. It adds the calculated Hessian values to each
    configuration in the trajectory.

    Parameters
    ----------
    summary
        A dictionary containing information about the molecular trajectory.
    **calc_kwargs
        Custom kwargs for the Equiformer calculator.

    Returns
    -------
    dict[str, Any]
        The modified summary dictionary with added Hessian values.
    """
    calc_defaults = {}
    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)
    
    for i, atoms in enumerate(summary["trajectory"]):
        calc = EquiformerASECalculator(**calc_flags)
        hessian = calc.get_hessian_autodiff(atoms)
        summary["trajectory_results"][i]["hessian"] = hessian

    return summary
