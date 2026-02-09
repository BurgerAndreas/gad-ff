#!/usr/bin/env python3
import pubchempy as pcp
import pandas as pd
import os
import re
import time
from tqdm import tqdm

ALLOWED_ELEMENTS = {"H", "C", "N", "O"}


def parse_formula(formula):
    """Parse a molecular formula string into a dict of {element: count}."""
    return dict(
        (el, int(n) if n else 1)
        for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if el
    )


def total_atoms(formula_dict):
    return sum(formula_dict.values())


def only_allowed_elements(formula_dict):
    return set(formula_dict.keys()).issubset(ALLOWED_ELEMENTS)


def fetch_geometries(max_per_natoms=5, natoms_min=30, natoms_max=100, max_cid=100000):
    output_dir = "geometries"
    os.makedirs(output_dir, exist_ok=True)

    found_counts = {n: 0 for n in range(natoms_min, natoms_max + 1)}
    total_to_find = len(found_counts) * max_per_natoms
    total_found = 0
    records = []

    cids = list(range(1, max_cid + 1))
    batch_size = 100

    for i in tqdm(range(0, len(cids), batch_size)):
        if total_found >= total_to_find:
            break

        batch_cids = cids[i : i + batch_size]
        print(f"Batch {i // batch_size + 1} | found {total_found}/{total_to_find}")

        try:
            compounds = pcp.get_compounds(batch_cids)
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            time.sleep(2)
            continue

        for c in compounds:
            if not c.molecular_formula:
                continue

            fd = parse_formula(c.molecular_formula)
            if not only_allowed_elements(fd):
                continue

            natoms = total_atoms(fd)
            if natoms not in found_counts or found_counts[natoms] >= max_per_natoms:
                continue

            filename = os.path.join(output_dir, f"natoms_{natoms}_cid_{c.cid}.sdf")
            if os.path.exists(filename):
                found_counts[natoms] += 1
                total_found += 1
                records.append(
                    {
                        "file": filename,
                        "natoms": natoms,
                        "formula": c.molecular_formula,
                        "pubchem_cid": c.cid,
                        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{c.cid}",
                    }
                )
                continue

            try:
                pcp.download("SDF", filename, c.cid, record_type="3d", overwrite=False)
                print(
                    f"  Downloaded CID {c.cid} | {c.molecular_formula} | {natoms} atoms"
                )
                found_counts[natoms] += 1
                total_found += 1
                records.append(
                    {
                        "file": filename,
                        "natoms": natoms,
                        "formula": c.molecular_formula,
                        "pubchem_cid": c.cid,
                        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{c.cid}",
                    }
                )
            except pcp.NotFoundError:
                pass

            time.sleep(0.2)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "geometries.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} geometries. CSV at {csv_path}")


if __name__ == "__main__":
    fetch_geometries(max_per_natoms=5, natoms_min=30, natoms_max=100)
