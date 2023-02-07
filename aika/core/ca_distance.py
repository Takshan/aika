# Copyright 2023 Rahul Brahma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from Bio.PDB import PDBParser
from .parse_sdf import parse_sdf
import contextlib


RESIDUE_VOCAB = {
    "GLY": 1,
    "ALA": 2,
    "VAL": 3,
    "LEU": 4,
    "ILE": 5,
    "MET": 6,
    "PHE": 7,
    "TRP": 8,
    "PRO": 9,
    "SER": 10,
    "THR": 11,
    "CYS": 12,
    "TYR": 13,
    "ASN": 14,
    "GLN": 15,
    "LYS": 16,
    "ARG": 17,
    "HIS": 18,
    "ASP": 19,
    "GLU": 20,
    "UNK": -1,  # unknown
}


def sc_ca_distance(
    protein, ligand, cutoff=4, res_type=True, removeHs=False, sanitize=False, addHs=True
):
    
    """Calculates the distance between the side chain and the ligand.

    Returns:
        np.ndarray: CA distance matrix
    """    
    
    parser = PDBParser()
    structure = parser.get_structure("protein", protein)
    # mol = Chem.MolFromMolFile(ligand, removeHs=False, sanitize=False)
    # mol.UpdatePropertyCache(strict=False)
    # if add_H:
    #     mol =Chem.AddHs(mol)
    # ligand_coords=[]
    # for i in range(1, mol.GetNumAtoms()):
    #     pos = mol.GetConformer().GetAtomPosition(i)
    #     ligand_coords.append([pos.x, pos.y, pos.z])
    #     # print(pos.x, pos.y, pos.z)
    # ligand_coords = np.array(ligand_coords)
    ligand_coords = parse_sdf(ligand, addHs=addHs, removeHs=removeHs, sanitize=sanitize)

    sc_coords = []
    res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom.name, atom.get_coord())
                    if (
                        np.linalg.norm(atom.get_coord() - ligand_coords, axis=1).min()
                        < cutoff
                    ):
                        # print(residue.child_dict["CA"].get_coord())
                        # print(residue.child_dict)
                        with contextlib.suppress(KeyError):
                            sc_coords.extend([residue.child_dict["CA"].get_coord()])
                            res.append(residue)
                        break
                        # sc_coords.extend(atom.get_coord() for atom in residue if atom.name != 'CA')
    sc_coords = np.array(sc_coords)
    # distance_matrix = np.zeros((len(sc_coords), len(sc_coords)))
    ca_distance_matrix = np.zeros((len(sc_coords), len(sc_coords)))
    for i, coord in enumerate(sc_coords):
        ca_distance_matrix[i] = np.linalg.norm(coord - sc_coords, axis=1)
    if res_type:
        ca_distance_matrix = np.concatenate(
            (ca_distance_matrix, np.zeros((len(sc_coords), 1))), axis=1
        )
        for i, r in enumerate(res):
            ca_distance_matrix[i, len(sc_coords) :] = RESIDUE_VOCAB[r.resname] or -1
    # print(len(sc_coords))
    # print(len(res), res)
    return ca_distance_matrix
