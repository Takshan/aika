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

from rdkit import Chem
import numpy as np


def parse_sdf(file_path, removeHs=False, sanitize=False,addHs=True, return_smiles=False, rmeoveH_smiles=True):
    sdf = []
    lig = Chem.SDMolSupplier(file_path, sanitize=sanitize, removeHs=removeHs)
    for mol in lig:
        mol = Chem.AddHs(mol) if addHs else Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
            sdf.append(pos)
        if return_smiles:
            if rmeoveH_smiles:
                mol = AllChem.RemoveHs(mol)
            return Chem.MolToSmiles(mol)
    return np.array(sdf)