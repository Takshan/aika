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
from sklearn.preprocessing import StandardScaler as Scaler
import tqdm
from rdkit import Chem
from rdkit.Chem import Lipinski, Crippen, MolSurf, AllChem, Descriptors as desc
import pandas as pd


def PhyChem(smiles):
    """Calculating the 19D physicochemical descriptors for each molecules,
    the value has been normalized with Gaussian distribution.

    Arguments:
        smiles (list): list of SMILES strings.
    Returns:
        props (ndarray): m X 19 matrix as nomalized PhysChem descriptors.
            m is the No. of samples
    """
    props = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        try:
            MW = desc.MolWt(mol)
            LOGP = Crippen.MolLogP(mol)
            HBA = Lipinski.NumHAcceptors(mol)
            HBD = Lipinski.NumHDonors(mol)
            rotable = Lipinski.NumRotatableBonds(mol)
            amide = AllChem.CalcNumAmideBonds(mol)
            bridge = AllChem.CalcNumBridgeheadAtoms(mol)
            heteroA = Lipinski.NumHeteroatoms(mol)
            heavy = Lipinski.HeavyAtomCount(mol)
            spiro = AllChem.CalcNumSpiroAtoms(mol)
            FCSP3 = AllChem.CalcFractionCSP3(mol)
            ring = Lipinski.RingCount(mol)
            Aliphatic = AllChem.CalcNumAliphaticRings(mol)
            aromatic = AllChem.CalcNumAromaticRings(mol)
            saturated = AllChem.CalcNumSaturatedRings(mol)
            heteroR = AllChem.CalcNumHeterocycles(mol)
            TPSA = MolSurf.TPSA(mol)
            valence = desc.NumValenceElectrons(mol)
            mr = Crippen.MolMR(mol)
            # charge = AllChem.ComputeGasteigerCharges(mol)
            prop = [
                MW,
                LOGP,
                HBA,
                HBD,
                rotable,
                amide,
                bridge,
                heteroA,
                heavy,
                spiro,
                FCSP3,
                ring,
                Aliphatic,
                aromatic,
                saturated,
                heteroR,
                TPSA,
                valence,
                mr,
            ]
        except Exception:
            print(smile)
            prop = [0] * 19
        props.append(prop)
    props = np.array(props)
    props = Scaler().fit_transform(props)
    return props


def properties(fnames, labels, is_active=False):
    """Five structural properties calculation for each molecule in each given file.
    These properties contains No. of Hydrogen Bond Acceptor/Donor, Rotatable Bond,
    Aliphatic Ring, Aromatic Ring and Heterocycle.

    Arguments:
        fnames (list): the file path of molecules.
        labels (list): the label for each file in the fnames.
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        df (DataFrame): the table contains three columns; 'Set' is the label
            of fname the molecule belongs to, 'Property' is the name of one
            of five properties, 'Number' is the property value.
    """

    props = []
    for i, fname in enumerate(fnames):
        df = pd.read_table(fname)
        if "SCORE" in df.columns:
            df = df[df.SCORE > (0.5 if is_active else 0)]
        elif "PCHEMBL_VALUE" in df.columns:
            df = df[df.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
        df = df.drop_duplicates(subset="CANONICAL_SMILES")
        if len(df) > int(1e5):
            df = df.sample(int(1e5))
        for smile in tqdm(df.CANONICAL_SMILES):
            mol = Chem.MolFromSmiles(smile)
            HA = Lipinski.NumHAcceptors(mol)
            props.append([labels[i], "Hydrogen Bond\nAcceptor", HA])
            HD = Lipinski.NumHDonors(mol)
            props.append([labels[i], "Hydrogen\nBond Donor", HD])
            RB = Lipinski.NumRotatableBonds(mol)
            props.append([labels[i], "Rotatable\nBond", RB])
            RI = AllChem.CalcNumAliphaticRings(mol)
            props.append([labels[i], "Aliphatic\nRing", RI])
            AR = Lipinski.NumAromaticRings(mol)
            props.append([labels[i], "Aromatic\nRing", AR])
            HC = AllChem.CalcNumHeterocycles(mol)
            props.append([labels[i], "Heterocycle", HC])
    df = pd.DataFrame(props, columns=["Set", "Property", "Number"])
    return df
