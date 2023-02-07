from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from seqvec.data.h5df_loader import load_dict_from_hdf5

from aika.utilities.utilis import get_pad_array
from .chem import morgan_fingerprint
from .smile2vec import Smi2Vec

console = Console()


class PDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_encoding: Union[str, None] = None,
        smiles_pharmacophore: Union[str, None] = None,
        WORD2VEC_MODEL: Union[str, None] = None,
    ) -> None:
        try:
            dataframe = pd.read_csv(data)
        except Exception as _error:
            # console.print(f"loading data from {data} .{_error}",style="red")
            dataframe = data
        self.data = dataframe
        protein_encoding: str = protein_encoding or (
            "/home/lab09/BindingAffinity/reference/NEW_4A_DM_18030.h5"
        )
        smiles_pharmacophore: str = smiles_pharmacophore or (
            "/home/lab09/BindingAffinity/reference/LATEST_NEW_PHARMACOPHORE4.h5"
        )
        self.protein_encoding = load_dict_from_hdf5(protein_encoding)
        self.ligand_pharmacophore = load_dict_from_hdf5(smiles_pharmacophore)
        # self.protein_encoding = h5py.File(protein_encoding, "r")
        # self.ligand_pharmacophore = h5py.File(smiles_pharmacophore, "r")
        WORD2VEC_MODEL: str = (
            WORD2VEC_MODEL or "/home/lab09/seq2vec/reference/model_300dim.pkl"
        )
        self.smiles_vectorizer = Smi2Vec(word2vec_path=WORD2VEC_MODEL)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.data.iloc[
            [index],
        ]
        _id = row.PDB_code.values[0]
        smile = row.canonical_smile.values[0]
        mfp = morgan_fingerprint(smile, radius=3, nBits=512)
        mfp = torch.Tensor(mfp).reshape(1, -1)
        smiles_vec = torch.Tensor(self.smiles_vectorizer.get_mol2vec(smile).vec)
        protein_encode = self.protein_encoding[_id][:]
        # print(protein_encode.shape, "n")
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        amino_acids = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        # aa show (1,40)
        pad_aa = torch.tensor(get_pad_array(amino_acids, (1, 40), 0))
        # one hot encoding
        one_hot = torch.nn.functional.one_hot(pad_aa.long(), num_classes=21)
        protein_encode = protein_encode[:, :-1]
        # print(protein_encode.shape, "m")
        padded_protein = torch.tensor(get_pad_array(protein_encode, (40, 40), 0))
        y = row["-logKd/Ki"].values[0]
        assert ligand_pharmacophore.shape == (
            1,
            3348,
        ), f"Error in {ligand_pharmacophore.shape}: {_id}"
        assert padded_protein.shape == (
            40,
            40,
        ), f"Erorr in {padded_protein.shape}: {_id}"

        return (
            padded_protein,
            ligand_pharmacophore,
            smiles_vec,
            one_hot,
            mfp,
        ), torch.from_numpy(np.array(y)).to(torch.float32)


class PDBDataset1(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_encoding: Union[str, None] = None,
        smiles_pharmacophore: Union[str, None] = None,
        WORD2VEC_MODEL: Union[str, None] = None,
    ) -> None:
        try:
            dataframe = pd.read_csv(data)
        except Exception as _er:
            # console.print(f"loading data from {data} .{_er}",style="red")
            dataframe = data
        self.data = dataframe
        smiles_pharmacophore: str = smiles_pharmacophore
        self.protein_encoding = load_dict_from_hdf5(protein_encoding)
        self.ligand_pharmacophore = load_dict_from_hdf5(smiles_pharmacophore)
        # self.protein_encoding = h5py.File(protein_encoding, "r")
        # self.ligand_pharmacophore = h5py.File(smiles_pharmacophore, "r")
        self.smiles_vectorizer = Smi2Vec(word2vec_path=WORD2VEC_MODEL)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[
            [index],
        ]
        _id = row.PDB_code.values[0]
        smile = row.canonical_smile.values[0]
        mfp = morgan_fingerprint(smile, radius=3, nBits=512)
        mfp = torch.tensor(mfp).reshape(1, -1)
        smiles_vec = self.smiles_vectorizer.get_mol2vec(smile).vec
        smiles_vec = torch.Tensor(smiles_vec)
        protein_encode = self.protein_encoding[_id][:]
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        amino_acids = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        pad_aa = torch.tensor(get_pad_array(amino_acids, (1, 40), 0))
        one_hot = torch.nn.functional.one_hot(pad_aa.long(), num_classes=20)
        protein_encode = protein_encode[:, :-1]
        padded_protein = torch.tensor(get_pad_array(protein_encode, (40, 40), 0))
        y_label = row["-logKd/Ki"].values[0]
        assert ligand_pharmacophore.shape == (
            1,
            3348,
        ), f"Error in {ligand_pharmacophore.shape}: {_id}"
        assert padded_protein.shape == (
            40,
            40,
        ), f"Erorr in {padded_protein.shape}: {_id}"
        return (
            padded_protein,
            ligand_pharmacophore,
            smiles_vec,
            one_hot,
            mfp,
        ), torch.from_numpy(np.array(y_label)).to(torch.float32)



class PDBDataset4(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_encoding: Union[str, None] = None,
        smiles_pharmacophore: Union[str, None] = None,
        WORD2VEC_MODEL: Union[str, None] = None,
    ) -> None:
        try:
            dataframe = pd.read_csv(data)
        except Exception as _er:
            # console.print(f"loading data from {data} .{_er}",style="red")
            dataframe = data
        self.data = dataframe
        smiles_pharmacophore: str = smiles_pharmacophore
        self.protein_encoding = load_dict_from_hdf5(protein_encoding)
        self.ligand_pharmacophore = load_dict_from_hdf5(smiles_pharmacophore)

        self.all_dm = load_dict_from_hdf5("/DATALAKE/Datasets/junk_bindingaffinity/pdbbind_12dm_file.h5")
        # self.protein_encoding = h5py.File(protein_encoding, "r")
        # self.ligand_pharmacophore = h5py.File(smiles_pharmacophore, "r")
        self.smiles_vectorizer = Smi2Vec(word2vec_path=WORD2VEC_MODEL)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[
            [index],
        ]
        _id = row.PDB_code.values[0]
        smile = row.canonical_smile.values[0]
        mfp = morgan_fingerprint(smile, radius=3, nBits=512)
        mfp = torch.tensor(mfp).reshape(1, -1)
        smiles_vec = self.smiles_vectorizer.get_mol2vec(smile).vec
        smiles_vec = torch.Tensor(smiles_vec)
        protein_encode = self.protein_encoding[_id][:]
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        amino_acids = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        pad_aa = torch.tensor(get_pad_array(amino_acids, (1, 40), 0))
        one_hot = torch.nn.functional.one_hot(pad_aa.long(), num_classes=21)
        protein_encode = protein_encode[:, :-1]
        padded_protein = torch.tensor(get_pad_array(protein_encode, (40, 40), 0))
        
        y_label = row["-logKd/Ki"].values[0]
        
        # dm 12
        all_dm = self.all_dm[_id][()]
        all_dm = torch.tensor(get_pad_array(all_dm, (100, 100), 0))
        
        # print(all_dm.shape)
        
        
        
        assert ligand_pharmacophore.shape == (
            1,
            3348,
        ), f"Error in {ligand_pharmacophore.shape}: {_id}"
        assert padded_protein.shape == (
            40,
            40,
        ), f"Erorr in {padded_protein.shape}: {_id}"
        return (
            padded_protein,
            ligand_pharmacophore,
            smiles_vec,
            one_hot,
            mfp,
            all_dm
        ), torch.from_numpy(np.array(y_label)).to(torch.float32)
        
        

class PDBDataset5(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_encoding: Union[str, None] = None,
        smiles_pharmacophore: Union[str, None] = None,
        WORD2VEC_MODEL: Union[str, None] = None,
    ) -> None:
        try:
            dataframe = pd.read_csv(data)
        except Exception as _er:
            # console.print(f"loading data from {data} .{_er}",style="red")
            dataframe = data
        self.data = dataframe
        smiles_pharmacophore: str = smiles_pharmacophore
        self.protein_encoding = load_dict_from_hdf5(protein_encoding)
        self.ligand_pharmacophore = load_dict_from_hdf5(smiles_pharmacophore)

        self.all_cc = load_dict_from_hdf5("/home/lab09/BindingAffinity/reference/signature_all_17441.h5")
        # self.protein_encoding = h5py.File(protein_encoding, "r")
        # self.ligand_pharmacophore = h5py.File(smiles_pharmacophore, "r")
        self.smiles_vectorizer = Smi2Vec(word2vec_path=WORD2VEC_MODEL)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[
            [index],
        ]
        _id = row.PDB_code.values[0]
        smile = row.canonical_smile.values[0]
        mfp = morgan_fingerprint(smile, radius=4, nBits=512)
        mfp = torch.tensor(mfp).reshape(1, -1)
        smiles_vec = self.smiles_vectorizer.get_mol2vec(smile).vec
        smiles_vec = torch.Tensor(smiles_vec)
        protein_encode = self.protein_encoding[_id][:]
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        amino_acids = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        pad_aa = torch.tensor(get_pad_array(amino_acids, (1, 40), 0))
        one_hot = torch.nn.functional.one_hot(pad_aa.long(), num_classes=21)
        protein_encode = protein_encode[:, :-1]
        padded_protein = torch.tensor(get_pad_array(protein_encode, (40, 40), 0))
        
        y_label = row["-logKd/Ki"].values[0]
        
        # dm 12
        all_cc = self.all_cc[_id][()]
        assert ligand_pharmacophore.shape == (
            1,
            3348,
        ), f"Error in {ligand_pharmacophore.shape}: {_id}"
        assert padded_protein.shape == (
            40,
            40,
        ), f"Erorr in {padded_protein.shape}: {_id}"
        return (
            padded_protein,
            ligand_pharmacophore,
            smiles_vec,
            one_hot,
            mfp,
            all_cc
        ), torch.from_numpy(np.array(y_label)).to(torch.float32)
        
        
class PDBDataset6(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_encoding: Union[str, None] = None,
        smiles_pharmacophore: Union[str, None] = None,
        WORD2VEC_MODEL: Union[str, None] = None,
    ) -> None:
        try:
            dataframe = pd.read_csv(data)
        except Exception as _er:
            # console.print(f"loading data from {data} .{_er}",style="red")
            dataframe = data
        self.data = dataframe
        smiles_pharmacophore: str = smiles_pharmacophore
        self.protein_encoding = load_dict_from_hdf5(protein_encoding)
        self.ligand_pharmacophore = load_dict_from_hdf5(smiles_pharmacophore)

        self.all_cc = load_dict_from_hdf5("/home/lab09/BindingAffinity/reference/signature_all_17441.h5")
        # self.protein_encoding = h5py.File(protein_encoding, "r")
        # self.ligand_pharmacophore = h5py.File(smiles_pharmacophore, "r")
        self.smiles_vectorizer = Smi2Vec(word2vec_path=WORD2VEC_MODEL)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[
            [index],
        ]
        _id = row.PDB_code.values[0]
        smile = row.canonical_smile.values[0]
        mfp = morgan_fingerprint(smile, radius=4, nBits=512)
        mfp = torch.tensor(mfp).reshape(1, -1)
        smiles_vec = self.smiles_vectorizer.get_mol2vec(smile).vec
        smiles_vec = torch.Tensor(smiles_vec)
        protein_encode = self.protein_encoding[_id][:]
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        amino_acids = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        pad_aa = torch.tensor(get_pad_array(amino_acids, (1, 40), 0))
        # one_hot = torch.nn.functional.one_hot(pad_aa.long(), num_classes=21)
        protein_encode = protein_encode[:, :-1]
        padded_protein = torch.tensor(get_pad_array(protein_encode, (40, 40), 0))
        
        y_label = row["-logKd/Ki"].values[0]
        
        # dm 12
        all_cc = self.all_cc[_id][()]
        assert ligand_pharmacophore.shape == (
            1,
            3348,
        ), f"Error in {ligand_pharmacophore.shape}: {_id}"
        assert padded_protein.shape == (
            40,
            40,
        ), f"Erorr in {padded_protein.shape}: {_id}"
        return (
            padded_protein,
            ligand_pharmacophore,
            smiles_vec,
            pad_aa,
            mfp,
            all_cc
        ), torch.from_numpy(np.array(y_label)).to(torch.float32)        