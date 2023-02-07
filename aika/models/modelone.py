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

# dataloader + model here
import os
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from rich.console import Console
from seqvec.data.h5df_loader import load_dict_from_hdf5
from aika.utilities.utilis import get_pad_array
from aika.core.chem import morgan_fingerprint
from aika.core.smile2vec import Smi2Vec

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
            df = pd.read_csv(data)
        except Exception as error:
            # console.print(f"loading data from {data} .")
            df = data
        self.data = df

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
        mfp = torch.tensor(morgan_fingerprint(smile, radius=3, nBits=512)).reshape(
            1, -1
        )
        smiles_vec = torch.Tensor(self.smiles_vectorizer.get_mol2vec(smile).vec)
        protein_encode = self.protein_encoding[_id][:]
        # print(protein_encode.shape, "n")
        ligand_pharmacophore = torch.Tensor(self.ligand_pharmacophore[_id][:])
        aa = torch.Tensor(protein_encode[:, -1]).reshape(1, -1)
        # aa show (1,40)
        pad_aa = torch.tensor(get_pad_array(aa, (1, 40), 0))
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


class Model3(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.protein_emb = nn.Embedding(50, 1)
        self.protein_transformer_layer = nn.TransformerEncoderLayer(
            d_model=1600, nhead=8
        )
        self.protein_transformer = nn.TransformerEncoder(
            self.protein_transformer_layer, num_layers=8
        )

        self.smile_tranformer_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4)
        self.smile_tranformer = nn.TransformerEncoder(
            self.smile_tranformer_layer, num_layers=4
        )
        self.ff = component.FeedForward(5088 + 512, 1024, 0.3)
        self.layer = nn.Sequential(
            nn.Linear(5088 + 512, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, aa, mfp = (
            x[0].unsqueeze(1).to(torch.int32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
        )
        protein_seq = protein_sequence.view(protein_sequence.size(0), 1, -1)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa.shape, protein_seq.shape)

        protein_emb = self.protein_emb(protein_seq)
        protein_emb = protein_emb.squeeze(-1)
        protein_transformed = self.protein_transformer(protein_emb)

        out = torch.cat([protein_transformed, ligand_smile, smi_vect, aa, mfp], dim=-1)
        # print(out.shape, "out")
        out = self.ff(out)
        return self.layer(out)
