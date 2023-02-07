import pytorch_lightning  as pl
import torch
from torch import nn
from typing import Any

from aika.models import component

class PDBBindTransformerModel(pl.LightningModule):
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
        self.smile_tranformer = nn.TransformerEncoder(self.smile_tranformer_layer, num_layers=4)
        self.ff = component.FeedForward(5088, 1024, 0.3)
        self.layer = nn.Sequential(
            nn.Linear(5088, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, aa = (
            x[0].unsqueeze(1).to(torch.int32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
        )
        protein_seq = protein_sequence.view(protein_sequence.size(0), 1, -1)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa.shape, protein_seq.shape)

        # int pro

        protein_emb = self.protein_emb(protein_seq)
        protein_emb = protein_emb.squeeze(-1)
        protein_transformed = self.protein_transformer(protein_emb)

        out = torch.cat([protein_transformed, ligand_smile, smi_vect, aa], dim=-1)
        # print(out.shape, "out")
        out = self.ff(out)
        return self.layer(out)
    
    
