import pytorch_lightning as pl
import torch
from torch import nn

from aika.models.component import FeedForward, PositionalEncoding


class PDBBindingModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512,1024, kernel_size=2, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            # nn.Linear(512, 1),
        )
        # self.ff2 = FeedForward(self.model_dim+2998)
        # self.output_layer = nn.Linear(4128, 1)
        self.ff = nn.Sequential(
            # nn.Linear(512 + 2988, 1024),
            nn.Linear(4600, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # protein_sequence, ligand_smile, fps = x[0].unsqueeze(1).to(
        #     torch.float32), x[1].unsqueeze(1).to(torch.float32), x[2].unsqueeze(1)
        protein_sequence, ligand_smile, smi_vect, aa = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
        )
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa.shape)

        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.unsqueeze(1).shape,)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                ligand_smile,
                smi_vect,
                # aa
            ],
            dim=-1,
        )
        out = self.ff(out)
        return out


class PDBBindingModel2(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512,1024, kernel_size=2, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            # nn.Linear(512, 1),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512,1024, kernel_size=2, padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            # nn.Linear(512, 1),
        )
        # self.ff2 = FeedForward(self.model_dim+2998)
        # self.output_layer = nn.Linear(4128, 1)
        self.ff = nn.Sequential(
            # nn.Linear(512 + 2988, 1024),
            nn.Linear(5368, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # protein_sequence, ligand_smile, fps = x[0].unsqueeze(1).to(
        #     torch.float32), x[1].unsqueeze(1).to(torch.float32), x[2].unsqueeze(1)
        protein_sequence, ligand_smile, smi_vect, aa = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
        )
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa.shape)
        aa_conv = self.conv_layers1(aa).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.unsqueeze(1).shape,)
        # print(aa_conv.shape, protein_sequence.shape)#1152
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                ligand_smile,
                smi_vect,
                aa_conv,
            ],
            dim=-1,
        )
        out = self.ff(out)
        return out


class PDBBindingModel3(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )

        self.ff = nn.Sequential(
            nn.Linear(5368 + 512, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                ligand_smile,
                smi_vect,
                aa_conv,
                mfp,
            ],
            dim=-1,
        )
        return self.ff(out)


class PDBBindingModel4(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.ff = nn.Sequential(
            nn.Linear(5368 + 512 + 1024, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, dm = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        dm_conv = self.conv_layers2(dm).unsqueeze(1)

        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                ligand_smile,
                smi_vect,
                aa_conv,
                mfp,
                dm_conv,
            ],
            dim=-1,
        )
        return self.ff(out)


class PDBBindingModel5(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.ff = nn.Sequential(
            nn.Linear(9080, 1024),
            # nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            # nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            # nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                ligand_smile,
                smi_vect,
                aa_conv,
                mfp,
                cc,
            ],
            dim=-1,
        )
        return self.ff(out)


class PDBBindingModel51(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Flatten(),
        )
        # self.conv_layers2 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(128, 512, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(1024, 2048, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(2048, 2048, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        # )
        self.ff = nn.Sequential(
            nn.Linear(1920, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200 + 512, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(256 + 100, 128),
            # nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                # ligand_smile,
                # smi_vect,
                aa_conv,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        out = self.ff(out)
        lig = torch.cat([mfp, cc], dim=-1)
        lig = self.ff1(lig)
        out = torch.cat([out, lig, smi_vect], dim=-1)
        return self.ff2(out)


class PDBBindingModel5111(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # nn.Flatten(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            nn.Dropout(0.3),
            # nn.MaxPool2d(2),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=, padding=0),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Flatten(),
        )

        self.ff = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200 + 512, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(1252, 512),
            nn.Linear(512, 128),
            # nn.Linear(256, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence).view(
            ligand_smile.size(0), 1, -1
        )
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, cc.shape, "<<<<")
        out = torch.cat(
            [
                protein_sequence,
                # ligand_smile,
                # smi_vect,
                aa_conv,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        # out = self.ff(out)
        lig = torch.cat([mfp, cc], dim=-1)
        lig = self.ff1(lig)
        # print(lig.shape, out.shape, "<<<<")
        out = torch.cat(
            [
                out,
                lig,
                smi_vect,
            ],
            dim=-1,
        )
        return self.ff2(out)


class PDBBindingModel512(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Flatten(),
        )
        # self.conv_layers2 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(128, 512, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(1024, 2048, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(2048, 2048, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        # )
        self.ff = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200 + 512, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(256 + 100, 128),
            # nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                # protein_sequence.view(ligand_smile.size(0), 1, -1),
                # ligand_smile,
                # smi_vect,
                aa_conv,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        out = self.ff(out)
        lig = torch.cat([mfp, cc], dim=-1)
        lig = self.ff1(lig)
        out = torch.cat([out, lig, smi_vect], dim=-1)
        return self.ff2(out)


class PDBBindingModel513(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.ff = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(256 + 100, 128),
            # nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                # protein_sequence.view(ligand_smile.size(0), 1, -1),
                # ligand_smile,
                # smi_vect,
                aa_conv,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        out = self.ff(out)
        lig = torch.cat([cc], dim=-1)
        lig = self.ff1(lig)
        out = torch.cat([out, lig, smi_vect], dim=-1)
        return self.ff2(out)


class PDBBindingModel511(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.ff = nn.Sequential(
            nn.Linear(1920, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200 + 512, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(256 + 100, 128),
            # nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.float32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        protein_sequence = self.conv_layers(protein_sequence)
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_conv.shape, mfp.shape, dm_conv.shape)
        out = torch.cat(
            [
                protein_sequence.view(ligand_smile.size(0), 1, -1),
                # ligand_smile,
                # smi_vect,
                aa_conv,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        out = self.ff(out)
        lig = torch.cat([mfp, cc], dim=-1)
        lig = self.ff1(lig)
        out = torch.cat([out, lig, smi_vect], dim=-1)
        return self.ff2(out)


class PDBBindingModel51112(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # nn.Flatten(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            # nn.MaxPool2d(2),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=, padding=0),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Flatten(),
        )

        self.ff = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )
        self.ff12 = nn.Sequential(
            nn.Linear(476, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(3200 + 512 + 100, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            nn.Linear(256, 1),
        )
        self.embedding = nn.Embedding(21, 16)
        self.feedf = FeedForward(896, 512)

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.int32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        # aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        aa_embeding = self.embedding(amino_acid).view(amino_acid.size(0), 1, -1)
        protein_sequence = self.conv_layers(protein_sequence).view(
            ligand_smile.size(0), 1, -1
        )
        # print(protein_sequence.shape, ligand_smile.shape, smi_vect.shape, aa_embeding.shape, mfp.shape, cc.shape, "<<<<")
        out = torch.cat(
            [
                protein_sequence,
                # ligand_smile,
                # smi_vect,
                aa_embeding,
                # mfp,
                # cc
            ],
            dim=-1,
        )
        # out = self.ff12(out)
        out = self.feedf(out)

        lig = torch.cat([mfp, cc, smi_vect], dim=-1)
        lig = self.ff1(lig)
        # print(lig.shape, out.shape, "<<<<")
        out = torch.cat([out, lig], dim=-1)
        return self.ff2(out)


class PDBBindingModel55(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # nn.Flatten(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            # nn.ReLU(),
            nn.ELU(),
            # nn.Dropout(0.3),
            # nn.MaxPool2d(2),
        )
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            # nn.ELU(),
            nn.Dropout(0.3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=, padding=0),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Flatten(),
        )

        self.ff = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(4196, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            # nn.Linear(256, 128),
            nn.Linear(128, 1),
        )
        self.embedding = nn.Embedding(21, 16)
        self.feedf = FeedForward(896, 512)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=996, nhead=12, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, 6)
        self.pos_encoder = PositionalEncoding(996, 0.1)

    def forward(self, x):
        protein_sequence, ligand_smile, smi_vect, amino_acid, mfp, cc = (
            x[0].unsqueeze(1).to(torch.float32),
            x[1].to(torch.float32),
            x[2].to(torch.float32),
            x[3].to(torch.int32),
            x[4].to(torch.float32),
            x[5].to(torch.float32).unsqueeze(1),
        )
        # aa_conv = self.conv_layers1(amino_acid).unsqueeze(1)
        aa_embeding = self.embedding(amino_acid).view(amino_acid.size(0), 1, -1)
        protein_sequence = self.conv_layers(protein_sequence).view(
            ligand_smile.size(0), 1, -1
        )
        prot_lig = torch.cat([protein_sequence,aa_embeding, smi_vect], dim=-1)
        # print(prot_lig.shape, "prot_lig")
        post_combine = self.pos_encoder(prot_lig)
        # print(post_combine.shape, "dsads")
        pos_post_combine = prot_lig + post_combine+5
        trans_combine = self.transformer(pos_post_combine)
        # print(trans_combine.shape, "combine")
       # lig_fp = torch.cat([mfp, cc], dim=-1)
        out = torch.cat([trans_combine, cc], dim=-1)
        return self.ff2(out)
