import os
from typing import Any, Optional, Tuple, Union
import pytorch_lightning as pl
import torch
from rich.console import Console
from sklearn.model_selection import KFold
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

console = Console()


class PDBFoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str,
        test: str,
        protein_encoding: Union[str, None] = None,
        smiles_encoding: Union[str, None] = None,
        smiles_vectorizer: Union[str, None] = None,
        batch_size: int = 32,
        num_workers: Union[None, int] = None,
        k: Union[int, None] = None,
        num_folds: int = 5,
        split_seed: int = 42,
        pin_memory: bool = True,
        cv: bool = False,
        split_size: Union[None, list] = None,
        dataset_name="KIBA",
        seed=42,
        dataloader=None,
    ) -> None:

        if split_size is None:
            split_size = [0.9, 0.1]
        if num_workers is None:
            num_workers = os.cpu_count()
        super().__init__()
        self.save_hyperparameters(logger=False)
        if k is not None:
            assert 0 <= k <= num_folds, "incorrect fold number"
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.cv: Optional[bool] = cv
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataloader = dataloader

    def prepare_data(self) -> None:
        # download
        # dont assign states here. not maitined the states
        pass

    def setup(self, stage=None):
        if self.data_train or self.data_val:
            return
        # self.data = PDBDataset(
        #     self.hparams.data,
        #     self.hparams.protein_encoding,
        #     self.hparams.smiles_encoding,
        #     self.hparams.smiles_vectorizer,
        # )
        # self.data_test = PDBDataset(self.hparams.test,
        #                             self.hparams.protein_encoding,
        #                             self.hparams.smiles_encoding,
        #                             self.hparams.smiles_vectorizer,)
        # if self.hparams.dataloader is None:
        #     self.hparams.dataloader = PDBDataset
        self.data = self.hparams.dataloader(
            self.hparams.data,
            self.hparams.protein_encoding,
            self.hparams.smiles_encoding,
            self.hparams.smiles_vectorizer,
        )
        self.data_test = self.hparams.dataloader(
            self.hparams.test,
            self.hparams.protein_encoding,
            self.hparams.smiles_encoding,
            self.hparams.smiles_vectorizer,
        )

        if not self.cv:
            all_index = self.data.data.index
            # print(self.UNIPROT_ID, self.data_test)
            data_train_index, data_val_index = random_split(
                all_index,
                self.hparams.split_size,
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.data_train, self.data_val = Subset(
                self.data, data_train_index
            ), Subset(self.data, data_val_index)
            if self.data_test is None or len(self.data_test) == 0:
                # splitting val into test and val 50-50
                data_test_index, data_val_index = random_split(
                    self.data_val.indices, [0.5, 0.5]
                )
                self.data_test = Subset(self.data, data_test_index)
                self.data_val = Subset(self.data, data_val_index)
                # data_test.index =  self.data_test.indices
            # print(self.data_train.indices.indices, self.data_train.indices,)
            # assert not (
            #     set(data_test_index) & set(data_train_index)
            # ), "test and train index overlap. data leakages"
            # assert not (
            #     set(data_test_index) & set(data_val_index)
            # ), "test and val index overlap. data leakages"

            # self.data_train, self.data_val =random_split(_temp_data, self.hparams.split_size)
            # print(max(self.data_train.indices), max(self.data_val.indices), max(self.data_test.index))
        else:
            # extact test data
            oneth = 1 / (self.hparams.num_folds + 1)
            oneth = round(oneth, 3)
            cvth_part = round(1 - oneth, 3)
            main_data_index, data_test_indexes = random_split(
                self.data,
                [cvth_part, oneth],
                generator=torch.Generator().manual_seed(self.seed),
            )
            # all_index = [
            #         x for x in self.data.data.index if x not in data_test_indexes]

            self.data_test = Subset(self.data, data_test_indexes)
            # self.main_data = Subset(self.data, main_data_index)
            kf = KFold(
                n_splits=self.hparams.num_folds,  # making 5 folds + 1 test
                shuffle=True,
                random_state=self.hparams.split_seed,
            )
            all_splits = list(kf.split(main_data_index))
            # remove one split with test
            # test_split = all_splits[-1]

            # all_splits = [all_splits[i] for i in range(
            #     len(all_splits)-1)]

            # # print(all_splits, len(self.data))
            # # pop one split as test
            # _, test_indexes = test_split # last splt is test
            # print(len(test_indexes), len(test_indexes[0]), len(test_indexes[0]))
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            self.data_train, self.data_val = Subset(self.data, train_indexes), Subset(
                self.data, val_indexes
            )
            # self.data_test = self.data_val

        # data infor
        size_train = len(self.data_train)
        size_val = len(self.data_val)
        size_test = len(self.data_test)
        size_data = len(self.data)
        console.rule("[bold red] Data Info")
        console.print(
            f"Total Data Size: {size_data}\n"
            f"Total length of train data: {size_train}\n"
            f"Total length of val data: {size_val}\n"
            f"Total length of test data: {size_test}\n"
        )
        # if not self.cv:
        #     assert size_data == size_train + size_val + size_test, "incorrect split "
        # else:
        #     console.rule("[bold red] Using CV. Using val as test data")

        if size_test <= 0:
            console.print("Test data is 0. Using val as test data")
            self.data_test = self.data_val
        # print(self.data_test.Target_ID.unique(), "test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )
