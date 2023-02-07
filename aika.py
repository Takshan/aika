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
import os
import subprocess

from rich.console import Console

from aika.models.trasnformer import PDBBindTransformerModel

console = Console()
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from rich.console import Console
from torch.utils.data.dataloader import DataLoader

import wandb
# datsets here
from aika.core import pdbdataset
from aika.core.folddataloader import PDBFoldDataModule
from aika.models.cnn import *
from aika.models.model_module import PDBModelModule
from aika.utilities.model_support import result_dct
from aika.utilities.utilis import config_to_args, parse_yaml
from aika.visual.plots import print_table


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            console.print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def main(args) -> None:
    # lightning seed_everything
    seed_everything(args.seed)
    subprocess.run(
        f'wandb {"online" if args.wandb else "disabled"}', shell=True
    )  # disenable or offline
    CHECKPOINT_PATH: str = args.default_root_dir or "./checkpoints"
    # DATASETS_PATH = path.join(path.dirname("__filename__"), "..", "..", "Datasets")
    # get random name for run
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdirs(CHECKPOINT_PATH, exist_ok=True)
    ## progress bar ##
    progress_bar: RichProgressBar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="bright_magenta",
        )
    )
    ## wandb logging ##
    wandb.init(project=args.project_name, tags=[args.tag])
    wandb_logger: WandbLogger = WandbLogger(
        log_model="all",
        project=args.project_name,
        save_dir=CHECKPOINT_PATH,
        settings=wandb.Settings(code_dir="../"),
    )
    wandb.config.update(args)
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    run_name = wandb.run.id

    early_stop_callback: EarlyStopping = EarlyStopping(
        monitor=args.es_monitor,
        min_delta=0.1,
        patience=args.patience,
        verbose=True,
        mode=args.monitor_mode,
    )

    lr_monitor: LearningRateMonitor = LearningRateMonitor(logging_interval="epoch")

    #######################
    optimizer_hparams: dict = {  # "lr": args.lr,
        "weight_decay": args.weight_decay,
        # "lr_scheduler": None,
    }

    models = {
        0: PDBBindingModel(),
        1: PDBBindTransformerModel(),
        2: PDBBindingModel2(),
        3: PDBBindingModel3(),
        4: PDBBindingModel4(),
        5: PDBBindingModel5(),
        51: PDBBindingModel51(),
        511: PDBBindingModel511(),
        512: PDBBindingModel512(),
        513: PDBBindingModel513(),
        5111: PDBBindingModel5111(),
        51112: PDBBindingModel51112(),
        55: PDBBindingModel55(),
        
        # 3 :Model3(),
    }
    dataloaders = {
        0: pdbdataset.PDBDataset,
        1: pdbdataset.PDBDataset1,
        3: pdbdataset.PDBDataset,
        4: pdbdataset.PDBDataset4,
        5: pdbdataset.PDBDataset5,
        6: pdbdataset.PDBDataset6,
    }
    dataset_dataloader = dataloaders[args.dataloader]
    model = PDBModelModule(
        model=models[args.model],
        optimizer_hparams=optimizer_hparams,
        decay_milestone=args.decay_milestone,
        learning_rate=args.lr,
    ).cuda()
    # console.print(pl.utilities.model_summary.ModelSummary(model, max_depth=-1))
    console.rule("Training model..", style="bold green")
    # if args.predict:
    # predictions = trainer.predict(model, data_loader)

    def train(
        model,
        k,
        save_name="_trained_model.ckpt",
        dataset_name="KIBA",
    ):

        model.apply(reset_weights)
        datamodule = PDBFoldDataModule(
            data=args.train,
            test=args.test,
            protein_encoding=args.protein_encoding,
            smiles_encoding=args.smiles_encoding,
            smiles_vectorizer=args.smiles2vector,
            k=k,
            num_folds=args.kfold,
            split_seed=args.seed,
            batch_size=args.batch_size,
            cv=args.cv,
            dataset_name=dataset_name,
            seed=args.seed,
            dataloader=dataset_dataloader,
        )
        # datamodule.prepare_data()
        console.rule(f"Fold {k+1} of {args.kfold}: {args.cv}", style="bold green")
        datamodule.setup()
        train_loader: DataLoader = datamodule.train_dataloader()
        val_loader: DataLoader = datamodule.val_dataloader()
        test_loader: DataLoader = datamodule.test_dataloader()
        # swa_callback = StochasticWeightAveraging(swa_lrs=5e-4, swa_epoch_start=1)

        checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
            save_weights_only=False,
            mode="min",
            save_top_k=1,
            verbose=True,
            monitor="loss",
            dirpath=CHECKPOINT_PATH,
            filename=f"{run_name}_CV{str(k)}" + "_ba_pdbbind_{loss:.2f}",
            every_n_train_steps=0,
            every_n_epochs=1,
            train_time_interval=None,
            save_on_train_epoch_end=True,
        )

        trainer: Trainer = Trainer.from_argparse_args(
            args,
            deterministic=True,
            #   precision=16,
            callbacks=[
                checkpoint_callback,
                lr_monitor,
                early_stop_callback,
                progress_bar,
                #  StochasticWeightAveraging(
                #      swa_lrs=1e-2)
                # swa_callback,
            ],
            logger=wandb_logger,
            num_sanity_val_steps=0,
            default_root_dir=CHECKPOINT_PATH,
            accelerator=args.cuda,  # "ddp" if args.cuda else None ,,
            #   auto_lr_find=True
        )
        if args.auto_lr_find:
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
            # # Results can be found in
            # print(lr_finder.results)
            # # Plot with
            fig = lr_finder.plot(suggest=True)
            fig.save_name = f"lr_finder_{k}.png"
            fig.tight_layout()
            fig.savefig(fig.save_name, dpi=300, format="png")
            wandb.log({"lr": wandb.Image(fig)})

            # # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()
            # # update hparams of the model
            model.hparams.lr = new_lr
            model.hparams.learning_rate = new_lr
            console.log(f"New lr: {new_lr}", style="bold green")

        try:
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
        except KeyError as error:
            # eror when resume with saving weight only
            console.log(f"{error}", style="bold red")
            print(args.resume_from_checkpoint)
            if args.resume_from_checkpoint:
                model = PDBModelModule.load_from_checkpoint(args.resume_from_checkpoint)
                model.train()
                trainer.fit(
                    model, train_dataloaders=train_loader, val_dataloaders=val_loader
                )
        score: list[dict] = trainer.test(
            ckpt_path="best", verbose=True, dataloaders=test_loader
        )
        # result = trainer.test(ckpt_path="best",dataloaders=val_loader)
        # results.append(score)
        console.rule(f"Fold {k+1} of {args.kfold} done..", style="bold green")
        trainer.save_checkpoint(os.path.join(CHECKPOINT_PATH, run_name + save_name))
        return score

    if not args.cv:
        args.kfold = 1
    console.log(f"Seed : {args.seed}", style="bold green")
    # print model summary
    # print(pl.utilities.model_summary.ModelSummary(model, max_depth=-1))
    # INPUT_SAMPLE = (500,100)
    # print(summary(model, INPUT_SAMPLE))
    results = []
    for k in range(args.kfold):
        _result: list[dict] = train(
            model,
            k,
            save_name=f"targated_trained_model_{k}.ckpt",
            dataset_name=args.ds,
        )
        results.append(_result)
    results: dict = result_dct(results)
    wandb.log({"results": results})
    print_table(results)
    # clean stuff
    # shutil.rmtree(ckpt_callback.logdir)
    wandb.finish()


if __name__ == "__main__":

    parser: ArgumentParser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--project_name", default="BA_PDBBind", type=str)
    parser.add_argument("--seed", default=143, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--kfold", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--es_monitor", default="val_loss", type=str, help="early stop monitor"
    )
    parser.add_argument("--patience", default=50, type=int, help="early stop patience")
    parser.add_argument(
        "--monitor_mode", default="min", type=str, help="early stop monitor mode"
    )
    parser.add_argument("--device", default=1, type=int, help="device")
    parser.add_argument(
        "--cuda",
        default="gpu",
        type=str,
        choices=["gpu", "ddp", "cpu", "tpu"],
        help="cuda",
    )
    parser.add_argument("--decay_milestone", default=[120, 150, 180, 250])
    parser.add_argument("--cv", action="store_true", help="use kfold cv")
    parser.add_argument(
        "--ds",
        default="KIBA",
        type=str,
        choices=[
            "KIBA",
            "DAVIS",
        ],
        help="DTI dataset",
    )
    # parser.add_argument("--split-seed", default=1234, type=int)
    parser.add_argument("--tag", default="default", type=str)
    parser.add_argument("--model", default=0, type=int)
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--dataloader", default=0, type=int)
    args: Namespace = parser.parse_args()
    yml = parse_yaml(args.config)
    args = config_to_args(yml, args)
    main(args)
