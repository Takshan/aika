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


# import torch
import random

CONFIG = {
    "model_name": "pdbbin_general_refine",
    # "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "dropout": random.uniform(0.01, 0.60),
    "max_length": 2250,
    "train_batch_size": 64,
    "valid_batch_size": 64,
    "epochs": 100,
    "folds": 5,
    "max_grad_norm": 1000,
    "weight_decay": 1e-6,
    "learning_rate": 1e-4,
    "loss_type": "rmse",
    "n_accumulate": 1,
    "label_cols": "-logKd/Ki",
    "seed": 42,
}
