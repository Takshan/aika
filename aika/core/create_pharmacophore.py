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
from rdkit import Chem
import h5py
from .pharmacophore import get_pharmacophore_fp
import tqdm
import json
import os
import parmap


from joblib import Parallel, delayed


def create_pharmacophore_data_joblib(
    smiles_pdbbind, filename="latest_pdbbind_pharmacophore_joblib.h5"
):
    # Use the Parallel function to apply the pharmacophore function in parallel
    cpu = os.cpu_count()
    results = Parallel(n_jobs=cpu)(
        delayed(get_pharmacophore_fp)(smiles_pdbbind[i], 100)
        for i in smiles_pdbbind.keys()
    )
    # Write the results to a h5py file and add tqdm progress bar
    with h5py.File(filename, "w") as f:
        for i, phar in tqdm.tqdm(enumerate(results)):
            f.create_dataset(list(smiles_pdbbind.keys())[i], data=phar)
        # for i, phar in enumerate(results):
        #     f.create_dataset(list(smiles_pdbbind.keys())[i], data=phar)


def create_pharmacophore_data(
    smiles_pdbbind, filename="latest_pdbbind_pharmacophore.h5"
):
    cpu = os.cpu_count()
    try:
        # multiprocessing pharmacophore
        phamr = parmap.map(
            get_pharmacophore_fp,
            smiles_pdbbind.values(),
            pm_pbar=True,
            pm_processes=cpu,
        )
        with h5py.File(filename, "w") as f:
            for i, phar in enumerate(phamr):
                f.create_dataset(list(smiles_pdbbind.keys())[i], data=phar)
    except Exception as er:
        print(er)
        print("Error in multiprocessing")
    print("Done")


def main(json_file, filename):
    smiles_json = json.load(open(json_file, "r"))
    print("Loading smiles")
    print(f"Total: {len(smiles_json)}")
    create_pharmacophore_data_joblib(smiles_json, filename=filename)
    os.system(f"mv {filename} /home/lab09/BindingAffinity/notebooks/")
    print("Done")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
