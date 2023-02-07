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

from multiprocessing import Pool, cpu_count
import os
from typing import Literal, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from rich.console import Console
console = Console()

def basedir_to_file(base_dir) -> Tuple[list, dict]:
    all_dirs: list[str] = os.listdir(base_dir)
    struct_dirs: list = []
    no_structured_dirs: list = []
    complete_dataset: dict = {}
    for i in all_dirs:
        x: Path = Path(f"{base_dir}/{i}")
        sample_protein: str = f"{x}/{x.name}_protein.pdb"
        sample_ligand: str = f"{x}/{x.name}_ligand.sdf"

        sample_protein, sample_ligand = os.path.abspath(
            sample_protein
        ), os.path.abspath(sample_ligand)
        if (os.path.exists(sample_protein)) and (os.path.exists(sample_ligand)):
            struct_dirs.append(sample_protein)
            complete_dataset[x.name] = {
                "protein": sample_protein,
                "ligand": sample_ligand,
            }

        else:
            no_structured_dirs.append(sample_protein)
            print(f"File not found {x.name}")
    # print(len(struct_dirs))
    return struct_dirs, complete_dataset


def index_to_df(file) -> pd.DataFrame:
    with open(file, "r") as f:
        pdbbind_index: list[str] = f.readlines()
    refined_dict: dict = {}

    for line in pdbbind_index:
        line: str = line.strip()
        if not line.startswith("#"):
            # pdbbind_index = json.loads(line)
            pdb_id = line.split()[0]
            resolution = line.split()[1]
            year = line.split()[2]
            logkd_ki = line.split()[3]
            exp = line.split()[4]
            exp_name, exp_value = exp.split("=")
            exp_value = float(exp_value[:-2])
            ligand_name = line.split()[-1].replace("(", "").replace(")", "").strip()
            refined_dict[pdb_id] = {
                "resolution": resolution,
                "year": year,
                "exp": exp,
                "ligand_name": ligand_name,
                "exp_name": exp_name,
                "exp_value": exp_value,
                "logkd_ki": logkd_ki,
            }
    return pd.DataFrame.from_dict(refined_dict, orient="index")



def get_one_code(mer, non_standard=False) -> str:
    ONE_LETTER: dict[str, str] = {
        "ALA": "A",
        "VAL": "V",
        "PHE": "F",
        "PRO": "P",
        "MET": "M",
        "ILE": "I",
        "LEU": "L",
        "ASP": "D",
        "GLU": "E",
        "LYS": "K",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "TYR": "Y",
        "HIS": "H",
        "CYS": "C",
        "ASN": "N",
        "GLN": "Q",
        "TRP": "W",
        "GLY": "G",
        "MSE": "M",
    }
    try:
        return ONE_LETTER[mer]
    except KeyError as e:
        if not non_standard:
            raise KeyError("Non standard residue found") from e
        print(f"Non standard residue {mer} found")
        NON_STANDARD_RES = {
            "CSD": "CYS",
            "HYP": "PRO",
            "BMT": "THR",
            "5HP": "GLU",
            "ABA": "ALA",
            "AIB": "ALA",
            "CSW": "CYS",
            "OCS": "CYS",
            "DAL": "ALA",
            "DAR": "ARG",
            "DSG": "ASN",
            "DSP": "ASP",
            "DCY": "CYS",
            "CRO": "CRO",
            "DGL": "GLU",
            "DGN": "GLN",
            "DHI": "HIS",
            "DIL": "ILE",
            "DIV": "VAL",
            "DLE": "LEU",
            "DLY": "LYS",
            "DPN": "PHE",
            "DPR": "PRO",
            "DSN": "SER",
            "DTH": "THR",
            "DTR": "DTR",
            "DTY": "TYR",
            "DVA": "VAL",
            "CGU": "GLU",
            "KCX": "LYS",
            "LLP": "LYS",
            "CXM": "MET",
            "FME": "MET",
            "MLE": "LEU",
            "MVA": "VAL",
            "NLE": "LEU",
            "PTR": "TYR",
            "ORN": "ALA",
            "SEP": "SER",
            "TPO": "THR",
            "PCA": "GLU",
            "SAR": "GLY",
            "CEA": "CYS",
            "CSO": "CYS",
            "CSS": "CYS",
            "CSX": "CYS",
            "CME": "CYS",
            "TYS": "TYR",
            "TPQ": "PHE",
            "STY": "TYR",
        }
        ##ref https://www.globalphasing.com/buster/manual/maketnt/manual/lib_val/library_validation.html

        mer = NON_STANDARD_RES[mer]
        return ONE_LETTER[mer]


def parse_seq(inpdb_filename):
    if not isinstance(inpdb_filename, list):
        inpdb_filename: list[str] = [inpdb_filename]
    ca_pattern: Pattern[str] = re.compile("^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])")  # type: ignore
    # ca_pattern=re.compile("^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])|^HETATM\s{0,4}\d{1,5}\s{2}CA\s[\sA](MSE)\s([\s\w])")

    THREE_TO_ONE_CODE: dict[str, str] = {
        "ALA": "A",
        "VAL": "V",
        "PHE": "F",
        "PRO": "P",
        "MET": "M",
        "ILE": "I",
        "LEU": "L",
        "ASP": "D",
        "GLU": "E",
        "LYS": "K",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "TYR": "Y",
        "HIS": "H",
        "CYS": "C",
        "ASN": "N",
        "GLN": "Q",
        "TRP": "W",
        "GLY": "G",
        "MSE": "M",
    }
    id_list: dict = {}
    for pdb_file in inpdb_filename:
        filename = os.path.basename(pdb_file).split("_")[0]
        chain_dict = {}
        chain_list = []

        with open(pdb_file, "rU") as fp:
            for line in fp.read().splitlines():
                if line.startswith("ENDMDL"):
                    break
                if match_list := ca_pattern.findall(line):
                    resn = match_list[0][0]  # +match_list[0][2]
                    chain = match_list[0][1]  # +match_list[0][3]
                    if chain in chain_dict:
                        # chain_dict[chain]+=THREE_TO_ONE_CODE[resn]
                        chain_dict[chain] += get_one_code(resn)

                    else:
                        # chain_dict[chain]=THREE_TO_ONE_CODE[resn]
                        chain_dict[chain] = get_one_code(resn)
                        chain_list.append(chain)
        id_list[filename] = chain_dict
    return id_list


def get_seq(pdb_filename) -> str:
    seq_dict: dict = parse_seq(pdb_filename)
    return "".join("".join(seq_dict[id].values()) for id in seq_dict)


# one hot encoding protein residues
def compute_protein_onehot(seq):
    AA: Literal["ACDEFGHIKLMNPQRSTVWY"] = "ACDEFGHIKLMNPQRSTVWY"
    enocde = np.empty((len(seq), len(AA)))
    for i in range(len(seq)):
        enocde[i] = np.array([1 if seq[i] == aa else 0 for aa in AA])
    return enocde


def get_protein_onehot(seq, pad_size=2000, zeroes=False):
    seq = seq.upper()
    if len(seq) > pad_size:
        seq = seq[:pad_size]
    onehot = compute_protein_onehot(seq)
    if len(onehot) < pad_size:
        if zeroes:
            pad = np.zeros((pad_size - len(onehot), len(onehot[0])))
            onehot = np.concatenate((onehot, pad))
        else:
            onehot = get_pad_array(onehot, pad_size=(pad_size, 20))
    return onehot


def plot_data(lengths, error=None, posy=850):
    if not error:
        error = []
    # plot the histogram of lengths
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=100)
    ax.axvline(max(lengths), color="red")
    ax.text(max(lengths), 0, f"max length: {max(lengths)}", color="red")
    ax.axvline(min(lengths), color="green")
    ax.text(min(lengths), posy, f"min length: {min(lengths)}", color="green")
    plt.title(
        f"Distribution of protein sequence lengths from core PDBbind: {len(lengths)-len(error)}"
    )
    plt.show()


def check_min_max_seq(file_names):
    error = []
    lengths = []

    for i in file_names:
        x = Path(file_names[i]["protein"])
        sample = x
        try:
            if x.name == "1e66":
                _ = 543
                lengths.append(_)
            elif x.name == "1gpn":
                _ = 537
                lengths.append(_)
            else:
                _ = get_seq(sample)

                lengths.append(len(_))
                if len(_) < 5:
                    print(sample)
            # print(len(_))
        except Exception as e:
            print(sample, e)
            error.append(sample)
    return error, lengths


# def get_pad_array(cmap, pad_size=(600, 600), constant=-1):
#     assert len(pad_size) == 2, "Padding size bust be 2D"
#     try:
#         cmap = np.pad(
#             cmap,
#             ((0, pad_size[0] - cmap.shape[0]), (0, pad_size[1] - cmap.shape[1])),
#             "constant",
#             constant_values=constant,
#         )
#         return cmap
#     except Exception as e:
#         print(e)
#         return None


def compute_contact_map(pdb_id, cutoff=8, pad_size=(2000, 2000)):
    with open(pdb_id, "r") as f:
        content = f.readlines()

    seq_dict = get_seq([pdb_id])
    # all_seq = []
    all_seq = "".join(list(seq_dict))
    coords = []
    ## contact map for CA
    #     for line in content:
    #         if line.startswith("ATOM") and line.split()[2].strip() == "CA":
    #             coords.extend([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    # using regex
    for line in content:
        if re.search(r"^ATOM\s+\d+\s+CA\s+", line):
            coords.extend(
                [[float(line[30:38]), float(line[38:46]), float(line[46:54])]]
            )
        # else:
        #     if "CA" in line: print(line)

    coords = np.array(coords)
    # get contact map
    length_seq = len(all_seq)
    assert (
        length_seq == coords.shape[0]
    ), f"Size miss matched in seq length-{length_seq} and c-alpha{coords.shape} for {pdb_id}"
    contact_map = np.zeros((length_seq, length_seq))
    for i, j in itertools.product(range(length_seq), range(length_seq)):
        if i != j:
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                contact_map[i, j] = 1
    #     return contact_map
    ## padding cmap
    pcamp = get_pad_array(contact_map, pad_size=pad_size)
    assert pcamp.shape == pad_size, "Padding Failed. Pad size not met requirement"
    return pcamp  # , length_seq


def compute_distance_apart(protein, ligand):
    protein = protein.reshape(-1, 3)
    ligand = ligand.reshape(-1, 3)
    distance = []
    for i in range(protein.shape[0]):
        distance.extend(
            np.linalg.norm(protein[i] - ligand[j]) for j in range(ligand.shape[0])
        )
    return np.array(distance)


def contact_map(pdb_id, cutoff=8, pad_size=(2000, 2000)):
    if not isinstance(pdb_id, list):
        pdb_id = [pdb_id]
    pool = Pool(cpu_count() - 1)
    jobs = [
        pool.apply_async(
            compute_contact_map,
            (
                i,
                cutoff,
                pad_size,
            ),
        )
        for i in pdb_id
    ]
    pool.close()
    pool.join()
    list_cmaps = [job.get() for job in jobs]
    return np.stack(list_cmaps, axis=0)


def get_contact_map(pdb_id, cutoff=8, pad_size=(2200, 2200)):
    return compute_contact_map(
        pdb_id,
        cutoff,
        pad_size,
    )



def get_pad_array(cmap, pad_size=(600, 600), constant=-1):
    assert len(pad_size) == 2, "Padding size bust be 2D"
    try:
        if cmap.shape[0] > pad_size[0]:
            cmap = cmap[: pad_size[0], :]

        if cmap.shape[1] > pad_size[1]:
            cmap = cmap[:, : pad_size[1]]
        cmap = np.pad(
            cmap,
            ((0, pad_size[0] - cmap.shape[0]), (0, pad_size[1] - cmap.shape[1])),
            "constant",
            constant_values=constant,
        )
        return cmap
    except Exception as e:
        print(e)
        return None
    
    


def parse_yaml(filename):
    with open(filename, "r") as stream:
        try:
            console.print("Cofiguration file loaded")
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def config_to_args(config, args):
    # args = argparse.Namespace()
    for k, v in config.items():
        setattr(args, k, v)
    return args
