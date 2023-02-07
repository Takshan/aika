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

TEMPLATE = """
#$ -N PHAM6
#$ -V
#$ -S /bin/bash
#$ -q gp2
#$ -pe 8cpu 8
#$ -o /home/lab09/DEV/BindingAffinity/dump/
#$ -e  /home/lab09/DEV/BindingAffinity/dump/
#$ -cwd
echo "Creating Pharmacophore...."
/home/lab09/.conda/envs/seq2vec/bin/python /home/lab09/DEV/BindingAffinity/notebooks/create_pharmacophore.py /home/lab09/DEV/BindingAffinity/NEW_PHAMRMA_SMILES_Mol_18030_pdbbind_2020_all_gen_ref_core_clean_drops_ro5.json  /scratch/lab09/NEW_PHARMACOPHORE_RO5_17740_pdb_code_smiles.h5
echo "Done Finally..."
"""

def submit_jobs():
    with open("submit_jobs.sh", "w") as f:
        f.write(TEMPLATE)
    # os.system("qsub submit_jobs.sh")