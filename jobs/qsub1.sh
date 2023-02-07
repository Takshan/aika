#$ -N MODEL0
#$ -V
#$ -S /bin/bash
#$ -q all.q@n152
#$ -pe 8cpu 8
#$ -o /home/lab09/BindingAffinity/dump/
#$ -e  /home/lab09/BindingAffinity/dump/
#$ -cwd
echo " Setting environment variables..."
export PATH="/share/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/share/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_LAUNCH_BLOCKING=1
export WANDB_BASE_URL=http://203.230.60.158:8080
export WANDB_API_KEY=local-37a32c308b849560eaf09169f6060205b2599dc7
PROJECTHOME=/home/lab09/BindingAffinity
CONDA=/home/lab09/.conda/envs/seq2vec/bin/python
DB=/scratch/lab09/reference
echo "Copying data to /scratch..."
scp -r $PROJECTHOME/reference $DB
echo "Training..."
$CONDA $PROJECTHOME/aiba.py --model 0 --batch_size 512 --config $PROJECTHOME/config.yaml --max_epoch 2000 --check_val_every_n_epoch 100 --wandb 
echo "Removing data from /scratch..."
rm -rf $DB
echo "Done Finally..."