#!/usr/bin/bash -li

#SBATCH --job-name=train_nette_xrnx50
#SBATCH --mail-user=emanuele@ballarin.cc
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=lovelace
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

sleep 3

export HOME="/u/emaballarin/"
export CODEHOME="$HOME/Downloads/NETTE/semigood-baselines/src"
export MYPYTHON="$HOME/pixies/minilit/.pixi/envs/default/bin/python"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo " "
echo "hostname="$(hostname)
echo "WORLD_SIZE="$WORLD_SIZE
echo "OMP_NUM_THREADS="$OMP_NUM_THREADS
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo " "

cd "$CODEHOME" || exit

echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "START TIME "$(date +'%Y_%m_%d-%H_%M_%S')
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
srun "$MYPYTHON" -O "$CODEHOME/nette_xrnx_50.py" --save --wandb --tgnotif
echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "STOP TIME "$(date +'%Y_%m_%d-%H_%M_%S')
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
