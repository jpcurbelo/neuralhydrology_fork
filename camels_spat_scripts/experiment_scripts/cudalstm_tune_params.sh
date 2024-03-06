#!/bin/bash
#SBATCH --gpus-per-node=1        # Number of GPU(s) per node
#SBATCH --ntasks-per-node=16     # 
#SBATCH --exclusive
#SBATCH --mem=40G                    # memory per node
#SBATCH --time=0-01:00:00
#SBATCH --output=batch1_cudalstm_usa_%j.out
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL

module load cuda cudnn 
module load python/3.11 scipy-stack

virtualenv --no-download $SLURM_TMPDIR/venv-nh
source $SLURM_TMPDIR/venv-nh/bin/activate

nvidia-smi
python cudalstm_tune_params.py --options_batch batch1