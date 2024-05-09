#!/bin/bash
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 72ba5c5a77837d309ef3436192fb6583049cbdcc
#SBATCH --gpus-per-node=1        # Number of GPU(s) per node
#SBATCH --ntasks-per-node=16     # 
#SBATCH --exclusive
#SBATCH --mem=40G                    # memory per node
#SBATCH --time=0-01:00:00
#SBATCH --output=batch1_cudalstm_usa_%j.out
<<<<<<< HEAD
=======

#SBATCH --job-name=lstm_usa_b3   # Specify your desired job name here
#SBATCH --gpus-per-node=1        # Number of GPU(s) per node
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.                  # memory per node
#SBATCH --time=7-00:00:00
#SBATCH --output=cudalstm_usa_b3_%j.out
>>>>>>> 4f8b191... Added postprocessing - find best models and plot basins with neg NSE
=======
>>>>>>> 72ba5c5a77837d309ef3436192fb6583049cbdcc
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL

module load cuda cudnn 
module load python/3.11 scipy-stack

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 72ba5c5a77837d309ef3436192fb6583049cbdcc
virtualenv --no-download $SLURM_TMPDIR/venv-nh
source $SLURM_TMPDIR/venv-nh/bin/activate

nvidia-smi
<<<<<<< HEAD
python cudalstm_tune_params.py --options_batch batch1
=======
source /home/jcurbelo/projects/def-spiteri/jcurbelo/neuralhydrology_fork/venv-nh/bin/activate

nvidia-smi
python cudalstm_tune_params.py --options_batch batch3
deactivate
>>>>>>> 4f8b191... Added postprocessing - find best models and plot basins with neg NSE
=======
python cudalstm_tune_params.py --options_batch batch1
>>>>>>> 72ba5c5a77837d309ef3436192fb6583049cbdcc
