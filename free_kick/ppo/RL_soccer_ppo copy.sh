#!/bin/bash
#SBATCH --job-name=rl_soccer_ppo             # Job name
#SBATCH --output=rl_soccer_ppo.out            # Output file
#SBATCH --error=rl_soccer_ppo.err             # Error file
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=2              # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory per node
#SBATCH --time=24:00:00                # Time limit hh:mm:ss
#SBATCH --partition=gpu                # Partition name
#SBATCH --gres=gpu:a100:1               # Request one GPU

# Load necessary modules
module load Anaconda3/2024.02-1
source activate RL
module load GCC/13.2.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

python -c "import torch; torch.cuda.empty_cache()"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to the directory containing your script
cd "/scratch/user/zahidhussain909/ammar/GPS_Spoofing_RL/DQN Drone"

# Run your script
python3 -u -m Image.train_img
