#!/bin/bash
#SBATCH --job-name=rl_soccer_a2c             # Job name
#SBATCH --output=rl_soccer_a2c.out            # Output file
#SBATCH --error=rl_soccer_a2c.err             # Error file
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=2              # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory per node
#SBATCH --time=12:00:00                # Time limit hh:mm:ss


# Load necessary modules
module load Anaconda3/2024.02-1
source activate RL
module load GCC/13.2.0

# Change to the directory containing your script
cd "/scratch/user/zahidhussain909/RL_Soccer/free_kick/ddpg"

# Run your script
python3 -u train_a2c.py
