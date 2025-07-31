#!/bin/bash                                                                                                                                                                                                        
                                                                                                                                                                                                                   
#SBATCH -J NSGA2

# SBATCH -N 1
# SBATCH --ntasks-per-node=1

#SBATCH -n 1
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00

#SBATCH --partition=long2

#SBATCH --begin=now

#SBATCH --output=/dev/null   # Redirige la sortie standard vers /dev/null
#SBATCH --error=/dev/null    # Redirige les erreurs standard vers /dev/null




module load python
source activate pymoo-env
python pareto_optim.py

