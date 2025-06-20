#!/bin/bash                                                                                                                                                                                                        
                                                                                                                                                                                                                   
#SBATCH -J LDVM_data_baze

# SBATCH -N 1
# SBATCH --ntasks-per-node=1

#SBATCH -n 1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00

#SBATCH --partition=short

#SBATCH --begin=now

#SBATCH --output=/dev/null   # Redirige la sortie standard vers /dev/null
#SBATCH --error=/dev/null    # Redirige les erreurs standard vers /dev/null




module python
source activate flutter_env
python ldvm_back_up.py


