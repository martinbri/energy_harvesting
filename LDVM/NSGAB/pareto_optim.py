import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from problem_multi_obj import Eta_Thrust_parreto
from callbacks import MyCallback
import os
from datetime import datetime

config = {
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.33,
        'cm_pvt': 0.33,
        'foil_name': '../naca0015_airfoil.dat',
        're_ref': 1100,
        'lesp_crit':0.19,
        'motion_file_name': 'motion_pr_amp45_k0.2.dat',
        'force_file_name': 'force_pr_amp45_k0.2_le.csv',
        'flow_file_name': 'flow.csv',
        'n_pts_flow': 100,
        'rho':1.225,
        'nu': 1.566e-5,
        'n_div': 70,
    }
problem = Eta_Thrust_parreto(config)


algorithm = NSGA2(pop_size=24,
                  n_offsprings=24,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20),
                  eliminate_duplicates=True)

                  
termination = get_termination("n_gen", 100)
# Create a directory with the current date in yyyy_mm_dd format
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path_save = os.path.join(os.getcwd(), f"paretto_results/Results_NSGA2_{current_time}")
print(f"Results will be saved in: {path_save}")

os.makedirs(path_save, exist_ok=True)
callbacks = MyCallback(path_save=path_save)

res = minimize(problem,
               algorithm,
               termination,
               seed=5,
               save_history=True,
               callback=callbacks,
               verbose=True)


