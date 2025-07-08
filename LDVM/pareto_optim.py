import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from problem_multi_obj import MyProblem

config = {
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.33,
        'cm_pvt': 0.33,
        'foil_name': 'naca0015_airfoil.dat',
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
problem = MyProblem(config)


algorithm = NSGA2(pop_size=24,
                  n_offsprings=50,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20),
                  eliminate_duplicates=True)

                  
termination = get_termination("n_gen", 100)

res = minimize(problem,
               algorithm,
               termination,
               seed=5,
               save_history=True,
               verbose=True)


