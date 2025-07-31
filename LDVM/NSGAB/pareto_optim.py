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
import yaml



# Load the configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

problem = Eta_Thrust_parreto(config)


pop_size = config['pop_size']
n_generations = config['n_generations']


algorithm = NSGA2(pop_size=pop_size ,
                  n_offsprings=24,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20),
                  eliminate_duplicates=True)

                  
termination = get_termination("n_gen", n_generations)
# Create a directory with the current date in yyyy_mm_dd format
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path_save = os.path.join(os.getcwd(), f"paretto_results/Results_NSGA2_{current_time}")


# Save the config.yaml file in the path_save directory
os.makedirs(path_save, exist_ok=True)
config_save_path = os.path.join(path_save, "config.yaml")
with open(config_save_path, 'w') as file:
    yaml.dump(config, file)
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


