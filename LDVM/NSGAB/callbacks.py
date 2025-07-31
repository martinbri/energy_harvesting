import numpy as np
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

import os


class MyCallback(Callback):

    def __init__(self,path_save) -> None:
        super().__init__()
        self.data["All"] = []
        self.data['All_designs'] = []
        self.path_save = path_save 
        # Ensure the directory exists
        
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
    def notify(self, algorithm):
        self.data["All"].append([algorithm.pop.get("F")])
        self.data['All_designs'].append([algorithm.pop.get("X")])
        print(f"Current generation: {algorithm.n_gen}, Population size: {len(algorithm.pop)}")
        # Optionally, you can plot the current front
        self.plot_pareto(algorithm)
        self.plot_designs(algorithm)
        
        
        # Save self.data to a file
        np.save(f"{self.path_save}/callback_data.npy", self.data, allow_pickle=True)
    
    
    def plot_pareto(self,algorithm):
        print(f"Plotting Pareto front for generation {algorithm.n_gen}")
        print(f"Number of designs in this generation: {len(algorithm.pop)}")
        print(f"Len of All: {len(self.data['All'])}")
        
        all_f = np.array(self.data["All"]).reshape(-1, 2)
        all_f = all_f[~np.any(all_f == 1e6, axis=1)]
        
        
        fig,ax =plt.subplots(figsize=(4, 3),tight_layout=True)
        ax.scatter(-all_f[:, 0],-all_f[:, 1], c='blue', marker='o', label='Pareto Front')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$C_T$')
        fig.savefig(f"{self.path_save}/pareto_front_genration_{algorithm.n_gen}.png")
        plt.close()
    def plot_designs(self, algorithm):
        all_designs = np.array(self.data['All_designs']).reshape(-1, 4)
       
        all_f = np.array(self.data["All"]).reshape(-1, 2)
        mask = np.any(all_f == 1e6, axis=1)
        all_f = all_f[~mask]
        all_designs = all_designs[~mask]
        
        fig, ax = plt.subplots(2,2,figsize=(8, 6), tight_layout=True)
        ax[0,0].scatter(all_designs[:, 0], -all_f[:, 0], c='red', marker='x', label='Designs')
        ax[0,0].set_xlabel(r'$$k$')
        ax[0,0].set_ylabel(r'$\eta$')
        ax_=ax[0,0].twinx()
        ax_.scatter(all_designs[:, 0], -all_f[:, 1], c='blue', marker='o', label='Thrust Coefficient')
        ax_.set_ylabel(r'$C_T$')        
        
        ax[0,1].scatter(all_designs[:, 0], -all_f[:, 1], c='red', marker='x', label='Designs')
        ax[0,1].set_xlabel(r'$\alpha_0$')
        ax[0,1].set_ylabel(r'$\eta$')
        ax_=ax[0,1].twinx()
        ax_.scatter(all_designs[:, 1], -all_f[:, 1], c='blue', marker='o', label='Thrust Coefficient')
        ax_.set_ylabel(r'$C_T$')
        
        
        ax[1,0].scatter(all_designs[:, 2], -all_f[:, 0], c='red', marker='x', label='Designs')
        ax[1,0].set_xlabel(r'$h_0$')
        ax[1,0].set_ylabel(r'$\eta$')
        ax_=ax[1,0].twinx()
        ax_.scatter(all_designs[:, 2], -all_f[:, 1], c='blue', marker='o', label='Thrust Coefficient')
        ax_.set_ylabel(r'$C_T$')
        
        
        ax[1,1].scatter(all_designs[:, 3], -all_f[:, 0], c='red', marker='x', label='Designs')
        ax[1,1].set_xlabel(r'$\Psi$')
        ax[1,1].set_ylabel(r'$\eta$')
        ax_=ax[1,1].twinx()
        ax_.scatter(all_designs[:, 3], -all_f[:, 1], c='blue', marker='o', label='Thrust Coefficient')
        ax_.set_ylabel(r'$C_T$')
        fig.savefig(f"{self.path_save}/designs_genration_{algorithm.n_gen}.png")
        plt.close()


