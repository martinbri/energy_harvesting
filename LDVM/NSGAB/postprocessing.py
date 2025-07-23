import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D



class PostProcessing:
    def __init__(self, path_data):
        self.path_data = path_data
        try:
            self.data=np.load(self.path_data, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading data from {self.path_data}: {e}")

    
    
    
    
    def cluster_designs(self):
        
        pareto= np.array(self.data["All"][-1]).reshape(-1, 2)
                
        for p_ in pareto :
            pass
            
    
    
    
    
    
    
    
    
    def plot_i_generation_pareto(self,i_gen,data_save):
        if i_gen < 0 or i_gen >= len(self.data['All']):
            print(f"Generation {i_gen} is out of bounds. Valid range: 0 to {len(self.data['All'])-1}")
            return
        
        all_f = np.array(self.data["All"][i_gen]).reshape(-1, 2)
        all_f = -all_f[~np.any(all_f == 1e6, axis=1)]
        
        
        
        fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        ax.scatter(-all_f[:, 0], -all_f[:, 1], c='blue', marker='o', label='Pareto Front')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$C_T$')
        plt.title(f'Pareto front for generation {i_gen}')
        fig.savefig(f"{data_save}/pareto_front_generation_{i_gen}.png")
        plt.show()
    def plot_i_generation_design(self,i_gen,data_save):
        if i_gen < 0 or i_gen >= len(self.data['All']):
            print(f"Generation {i_gen} is out of bounds. Valid range: 0 to {len(self.data['All'])-1}")
            return
        
        all_f = np.array(self.data["All"][i_gen]).reshape(-1, 2)
        mask= np.any(all_f == 1e6, axis=1)
        all_f = -all_f[~mask]
        designs = np.array(self.data["All_designs"][i_gen]).reshape(-1, 4)
        designs = designs[~mask]
        
        omega = 2 * designs[:, 0] 
        
    
        f= omega/2/ np.pi # Frequency in Hz
        St=f *designs[:, 2]*2/1 # Assuming velocity is 1  
        
        fig, ax = plt.subplots(3, 2, figsize=(12, 9), tight_layout=True)
        ax[0, 0].scatter(St, all_f[:, 0], c='red', marker='x', label=r'$\eta$')
        ax[0, 0].set_xlabel(r'$S_t$')
        ax[0, 0].set_ylabel(r'$\eta$')
        ax_ = ax[0, 0].twinx()
        ax_.scatter(St, all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.set_ylabel(r'$C_T$')
        ax[0, 0].legend()
        #ax_.legend(loc='upper left')

        ax[0, 1].scatter(designs[:, 1]*180/np.pi, all_f[:, 0], c='red', marker='x', label=r'$\eta$')
        ax[0, 1].set_xlabel(r'$\alpha_0$')
        ax[0, 1].set_ylabel(r'$\eta$')
        ax_ = ax[0, 1].twinx()
        ax_.scatter(designs[:, 1]*180/np.pi, all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.legend()
        ax_.set_ylabel(r'$C_T$')
        ax[1, 0].scatter(designs[:, 2], all_f[:, 0], c='red', marker='x', label=r'$\eta$')
        ax[1, 0].set_xlabel(r'$h_0/c$')
        ax[1, 0].set_ylabel(r'$\eta$')
        ax_ = ax[1, 0].twinx()
        ax_.scatter(designs[:, 2], all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.set_ylabel(r'$C_T$')
        ax[1, 1].scatter(designs[:, 3], all_f[:, 0], c='red', marker='x', label=r'$\eta$')
        ax[1, 1].set_xlabel(r'$\Psi$')
        ax[1, 1].set_ylabel(r'$\eta$')
        ax_ = ax[1, 1].twinx()
        ax_.scatter(designs[:, 3], all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.set_ylabel(r'$C_T$')
        
        
        ax[2, 0].scatter(designs[:,0], all_f[:, 0], c='red', marker='x', label=r'$\eta$')
        ax[2, 0].set_xlabel(r'$S_t$')
        ax[2, 0].set_ylabel(r'$\eta$')
        ax_ = ax[2, 0].twinx()
        ax_.scatter(designs[:,0], all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.set_ylabel(r'$C_T$')
        ax[2, 0].legend()
        fig.savefig(f"{data_save}/designs_generation_{i_gen}.png")
        plt.show()
data_save='/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_10_10_05_41/callback_data.npy'
pp= PostProcessing(data_save)

pp.plot_i_generation_design(99,'/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_10_10_05_41')

print(pp.data['All_designs'][99])