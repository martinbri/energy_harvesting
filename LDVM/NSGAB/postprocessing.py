import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ldvm_back_up import ldvm
from matplotlib.animation import FuncAnimation

import yaml



class PostProcessing:
    def __init__(self, path_data,config_data):
        self.path_data = path_data
        try:
            self.data=np.load(self.path_data, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading data from {self.path_data}: {e}")
            
        print(config_data)
       
        with open(config_data, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(f"Error loading config data from {config_data}: {e}")
        self.ldvm_instance = ldvm(self.config)

    
    def make_animmation(self,k,alpha0,h0,phi,data_save):
        self.ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=phi,save=False)


        #ldvm_instance.load_motion()
        self.ldvm_instance.initialize_computation()
        ani=self.ldvm_instance.make_ldvm_animation(n_frame=self.ldvm_instance.n_period*self.ldvm_instance.ppp-1, add_reference=False,colorscale=True,fixed_airfoil=True)
    
        # Move the generated animation to the data path
        destination_path = os.path.join(data_save, f'ldvm_animation_k_{k}_alpha0_{alpha0*180/np.pi}_h0_{h0}_Phi_{phi}.gif')
        
        ani.save(destination_path, writer='pillow', fps=20)
        # animation_path = '../ldvm_animation.gif'
        # if os.path.exists(animation_path):
            
        #     os.rename(animation_path, destination_path)
        #     print(f"Animation moved to {destination_path}")
        # else:
        #     print("Animation file not found.")
    def cluster_designs(self):
        
        pareto= np.array(self.data["All"][-1]).reshape(-1, 2)
                
        for p_ in pareto :
            pass
        
    def plot_colored_pareto(self, data_save,x_min=-1, y_min=-2, x_max=1, y_max=2.7):
        pop_size=self.config["pop_size"]
        n_gens=self.config["n_generations"]
        print(len(self.data["All"]),type(self.data["All"]))
        all_f = -np.array(self.data["All"]).reshape(-1,pop_size, 2)
        print(all_f.shape)
        #all_f = -all_f[~np.any(all_f == 1e6, axis=2)]
        print(all_f.shape)
        
        fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        n_gens, pop_size, _ = all_f.shape
        c = np.tile(np.arange(1, n_gens + 1), pop_size)
        scatter = ax.scatter(all_f[:,:, 0], all_f[:,:, 1], c=c, cmap='Blues', marker='o', label='Pareto Front',s=10)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Generation number')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$C_T$')
        plt.savefig(f"{data_save}/colored_pareto_front.png")
        
        
        
            
    
    
    
    
    
    def plot_lift(self, data_save):
        self.ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=phi,save=False)


        #ldvm_instance.load_motion()
        self.ldvm_instance.initialize_computation()
    
    
    
    
    
    def plot_i_generation_pareto(self,i_gen,data_save):
        if i_gen < 0 or i_gen >= len(self.data['All']):
            print(f"Generation {i_gen} is out of bounds. Valid range: 0 to {len(self.data['All'])-1}")
            return
        
        all_f = np.array(self.data["All"][i_gen]).reshape(-1, 2)
        all_f = -all_f[~np.any(all_f == 1e6, axis=1)]
        
        
        
        fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        ax.scatter(all_f[:, 0], all_f[:, 1], c='blue', marker='o', label='Pareto Front')
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
        ax[2, 0].set_xlabel(r'$k$')
        ax[2, 0].set_ylabel(r'$\eta$')
        ax_ = ax[2, 0].twinx()
        ax_.scatter(designs[:,0], all_f[:, 1], c='blue', marker='o', label=r'$C_T$')
        ax_.set_ylabel(r'$C_T$')
        ax[2, 0].legend()
        fig.savefig(f"{data_save}/designs_generation_{i_gen}.png")
        plt.show()
folder='/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_28_10_29_33'
folder='/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_28_11_39_22'
folder='/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_28_16_52_30'
folder='/scratch/disc/b.martin/Documents/energy_harvesting/LDVM/NSGAB/paretto_results/Results_NSGA2_2025_07_28_11_40_28'
data_save=os.path.join(folder,'callback_data.npy')
config= os.path.join(folder,'config.yaml')
pp= PostProcessing(data_save,config)
#pp.make_animmation(0.99993335,0.70839055,0.99949244,1.87209392,folder)
#pp.make_animmation(0.3231061 , 0.46522308, 0.99998693, 1.36552392,folder)
#pp.make_animmation(0.80997988, 0.82344625, 0.99998003, 1.45602329,folder)
# pp.make_animmation(0.3055602 , 0.74079869, 1.99735096, 1.42931189,folder)


# pp.plot_colored_pareto(folder,x_min=-1, y_min=-5, x_max=1, y_max=12)
# pp.plot_i_generation_pareto(300,folder)
# pp.plot_i_generation_design(300,folder)
data_plot=pp.data['All_designs'][399][0]

for i in [24]:
    pp.make_animmation(data_plot[i][0],data_plot[i][1],data_plot[i][2],data_plot[i][3],folder)
    print(data_plot[i][0],data_plot[i][1],data_plot[i][2],data_plot[i][3])
print(data_plot,len(data_plot[0]))