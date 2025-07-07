import numpy as np
import matplotlib.pyplot as plt
from ldvm_back_up import ldvm
import os
import time


class PostProcessing:
    def __init__(self,data_path, config=None):
        self.config = config
        self.ldvm = ldvm(config)
        
        self.data = np.load(data_path)
        
        self.data=self.data#[:80,:,:]  # Remove the first 75 points to avoid transients
    def plot_efficiency(self,folder_save):
        """
        Plot the efficiency of the LDVM instance.
        """
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        
        
        ## Strouhal number is defined as St = f * 2h0 / U, where f is the frequency.
        
        efficiency = -self.data[:,:,-1]
    
        omega=2*self.ldvm.u_ref*self.data[:,:,0]/self.ldvm.chord
        f= omega/(2*np.pi)
        St=np.abs( f * 2 * self.data[:,:,2] / self.ldvm.u_ref)
        period=2*np.pi/omega
        D=period*self.ldvm.u_ref
        d_wake=1./self.ldvm.n_div*self.ldvm.chord
        ppp=D/d_wake


                
        
        # TEV_position= self.data[:,:,2]*np.sin(self.data[:,:,0]*t-self.data[:,:,3])- (1-self.ldvm.pvt)*self.ldvm.chord* np.sin(self.data[:,:,1]*np.sin(self.data[:,:,0]*t)) 
        
        # print(TEV_position.shape)
        #max_TEV_dist=
        
        # d
        
        fix,ax = plt.subplots(2, 2, figsize=(8, 6),tight_layout=True, sharey=True)
        ax[0, 0].scatter(self.data[:,:,0], efficiency,color=(0.5,0,0.5),s=5)
        #ax[0, 0].set_xlabel(r'$St = \frac{\omega*2h_0}{2\pi U} $')
        ax[0, 0].set_xlabel(r'$k$')
        ax[0, 0].set_ylabel(r'$\eta$')    
        ax[0, 0].set_xlim(0, 0.5)
        ax[0,0].set_ylim(0.5, 1.05)
        ax[0, 0].text(0.95, 0.95, '(a)', transform=ax[0, 0].transAxes, 
                  fontsize=12, verticalalignment='top', horizontalalignment='right')

        ax[0, 1].scatter(self.data[:,:,1]*180/np.pi, efficiency,color=(0.,0.5,0.),s=5)
        ax[0, 1].set_xlabel(r'$\alpha_0(\circ)$')
        ax[0,1].set_ylim(0.50, 1.05)
        ax[0, 1].set_xlim(-15, 15)
        ax[0, 1].set_ylabel(r'$\eta$')   
        
        ax[0, 1].text(0.95, 0.95, '(b)', transform=ax[0, 1].transAxes, 
                  fontsize=12, verticalalignment='top', horizontalalignment='right')
        ax[1, 0].scatter(self.data[:,:,2]/self.ldvm.u_ref, efficiency,color=(0.0,0.0,1.),s=5)
        ax[1, 0].set_xlabel(r'$h_0/c$')
        ax[1, 0].set_ylim(0.50, 1.05)
        ax[1, 0].set_ylabel(r'$\eta$')   
        ax[1, 0].text(0.95, 0.95, '(c)', transform=ax[1, 0].transAxes, 
                  fontsize=12, verticalalignment='top', horizontalalignment='right')
        ax[1, 1].scatter(self.data[:,:,3]*180/np.pi, efficiency,color=(1.0,0.,0.),s=5)
        ax[1, 1].set_xlabel(r'$\Psi$')
        ax[1, 1].set_ylim(0.50, 1.05)
        ax[1, 1].set_ylabel(r'$\eta$')   
        ax[1, 1].text(0.95, 0.95, '(d)', transform=ax[1, 1].transAxes, 
                  fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.savefig(os.path.join(folder_save, 'efficiency_plot.png'))
        plt.close(fix)        
    

if __name__ == "__main__":
    # Example usage
    config = {
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.25,
        'cm_pvt': 0.25,
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
    data_path ='results/cma_run_2025-07-03_09-25-56'
    post_processing = PostProcessing(data_path+'/results.npy',config)
    print("PostProcessing instance created.")
    folder_save = data_path+'/plots'
    
    post_processing.plot_efficiency(folder_save)
    print("Efficiency plot created and saved in:", folder_save)
    
    
