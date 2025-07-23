import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ldvm_back_up import ldvm

def validate_with_theodorsen(k,alpha0,h0,Phi,save_fig=False, n_period=10,path_save='.'):
    config = {
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.5,
        'cm_pvt': 0.5,
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
    ldvm_instance = ldvm(config)
    omega=2*ldvm_instance.u_ref*k/ldvm_instance.chord
    period=2*np.pi/omega
    D=period*ldvm_instance.u_ref

    d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

    ppp=int(D/d_wake)
    #ldvm_instance.load_motion()
   
    print('ppp =', ppp)

    ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=Phi,ppp=ppp,save=False,n_period=n_period)
    ldvm_instance.initialize_computation()
    
    
    cl_history = [0]
    cd_history = [0]
    cm_history = [0]
    cn_history = [0]
    time_theo,alpha_theo,h_theo,cl_theodorsen,cm_theodorsen=ldvm_instance.theodorsen_loads(k,alpha0,h0,Phi)
    for i in range(n_period*ppp-1):
        cl, cd, cm, lesp, re_le,cn=ldvm_instance.step()
        cl_history.append(cl)
        cd_history.append(cd)
        cm_history.append(cm)  # Assuming cm is not calculated in this example
        cn_history.append(cn)  # Assuming cn is not calculated in this example
    
    cl_history = np.array(cl_history)
    cd_history = np.array(cd_history)
    cm_history = np.array(cm_history)
    cn_history = np.array(cn_history)
    print(pd.Timestamp.now())
    meta_data = {
        'k': str(k),
        'alpha0': str(alpha0*180/np.pi),  # Convert to degrees for metadata
        'h0/c': str(h0/ldvm_instance.chord),  # Normalize by chord length
        'phi': str(Phi*180/np.pi),  # Convert to degrees for metadata
        'n_period': str(n_period),
        'ppp': str(ppp),
        'lesp_crit': str(ldvm_instance.lesp_crit),
        're_ref': str(ldvm_instance.re_ref),
        'rho': str(ldvm_instance.rho),
        'foil_name': str(ldvm_instance.foil_name),
        'n_div': str(ldvm_instance.n_div),
        'pvt': str(ldvm_instance.pvt),
        'cm_pvt': str(ldvm_instance.cm_pvt)
    }
    
    
    
    ## Plotting the results
    fig,ax= plt.subplots(figsize=(4, 3),tight_layout=True)
    ax.plot(ldvm_instance.time[-ppp:],cl_history[-ppp:],color=(1,0,0), label='LDVM CL')
    ax.plot(time_theo[-ppp:],cl_theodorsen[-ppp:],color=(0,0,1), label='Theodorsen CL')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$C_L$')
    fig.savefig(f'{path_save}/cl_vs_time_k_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.png', metadata=meta_data)
    fig.savefig(f'{path_save}/cl_vs_time_k_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.pdf', metadata=meta_data)

    fig,ax= plt.subplots(figsize=(4, 3),tight_layout=True)
    ax.plot(ldvm_instance.alpha[-ppp:]*180/np.pi,cl_history[-ppp:],color=(1,0,0),  label='LDVM CL')
    ax.plot(alpha_theo[-ppp:]*180/np.pi,cl_theodorsen[-ppp:],color=(0,0,1) , label='Theodorsen CL')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$C_L$')
    fig.savefig(f'{path_save}/cl_vs_alpha_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.png', metadata=meta_data)
    fig.savefig(f'{path_save}/cl_vs_alpha__{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.pdf', metadata=meta_data)
    
    fig,ax= plt.subplots(figsize=(4, 3),tight_layout=True)
    ax.plot(ldvm_instance.time[-ppp:],cm_history[-ppp:],color=(1,0,0),  label='LDVM Cm')
    ax.plot(time_theo[-ppp:],cm_theodorsen[-ppp:],color=(0,0,1),  label='Theodorsen Cm')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$C_m$')
    
    
    fig.savefig(f'{path_save}/cm_vs_time_k_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.png', metadata=meta_data)
    fig.savefig(f'{path_save}/cm_vs_time_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.pdf', metadata=meta_data)
    fig,ax= plt.subplots(figsize=(4, 3),tight_layout=True)
    ax.plot(ldvm_instance.alpha[-ppp:]*180/np.pi,cm_history[-ppp:],color=(0,0,1),  label='LDVM Cm')
    ax.plot(alpha_theo[-ppp:]*180/np.pi,cm_theodorsen[-ppp:],color=(0,0,1),  label='Theodorsen Cm')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$C_m$')
    fig.savefig(f'{path_save}/cm_vs_alpha_k_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.png', metadata=meta_data)
    fig.savefig(f'{path_save}/cm_vs_alpha_k_{k}_alpha0_{alpha0}_h0_{h0}_Phi_{Phi}_pvt_{ldvm_instance.pvt}.pdf', metadata=meta_data)
    
def validate_with_andersen_results(k,alpha0,h0,Phi,save_fig=False, n_period=10,path_save='.'):
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
    ldvm_instance = ldvm(config)

    #ldvm_instance.load_motion()
   

    ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=Phi,save=False,n_period=n_period)
    print('ppp =', ldvm_instance.ppp)
    ldvm_instance.initialize_computation()
    
    
    cl_history = [0]
    cd_history = [0]
    cm_history = [0]
    cn_history = [0]
    for i in range(n_period*ldvm_instance.ppp-1):
        cl, cd, cm, lesp, re_le,cn=ldvm_instance.step()
        cl_history.append(cl)
        cd_history.append(cd)
        cm_history.append(cm)  # Assuming cm is not calculated in this example
        cn_history.append(cn)  # Assuming cn is not calculated in this example
    print('NLEV:',ldvm_instance.n_lev)
    cl_history = np.array(cl_history)
    cd_history = np.array(cd_history)
    cm_history = np.array(cm_history)
    cn_history = np.array(cn_history)
    eta,ct,cp=ldvm_instance.compute_thrust_efficiency(cl_history,cd_history,cm_history,k,ldvm_instance.ppp)
    print(r"$\eta$:", eta)
    print(r"$C_T$:", ct)
    print(r"$C_P$:", cp)
    return eta,ct,cp




if __name__ == "__main__":
    k = 2.1
    alpha0 = np.deg2rad(15)
    h0 = 0.25
    Phi = np.deg2rad(90)
    #validate_with_theodorsen(k, alpha0, h0, Phi, save_fig=True, n_period=10, path_save='theodorsen_validation/pitch/')
    
    validate_with_andersen_results(k, alpha0, h0, Phi, save_fig=True, n_period=5, path_save='andersen_validation/pitch/')