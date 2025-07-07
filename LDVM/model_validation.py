import pandas as pd
import numpy as np
import os
from ldvm_back_up import ldvm

import matplotlib.pyplot as plt



def validate_model():
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
    
    data_paper= pd.read_csv('paper_result//form_data_h0_0.25_alpha0_15_deg_phase_90_degree.csv', sep=',')
    
    St_Ct=data_paper['C_T.2'].dropna().values[1:]
    eta_list = []
    ct_list = []
    cp_list = []
    for St in St_Ct:
        St=float(St)
        f=St*ldvm_instance.u_ref/2/(0.25*ldvm_instance.chord)
        omega=2*np.pi*f
        k=omega*ldvm_instance.chord/(2*ldvm_instance.u_ref)
        period=2*np.pi/omega    
        alpha0=15*np.pi/180
        h0= 0.25*ldvm_instance.chord
        phi= 90*np.pi/180

        D=period*ldvm_instance.u_ref
    
        d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

        ppp=int(D/d_wake)
        #ldvm_instance.load_motion()
    
        ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=phi,ppp=ppp)
    

        #ldvm_instance.load_motion()
        ldvm_instance.initialize_computation()
        #ldvm_instance.make_ldvm_animation(add_reference=True,colorscale=True)

        cl_history = []
        cd_history = []
        cm_history = []

        for i in range(ppp-1):
            #print(ldvm_instance.alpha[:i]*180/np.pi)
            cl, cd, cm, lesp, re_le,cn=ldvm_instance.step()
            cl_history.append(cl)
            cd_history.append(cd)
            cm_history.append(cm)  # Assuming cm is not calculated in this example
        
        eta,ct,cp= ldvm_instance.compute_thrust_efficiency(np.array(cl_history), np.array(cd_history), np.array(cm_history), k,ppp)
        eta_list.append(eta)
        ct_list.append(ct)
        cp_list.append(cp)
    fig,ax=plt.subplots(1,1,figsize=(4,3),tight_layout=True)
    ax.scatter(np.array(St_Ct,dtype=float),np.array(ct_list,dtype=float),label='LDVM',color=(0.5,0.,0.5),s=5)
    ax.scatter(np.array(data_paper['C_T.2'].dropna().values[1:],dtype= float),np.array(data_paper['C_T.3'].dropna().values[1:],dtype=float),label='Experimental Data',color=(1,0.0,0.0),s=5)
    ax.scatter(np.array(data_paper['C_T'].dropna().values[1:],dtype=float),np.array(data_paper['C_T.1'].dropna().values[1:],dtype=float),label='Model Data',color=(0.0,0.0,1.0),s=5)
    ax.scatter(np.array(data_paper['C_T.4'].dropna().values[1:],dtype=float),np.array(data_paper['C_T.5'].dropna().values[1:],dtype=float),label='analytical  Data',color=(0.0,0.5,0.0),s=5)        
    ax.set_xlabel(r'$St$')
    ax.set_ylabel(r'$C_t$')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.legend()
    fig.savefig('paper_result/ct_vs_st.png',dpi=300)
    fig.savefig('paper_result/ct_vs_st.pdf',dpi=300)
    
    
    fig,ax=plt.subplots(1,1,figsize=(4,3),tight_layout=True)
    ax.scatter(np.array(St_Ct,dtype=float),np.array(cp_list,dtype=float),label='LDVM',color=(0.5,0.,0.5),s=5)
    ax.scatter(np.array(data_paper['C_p.2'].dropna().values[1:],dtype= float),np.array(data_paper['C_p.3'].dropna().values[1:],dtype=float),label='Experimental Data',color=(1,0.0,0.0),s=5)
    ax.scatter(np.array(data_paper['C_p'].dropna().values[1:],dtype=float),np.array(data_paper['C_p.1'].dropna().values[1:],dtype=float),label='Model Data',color=(0.0,0.0,1.0),s=5)
    ax.scatter(np.array(data_paper['C_p.4'].dropna().values[1:],dtype=float),np.array(data_paper['C_p.5'].dropna().values[1:],dtype=float),label='analytical  Data',color=(0.0,0.5,0.0),s=5)        
    ax.set_xlabel(r'$St$')
    ax.set_ylabel(r'$C_p$')
    
    fig.savefig('paper_result/cp_vs_st.png',dpi=300)
    fig.savefig('paper_result/cp_vs_st.pdf',dpi=300)
    fig,ax=plt.subplots(1,1,figsize=(4,3),tight_layout=True)
    ax.scatter(np.array(St_Ct,dtype=float),np.array(eta_list,dtype=float),label='LDVM',color=(0.5,0.,0.5),s=5)
    ax.scatter(np.array(data_paper['\eta.2'].dropna().values[1:],dtype= float),np.array(data_paper['\eta.3'].dropna().values[1:],dtype=float),label='Experimental Data',color=(1,0.0,0.0),s=5)
    ax.scatter(np.array(data_paper['\eta'].dropna().values[1:],dtype=float),np.array(data_paper['\eta.1'].dropna().values[1:],dtype=float),label='Model Data',color=(0.0,0.0,1.0),s=5)
    ax.scatter(np.array(data_paper['\eta.4'].dropna().values[1:],dtype=float),np.array(data_paper['\eta.5'].dropna().values[1:],dtype=float),label='analytical  Data',color=(0.0,0.5,0.0),s=5)        
    ax.set_xlabel(r'$St$')
    ax.set_ylabel(r'$\eta$')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.legend()
    fig.savefig('paper_result/eta_vs_st.png',dpi=300)
    fig.savefig('paper_result/eta_vs_st.pdf',dpi=300)
validate_model()