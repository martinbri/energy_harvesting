import numpy as np
import matplotlib.pyplot as plt
import cma
from ldvm_back_up import ldvm
import time
from scipy.integrate import trapezoid

import multiprocessing
def evaluate(x):
    """
    Evaluate the performance of the LDVM instance with given parameters.
    The parameters are unpacked from the input vector x.
    """
    ldvm_instance = ldvm(config)
    

    
    k, alpha0, h0, phi = x
    omega=2*ldvm_instance.u_ref*k/ldvm_instance.chord
    period=2*np.pi/omega
    D=period*ldvm_instance.u_ref
    
    d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

    ppp=int(D/d_wake)
    print('ppp:', ppp)
    ldvm_instance.initialize_computation()
    ldvm_instance.make_parameterized_motions(k=k, alpha0=alpha0, h0=h0, phi=phi, ppp=ppp)
    
    cl_history = []
    cd_history = []
    cm_history = []
    for i in range(ppp - 1):
        cl, cd, cm, lesp, re_le, cn = ldvm_instance.step()
        cl_history.append(cl)
        cd_history.append(cd)
        cm_history.append(cm)
    cl_history = np.array(cl_history)
        
    thrust=-trapezoid(cl_history,dx=period/ppp)
    # thrust = 0
    # for i in range(len(cl_history) - 1):
    #     thrust += 0.5 * (cl_history[i] + cl_history[i + 1]) * (period / ppp)
    
    # print('Thrust:', thrust)
    

    
    return thrust

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
    x0= np.zeros(4)  # Initialize x with 4 parameters
    x0[0] = 0.2  # k
    x0[1] = 50*np.pi/180 # alpha0
    x0[2] = 0.2  # h0
    x0[3] = 0.0  # phi

    lower_bounds = [0.05, -90 * np.pi / 180, -3 , 0]
    upper_bounds = [5,  90 * np.pi / 180,  3 , 2 * np.pi]
    sigma0=0.5
    popsize=48
    opts = {
    'bounds': [lower_bounds, upper_bounds],
    'popsize': popsize,
    'maxiter': 500,
    'verb_disp': 1,
    'seed': 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    

    while not es.stop():
        solutions = es.ask()
        fitnesses = [evaluate(x) for x in solutions]  # Sequential evaluation
        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()

    print("✅ Solution trouvée :", es.result.xbest)
    print("🎯 Valeur minimale :", evaluate(es.result.xbest))
    
    

    
    
    
    

    