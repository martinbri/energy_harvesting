from pymoo.core.problem import Problem,ElementwiseProblem
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldvm_back_up import ldvm
import time
from scipy.integrate import trapezoid
from pymoo.core.callback import Callback
import sys
class Eta_Thrust_parreto(ElementwiseProblem):
    def __init__(self,config ):
        
        
        up_bound=config['up_bound']
        low_bound=config['low_bound']
        super().__init__(
            n_var=4,         # Number of decision variables
            n_obj=2,         # Number of objectives
            n_constr=0,      # Number of constraints (set >0 if you have constraints)
            xl=np.array(low_bound),  # Lower bounds
            xu=np.array(up_bound)   # Upper bounds
        )
        self.config=config
        
        
        

    def _evaluate(self, x, out, *args, **kwargs):
        ldvm_instance = ldvm(self.config)
        time_deb= time.time()
        # print(f"[PID {os.getpid()}] évalue {x}")
    


        k, alpha0, h0, phi = x
        
        print("Evaluation en cours pour x =", x)
        o#mega=2*ldvm_instance.u_ref*k/ldvm_instance.chord
        #period=2*np.pi/omega
        #D=period*ldvm_instance.u_ref
    
    
        #Xhi=alpha0/np.arctan(h0*omega/ldvm_instance.u_ref)
    
        if np.abs(Xhi) >=1:
            print("Xhi >= 1, entering extraction mode...skipping evaluation")
            out["F"] = np.array([1e6,1e6])
         
        else: 
            #d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

            #ppp=int(D/d_wake)
            #print('ppp:', ppp)
            ldvm_instance.initialize_computation()
            ldvm_instance.make_parameterized_motions(k=k, alpha0=alpha0, h0=h0, phi=phi, save=False)
    
            cl_history = [0]
            cd_history = [0]
            cm_history = [0]
            for i in range(ldvm_instance.n_period*ldvm_instance.ppp - 1):
                cl, cd, cm, lesp, re_le, cn = ldvm_instance.step()
                cl_history.append(cl)
                cd_history.append(cd)
                cm_history.append(cm)
            cd_history = np.array(cd_history)
    
            cl_history = np.array(cl_history)
            cm_history = np.array(cm_history)
            
            eta,ct,cp = ldvm_instance.compute_thrust_efficiency(cl_history, cd_history, cm_history,k,ldvm_instance.ppp)
    

    
            print("Thrust contribution", ct)
            print("efficiency",eta)
            print('Xhi:', Xhi)
    


            out["F"] = np.array([-eta, -ct])