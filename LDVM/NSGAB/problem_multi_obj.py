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
    def __init__(self,config):
        super().__init__(
            n_var=4,         # Number of decision variables
            n_obj=2,         # Number of objectives
            n_constr=0,      # Number of constraints (set >0 if you have constraints)
            xl=np.array([0.1, 0 * np.pi / 180, 0, -np.pi]),  # Lower bounds
            xu=np.array([1.0, 80 * np.pi / 180, 1, np.pi])   # Upper bounds
        )
        self.config=config
        

    def _evaluate(self, x, out, *args, **kwargs):
        ldvm_instance = ldvm(self.config)
        time_deb= time.time()
        # print(f"[PID {os.getpid()}] évalue {x}")
    

    
        k, alpha0, h0, phi = x
        omega=2*ldvm_instance.u_ref*k/ldvm_instance.chord
        period=2*np.pi/omega
        D=period*ldvm_instance.u_ref
    
    
        Xhi=alpha0/np.arctan(h0*omega/ldvm_instance.u_ref)
    
        if np.abs(Xhi) >=1:
            print("Xhi >= 1, entering extraction mode...skipping evaluation")
            out["F"] = np.array([1e6,1e6])
         
        else: 
            d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

            ppp=int(D/d_wake)
            #print('ppp:', ppp)
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
            cd_history = np.array(cd_history)
    
            drag= np.array(cd_history[3:]) * ldvm_instance.rho * ldvm_instance.u_ref**2 * ldvm_instance.chord / 2
            lift = np.array(cl_history[3:]) * ldvm_instance.rho * ldvm_instance.u_ref**2 * ldvm_instance.chord / 2
            moment = np.array(cm_history[3:])*1/2 * ldvm_instance.rho * ldvm_instance.u_ref**2 * ldvm_instance.chord**2
    
            input_power= 1/period*trapezoid(np.abs(lift * ldvm_instance.hdot[4:]) +np.abs(moment * ldvm_instance.alphadot[4:]),dx=period/ppp)
            propulsion_force = 1/period*trapezoid(-drag, dx=period/ppp)   
    

    
            print("Thrust contribution", trapezoid(-drag* ldvm_instance.u_ref, dx=period/ppp))
            print("Lift contribution", -trapezoid(lift * ldvm_instance.hdot[4:], dx=period/ppp))
            print("Moment contribution", -trapezoid(moment * ldvm_instance.alphadot[4:], dx=period/ppp))
            print('Xhi:', Xhi)
    
            ct= propulsion_force / (0.5 * ldvm_instance.rho * ldvm_instance.u_ref**2 * ldvm_instance.chord)
            eta= propulsion_force*ldvm_instance.u_ref/input_power

            print("Evaluation terminée pour x =", x, "-> eta, thrust:", eta,propulsion_force, "Temps écoulé:", time.time() - time_deb)

            out["F"] = np.array([-eta, -ct])