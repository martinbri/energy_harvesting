import numpy as np
import matplotlib.pyplot as plt
from ldvm_back_up import ldvm
from scipy.integrate import solve_ivp
import pandas


class two_DOF_ldvm:
    def __init__(self, config=None):

        if config is not None:
            self.m=np.float64(config['mass'])
            self.c=np.float64(config['chord'])
            self.kh=np.float64(config['kh'])
            self.kalpha=np.float64(config['kalpha'])
            self.pvt=np.float64(config['pvt'])
            self.u_ref=np.float64(config['u_ref'])
            self.cm_pvt=np.float64(config['cm_pvt'])
            #np.float64(config['Ialpha'])
            self.xg=np.float64(config['xg'])
            self.foil_name=config['foil_name']
            self.condition_initial=config['condition_initial']
            self.span=np.float64(config['span'])
            self.rho=np.float64(config['rho'])  

            
        else:
            self.m=3
            self.c=1
            self.kh=200
            self.kalpha=4
            self.pvt=0.0
            self.u_ref=10
            self.cm_pvt=0.0
            self.Ialpha=1
            self.xg=self.c/2
            self.foil_name='foil.dat'
            self.condition_initial={'h0':self.c/10, 'alpha0':0.0, 'hdot':0.0, 'alphadot0':0.0}
            self.span=1.0
            self.rho=1.225
        self.dt_computation=1/1000  # Time step for computation, based on the chord length and reference speed
        self.xf=self.pvt*self.c
        self.Ialpha=self.m/3*(self.c*self.c-3*self.c*self.xf+3*self.xf*self.xf)
        print("natural frequency",np.sqrt(self.kh/self.m))
        print("natural frequency alpha",np.sqrt(self.kalpha/self.Ialpha))
        print("dt_computation",self.dt_computation)
        print("pvt",self.pvt)
        print("foil_name",self.foil_name)
        print("Ialpha",self.Ialpha)
        print("xg",self.xg)
        print("span",self.span)
        print("rho",self.rho)
        print("Reynolds number",self.u_ref*self.c/self.rho)
        print("mass",self.m)
        print("chord",self.c)
        print("kh",self.kh)
        print('Ialpha',self.Ialpha)
        print("kalpha",self.kalpha)
        print('S',self.m*(self.xg-self.xf))
        input("Press Enter to continue...")


        self.ldvm=ldvm(config=config)
        
        self.ldvm.initialize_computation()
    def make_mass_matrix(self):
        s=self.m*(self.xg-self.xf)
        Ialpha=self.Ialpha 
        print("m",self.m)
        print("s",s)
        print("Ialpha",Ialpha)
        print("span",self.span)
        print("xg",self.xg)
        print("xf",self.xf)
        print("c",self.c)
        print("pvt",self.pvt)
        print("Ialpha",Ialpha)
        print("mass matrix")
        #input("Press Enter to continue...")
        A=np.array([[self.m,s],
                   [s,Ialpha]])
        A=A/self.span
        self.A=A
        return A
    def make_stiffness_matrix(self):
        print("kh",self.kh)
        print("kalpha",self.kalpha)
        print("span",self.span)
        E=np.array([[self.kh,0],
                   [0,self.kalpha]])
        E=E/self.span
        print("E",E)
  
        return E    
    def make_damping_matrix(self):
        return self.make_stiffness_matrix()/1000
    
    def make_initial_conditions(self,data=None):
        h0=self.condition_initial['h0']
        alpha0=self.condition_initial['alpha0']
        hdot=self.condition_initial['hdot']
        alphadot0=self.condition_initial['alphadot0']
        X0=np.array([hdot,alphadot0,h0,alpha0])
        self.t_values=[]
        self.t=0.0
        self.t_minus_1=-self.dt_computation
        if data is not None:
            self.X_values=np.array(data)
            return
        self.X_values=np.array([X0])
        return X0
    def make_aero_matrix(self):
        b=self.c/2
        xcg=self.xf - self.xg
        a=xcg/b
        B = b**2 * np.array([
        [np.pi,         -np.pi * a * b],
        [-np.pi * a * b, np.pi * b**2 * (1/8 + a**2)]])
        
        D1 = b**2 * np.array([
            [0,                np.pi],
            [0, np.pi * (1/2 - a) * b]])

        D2 = np.array([
        [2 * np.pi * b,                2 * np.pi * b**2 * (1/2 - a)],
        [-2 * np.pi * b**2 * (a + 1/2), -2 * np.pi * b**3 * (a + 1/2) * (1/2 - a)]])
        F = np.array([
            [0,                2 * np.pi * b],
            [0, -2 * np.pi * b**2 * (a + 1/2)]])
    
        return B, D1, D2, F


    def make_first_order_matrices(self):
        A=self.make_mass_matrix()
        E=self.make_stiffness_matrix()
        C=self.make_damping_matrix()
        print("A",A)
        print("E",E)
        print("C",C)
        

        A_inv=np.linalg.inv(A)
        self.A_inv=A_inv
        l1=np.hstack((-np.dot(A_inv,C),-np.dot(A_inv,E)))
        l2=np.hstack((np.eye(2),np.zeros((2,2))))
        print('A_inv', A_inv)
        
        print("l1",l1.shape)
        print("l2",l2)

        Q=np.vstack((l1,l2))
        self.Q=Q
        print("Q",Q)
        input("Press Enter to continue...")

        return Q
    
    def ODE(self,t, x,loads):

        RHS = np.dot(self.A_inv,loads)  # Compute the right-hand side of the ODE
        RHS = np.hstack((RHS, np.zeros(2)))  # Append zeros for the second order terms

        return np.dot(self.Q,x)+RHS
    def one_ODE_resolution_step(self,t_span,X,loads):
        """Solve the system over the wanted time lapse tspan.

        Args:
            t_span (tupple): min and max time required for the solution
            X (array): current state

        Returns:
            float,array: updated time and state
        """
        
        #t_eval=np.linspace(t_span[0],t_span[1],int((t_span[1]-t_span[0])/self.dt_computation)+1)
        
        
        
        solution = solve_ivp(self.ODE, (t_span), X, args=(loads,), method='RK45')
        self.t_minus_1=self.t  # Update the previous time
        print(t_span)
        print("solution.t",solution)

        self.t=solution.t[-1]
        self.t_values.append(solution.t[-1])
        self.X_values=np.vstack((self.X_values,solution.y[:, -1]))  # Stocke la dernière valeur obtenue
        return solution.t[-1],solution.y[:, -1]
    

    def compute_current_loads(self, t,X, t_minus_1,u):
        """Compute the current loads based on the current state and time.

        Args:
            t (float): current time
            X (array): current state

        Returns:
            array: computed loads according to the ldvm model
        """
        print("t", t)
        print("t_minus_1", t_minus_1)
        print("X", X)
        h, alpha, hdot, alphadot = X
        cl,cd,cm = self.ldvm.step( h=-X[2], alpha=X[3], hdot=-X[0], alphadot=X[1],t=t,t_minus_1=t_minus_1,u=u)# There is a minus sign on h because the ldvm model uses positive h for upwards motion 
        l=cl*(0.5*self.rho*self.u_ref**2*self.c)
        m=cm*(0.5*self.rho*self.u_ref**2*self.c**2)

        print("-l", -l) 
        print("m", m)
        print("u_ref", )
        print("cl", cl)
        print("cm", cm)
        #input("Press Enter to continue...")
        
        loads = np.array([-l,m])

        return loads
if __name__ == "__main__":
    chord = 0.15
    u=15.3
    pvt=0.25
    config = {
        'mass': 3.0,
        'chord': chord,
        'kh': 200,
        'kalpha': 4,
        'pvt': pvt,
        'u_ref': u,
        'cm_pvt': pvt,
        'Ialpha': 0.0098,
        'xg': 0.5*chord,
        'foil_name': 'sd7012.dat',
        'condition_initial': {'h0': 0.015, 'alpha0': 0.0, 'hdot': 0.0, 'alphadot0': 0.0},
        'span': 0.45,
        'rho': 1.225,
        're_ref':30000,
        'lesp_crit':50
    }
    model = two_DOF_ldvm(config)



  

    data=pandas.read_csv('motion_pr_amp45_k0.2.dat',delim_whitespace=True)
    #print("data",data)
    times=data['time'].values
    h=data['alpha'].values/100
    alpha=data['h'].values*np.pi/180
    hdot=(h[1:]-h[:-1])/(times[1:]-times[:-1])
    alphadot=(alpha[1:]-alpha[:-1])/(times[1:]-times[:-1])
    hdot=np.hstack((0,hdot))
    lphadot=np.hstack((0,alphadot))
    X=np.array([hdot[0],alphadot[0],h[0],alpha[0]])
    X=model.make_initial_conditions()#data=np.array([X]))
    print(model.X_values)

    print("X",X)
    
    
    # t=times[0]
    # print("X",X)


    
    Q=model.make_first_order_matrices()
    
    X_t_minus_1=np.copy(X)

    l_theo_store=[]
    m_theo_store=[]
    l_ldvm_store=[]
    m_ldvm_store=[]



    
    for i in range(1,10000):
        b=model.c/2
        xcg=model.xf - model.xg
        a=xcg/b
        X_second= (X[:2]-X_t_minus_1[:2])/model.dt_computation
        print(model.dt_computation)
        

        loads=model.compute_current_loads(X=X,t=model.t,t_minus_1=model.t_minus_1,u=model.u_ref)
        l_ldvm_store.append(-loads[0]/(0.5*model.rho*model.u_ref**2*model.c))
        m_ldvm_store.append(loads[1]/(0.5*model.rho*model.u_ref**2*model.c**2))

        

        lift_theo=model.rho*(b)**2*(model.u_ref*np.pi*X[1]+np.pi*X_second[0]-np.pi*b*a*X_second[1])+2*np.pi*model.rho*(model.c/2)*model.u_ref*(model.u_ref*X[3]+X[0]+model.c/2*(0.5-a)*X[1])
        m_theo=-model.rho*(b)**2*(-a*np.pi*b*X_second[0]+np.pi*b**2*(1/8*a**2)*X_second[0]+np.pi*(0.5-a)*model.u_ref*b*X[1])+2*np.pi*model.rho*b**2*(1/2+a)*model.u_ref*(model.u_ref*X[3]+X[0]+model.c/2*(0.5-a)*X[1])
       
        l_theo_store.append(lift_theo/(0.5*model.rho*model.u_ref**2*model.c))
        m_theo_store.append(m_theo/(0.5*model.rho*model.u_ref**2*model.c**2))

        # input('dd')


        
        t,X=model.one_ODE_resolution_step((model.t,model.t+model.dt_computation),X,loads)#=np.array([-loads[0],loads[1]]))
        print("lift_theo",lift_theo)
        print("m_theo",m_theo)
        print('loads',loads)
        print("X",X)
        
        #t,X=times[i],np.array([hdot[i],alphadot[i],h[i],alpha[i]])
        X_t_minus_1=np.copy(X)
        #model.t_values.append(times[i])
        #model.X_values=np.vstack((model.X_values,X))  # Stocke la dernière valeur obtenue
        
        

        
    print(f"Step {i}, Time: {model.t:.2f}, State: {X}")
    
    save_data = np.column_stack((np.array([0]+model.t_values), model.X_values[:,3],model.X_values[:,2],np.ones(len(model.t_values)+1))) #minus sign on h because the ldvm model uses positive h for upwards motion


    np.savetxt('flutter_data.dat', save_data, delimiter=' ')



    resu_ldvm=np.loadtxt('../LDVM_v2_original.5/force_float.dat')

    plt.figure()


    plt.plot(model.t_values, model.ldvm.bound_circ_save, label='Bound Circulation')
    plt.plot(resu_ldvm[:,0], resu_ldvm[:,4], label='Bound Circulation LDVM')
    plt.xlabel('Time')
    plt.ylabel('Bound Circulation')
    plt.title('Bound Circulation vs Time')
    plt.legend()

    plt.figure()
    plt.plot(model.t_values,model.X_values[1:,2]/model.c*2,label='h/b')
    plt.legend()
    plt.figure()
    plt.plot(model.t_values,model.X_values[1:,3]*180/np.pi,label='alpha')
    plt.legend()

    
    plt.figure()
    print("l_theo_store",len(l_theo_store))
    print("l_ldvm_store",len(l_ldvm_store))
    print(("len(model.t_values)",len(model.t_values)))
    plt.plot(resu_ldvm[:,0], resu_ldvm[:,8], label='Lift LDVM fortran')

    plt.plot(model.t_values,l_theo_store, label='Lift theo')
    plt.plot(model.t_values,l_ldvm_store, label='Lift LDVM')
    data_ldvm=np.loadtxt('/home/disc/b.martin/Documents/energy_harvesting/LDVM_v2_original.5/force_pr_amp45_k0.2_le.dat')
    print("data_ldvm",data_ldvm.shape)
    cl_lit=data_ldvm[:,8]
    #plt.plot(data_ldvm[1:,0],cl_lit[:-1], label='Lift literature')
    plt.legend()


    plt.figure(tight_layout=True)
    plt.plot(model.t_values, m_theo_store, label='Moment theo')
    plt.plot(model.t_values, np.array(m_ldvm_store), label='Moment LDVM')
    plt.plot(resu_ldvm[:,0], resu_ldvm[:,10], label='Moment LDVM fortran')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Moment coefficient')
    plt.title('Moment Comparison')
    plt.show()


