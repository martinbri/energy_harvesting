import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import pandas
from scipy.integrate import trapezoid

class ldvm:
    def __init__(self, config=None):

        self.eps=1e-6 #Tolerance or iteration
        iter_max=100 #Max iteration
        self.v_core=0.02 #Non dimensional core radius of point vortices
        self.n_div=70 # No. of divisions along chord on airfoil
        self.n_aterm=70 #Number of fourier terms used to compute vorticity at a location on chord  
        self.del_dist=10
        self.iter_max=100
        self.kelv_enf=0
        if config is not None:
            self.input_file_name=config['input_file_name']
            self.u_ref=config['u_ref']
            self.chord=config['chord']
            self.pvt=config['pvt']
            self.cm_pvt=config['cm_pvt']
            self.foil_name=config['foil_name']
            self.re_ref=config['re_ref']
            self.lesp_crit=config['lesp_crit']

            self.motion_file_name=config['motion_file_name']
            self.force_file_name=config['force_file_name']
            self.flow_file_name=config['flow_file_name']
            self.n_pts_flow=config['n_pts_flow']
        else:
            self.input_file_name='input.csv'
            self.u_ref=1.0
            self.chord=1.0
            self.pvt=0.25
            self.cm_pvt=0.25
            self.foil_name='NACA0012'
            self.re_ref=1e6
            self.lesp_crit=0.5

            self.motion_file_name='motion.csv'
            self.force_file_name='force.csv'
            self.flow_file_name='flow.csv'
            self.n_pts_flow=100


        
        
        
        ##Dimmensionalize parameters
        self.vcore=self.v_core*self.chord
        self.del_dist=self.del_dist*self.chord



    def load_motion(self):

        # Load motion data from file
        try:
            motion_data = pandas.read_csv(self.motion_file_name,delim_whitespace=True)#pandas.read_csv(self.motion_file_name, sep=',')
        except pandas.errors.ParserError:
            raise ValueError(f"The file '{self.motion_file_name}' is not a valid CSV file or is improperly formatted.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{self.motion_file_name}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading the file: {e}")
        
        # Check if the required columns are present
        required_columns = ['time', 'alpha', 'h', 'u']
        for col in required_columns:
            if col not in motion_data.columns:
                raise ValueError(f"The required column '{col}' is missing from the motion data file.")
            
        self.time = motion_data['time'].values*self.chord/self.u_ref

        self.alpha = motion_data['alpha'].values*np.pi/180
        self.h = motion_data['h'].values*self.chord

        self.u = motion_data['u'].values*self.u_ref

        self.dtheta=np.pi/(self.n_div-1)
        self.theta = np.linspace(0, np.pi, self.n_div)
        self.x=(self.chord/2.)*(1-np.cos(self.theta))

        ## ADD Camber computation stuff
        self.alphadot=np.diff(self.alpha)/np.diff(self.time)
        self.hdot=np.diff(self.h)/np.diff(self.time)
        self.alphadot=np.concatenate(([self.alphadot[0]], self.alphadot))
        self.hdot=np.concatenate(([self.hdot[0]], self.hdot))
      

        
    def initialize_computation(self):
        # Initialize computation parameters
        self.n_lev=0
        self.n_tev=0
        self.aterm=np.zeros(self.n_aterm)
        self.aterm_prev=np.zeros(self.n_aterm)
        self.bound_vortex_pos=np.zeros((self.n_div, 3))
        self.levflag=0
        self.dist_wind=0
        self.i_step=0

        self.tev=np.empty((0, 3))
        self.lev=np.empty((0, 3))


        self.cam=0
        self.cam_slope=0


    def calc_downwash_boundcirc(self):
        # Placeholder for the actual downwash calculation
        # This function should compute the downwash based on the bound circulation
        # and update the relevant variables accordingly.
        uind=np.zeros((1,self.n_div))
        wind=np.zeros((1,self.n_div))

        ##Compute induced velocity at each point on the airfoil

        
        
        # Compute wake induced velocity
        xdist_TEV_Bound=np.tile(self.tev[:,1], (len(self.bound_vortex_pos[:,1]), 1)).T- np.tile(self.bound_vortex_pos[:,1], (len(self.tev[:,1]), 1))
        zdist_TEV_Bound=np.tile(self.tev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))
        dist=xdist_TEV_Bound**2+zdist_TEV_Bound**2
        Gamma=(self.tev[:,0]).reshape(1,-1)
        Ustar=(-zdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind+=Gamma@Ustar ##Warning sum is to be done in the appropriate direction +multiplication issue

        wind+=-Gamma@Wstar ##Warning sum is to be done in the appropriate direction +multiplication issue

        # Compute lev induced velocity

        xdist_LEV_Bound=np.tile(self.lev[:,1], (len(self.bound_vortex_pos[:,1]), 1)).T- np.tile(self.bound_vortex_pos[:,1], (len(self.lev[:,1]), 1))
        zdist_LEV_Bound=np.tile(self.lev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.lev[:,2]), 1))
        dist=xdist_LEV_Bound**2+zdist_LEV_Bound**2
        Gamma=(self.lev[:,0]).reshape(1,-1)
        Ustar=(-zdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind+=Gamma@Ustar ##Warning sum is to be done in the appropriate direction +multiplication issue
        wind+=-Gamma@Wstar ##Warning sum is to be done in the appropriate direction +multiplication issue


        # Compute the downwash  
        downwash=(-self.u[self.i_step]*np.sin(self.alpha[self.i_step]))+\
            (-uind*np.sin(self.alpha[self.i_step]))+\
            (self.hdot[self.i_step]*np.cos(self.alpha[self.i_step]))+\
            (-wind*np.cos(self.alpha[self.i_step]))+\
            (-self.alphadot[self.i_step]*(self.x-self.pvt*self.chord))+\
            (self.cam_slope*((uind*np.cos(self.alpha[self.i_step]))+(self.u[self.i_step]*np.cos(self.alpha[self.i_step]))+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step]))+(-wind*np.sin(self.alpha[self.i_step]))))
        

        aterm0 = trapezoid(downwash, dx=self.dtheta)
        aterm1 = trapezoid(downwash * np.cos(self.theta), dx=self.dtheta)
        bound_circ=self.u_ref*self.chord*np.pi*(aterm0+(aterm1/2.))
        return aterm0, aterm1, downwash,bound_circ,uind,wind



    def step(self):
        
        # Perform a single step of the computation
        self.i_step+=1
        print("Step: {}, number of lev {}, number of tev {}, time {},tmax ={}".format(self.i_step, self.n_lev, self.n_tev,self.time[self.i_step],self.time[-1]))
        #Calculate bound vortex positions at this time step
        self.dist_wind=self.dist_wind+(self.u[self.i_step-1]*(self.time[self.i_step]-self.time[self.i_step-1]))
        self.bound_vortex_pos[:,1]=-((self.chord-self.pvt*self.chord)+((self.pvt*self.chord-self.x)*np.cos(self.alpha[self.i_step]))+self.dist_wind) + (self.cam*np.sin(self.alpha[self.i_step]))
        self.bound_vortex_pos[:,2]=self.h[self.i_step]+((self.pvt*self.chord-self.x)*np.sin(self.alpha[self.i_step]))+(self.cam*np.cos(self.alpha[self.i_step]))


        #TEV shed at every time step
        
        tev_iter=np.zeros(101)
        lev_iter=np.zeros(101)
        kutta=np.zeros(100)
        kelv=np.zeros(100)
        tev_iter[0]=0
        tev_iter[1]=-0.01

        if self.n_tev==0:
            x_tev=self.bound_vortex_pos[self.n_div-1,1]+0.5*self.u[self.i_step]*(self.time[self.i_step]-self.time[self.i_step-1])
            y_tev= self.bound_vortex_pos[self.n_div-1,2]
            self.tev=np.concatenate((self.tev, np.array([[0, x_tev, y_tev]])), axis=0)
            

        else:
            
            
            x_tev=self.bound_vortex_pos[self.n_div-1,1]+((1./3.)*(self.tev[self.n_tev-1,1]-self.bound_vortex_pos[self.n_div-1,1]))
            y_tev=self.bound_vortex_pos[self.n_div-1,2]+((1./3.)*(self.tev[self.n_tev-1,2]-self.bound_vortex_pos[self.n_div-1,2]))
            self.tev=np.concatenate((self.tev, np.array([[0, x_tev, y_tev]])), axis=0)



        #Iterating to find AO value assuming no LEV is formed
        iter=0
        while (iter<self.iter_max-1):
            iter=iter+1
            
            

            
            
            self.tev[self.n_tev,0]=tev_iter[iter]
            aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc()

            kelv[iter]=self.kelv_enf
            
            if self.lev.size>0:
                kelv[iter]+=np.sum(self.lev[:,0])
            if self.tev.size>0:
                
                
                kelv[iter]+=np.sum(self.tev[:,0])

            print("curent kelmvin condition",kelv[iter])
            kelv[iter]=kelv[iter]+bound_circ
            if (abs(kelv[iter])<self.eps) :

                break

            
            dkelv=(kelv[iter]-kelv[iter-1])/(tev_iter[iter]-tev_iter[iter-1])
            tev_iter[iter+1]=tev_iter[iter]-(kelv[iter]/dkelv)  
            
            #print("tev_iter[iter+1]",tev_iter[iter],tev_iter[iter-1],kelv[iter],kelv[iter-1],iter)
            
        if (iter>=self.iter_max):
            print('1D iteration failed, the residual is ', abs(kelv[iter]))

        aterm2 =(2./(self.u_ref*np.pi))* trapezoid(downwash * np.cos(2 * self.theta), dx=self.dtheta)
        aterm3 = (2./(self.u_ref*np.pi))* trapezoid(downwash * np.cos(3 * self.theta), dx=self.dtheta)   
      
        adot0=(aterm0-self.aterm_prev[0])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot1=(aterm1-self.aterm_prev[1])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot2=(aterm2-self.aterm_prev[2])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot3=(aterm3-self.aterm_prev[3])/(self.time[self.i_step]-self.time[self.i_step-1])
      
        le_vel_x=(self.u[self.i_step])-(self.alphadot[self.i_step]*np.sin(self.alpha[self.i_step])*self.pvt*self.chord)+uind[0,0]
        
        
        le_vel_y=-(self.alphadot[self.i_step]*np.cos(self.alpha[self.i_step])*self.pvt*self.chord)-(self.hdot[self.i_step])+wind[0,0]
        
        vmag=np.sqrt(le_vel_x*le_vel_x+le_vel_y*le_vel_y)
        re_le=self.re_ref*vmag/self.u_ref
        lesp=aterm0
        #2D iteration if LESP_crit is exceeded

        print("lesp",lesp)
        print("lesp crit",self.lesp_crit)
        print("current incidence",self.alpha[self.i_step])
        print("current time",self.time[self.i_step])
        if (abs(lesp)>self.lesp_crit):
            print("A LEV is formed")


            if (lesp>0) :
                lesp_cond=self.lesp_crit
            else:
                lesp_cond=-self.lesp_crit
            
            
            tev_iter[0]=0
            tev_iter[1]=-0.01
            lev_iter[0]=0
            lev_iter[1]=0.01
     
            if (self.levflag==0) :
                x_lev=self.bound_vortex_pos[0,1]+(0.5*le_vel_x*(self.time[self.i_step]-self.time[self.i_step-1]))
                y_lev=self.bound_vortex_pos[0,2]+(0.5*le_vel_y*(self.time[self.i_step]-self.time[self.i_step-1]))
            else:
                x_lev=self.bound_vortex_pos[0,1]+((1./3.)*(self.lev[self.n_lev-1,1]-self.bound_vortex_pos[0,1]))
                y_lev=self.bound_vortex_pos[0,2]+((1./3.)*(self.lev[self.n_lev-1,2]-self.bound_vortex_pos[0,2]))
            print(self.lev.shape)
            print(x_lev,y_lev,le_vel_x,le_vel_y)
            self.lev=np.concatenate((self.lev, np.array([[0, x_lev, y_lev]])), axis=0)
            self.levflag=1
            #Update xdist and zdist arrays I Dont need to do that here



            iter =0 
            while (iter<self.iter_max):
                

                iter=iter+1
                
                #Advancing with tev strength
                self.lev[self.n_lev,0]=lev_iter[iter-1]
                self.tev[self.n_tev,0]=tev_iter[iter]

                aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc()

                kelv_tev=self.kelv_enf
                kelv_tev+=np.sum(self.lev[:,0])
                kelv_tev+=np.sum(self.tev[:,0])
                kelv_tev+=bound_circ

                kutta_tev=aterm0-lesp_cond
                dkelv_tev=(kelv_tev-kelv[iter-1])/(tev_iter[iter]-tev_iter[iter-1])
                
                dkutta_tev=(kutta_tev-kutta[iter-1])/(tev_iter[iter]-tev_iter[iter-1])
                #Advancing with lev strength 
                self.lev[self.n_lev,0]=lev_iter[iter]
                self.tev[self.n_tev,0]=tev_iter[iter-1] 

                aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc()
                kelv_lev=self.kelv_enf
                kelv_lev+=np.sum(self.lev[:,0])
                kelv_lev+=np.sum(self.tev[:,0])
                kelv_lev+=bound_circ

                kutta_lev=aterm0-lesp_cond
                dkelv_lev=(kelv_lev-kelv[iter-1])/(lev_iter[iter]-lev_iter[iter-1])

                dkutta_lev=(kutta_lev-kutta[iter-1])/(lev_iter[iter]-lev_iter[iter-1])

                #Advancing with both
                self.lev[self.n_lev,0]=lev_iter[iter]
                self.tev[self.n_tev,0]=tev_iter[iter]

                aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc()
                kelv[iter]=self.kelv_enf
                kelv[iter]+=np.sum(self.lev[:,0])
                kelv[iter]+=np.sum(self.tev[:,0])
                kelv[iter]+=bound_circ
                kutta[iter]=aterm0-lesp_cond
                if (abs(kelv[iter])<self.eps and abs(kutta[iter])<self.eps):
                    break
                tev_iter[iter+1]=tev_iter[iter]-((1/(dkelv_tev*dkutta_lev-dkelv_lev*dkutta_tev))*((dkutta_lev*kelv[iter])-(dkelv_lev*kutta[iter])))
                lev_iter[iter+1]=lev_iter[iter]-((1/(dkelv_tev*dkutta_lev-dkelv_lev*dkutta_tev))*((-dkutta_tev*kelv[iter])+(dkelv_tev*kutta[iter])))
                
            if (iter>=self.iter_max):
                    print('2D iteration failed, the residuals are kelvin :{}, kutta {}'.format(abs(kelv[iter]),abs(kutta[iter])))
            self.n_lev=self.n_lev+1
        else:
            self.levflag=0
        

                 
        #To remove any massive starting vortices
        if (self.i_step==1) :
            self.tev[0,0]=0
        
        #Calculate fourier terms and bound vorticity
        
        for i_aterm in range(self.n_aterm):  # includes n_aterm
            integrand = downwash * np.cos(i_aterm * self.theta)
            integral = trapezoid(integrand, dx=self.dtheta)
            self.aterm[i_aterm] = (2.0 / (self.u_ref * np.pi)) * integral

        self.aterm_prev=self.aterm.copy()
        #Calculate bound_vortex strengths

        gamma = np.zeros(self.n_div)
 
        gamma+=(self.aterm[0]*(1+np.cos(self.theta)))
        for i_aterm in range(1, self.n_aterm):
            gamma+=(self.aterm[i_aterm]*np.sin(i_aterm*self.theta)*np.sin(self.theta))

        bound_int=np.zeros((self.n_div-1,3))
        bound_int[:,0]=((gamma[:-1]+gamma[1:])/2)*self.dtheta
        bound_int[:,1]=(self.bound_vortex_pos[:-1,1]+self.bound_vortex_pos[1:,1])/2
        bound_int[:,2]=(self.bound_vortex_pos[:-1,2]+self.bound_vortex_pos[1:,2])/2

        # Wake Rollup
        # Update Tev numbers
        self.n_tev=self.n_tev+1  

        uind_tev=np.zeros(self.n_tev) # Vitesse induite sur les TEV
        wind_tev=np.zeros(self.n_tev)
        
        xdist_TEV_TEV=np.tile(self.tev[:,1], (len(self.tev[:,1]), 1)).T- np.tile(self.tev[:,1], (len(self.tev[:,1]), 1))
        zdist_TEV_TEV=np.tile(self.tev[:,2], (len(self.tev[:,2]), 1)).T- np.tile(self.tev[:,2], (len(self.tev[:,2]), 1))
        dist=xdist_TEV_TEV**2+zdist_TEV_TEV**2
        Gamma=(self.tev[:,0]).reshape(1,-1)
        Ustar=(-zdist_TEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_tev=uind_tev+Gamma@Ustar #Warning sum is to be done in the appropriate direction +multiplication issue
        
        wind_tev=wind_tev-Gamma@Wstar #Warning sum is to be done in the appropriate direction +multiplication issue

        
        
        xdist_LEV_TEV=np.tile(self.lev[:,1], (len(self.tev[:,1]), 1)).T- np.tile(self.tev[:,1], (len(self.lev[:,1]), 1))
        zdist_LEV_TEV=np.tile(self.lev[:,2], (len(self.tev[:,2]), 1)).T- np.tile(self.tev[:,2], (len(self.lev[:,2]), 1))
        dist=xdist_LEV_TEV**2+zdist_LEV_TEV**2
        Gamma=(self.lev[:,0]).reshape(1,-1)
        

        Ustar=(-zdist_LEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind_tev=uind_tev+Gamma@Ustar# Warning sum is to be done in the appropriate direction +multiplication issue
        wind_tev=wind_tev-Gamma@Wstar #Warning sum is to be done in the appropriate direction +multiplication issue
        

        #Profile sur TEV
        bound_int_xdist=np.tile(bound_int[:,1], (len(self.tev[:,1]),1)).T-np.tile(self.tev[:,1],(len(bound_int[:,1]),1))
        bound_int_zdist=np.tile(bound_int[:,2], (len(self.tev[:,2]),1)).T-np.tile(self.tev[:,2],(len(bound_int[:,2]),1))
        
        

        dist=bound_int_xdist**2+bound_int_zdist**2
        Gamma=(bound_int[:,0]).reshape(1,-1)

        Ustar=(-bound_int_zdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-bound_int_xdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        print(Gamma.shape,Ustar.shape)
        
        uind_tev=uind_tev+Gamma@Ustar
        wind_tev=wind_tev-Gamma@Wstar
        
        
        
        # Vitesse induite sur les LEV

        
        uind_lev=np.zeros(self.n_lev) 
        wind_lev=np.zeros(self.n_lev)
        #Vitesse LEV sur LEV

        xdist_LEV_LEV=np.tile(self.lev[:,1], (len(self.lev[:,1]),1)).T-np.tile(self.lev[:,1], (len(self.lev[:,1]),1))
        zdist_LEV_LEV=np.tile(self.lev[:,2], (len(self.lev[:,2]),1)).T-np.tile(self.lev[:,2], (len(self.lev[:,2]),1))
        Gamma=(self.lev[:,0]).reshape(1,-1)

      
        dist=xdist_LEV_LEV**2+zdist_LEV_LEV**2
        Ustar=(-zdist_LEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar
        
            
        #Vitesse TEV sur LEV
        xdist_TEV_LEV=np.tile(self.tev[:,1], (len(self.lev[:,1]), 1)).T- np.tile(self.lev[:,1], (len(self.tev[:,1]), 1))
        zdist_TEV_LEV=np.tile(self.tev[:,2], (len(self.lev[:,2]), 1)).T- np.tile(self.lev[:,2], (len(self.tev[:,2]), 1))

        dist=xdist_TEV_LEV**2+xdist_TEV_LEV**2
        Gamma=(self.tev[:,0]).reshape(1,-1)

        Ustar=(-zdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))


        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar
        

        #Compute airfoil on LEV

        bound_int_xdist=np.tile(bound_int[:,1], (len(self.lev[:,1]),1)).T-np.tile(self.lev[:,1],(len(bound_int[:,1]),1))
        bound_int_zdist=np.tile(bound_int[:,2], (len(self.lev[:,2]),1)).T-np.tile(self.lev[:,2],(len(bound_int[:,2]),1))
        dist=bound_int_xdist**2+bound_int_zdist**2

        Gamma=(bound_int[:,0]).reshape(1,-1)
        Ustar=(-bound_int_zdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-bound_int_xdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        
        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar

        



        dt=self.time[self.i_step]-self.time[self.i_step-1]


        ##Update TEV and LEV positions
        self.tev[:,1]=self.tev[:,1]+(uind_tev*dt)
        self.tev[:,2]=self.tev[:,2]+(wind_tev*dt)
        self.lev[:,1]=self.lev[:,1]+(uind_lev*dt)
        self.lev[:,2]=self.lev[:,2]+(wind_lev*dt)


        # Cropping LEV and tEV arrays is not done here


        
        #Load coefficient calculation (nondimensional units)

        cnc=(2*np.pi*((self.u[self.i_step]*np.cos(self.alpha[self.i_step])/self.u_ref)+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step])/self.u_ref))*(self.aterm[0]+self.aterm[1]/2))

        cnnc=(2*np.pi*((3*self.chord*adot0/(4*self.u_ref))+(self.chord*adot1/(4*self.u_ref))+(self.chord*adot2/(8*self.u_ref))))

        cs=2*np.pi*self.aterm[0]*self.aterm[0]

        #The components of normal force and moment from induced velocities are calulcated in dimensional units and nondimensionalized later

        nonl=0
        nonl_m=0


        #nonl=np.sum(((uind*np.cos(self.alpha[self.i_step]))-(wind*np.sin(self.alpha[self.i_step])))*bound_int[:,0])
        #nonl_m=np.sum(((uind*np.cos(self.alpha[self.i_step]))-(wind*np.sin(self.alpha[self.i_step])))*(self.x[1:])*bound_int[:,0])


        nonl=nonl*(2/(self.u_ref*self.u_ref*self.chord))

        nonl_m=nonl_m*(2/(self.u_ref*self.u_ref*self.chord*self.chord))



        cn=cnc+cnnc+nonl

        cl=cn*np.cos(self.alpha[self.i_step])+cs*np.sin(self.alpha[self.i_step])
        cd=cn*np.sin(self.alpha[self.i_step])-cs*np.cos(self.alpha[self.i_step])

        
        #cm=cn*self.cm_pvt-(2*np.pi*(((self.u[self.i_step]*np.cos(self.alpha[self.i_step])/self.u_ref)+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step])/self.u_ref))*((self.aterm[0]/4)+(self.aterm[1]/4)-(self.aterm[2]/8))+(self.chord/self.u_ref)*((7*adot[0]/16)+(3*adot[1]/16)+(adot[2]/16)-(adot[3]/64))))-nonl_m



        return cl, cd, 0, lesp, re_le,







if __name__ == "__main__":
    # Example usage
    config = {
        'input_file_name': 'input.csv',
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.0,
        'cm_pvt': 0.0,
        'foil_name': 'sd7003.dat',
        're_ref': 30000,
        'lesp_crit': 0.18,
        'motion_file_name': '../LDVM_paul/motion_pr_amp45_k0.2.dat',
        'force_file_name': 'fforce_pr_amp45_k0.2_le.csv',
        'flow_file_name': 'flow.csv',
        'n_pts_flow': 100
    }

    ldvm_instance = ldvm(config)
    ldvm_instance.load_motion()
    ldvm_instance.initialize_computation()
    for i in range(499):
        #print(ldvm_instance.alpha[:i]*180/np.pi)
        ldvm_instance.step()
        if i%5 ==0:
            

            plt.plot(ldvm_instance.bound_vortex_pos[:,1], ldvm_instance.bound_vortex_pos[:,2], 'b-',markersize=2)
            plt.plot(ldvm_instance.tev[:,1],ldvm_instance.tev[:,2],'ko',markersize=2)

            plt.plot(ldvm_instance.lev[:,1],ldvm_instance.lev[:,2],'ro',markersize=2)

            

            plt.xlim(ldvm_instance.bound_vortex_pos[0,1]-ldvm_instance.chord/2,ldvm_instance.bound_vortex_pos[-1,1]+ldvm_instance.chord/2)
            plt.ylim(-1.5,1.0)
            
            
            plt.show()
 



   