import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas

class ldvm:
    def __init__(self, config=None):

        self.eps=1e-6 #Tolerance or iteration
        iter_max=100 #Max iteration
        self.v_core=0.02 #Non dimensional core radius of point vortices
        self.n_div=70 # No. of divisions along chord on airfoil
        self.n_aterm=70 #Number of fourier terms used to compute vorticity at a location on chord  
        self.del_dist=10
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

        dtheta=np.pi/(self.n_div-1)
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

        self.tev=np.zeros((3000, 3))


        self.cam=0



    def step(self):
        
        # Perform a single step of the computation
        self.i_step+=1
        print("Step: {}, number of lev {}, number of tev {}".format(self.i_step, self.n_lev, self.n_tev))
        #Calculate bound vortex positions at this time step
        self.dist_wind=self.dist_wind+(self.u[self.i_step-1]*(self.time[self.i_step]-self.time[self.i_step-1]))
        self.bound_vortex_pos[:,1]=-((self.chord-self.pvt*self.chord)+((self.pvt*self.chord-self.x)*np.cos(self.alpha[self.i_step]))+self.dist_wind) + (self.cam*np.sin(self.alpha[self.i_step]))
        self.bound_vortex_pos[:,2]=self.h[self.i_step]+((self.pvt*self.chord-self.x)*np.sin(self.alpha[self.i_step]))+(self.cam*np.cos(self.alpha[self.i_step]))


        #TEV shed at every time step
        self.n_tev=self.n_tev+1
        self.tev_iter=np.zeros(100)
        self.tev_iter[0]=0
        self.tev_iter[1]=-0.01

        if self.n_tev==1:
            self.tev[self.n_tev-1,1]=self.bound_vortex_pos[self.n_div-1,1]+0.5*self.u[self.i_step]*(self.time[self.i_step]-self.time[self.i_step-1])
            self.tev[self.n_tev-1,2]= self.bound_vortex_pos[self.n_div-1,2]

        else:
            self.tev[self.n_tev-1,1]=self.bound_vortex_pos[self.n_div-1,2]+((1./3.)*(self.tev(self.n_tev-2,2)-self.bound_vortex_pos[self.n_div-1,2]))
            self.tev[self.n_tev-1,3]=self.bound_vortex_pos(self.n_div-1,3)+((1./3.)*(self.tev(self.n_tev-2,3)-self.bound_vortex_pos(self.n_div,3)))

        #Precalculate xdist and zdist arrays

        self.xdist=np.zeros((3,3,self.n_div, self.n_tev))

        self.xdist[0,1,:,:]= np.tile(self.tev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))
        
        
        self.xdist[0,2,:,:]= np.tile(self.lev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))


        self.xdist[1,1,:,:]= np.tile(self.lev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))   


        self.xdist[1,2,:,:]= np.tile(self.lev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))  



        



if __name__ == "__main__":
    # Example usage
    config = {
        'input_file_name': 'input.csv',
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.0,
        'cm_pvt': 0.25,
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
    ldvm_instance.step()


   