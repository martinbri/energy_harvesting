import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import pandas
from scipy.integrate import trapezoid
import time
import matplotlib.colorbar as cbar
from matplotlib.animation import FuncAnimation
from scipy.special import hankel2
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from multiprocessing import Pool, cpu_count
import matplotlib.gridspec as gridspec

parameters = {'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize':12,
          'ytick.labelsize':12}
plt.rcParams.update(parameters)

class ldvm:
    def __init__(self, config=None):

        self.eps=1e-6 #Tolerance or iteration
        self.v_core=0.02 #Non dimensional core radius of point vortices
         # No. of divisions along chord on airfoil
        self.n_aterm=45 #Number of fourier terms used to compute vorticity at a location on chord
        self.del_dist=5.00
        self.iter_max=100
        self.kelv_enf=0.0
        if config is not None:
            self.u_ref=np.float64(config['u_ref'])
            self.chord=np.float64(config['chord'])
            self.pvt=np.float64(config['pvt'])
            self.n_div=np.int64(config['n_div'])
            self.cm_pvt=np.float64(config['cm_pvt'])

            self.foil_name=config['foil_name']
            self.re_ref=np.float64(config['re_ref'])
            self.lesp_crit=np.float64(config['lesp_crit'])

            self.motion_file_name=config['motion_file_name']
            self.force_file_name=config['force_file_name']
            self.flow_file_name=config['flow_file_name']
            self.n_pts_flow=config['n_pts_flow']
            self.rho=config['rho']
        else:
            self.u_ref=np.float64(1.0)
            self.chord=np.float64(1.0)
            self.pvt=np.float64(0.25)
            self.cm_pvt=0.25
            self.foil_name='NACA0012'
            self.re_ref=float(1e6)
            self.lesp_crit=0.5

            self.motion_file_name='motion.csv'
            self.force_file_name='force.csv'
            self.flow_file_name='flow.csv'
            self.n_pts_flow=100



        self.dtheta=np.pi/(self.n_div-1)
        self.theta = np.linspace(0, np.pi, self.n_div)
        self.x=(self.chord/2.)*(1-np.cos(self.theta))

        self.W_trapz_integration=self.make_trapezoid_integration_matrix(self.n_div)

        ##Dimmensionalize parameters
        self.v_core=self.v_core*self.chord
        self.del_dist=self.del_dist*self.chord



    def load_motion(self):
        ff

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



        ## ADD Camber computation stuff
        self.alphadot=np.diff(self.alpha)/np.diff(self.time)
        self.hdot=np.diff(self.h)/np.diff(self.time)
        self.alphadot=np.concatenate(([self.alphadot[0]], self.alphadot))
        self.hdot=np.concatenate(([self.hdot[0]], self.hdot))


    def make_trapezoid_integration_matrix(self,n_div):
        W=np.ones(n_div)
        W[0]=0.5
        W[-1]=0.5
        return W
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


        self.bound_circ_save=[]


        self.cam,self.cam_slope=self.calc_camber_slope()

    def make_traz_int(self,n_div):
        W=np.ones((n_div, 3))  # Initialize W with ones
    def calc_camber_slope(self, plot=False):
        from scipy.interpolate import CubicSpline
        #Constructing camber slope from airfoil file
        if self.foil_name== 'flate_plate':
            return np.zeros(self.n_div),np.zeros(self.n_div)

        data_profile=np.loadtxt(self.foil_name)
        xcoord = data_profile[:, 0]
        ycoord = data_profile[:, 1]
        n_coord = len(xcoord)



        xcoord_sum = np.zeros(n_coord)



        # Compute cumulative distance
        for i in range(1, n_coord):
            xcoord_sum[i] = xcoord_sum[i - 1] + abs(xcoord[i] - xcoord[i - 1])


        cs = CubicSpline(xcoord_sum, ycoord)#, bc_type="natural")  # 'natural' = second derivative zero at ends
        ysplined = cs(xcoord_sum, 2)
        y_coord_ans=np.zeros(2*self.n_div)
        xreq=np.zeros(2*self.n_div)
        xreq[:self.n_div]=self.x/self.chord
        y_coord_ans[:self.n_div] = cs(xreq[:self.n_div])

        xreq[self.n_div:]=self.x[self.n_div-1]/self.chord+self.x/self.chord
        y_coord_ans[self.n_div:] = cs(xreq[self.n_div:])


        cam=np.zeros(self.n_div)
        cam=(y_coord_ans[:self.n_div][::-1]+y_coord_ans[self.n_div:])/2
        cam=cam*self.chord
        cam_slope=np.zeros(self.n_div)
        cam_slope[0]=(cam[1]-cam[0])/(self.x[1]-self.x[0])
        cam_slope[1:]=(cam[1:]-cam[:-1])/(self.x[1:]-self.x[:-1])
        if plot:

            plt.figure(figsize=(6, 6))
            plt.plot(xcoord,ycoord,'k-', label='Airfoil Profile')
            plt.axis('equal')
            plt.plot(xreq[self.n_div:]-1, y_coord_ans[self.n_div:], 'bo', label='interpolated',markersize=2)
            plt.plot(xreq[:self.n_div], y_coord_ans[:self.n_div][::-1], 'bo', label='interpolated',markersize=2)
            plt.plot(self.x, cam, 'g-', label='camber')
            plt.legend()
            plt.savefig('camber_slope.png', dpi=300)
            plt.show()

        return cam, cam_slope

    def calc_downwash_boundcirc(self):




        uind=np.zeros((1,self.n_div))
        wind=np.zeros((1,self.n_div))

        # Compute wake induced velocity
        x_tev = self.tev[:, 1]               # shape (n_tev,)
        z_tev = self.tev[:, 2]               # shape (n_tev,)

        x_bound = self.bound_vortex_pos[:, 1]  # shape (n_bound_vortex_pos,)
        z_bound = self.bound_vortex_pos[:, 2]  # shape (n_bound_vortex_pos,)
        xdist_TEV_Bound = x_tev[:, None] - x_bound[None, :]   # shape (n_tev, n_bound_vortex_pos)
        zdist_TEV_Bound = z_tev[:, None] - z_bound[None, :]
        dist=xdist_TEV_Bound**2+zdist_TEV_Bound**2
        Gamma=(self.tev[:,0]).reshape(1,-1)

        Ustar=(-zdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind=uind+Gamma@Ustar

        wind=wind-Gamma@Wstar
        # Compute lev induced velocity
        x_lev = self.lev[:, 1]               # shape (n_lev,)
        z_lev = self.lev[:, 2]               # shape (n_lev,)

        x_bound = self.bound_vortex_pos[:, 1]  # shape (n_bound_vortex_pos,)
        z_bound = self.bound_vortex_pos[:, 2]  # shape (n_bound_vortex_pos,)

        xdist_LEV_Bound = x_lev[:, None] - x_bound[None, :]   # shape (n_lev, n_bound_vortex_pos)
        zdist_LEV_Bound = z_lev[:, None] - z_bound[None, :]   # shape (n_lev, n_bound_vortex_pos)

        dist=xdist_LEV_Bound**2+zdist_LEV_Bound**2
        Gamma=(self.lev[:,0]).reshape(1,-1)
        Ustar=(-zdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))


        uind=uind+Gamma@Ustar
        wind=wind-Gamma@Wstar
        # Compute the downwash
        downwash=(-self.u[self.i_step]*np.sin(self.alpha[self.i_step]))+\
            (-uind*np.sin(self.alpha[self.i_step]))+\
            (self.hdot[self.i_step]*np.cos(self.alpha[self.i_step]))+\
            (-wind*np.cos(self.alpha[self.i_step]))+\
            (-self.alphadot[self.i_step]*(self.x-self.pvt*self.chord))+\
            (self.cam_slope*((uind*np.cos(self.alpha[self.i_step]))+(self.u[self.i_step]*np.cos(self.alpha[self.i_step]))+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step]))+(-wind*np.sin(self.alpha[self.i_step]))))


        aterm0=self.W_trapz_integration.reshape(1, -1) @ downwash.reshape(-1, 1)*self.dtheta  # Trapezoidal integration for aterm0
        aterm1=self.W_trapz_integration.reshape(1, -1) @ (downwash * np.cos(self.theta)).reshape(-1, 1) * self.dtheta  # Trapezoidal integration for aterm1

        aterm0=(-1./(self.u_ref*np.pi))*np.squeeze(aterm0)
        aterm1=(2./(self.u_ref*np.pi))*np.squeeze(aterm1)
        bound_circ=self.u_ref*self.chord*np.pi*(aterm0+(aterm1/2.))





        return aterm0, aterm1, downwash,bound_circ,uind,wind


    def one_D_tev_shedding(self):
        # Perform tev shedding assuming LEV is not formed.
        #TEV shed at every time step
        tev_iter=np.zeros(101)
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
            kelv[iter]=kelv[iter]+bound_circ
            if (abs(kelv[iter])<self.eps) :
                break
            dkelv=(kelv[iter]-kelv[iter-1])/(tev_iter[iter]-tev_iter[iter-1])
            tev_iter[iter+1]=tev_iter[iter]-(kelv[iter]/dkelv)
        if (iter>=self.iter_max):
            print('1D iteration failed, the residual is ', abs(kelv[iter]))
        return downwash,aterm0,aterm1,bound_circ,uind,wind

    def two_D_lev_tev_shedding(self,le_vel_x,le_vel_y,lesp):

        tev_iter=np.zeros(101)
        lev_iter=np.zeros(101)
        kelv=np.zeros(100)

        kutta=np.zeros(100)

        #2D iteration if LESP_crit is exceeded



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
        self.lev=np.concatenate((self.lev, np.array([[0, x_lev, y_lev]])), axis=0)
        self.levflag=1


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


        return aterm0, aterm1, downwash, bound_circ,uind,wind

    def wake_rollup(self,bound_int):
                # Update Tev numbers
        self.n_tev=self.n_tev+1

        uind_tev=np.zeros(self.n_tev) # Vitesse induite sur les TEV

        wind_tev=np.zeros(self.n_tev)

        x = self.tev[:, 1]
        z = self.tev[:, 2]
        xdist_TEV_TEV = x[:, None] - x[None, :]  # shape (n_tev, n_tev)
        zdist_TEV_TEV = z[:, None] - z[None, :]
        dist=xdist_TEV_TEV**2+zdist_TEV_TEV**2
        Gamma=(self.tev[:,0]).reshape(1,-1)
        Ustar=(-zdist_TEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_tev=uind_tev+Gamma@Ustar
        wind_tev=wind_tev-Gamma@Wstar

        # LEV induced velocity on TEV
        x_lev = self.lev[:, 1]  # shape (n_lev,)
        z_lev = self.lev[:, 2]  # shape (n_lev,)

        x_tev = self.tev[:, 1]  # shape (n_tev,)
        z_tev = self.tev[:, 2]  # shape (n_tev,)

        # Broadcasting pour différence entre chaque lev et chaque tev
        xdist_LEV_TEV = x_lev[:, None] - x_tev[None, :]  # shape (n_lev, n_tev)
        zdist_LEV_TEV = z_lev[:, None] - z_tev[None, :]  # shape (n_lev, n_tev)
        dist=xdist_LEV_TEV**2+zdist_LEV_TEV**2
        Gamma=(self.lev[:,0]).reshape(1,-1)


        Ustar=(-zdist_LEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_TEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind_tev=uind_tev+Gamma@Ustar# Warning sum is to be done in the appropriate direction +multiplication issue
        wind_tev=wind_tev-Gamma@Wstar #Warning sum is to be done in the appropriate direction +multiplication issue

        #Profile sur TEV
        x_bound = bound_int[:, 1]  # shape (n_bound_int,)
        z_bound = bound_int[:, 2]  # shape (n_bound_int,)

        x_tev = self.tev[:, 1]     # shape (n_tev,)
        z_tev = self.tev[:, 2]     # shape (n_tev,)

        bound_int_xdist = x_bound[:, None] - x_tev[None, :]  # shape (n_bound_int, n_tev)
        bound_int_zdist = z_bound[:, None] - z_tev[None, :]  # shape (n_bound_int, n_tev)

        dist=bound_int_xdist**2+bound_int_zdist**2
        Gamma=(bound_int[:,0]).reshape(1,-1)

        Ustar=(-bound_int_zdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-bound_int_xdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        #print(Gamma.shape,Ustar.shape)

        uind_tev=uind_tev+Gamma@Ustar
        wind_tev=wind_tev-Gamma@Wstar

        # Vitesse induite sur les LEV

        uind_lev=np.zeros(self.n_lev)
        wind_lev=np.zeros(self.n_lev)


        #Vitesse LEV sur LEV

        x_lev = self.lev[:, 1]  # shape (n_lev,)
        z_lev = self.lev[:, 2]  # shape (n_lev,)

        xdist_LEV_LEV = x_lev[:, None] - x_lev[None, :]  # shape (n_lev, n_lev)
        zdist_LEV_LEV = z_lev[:, None] - z_lev[None, :]  # shape (n_lev, n_lev)
        Gamma=(self.lev[:,0]).reshape(1,-1)

        dist=xdist_LEV_LEV**2+zdist_LEV_LEV**2
        Ustar=(-zdist_LEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar

        #Vitesse TEV sur LEV
        x_tev = self.tev[:, 1]  # (n_tev,)
        z_tev = self.tev[:, 2]  # (n_tev,)

        x_lev = self.lev[:, 1]  # (n_lev,)
        z_lev = self.lev[:, 2]  # (n_lev,)

        xdist_TEV_LEV = x_tev[:, None] - x_lev[None, :]  # (n_tev, n_lev)
        zdist_TEV_LEV = z_tev[:, None] - z_lev[None, :]
        dist=xdist_TEV_LEV**2+zdist_TEV_LEV**2
        Gamma=(self.tev[:,0]).reshape(1,-1)

        Ustar=(-zdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar

        x_bound_int = bound_int[:, 1]  # (n_bound_int,)
        z_bound_int = bound_int[:, 2]  # (n_bound_int,)

        x_lev = self.lev[:, 1]         # (n_lev,)
        z_lev = self.lev[:, 2]         # (n_lev,)

        bound_int_xdist = x_bound_int[:, None] - x_lev[None, :]  # (n_bound_int, n_lev)
        bound_int_zdist = z_bound_int[:, None] - z_lev[None, :]  # (n_bound_int, n_lev)
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
    def compute_forces(self, bound_int, uind, wind, adot0, adot1, adot2, adot3):
                #Load coefficient calculation (nondimensional units)

        cnc=(2*np.pi*((self.u[self.i_step]*np.cos(self.alpha[self.i_step])/self.u_ref)+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step])/self.u_ref))*(self.aterm[0]+self.aterm[1]/2))
        cnnc=(2*np.pi*((3*self.chord*adot0/(4*self.u_ref))+(self.chord*adot1/(4*self.u_ref))+(self.chord*adot2/(8*self.u_ref))))
        cs=2*np.pi*self.aterm[0]*self.aterm[0]
        #The components of normal force and moment from induced velocities are calulcated in dimensional units and nondimensionalized later
        # Vectorized computation
        cos_alpha = np.cos(self.alpha[self.i_step])
        sin_alpha = np.sin(self.alpha[self.i_step])

        # Compute the induced velocity term for all divisions
        induced_velocity = (uind[0, 1:] * cos_alpha) - (wind[0, 1:] * sin_alpha)

        # Compute non_l
        non_l = np.sum(induced_velocity * bound_int[:, 0])

        # Compute nonl_m
        nonl_m = np.sum(induced_velocity * self.x[1:] * bound_int[:, 0])


        non_l=non_l*(2/(self.u_ref*self.u_ref*self.chord))
        nonl_m=nonl_m*(2/(self.u_ref*self.u_ref*self.chord*self.chord))



        cn=cnc+cnnc+non_l
        cl=cn*np.cos(self.alpha[self.i_step])+cs*np.sin(self.alpha[self.i_step])
        cd=cn*np.sin(self.alpha[self.i_step])-cs*np.cos(self.alpha[self.i_step])


        cm=cn*self.cm_pvt-(2*np.pi*(((self.u[self.i_step]*np.cos(self.alpha[self.i_step])/self.u_ref)+(self.hdot[self.i_step]*np.sin(self.alpha[self.i_step])/self.u_ref))*((self.aterm[0]/4)+(self.aterm[1]/4)-(self.aterm[2]/8))+(self.chord/self.u_ref)*((7*adot0/16)+(3*adot1/16)+(adot2/16)-(adot3/64))))-nonl_m



        return cl, cd, cm, cn
    def theodorsen_loads(self,k,alpha0,h0,Phi):
        b=self.chord/2
        xcg=self.pvt*self.chord - b
        a= xcg/b #Non dimensional distance of the center of mass from the leading edge
        z= -h0*np.exp(1j*2*self.u_ref*k/self.chord*self.time-Phi) #Non dimensional height of the airfoil
        z_dot= 1j*2*self.u_ref*k/self.chord*z#Non dimensional height derivative of the airfoil
        z_second= 1j*2*self.u_ref*k/self.chord*z_dot#Non dimensional height second derivative of the airfoil

        alpha=alpha0*np.exp(1j*2*self.u_ref*k/self.chord*self.time)#-5*np.pi/180 #2*uvlm.U_inf*frr/uvlm.cw*t
        alpha_dot=1j*2*self.u_ref*k/self.chord*alpha
        alpha_second=1j*2*self.u_ref*k/self.chord*alpha_dot

        print("a",a)
        print('pvt', self.pvt)
        H1 = hankel2(1, k)  # Hankel de 2e espèce, ordre 1
        H0 = hankel2(0, k)  # Hankel de 2e espèce, ordre 0
        C=H1 / (H1 + 1j * H0)
        #Cl=2*np.pi*(alpha+z_dot/self.u_ref+(0.5-a)*alpha_dot*b/self.u_ref)*C+np.pi*(z_second*0.5/self.u_ref**2+alpha_dot*b/self.u_ref-a*alpha_second*(b/self.u_ref)**2)##Cl brunton
        # lift_theo=self.rho*(b)**2*(self.u_ref*np.pi*self.alphadot+np.pi*-self.h_second-np.pi*b*a*self.alpha_second)+2*np.pi*self.rho*(self.chord/2)*self.u_ref*(self.u_ref*self.alpha-self.hdot+self.chord/2*(0.5-a)*self.alphadot)* C

        # lift_theo=lift_theo.imag
        # cl= lift_theo/(0.5*self.rho*self.u_ref**2*self.chord)
        #m_theo=-self.rho*(b)**2*(-a*np.pi*b*-z_second+np.pi*b**2*(1/8+a**2)*alpha_second+np.pi*(0.5-a)*self.u_ref*b*alpha_dot+2*np.pi*self.rho*b**2*(1/2+a)*self.u_ref*(self.u_ref*alpha-z_dot+self.chord/2*(0.5-a)*alpha_dot)* C)
        #m_theo=m_theo.imag
        
        term1 = 2 * np.pi * self.rho * self.u_ref * b * (self.u_ref * alpha + z_dot + (0.5 - a) * b * alpha_dot) * C
        term2 = self.rho * np.pi * b**2 * (self.u_ref * alpha_dot + z_second - a * b * alpha_second)
        
        l = (term1 + term2).imag
        Cl = l / (0.5 * self.rho * self.u_ref**2 * self.chord)
        
        
        
        
        term1 = 2 * np.pi * self.rho * self.u_ref * b**2 * (0.5 + a) * (self.u_ref * alpha + z_dot + (0.5 - a) * b * alpha_dot) * C

        # Second term
        term2 = self.rho * np.pi * b**3 * (a * z_second - (0.5 - a) * self.u_ref * alpha_dot - (1/8 + a**2) * b * alpha_second)
        m=(term1 + term2).imag
        cm= m/(0.5*self.rho*self.u_ref**2*self.chord**2)

        # cp_lift=1/self.period/self.n_period*trapezoid(lift_theo*-self.hdot,dx=period/self.ppp)



        # cp_lift=cp_lift/(0.5 * self.rho * self.u_ref**3 * self.chord)



        # cp_mom=1/self.period/self.n_period*trapezoid(m_theo*self.alphadot,dx=period/self.ppp)


        # cp_mom=cp_mom/(0.5 * self.rho * self.u_ref**3 * self.chord**2)

        # print('Theodorsen Cp lift',cp_lift)
        # print('Theodorsen Cp moment',cp_mom)
        print('C,' , C)


        # plt.figure(figsize=(10, 6))
        # plt.plot(self.time, cl, label='Lift Theodorsen')
        # plt.savefig('lift_theodorsen.png', dpi=300)
        return self.time,alpha.imag,z.imag,Cl,cm

    def make_parameterized_motions(self,k,h0,alpha0,phi,save=False,n_period=2):
        # Create a parameterized motion for the airfoil
        self.n_period=n_period
    

    

    
        omega=2*self.u_ref*k/self.chord
        period=2*np.pi/omega
        D=period*self.u_ref
        d_wake=1./self.n_div*self.chord
        ppp=int(D/d_wake)
        self.period=period
        self.ppp=ppp
        print('period', period)
        self.time=np.linspace(0, n_period*period, n_period*ppp)

        self.dt=self.time[1]-self.time[0]
        self.alpha=alpha0*np.sin(omega*self.time)
        self.h=h0*np.sin(omega*self.time-phi)




        self.alphadot=omega*alpha0*np.cos(omega*self.time)
        self.hdot=omega*h0*np.cos(omega*self.time-phi)


        self.alpha_second=-omega**2*alpha0*np.sin(omega*self.time)
        self.h_second=-omega**2*h0*np.sin(omega*self.time-phi)


        self.u=self.u_ref*np.ones_like(self.time)
        
        print(save)
        if save:
            data_save=np.array([self.time*self.chord/self.u_ref, self.alpha*180/np.pi, self.h/self.chord, np.ones(n_period*ppp)]).T
            np.savetxt('motion_data_alpha_0_{}_h0_{}_k_{}_phi_{}_ppp_{}.dat'.format(alpha0*180/np.pi,h0,k,phi,ppp), data_save)
        # ff
        # print("omega = {}, period = {}, dt = {}".format(omega, period, self.dt))

    def step(self):

        # Perform a single step of the computation
        self.i_step+=1
        # if self.i_step%1000 ==0 :
        #     print("Step: {}, number of lev {}, number of tev {}, time {},tmax ={}".format(self.i_step, self.n_lev, self.n_tev,self.time[self.i_step],self.time[-1]))
        #Calculate bound vortex positions at this time step
        self.dist_wind=self.dist_wind+(self.u[self.i_step-1]*(self.time[self.i_step]-self.
        time[self.i_step-1]))

        self.bound_vortex_pos[:,1]=-((self.chord-self.pvt*self.chord)+((self.pvt*self.chord-self.x)*np.cos(self.alpha[self.i_step]))+self.dist_wind) + (self.cam*np.sin(self.alpha[self.i_step]))
        self.bound_vortex_pos[:,2]=self.h[self.i_step]+((self.pvt*self.chord-self.x)*np.sin(self.alpha[self.i_step]))+(self.cam*np.cos(self.alpha[self.i_step]))
        # print("Step: {}, position profgile {}".format(self.i_step, self.bound_vortex_pos[0,2]))
        # input('dd')

        downwash,aterm0,aterm1,bound_circ,uind,wind=self.one_D_tev_shedding()

        #Comupte the fourier terms

        self.aterm[2]=self.W_trapz_integration.reshape(1,-1)@(downwash*np.cos(2*self.theta)).reshape(-1,1)*self.dtheta

        self.aterm[3]=self.W_trapz_integration.reshape(1,-1)@(downwash*np.cos(3*self.theta)).reshape(-1,1)*self.dtheta
        self.aterm[2]=(2./(self.u_ref*np.pi))*self.aterm[2]
        self.aterm[3]=(2./(self.u_ref*np.pi))*self.aterm[3]

        adot0=(aterm0-self.aterm_prev[0])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot1=(aterm1-self.aterm_prev[1])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot2=(self.aterm[2]-self.aterm_prev[2])/(self.time[self.i_step]-self.time[self.i_step-1])
        adot3=(self.aterm[3]-self.aterm_prev[3])/(self.time[self.i_step]-self.time[self.i_step-1])



        le_vel_x=(self.u[self.i_step])-(self.alphadot[self.i_step]*np.sin(self.alpha[self.i_step])*self.pvt*self.chord)+uind[0,0]
        le_vel_y=-(self.alphadot[self.i_step]*np.cos(self.alpha[self.i_step])*self.pvt*self.chord)-(self.hdot[self.i_step])+wind[0,0]
        vmag=np.sqrt(le_vel_x*le_vel_x+le_vel_y*le_vel_y)
        re_le=self.re_ref*vmag/self.u_ref
        lesp=aterm0

        #Shed the TEV and LEV if LESP crit is exceeded
        if (abs(lesp)>self.lesp_crit):
            aterm0, aterm1, downwash, bound_circ,uind,wind=self.two_D_lev_tev_shedding(le_vel_x,le_vel_y,lesp)

        else:
            self.levflag=0



        #To remove any massive starting vortices

        if (self.i_step==1) :
            self.tev[0,0]=0

        #Calculate fourier terms and bound vorticity
        self.aterm[0] = aterm0
        self.aterm[1] = aterm1
        self.aterm[2:] = 0.0


        iaterm=np.arange(2,self.n_aterm).reshape(-1,1)
        cos_=(np.cos(iaterm*self.theta)*downwash[0,:]).T
        self.aterm[2:]=self.W_trapz_integration.reshape(1, -1) @ cos_ * self.dtheta
        self.aterm[2:]=(2./(self.u_ref*np.pi))*self.aterm[2:]
        self.aterm_prev=self.aterm.copy()





        #Calculate bound_vortex strengths
        gamma = np.zeros(self.n_div)
        gamma+=(self.aterm[0]*(1+np.cos(self.theta)))

        for i_aterm in range(1, self.n_aterm):
            gamma+=(self.aterm[i_aterm]*np.sin(i_aterm*self.theta)*np.sin(self.theta))
        bound_int=np.zeros((self.n_div-1,3))
        bound_int[:,0]=((gamma[1:]+gamma[:-1])/2)*self.dtheta
        bound_int[:,1]=(self.bound_vortex_pos[:-1,1]+self.bound_vortex_pos[1:,1])/2
        bound_int[:,2]=(self.bound_vortex_pos[:-1,2]+self.bound_vortex_pos[1:,2])/2
        # Wake Rollup

        self.wake_rollup(bound_int)



        cl, cd, cm,cn= self.compute_forces(bound_int, uind, wind, adot0, adot1, adot2, adot3)


        self.bound_circ_save.append(bound_circ)
        #Remove TEV and LEV if they are too far away
        del_dist=5*self.chord


        if (self.tev[0,1]-self.bound_vortex_pos[-1,1])>del_dist:
                self.kelv_enf=self.kelv_enf+self.tev[0,0]
                self.tev=np.delete(self.tev, 0, axis=0)
                self.n_tev=self.n_tev-1
        # Remove LEV if they are too far away
        if self.n_lev > 0 and (self.lev[0, 1] - self.bound_vortex_pos[-1, 1]) > self.del_dist:
            self.kelv_enf += self.lev[0, 0]
            self.lev = np.delete(self.lev, 0, axis=0)
            self.n_lev -= 1
        return cl, cd, cm, lesp, re_le,cn

    def make_ldvm_animation(self, add_reference=False,file_reference='../LDVM_v2_original.5/flow_pr_amp45_k0.2_le.dat',colorscale=False):
        # Create an animation of the LDVM simulation

        if add_reference:
            if colorscale:
                fig = plt.figure(figsize=(10, 10),tight_layout=True)
                gs = gridspec.GridSpec(200, 200,figure=fig)
                fig.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.02, wspace=0.2, hspace=0.2)
                ax = fig.add_subplot(gs[:80, :])
                ax2 = fig.add_subplot(gs[82:162, :])
                ax3 = fig.add_subplot(gs[170:175, :])
                ax3.set_xlabel('Circulation')
                ax3.set_yticks([])
            else:
                fig, axs = plt.subplots(2,1,figsize=(10, 8),tight_layout=True)
                ax = axs[0]
                ax2 = axs[1]
            ax2.set_xlim(-8, 1)
            ax2.set_ylim(-2, 2)
            ax2.set_yticks([])
            ax2.set_xticks([])
            self.ref_data = np.loadtxt(file_reference)
        else:
            if colorscale:
                fig = plt.figure(figsize=(10, 5),tight_layout=True)
                gs = gridspec.GridSpec(100, 50,figure=fig)
                fig.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.02, wspace=0.2, hspace=0.2)
                ax = fig.add_subplot(gs[:80, :])
                ax3 = fig.add_subplot(gs[85:90, :])
                ax3.set_xlabel('Circulation')

                ax3.set_yticks([])
            else:
                fig, ax = plt.subplots(1,1,figsize=(10, 4),tight_layout=True)
        ax.set_xlim(-8, 1)
        ax.set_ylim(-2, 2)
        ax.set_yticks([])
        ax.set_xticks([])
        bound_vortex_line, = ax.plot([], [], 'k-')
        if colorscale:
            cmap = plt.get_cmap('coolwarm')
            norm = Normalize(vmin=-0.05, vmax=0.05)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Only needed for colorbar
            lev_line = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=2)
            tev_line = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=2)
            if add_reference:
                lev_line_ref = ax2.scatter([], [], c=[], cmap=cmap, norm=norm, s=2)
                tev_line_ref = ax2.scatter([], [], c=[], cmap=cmap, norm=norm, s=2)
                bound_vortex_line_ref, = ax2.plot([], [], 'k-')
            colorbar = cbar.ColorbarBase(ax3,norm=norm,orientation='horizontal',cmap=cmap)
            ax3.set_xlabel('Circulation (m$^2$/s)')

        else:
            lev_line, = ax.plot([], [], 'ro', markersize=2)
            tev_line, = ax.plot([], [], 'bo', markersize=2)
            if add_reference:
                lev_line_ref, = ax2.plot([], [], 'ro', markersize=2)
                tev_line_ref, = ax2.plot([], [], 'bo', markersize=2)
                bound_vortex_line_ref, = ax2.plot([], [], 'k-')
        def init():
            bound_vortex_line.set_data([], [])
            if colorscale:

                lev_line.set_offsets(np.empty((0, 2)))
                tev_line.set_offsets(np.empty((0, 2)))
                if add_reference:
                    lev_line_ref.set_offsets(np.empty((0, 2)))
                    tev_line_ref.set_offsets(np.empty((0, 2)))

            else:
                lev_line.set_data([], [])
                tev_line.set_data([], [])

                if add_reference:
                    lev_line_ref.set_data([], [])
                    tev_line_ref.set_data([], [])
                    bound_vortex_line_ref.set_data([], [])


                    return lev_line, tev_line, bound_vortex_line, lev_line_ref, tev_line_ref, bound_vortex_line_ref
            return lev_line, tev_line, bound_vortex_line

        def update(frame):

            self.step()  # Perform a step to update the state
            if colorscale:
                lev_line.set_offsets(np.c_[self.lev[:, 1], self.lev[:, 2]])
                if self.lev.shape[0] > 0:
                    lev_line.set_array(self.lev[:, 0])  # Color by circulation


                # Update TEV points and colors
                tev_line.set_offsets(np.c_[self.tev[:, 1], self.tev[:, 2]])
                if self.tev.shape[0] > 0:
                    tev_line.set_array(self.tev[:, 0])  # Color by circulation

                if add_reference:
                    nan_rows = np.where(np.isnan(self.ref_data).all(axis=1))[0]
                    dat=self.ref_data[nan_rows[0]+1:nan_rows[1],:]
                    lev_line_ref.set_offsets(np.c_[dat[:self.n_lev, 1], dat[:self.n_lev, 2]])
                    if dat[:self.n_lev, 0].size > 0:
                        lev_line_ref.set_array(dat[:self.n_lev, 0])
                    tev_line_ref.set_offsets(np.c_[dat[self.n_lev:self.n_lev+self.n_tev, 1], dat[self.n_lev:self.n_lev+self.n_tev, 2]])
                    if dat[self.n_lev:self.n_lev+self.n_tev, 0].size > 0:
                        tev_line_ref.set_array(dat[self.n_lev:self.n_lev+self.n_tev, 0])
                    bound_vortex_line_ref.set_data(dat[-69:, 1], dat[-69:, 2])
                    self.ref_data = self.ref_data[nan_rows[1]:,:]  # Update
            else:
                lev_line.set_data(self.lev[:, 1], self.lev[:, 2])
                tev_line.set_data(self.tev[:, 1], self.tev[:, 2])
                if add_reference:
                    nan_rows = np.where(np.isnan(self.ref_data).all(axis=1))[0]
                    dat=self.ref_data[nan_rows[0]+1:nan_rows[1],:]
                    lev_line_ref.set_data(dat[:self.n_lev, 1], dat[:self.n_lev, 2])
                    tev_line_ref.set_data(dat[self.n_lev:self.n_lev+self.n_tev, 1], dat[self.n_lev:self.n_lev+self.n_tev, 2])
                    bound_vortex_line_ref.set_data(dat[-69:, 1], dat[-69:, 2])
                    self.ref_data = self.ref_data[nan_rows[1]:,:]  # Update ref_data to the next segment

            bound_vortex_line.set_data(self.bound_vortex_pos[:, 1], self.bound_vortex_pos[:, 2])
            print("frame",frame)
            if True:
                ax.text(
                0.95, 0.95, r"$t^*$ = {:.2f}".format(self.time[frame]),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            return lev_line, tev_line, bound_vortex_line

        ani = FuncAnimation(fig, update, frames=299, init_func=init, blit=True)
        #ani.save('ldvm_animation.mp4', writer='ffmpeg', fps=20)
        ani.save('ldvm_animation.gif', writer='pillow', fps=20)
        #plt.show()
   # def theodorsen_loads(self):
    def compute_thrust_efficiency(self,cl,cd,cm,k,ppp):
        # Compute thrust and efficiency

        omega=2*self.u_ref*k/self.chord
        period=2*np.pi/omega

        print('period =', period)

        drag= cd[-ppp:] * self.rho * self.u_ref**2 * self.chord / 2
        lift = cl[-ppp:]* self.rho * self.u_ref**2 * self.chord / 2
        moment = cm[-ppp:]*1/2 * self.rho * self.u_ref**2 * self.chord**2

        input_power= 1/period*trapezoid(-(lift[:] * self.hdot[-ppp:]) +(moment[:] * self.alphadot[-ppp:]),dx=period/ppp)
        propulsion_force = 1/period*trapezoid(-drag, dx=period/ppp)

        ct= propulsion_force / (0.5 * self.rho * self.u_ref**2 * self.chord)
        cp = input_power / (0.5 * self.rho * self.u_ref**3 * self.chord)

        print('lift power',1/period/self.n_period*trapezoid(lift * self.hdot[-ppp:] ,dx=period/ppp))
        print('moment power',1/period/self.n_period*trapezoid(moment * self.alphadot[-ppp:],dx=period/ppp))
        print('input power =', input_power)
        print('propulsion force =', propulsion_force)
        print( 'u_ref =', self.u_ref)




        eta= propulsion_force*self.u_ref/input_power
        return eta,ct,cp







if __name__ == "__main__":
    import time
    t_start = time.time()
    # Example usage
    config = {
        'u_ref': 1.0,
        'chord': 1.0,
        'pvt': 0.5,
        'cm_pvt': 0.5,
        'foil_name': 'naca0015_airfoil.dat',
        're_ref': 1100,
        'lesp_crit':50,
        'motion_file_name': 'motion_pr_amp45_k0.2.dat',
        'force_file_name': 'force_pr_amp45_k0.2_le.csv',
        'flow_file_name': 'flow.csv',
        'n_pts_flow': 100,
        'rho':1.225,
        'nu': 1.566e-5,
        'n_div': 140,
    }

    ldvm_instance = ldvm(config)


    #1.23518463  0.79142985 -1.46978111  1.69777453
    #[0.58292331 0.75559468 0.98962615 0.38373175
    a=[1.99368069,  0.75204751, -2.75259722,  1.18676272]
    a=[0.97261823, 0.67152885, 1.20500712, 0.54600165]
    a=[0.94347952, 0.43929832, 2.76788972, 0.06695997]
    a=[0.89093309, 0.65220004, 0.96688769, 0.45459513]
    a=[ 0.99331597, -0.43568051,  0.99158617, -0.63378655]
    a=[ 0.3707096 , -0.26096754, -0.04431886, -0.35439685]
    a=[ 0.42270996,  1.19806915,  0.38464972, -0.48963669]
    a=[ 0.89268608, -0.44271798,  0.49993434, -1.07366876]
    a=[ 0.15918461,  1.21793932,  0.08393507, -0.62190884]
    a=[ 0.5918461,  1.21793932,  0.08393507, -0]
    a=[ 0.96568032, -0.21449621,  0.4993959,   0.45835996]
    a=[ 5.22588681e-01, -8.39874544e-02, -3.74972611e-01, -1.75648481e-04]
    a=[ 6.20782931e-01,  1.09223340e-01, -3.70882902e-01, -2.77944469e-04]
    a= [ 0.25946782, -0.23134507,  0.4999472,  -1.81682425,]
    a=[ 0.10681548, -0.15764927,  0.75057282, -1.67788348]
    a=[0.4,  5*np.pi/180, 0.0000, 0*np.pi/180]




    #St=0.3
    #a=[2.1, 15*np.pi/180, 0.25, 90*np.pi/180] #0.3, 0.75559468, 0.98962615, 0.38373175

    
    k=a[0 ]#0.3#0.05
    alpha0=a[1]#0.75559468 #50*np.pi/180
    h0=a[2] #0.1
    phi=a[3]#0.38373175 #0.
    omega=2*ldvm_instance.u_ref*k/ldvm_instance.chord
    period=2*np.pi/omega
    D=period*ldvm_instance.u_ref

    d_wake=1./ldvm_instance.n_div*ldvm_instance.chord

    ppp=int(D/d_wake)
    #ldvm_instance.load_motion()
    n_period=10
    print('ppp =', ppp)

    ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=phi,ppp=ppp,save=False,n_period=n_period)


    #ldvm_instance.load_motion()
    ldvm_instance.initialize_computation()
    #ldvm_instance.make_ldvm_animation(add_reference=False,colorscale=True)






    cl_history = [0]
    cd_history = [0]
    cm_history = [0]
    cn_history = [0]
    alpha_theo,h_theo,cl_theodorsen,cm_theodorsen=ldvm_instance.theodorsen_loads(k,alpha0)
    for i in range(n_period*ppp-1):
        #print(ldvm_instance.alpha[:i]*180/np.pi)
        cl, cd, cm, lesp, re_le,cn=ldvm_instance.step()
        cl_history.append(cl)
        cd_history.append(cd)
        cm_history.append(cm)  # Assuming cm is not calculated in this example
        cn_history.append(cn)  # Assuming cn is not calculated in this example
    eta,ct,cp= ldvm_instance.compute_thrust_efficiency(np.array(cl_history), np.array(cd_history), np.array(cm_history), k, ppp)

    print('eta =', eta)
    print('ct =', ct)
    print('cp =', cp)

    print('N_LEV =', ldvm_instance.n_lev)

    #ldvm_instance.make_ldvm_animation(add_reference=True)
    t_end = time.time()
    print('Total time for {} steps:'.format(ldvm_instance.i_step), t_end - t_start, 'seconds')


    #data_loads=np.column_stack((ldvm_instance.time[:ldvm_instance.i_step],ldvm_instance.bound_circ_save[:],cl_history[:],cd_history[:], cm_history[:]))
    #np.savetxt('data_base/force_data_alpha_0_{}_h0_{}_k_{}_phi_{}_ppp_{}.dat'.format(alpha0*180/np.pi,h0,k,phi,ppp), data_loads, header='time bound_circ lift drag moment', fmt='%f %f %f %f %f')

    #data=np.loadtxt('../LDVM_v2_original.5/data_base/force_data_alpha_0_{}_h0_{}_k_{}_phi_{}_ppp_{}.dat'.format(alpha0*180/np.pi,h0,k,phi,ppp),skiprows=1)

    # gamma_lit=data[:,4]
    # cl_lit=data[:,8]
    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(ldvm_instance.time,ldvm_instance.alpha*180/np.pi,'r-',label='my LDVM',markersize=2)
    ax.set_xlabel('time')
    ax.set_ylabel('angle of attack (deg)')
    ax2 = ax.twinx()
    ax2.plot(ldvm_instance.time, ldvm_instance.h, 'b--', label='Height', markersize=2)
    ax2.set_ylabel('Height (m)')
    fig.savefig('angle_height.png', dpi=300)
    plt.figure()
    plt.plot(ldvm_instance.time[1:ldvm_instance.i_step+1],ldvm_instance.bound_circ_save,'r-',label='my LDVM',markersize=2)

    #plt.plot(data[-ppp:,0],gamma_lit[-ppp:],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('bound circulation')
    plt.legend()
    plt.savefig('bound_circ.png', dpi=300)
    plt.figure()

    plt.plot(ldvm_instance.time[-ppp:],cl_history[-ppp:],'r-',label='my LDVM',markersize=2)
    #plt.plot(ldvm_instance.time[1:ldvm_instance.i_step+1],cn_history,'b-',label='my LDVM',markersize=2)
    plt.plot(ldvm_instance.time[-ppp:],cl_theodorsen[-ppp:],'g--',label='Theodorsen lift',markersize=2)


   # plt.plot(data[-ppp:,0],cl_lit[-ppp:],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('cl')
    plt.legend()
    plt.savefig('lift.png', dpi=300)





    plt.figure()

    plt.plot(ldvm_instance.alpha[-ppp:]*180/np.pi,cl_history[-ppp:],'r-',label='my LDVM',markersize=2)
    #plt.plot(ldvm_instance.time[1:ldvm_instance.i_step+1],cn_history,'b-',label='my LDVM',markersize=2)
    plt.plot(alpha_theo*180/np.pi,cl_theodorsen,'g--',label='Theodorsen lift',markersize=2)


    #plt.plot(ldvm_instance.alpha[-ppp:],cl_lit[-ppp:],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('cl')
    plt.legend()
    plt.savefig('lift_hysteresis.png', dpi=300)

    plt.figure()
    plt.figure()
    plt.plot(ldvm_instance.time[-ppp:],cn_history[-ppp:],'r-',label='my LDVM',markersize=2)
    plt.plot(ldvm_instance.time[-ppp:],ldvm_instance.h[-ppp:],'b-',label='h',markersize=2)
    plt.plot(ldvm_instance.time[-ppp:],ldvm_instance.alpha[-ppp:],'b--',label='alpha',markersize=2)

    plt.xlabel('time')
    plt.ylabel('cn')
    plt.legend()
    plt.savefig('normal_force.png', dpi=300)
    plt.figure()
    plt.plot(ldvm_instance.time[-ppp:],cd_history[-ppp:],'r-',label='my LDVM',markersize=2)
    #plt.plot(ldvm_instance.time[-ppp:],np.sin(ldvm_instance.alpha[1:])*np.array(cn_history),'g-',label='cnsin(alpha)',markersize=2)
    #plt.plot(data[-ppp:,0],data[-ppp:,9],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('cd')
    plt.legend()
    plt.savefig('drag.png', dpi=300)
    plt.figure()
    plt.plot(ldvm_instance.time[-ppp:],cm_history[-ppp:],'r-',label='my LDVM',markersize=2)
    plt.plot(ldvm_instance.time[-ppp:],cm_theodorsen[-ppp:],'g--',label='Theodorsen moment',markersize=2)
    #plt.plot(data[-ppp:,0],data[-ppp:,10],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('cm')
    plt.legend()
    plt.savefig('moment.png', dpi=300)
    plt.show()


    plt.figure()
    plt.plot(ldvm_instance.alpha[-ppp:],cm_history[-ppp:],'r-',label='my LDVM',markersize=2)
    plt.plot(ldvm_instance.alpha[-ppp:],cm_theodorsen[-ppp:],'g--',label='Theodorsen moment',markersize=2)
    #plt.plot(data[-ppp:,0],data[-ppp:,10],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('cm')
    plt.legend()
    plt.savefig('moment_hysteresis.png', dpi=300)
    plt.show()




    plt.figure()
    plt.plot(ldvm_instance.time[-ppp:],cl_history[-ppp:]*ldvm_instance.hdot[-ppp:],'b-')
    plt.plot(ldvm_instance.time[-ppp:],cl_theodorsen[-ppp:]*ldvm_instance.hdot[-ppp:],'r-')
    plt.xlabel('time')
    plt.ylabel('power lift * velocity')
    plt.legend()
    plt.savefig('power_lift.png', dpi=300)
    plt.figure()

    plt.figure()
    plt.scatter(ldvm_instance.time[-ppp:],cm_history[-ppp:]*ldvm_instance.alphadot[-ppp:])
    plt.xlabel('time')
    plt.ylabel('power moment * velocity')
    plt.legend()
    plt.savefig('power_moment.png', dpi=300)



    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(ldvm_instance.time,ldvm_instance.alphadot*180/np.pi,'r-',label='my LDVM',markersize=2)
    ax.set_xlabel('time')
    ax.set_ylabel('angle of attack derivatice (deg)')
    ax2 = ax.twinx()
    ax2.plot(ldvm_instance.time, ldvm_instance.hdot, 'b--', label='Height', markersize=2)
    ax2.set_ylabel('Height derivative (m)')
    fig.savefig('angle_height_derivatives.png', dpi=300)



