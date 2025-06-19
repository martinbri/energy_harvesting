import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import pandas
from scipy.integrate import trapezoid
import time
import matplotlib.colorbar as cbar
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
        self.n_div=70 # No. of divisions along chord on airfoil
        self.n_aterm=45 #Number of fourier terms used to compute vorticity at a location on chord
        self.del_dist=10.00
        self.iter_max=100
        self.kelv_enf=0.0
        if config is not None:
            self.u_ref=np.float64(config['u_ref'])
            self.chord=np.float64(config['chord'])
            self.pvt=np.float64(config['pvt'])
            self.rho=np.float64(config['rho'])
            self.cm_pvt=np.float64(config['cm_pvt'])
            self.foil_name=config['foil_name']
            self.re_ref=np.float64(config['re_ref'])
            self.lesp_crit=np.float64(config['lesp_crit'])
            self.motion_file_name=config['motion_file_name']

        else:
            self.u_ref=np.float64(1.0)
            self.chord=np.float64(1.0)
            self.pvt=np.float64(0.25)
            self.rho=np.float64(1.225)
            self.cm_pvt=0.25
            self.foil_name='NACA0012'
            self.re_ref=float(1e6)
            self.lesp_crit=0.18


        #Geometric parameters
        self.dtheta=np.pi/(self.n_div-1)
        self.theta = np.linspace(0, np.pi, self.n_div)
        self.x=(self.chord/2.)*(1-np.cos(self.theta))

        ##Dimmensionalize parameters
        self.v_core=self.v_core*self.chord
        self.del_dist=self.del_dist*self.chord



    def make_parameterized_motions(self,k,h0,alpha0,phi,ppp):
        # Create a parameterized motion for the airfoil
        
        omega=2*self.u_ref*k/self.chord
        period=2*np.pi/omega
        self.time=np.linspace(0, period, ppp)
        self.dt=self.time[1]-self.time[0]
        self.alpha=alpha0*np.sin(omega*self.time)
        self.h=h0*np.sin(omega*self.time-phi) 
        



        self.alphadot=omega*alpha0*np.cos(omega*self.time)
        self.hdot=omega*h0*np.cos(omega*self.time-phi)
        self.hdot=np.diff(self.h)/np.diff(self.time)
       
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

        self.W_trapezoid_n_div=self.make_trapezoid_integration_matrix(n_div=self.n_div).reshape(1,-1)
        
        self.bound_circ_save=[]


        self.cam,self.cam_slope=self.calc_camber_slope()
        
        


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
            plt.show()
        return cam, cam_slope

    def calc_downwash_boundcirc(self,u,alpha,hdot,alphadot):

        uind=np.zeros((1,self.n_div))
        wind=np.zeros((1,self.n_div))
        # Compute wake induced velocity
        xdist_TEV_Bound=np.tile(self.tev[:,1], (len(self.bound_vortex_pos[:,1]), 1)).T- np.tile(self.bound_vortex_pos[:,1], (len(self.tev[:,1]), 1))
        zdist_TEV_Bound=np.tile(self.tev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.tev[:,2]), 1))
        dist=xdist_TEV_Bound**2+zdist_TEV_Bound**2
        Gamma=(self.tev[:,0]).reshape(1,-1)
        Ustar=(-zdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        uind=uind+Gamma@Ustar

        wind=wind-Gamma@Wstar
        # Compute lev induced velocity

        xdist_LEV_Bound=np.tile(self.lev[:,1], (len(self.bound_vortex_pos[:,1]), 1)).T- np.tile(self.bound_vortex_pos[:,1], (len(self.lev[:,1]), 1))
        zdist_LEV_Bound=np.tile(self.lev[:,2], (len(self.bound_vortex_pos[:,2]), 1)).T- np.tile(self.bound_vortex_pos[:,2], (len(self.lev[:,2]), 1))
        dist=xdist_LEV_Bound**2+zdist_LEV_Bound**2
        Gamma=(self.lev[:,0]).reshape(1,-1)
        Ustar=(-zdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_LEV_Bound)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))


        uind=uind+Gamma@Ustar
        wind=wind-Gamma@Wstar
        # Compute the downwash
        downwash=(-u*np.sin(alpha))+\
            (-uind*np.sin(alpha))+\
            (hdot*np.cos(alpha))+\
            (-wind*np.cos(alpha))+\
            (-alphadot*(self.x-self.pvt*self.chord))+\
            (self.cam_slope*((uind*np.cos(alpha))+(u*np.cos(alpha))+(hdot*np.sin(alpha))+(-wind*np.sin(alpha))))
        #Compute the bound circulation with for loop
        aterm0=0.0
        aterm1=0.0
        for i_div in range (1,self.n_div):
            aterm0=aterm0+(((downwash[0,i_div]+downwash[0,i_div-1])/2)*self.dtheta)
            aterm1=aterm1+(((downwash[0,i_div]*np.cos(self.theta[i_div])+downwash[0,i_div-1]*np.cos(self.theta[i_div-1]))/2)*self.dtheta)
        

        aterm0=(-1./(self.u_ref*np.pi))*aterm0
        aterm1=(2./(self.u_ref*np.pi))*aterm1
        bound_circ=self.u_ref*self.chord*np.pi*(aterm0+(aterm1/2.))

        # aterm0=self.W_trapezoid_n_div@(downwash.T)*self.dtheta      
        # aterm1=self.W_trapezoid_n_div@((downwash.T*np.cos(self.theta.reshape(-1,1))))*self.dtheta
        # aterm0=(-1./(self.u_ref*np.pi))*aterm0
        # aterm1=(2./(self.u_ref*np.pi))*aterm1
        # aterm0 = np.float64(np.squeeze(aterm0))
        # aterm1 = np.float64(np.squeeze(aterm1))
        # bound_circ=self.u_ref*self.chord*np.pi*(aterm0+(aterm1/2.))
        #print('aterm0,aterm1,bound_circ, ',aterm0,aterm1,bound_circ,type(aterm0),type(aterm1),type(bound_circ))
        #input('press enter to continue')

        return aterm0, aterm1, downwash,bound_circ,uind,wind

    def make_trapezoid_integration_matrix(self,n_div):
        W=np.ones(n_div)
        W[0]=0.5
        W[-1]=0.5
        return W
    def one_D_tev_shedding(self,t,t_minus_1,u,alpha,hdot,alphadot):
        # Perform tev shedding assuming LEV is not formed.
        #TEV shed at every time step
        tev_iter=np.zeros(101)
        kelv=np.zeros(100)
        tev_iter[0]=0
        tev_iter[1]=-0.01

        if self.n_tev==0:
            x_tev=self.bound_vortex_pos[self.n_div-1,1]+0.5*u*(t-t_minus_1)
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
            aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc(u,alpha,hdot,alphadot)
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

    def two_D_lev_tev_shedding(self,le_vel_x,le_vel_y,lesp,t,t_minus_1,u,alpha,hdot,alphadot):

        tev_iter=np.zeros(102)
        lev_iter=np.zeros(102)
        kelv=np.zeros(101)

        kutta=np.zeros(101)

        #2D iteration if LESP_crit is exceeded
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
            x_lev=self.bound_vortex_pos[0,1]+(0.5*le_vel_x*(t-t_minus_1))
            y_lev=self.bound_vortex_pos[0,2]+(0.5*le_vel_y*(t-t_minus_1))
        else:
            x_lev=self.bound_vortex_pos[0,1]+((1./3.)*(t-t_minus_1))
            y_lev=self.bound_vortex_pos[0,2]+((1./3.)*(t-t_minus_1))
        self.lev=np.concatenate((self.lev, np.array([[0, x_lev, y_lev]])), axis=0)
        self.levflag=1


        iter =0
        while (iter<self.iter_max):


            iter=iter+1

            #Advancing with tev strength
            self.lev[self.n_lev,0]=lev_iter[iter-1]

            self.tev[self.n_tev,0]=tev_iter[iter]

            aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc(u,alpha,hdot,alphadot)


            kelv_tev=self.kelv_enf
            kelv_tev+=np.sum(self.lev[:,0])
            kelv_tev+=np.sum(self.tev[:,0])
            kelv_tev+=bound_circ


            kutta_tev=aterm0-lesp_cond

            dkelv_tev=(kelv_tev-kelv[iter-1])/(tev_iter[iter]-tev_iter[iter-1])

            dkutta_tev=(kutta_tev-kutta[iter-1])/(tev_iter[iter]-tev_iter[iter-1])

            self.lev[self.n_lev,0]=lev_iter[iter]
            self.tev[self.n_tev,0]=tev_iter[iter-1]

            aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc(u,alpha,hdot,alphadot)
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

            aterm0, aterm1, downwash, bound_circ,uind,wind=self.calc_downwash_boundcirc(u,alpha,hdot,alphadot)
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

    def wake_rollup(self,bound_int,dt):
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

        uind_tev=uind_tev+Gamma@Ustar
        wind_tev=wind_tev-Gamma@Wstar

        # LEV induced velocity on TEV
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

        #print(Gamma.shape,Ustar.shape)

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

        dist=xdist_TEV_LEV**2+zdist_TEV_LEV**2
        Gamma=(self.tev[:,0]).reshape(1,-1)

        Ustar=(-zdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-xdist_TEV_LEV)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar

        bound_int_xdist=np.tile(bound_int[:,1], (len(self.lev[:,1]),1)).T-np.tile(self.lev[:,1],(len(bound_int[:,1]),1))
        bound_int_zdist=np.tile(bound_int[:,2], (len(self.lev[:,2]),1)).T-np.tile(self.lev[:,2],(len(bound_int[:,2]),1))
        dist=bound_int_xdist**2+bound_int_zdist**2

        Gamma=(bound_int[:,0]).reshape(1,-1)
        Ustar=(-bound_int_zdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))
        Wstar=(-bound_int_xdist)/(2*np.pi*np.sqrt(self.v_core**4+dist**2))

        uind_lev=uind_lev+Gamma@Ustar
        wind_lev=wind_lev-Gamma@Wstar


        ##Update TEV and LEV positions
        self.tev[:,1]=self.tev[:,1]+(uind_tev*dt)
        self.tev[:,2]=self.tev[:,2]+(wind_tev*dt)
        self.lev[:,1]=self.lev[:,1]+(uind_lev*dt)
        self.lev[:,2]=self.lev[:,2]+(wind_lev*dt)


        # Cropping LEV and tEV arrays is not done here
    def compute_forces(self, bound_int, uind, wind, adot0, adot1, adot2, adot3,u, alpha, hdot):
        #Load coefficient calculation (nondimensional units)

        cnc=2*np.pi*((u*np.cos(alpha)/self.u_ref)+(hdot*np.sin(alpha)/self.u_ref))*(self.aterm[0]+self.aterm[1]/2)
        cnnc=2*np.pi*((3*self.chord*adot0/(4*self.u_ref))+(self.chord*adot1/(4*self.u_ref))+(self.chord*adot2/(8*self.u_ref)))
        cs=2*np.pi*self.aterm[0]*self.aterm[0]
        #The components of normal force and moment from induced velocities are calulcated in dimensional units and nondimensionalized later
        non_l=0
        nonl_m=0
        #nonl=np.sum(((uind*np.cos(self.alpha[self.i_step]))-(wind*np.sin(self.alpha[self.i_step])))*bound_int[:,0])
        for i_div in range(1,self.n_div):
            non_l=non_l+(((uind[0,i_div]*np.cos(alpha))-(wind[0,i_div]*np.sin(alpha)))*bound_int[i_div-1,0])
            nonl_m=nonl_m+(((uind[0,i_div]*np.cos(alpha))-(wind[0,i_div]*np.sin(alpha)))*(self.x[i_div])*bound_int[i_div-1,0])
        non_l=non_l*(2/(self.u_ref*self.u_ref*self.chord))
        nonl_m=nonl_m*(2/(self.u_ref*self.u_ref*self.chord*self.chord))

        print('cnc, cnnc, cs, non_l, nonl_m', cnc, cnnc, cs, non_l, nonl_m)
        #input('dd')

        cn=cnc+cnnc+non_l
        cl=cn*np.cos(alpha)+cs*np.sin(alpha)
        cd=cn*np.sin(alpha)-cs*np.cos(alpha)
        cm=cn*self.cm_pvt-(2*np.pi*(((u*np.cos(alpha)/self.u_ref)+(hdot*np.sin(alpha)/self.u_ref))*((self.aterm[0]/4)+(self.aterm[1]/4)-(self.aterm[2]/8))+(self.chord/self.u_ref)*((7*adot0/16)+(3*adot1/16)+(adot2/16)-(adot3/64))))-nonl_m

        

        return cl, cd, cm, cn


    def step(self,t,t_minus_1, alpha, h, u, alphadot, hdot):

        print('t', t, 't_minus_1', t_minus_1, 'alpha', alpha, 'h', h, 'u', u, 'alphadot', alphadot, 'hdot', hdot)
        

       
        

        # Perform a single step of the computation
        self.i_step+=1
        print("Step: {}, number of lev {}, number of tev {}, time {}".format(self.i_step, self.n_lev, self.n_tev,t))
        #Calculate bound vortex positions at this time step
        print(self.dist_wind)
        self.dist_wind=self.dist_wind+(u*(t-t_minus_1))
        print('dist_wind, ', self.dist_wind)


        self.bound_vortex_pos[:,1]=-((self.chord-self.pvt*self.chord)+((self.pvt*self.chord-self.x)*np.cos(alpha))+self.dist_wind) + (self.cam*np.sin(alpha))
        self.bound_vortex_pos[:,2]=h+((self.pvt*self.chord-self.x)*np.sin(alpha))+(self.cam*np.cos(alpha))

        downwash,aterm0,aterm1,bound_circ,uind,wind=self.one_D_tev_shedding(t,t_minus_1,u,alpha,hdot,alphadot)

        #Comupte the fourier terms
        self.aterm[2]=0.0
        self.aterm[3]=0.0
        for i_aterm in range(2,4):
            for i_div in range(1, self.n_div):
                self.aterm[i_aterm]= self.aterm[i_aterm]+((((downwash[0,i_div]*np.cos(i_aterm*self.theta[i_div]))+(downwash[0,i_div-1]*np.cos(i_aterm*self.theta[i_div-1])))/2)*self.dtheta)


            self.aterm[i_aterm]=(2./(self.u_ref*np.pi))*self.aterm[i_aterm]
        adot0=(aterm0-self.aterm_prev[0])/(t-t_minus_1)
        adot1=(aterm1-self.aterm_prev[1])/(t-t_minus_1)
        adot2=(self.aterm[2]-self.aterm_prev[2])/(t-t_minus_1)
        adot3=(self.aterm[3]-self.aterm_prev[3])/(t-t_minus_1)



        le_vel_x=(u)-(alphadot*np.sin(alpha)*self.pvt*self.chord)+uind[0,0]
        le_vel_y=-(alphadot*np.cos(alpha)*self.pvt*self.chord)-(hdot)+wind[0,0]
        vmag=np.sqrt(le_vel_x*le_vel_x+le_vel_y*le_vel_y)
        re_le=self.re_ref*vmag/self.u_ref
        lesp=aterm0

        #Shed the TEV and LEV if LESP crit is exceeded
        if (abs(lesp)>self.lesp_crit):
            print("A LEV is formed")
            aterm0, aterm1, downwash, bound_circ,uind,wind=self.two_D_lev_tev_shedding(le_vel_x,le_vel_y,lesp, t, t_minus_1, u, alpha, hdot, alphadot)

        else:
            self.levflag=0



        #To remove any massive starting vortices

        if (self.i_step==1) :
            self.tev[0,0]=0

        #Calculate fourier terms and bound vorticity

        self.aterm[0] = aterm0
        self.aterm[1] = aterm1
        self.aterm[2:] = 0.0
        for i_aterm in range(2, self.n_aterm):
            self.aterm[i_aterm]=np.sum((((downwash[0,1:]*np.cos(i_aterm*self.theta[1:]))+(downwash[0,:-1]*np.cos(i_aterm*self.theta[:-1])))/2))*self.dtheta
            # For loop version
            # for i_div in range(1, self.n_div):

            #     self.aterm[i_aterm]= self.aterm[i_aterm]+((((downwash[0,i_div]*np.cos(i_aterm*self.theta[i_div]))+(downwash[0,i_div-1]*np.cos(i_aterm*self.theta[i_div-1])))/2)*self.dtheta)
            self.aterm[i_aterm]=(2./(self.u_ref*np.pi))*self.aterm[i_aterm]
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

        self.wake_rollup(bound_int,t-t_minus_1)



        cl, cd, cm,cn= self.compute_forces(bound_int, uind, wind, adot0, adot1, adot2, adot3, u, alpha, hdot)


        self.bound_circ_save.append(bound_circ)

        print(cl,cm)

        #Remove TEV and LEV if they are too far away
        del_dist=10*self.chord


        if (self.tev[0,1]-self.bound_vortex_pos[-1,1])>del_dist:
                self.kelv_enf=self.kelv_enf+self.tev[0,0]
                self.tev=np.delete(self.tev, 0, axis=0)
                self.n_tev=self.n_tev-1


        return cl, cd, cm

    def make_ldvm_animation(self, add_reference=False,n_frames=1000,file_reference='../LDVM_v2_original.5/flow_pr_amp45_k0.2_le.dat',colorscale=False):
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
            ax2.set_xlim(-6, 1/2)
            ax2.set_ylim(-1/8,1/8 )
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

            #self.step()  # Perform a step to update the state
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
                    print('dat',dat.shape)
                    
                    #lev_line_ref.set_data(dat[:self.n_lev, 1], dat[:self.n_lev, 2])
                    tev_line_ref.set_data(dat[:-69, 1], dat[:-69, 2])
                    bound_vortex_line_ref.set_data(dat[-69:, 1], dat[-69:, 2])
                    self.ref_data = self.ref_data[nan_rows[1]:,:]  # Update ref_data to the next segment

            bound_vortex_line.set_data(self.bound_vortex_pos[:, 1], self.bound_vortex_pos[:, 2])
            if False:
                ax.text(
                0.95, 0.95, r"$t^*$ = {:.2f}".format(self.time[frame]),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            return lev_line, tev_line, bound_vortex_line

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
        ani.save('ldvm_animation.mp4', writer='ffmpeg', fps=20)
        #plt.show()

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

        self.dt = self.time[1] - self.time[0]




if __name__ == "__main__":
    import time
    t_start = time.time()
    # Example usage
    config = {
        'u_ref': 1,
        'chord': 1.0,
        'pvt': 0.25,
        'rho': 1.225,
        'cm_pvt': 0.25,
        'foil_name': 'sd7012.dat',
        're_ref': 30000,
        'lesp_crit': 0.18,
        'motion_file_name':'motion_pr_amp45_k0.2.dat'
    }

    ldvm_instance = ldvm(config)
    ppp=100
    k=0.5
    alpha0=50*np.pi/180
    h0=0.1
    phi=0.0
    #ldvm_instance.load_motion()
    ldvm_instance.initialize_computation()
    
    #ldvm_instance.make_parameterized_motions(k=k,alpha0=alpha0,h0=h0,phi=phi,ppp=ppp)
    ldvm_instance.load_motion()
    cl_history = []
    cd_history = []
    cm_history = []
    
    t_minus_1=-ldvm_instance.dt

    for (t,alpha,h,alpha_dot,h_dot) in zip(ldvm_instance.time, ldvm_instance.alpha, ldvm_instance.h, ldvm_instance.alphadot, ldvm_instance.hdot):
        #print(ldvm_instance.alpha[:i]*180/np.pi)
        cl, cd, cm,=ldvm_instance.step(t=t,t_minus_1=t_minus_1,alpha=alpha,h=h,u=ldvm_instance.u_ref,alphadot=alpha_dot,hdot=h_dot)
        cl_history.append(cl)
        cd_history.append(cd)
        cm_history.append(cm)  # Assuming cm is not calculated in this example
        t_minus_1=t


    data_save=np.array([ldvm_instance.time*ldvm_instance.chord/ldvm_instance.u_ref, ldvm_instance.alpha*180/np.pi, ldvm_instance.h/ldvm_instance.chord, np.ones(500)]).T
    print(data_save)
    
    np.savetxt('motion_data_alpha_0_{}_h0_{}_k_{}_phi_{}.dat'.format(alpha0*180/np.pi,h0,k,phi), data_save)

    #ldvm_instance.make_ldvm_animation(add_reference=True)
    t_end = time.time()
    print('Total time for {} steps:'.format(ldvm_instance.i_step), t_end - t_start, 'seconds')
    
    data= np.loadtxt('force_data_alpha_0_{}_h0_{}_k_{}_phi_{}.dat'.format(alpha0*180/np.pi,h0,k,phi))
    
    fig, ax = plt.subplots(tight_layout=True)
    
    ax.plot(ldvm_instance.time,ldvm_instance.alpha*180/np.pi,'r-',label='my LDVM',markersize=2)
    ax.set_xlabel('time')
    ax.set_ylabel('angle of attack (deg)')
    ax2 = ax.twinx()
    ax2.plot(ldvm_instance.time, ldvm_instance.h, 'b--', label='Height', markersize=2)
    ax2.set_ylabel('Height (m)')
    fig.savefig('angle_height.png', dpi=300)
    
    
    cl_lit=data[:,8]
    plt.figure()
    plt.plot(ldvm_instance.time[:ldvm_instance.i_step+1],ldvm_instance.bound_circ_save,'r-',label='my LDVM',markersize=2)

    #plt.plot(data[:,0],gamma_lit,'b-',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('bound circulation')
    plt.plot(data[:,0],data[:,4],'b--',label='literature',markersize=2)
    plt.legend()
    plt.savefig('bound_circ.png', dpi=300)
    plt.figure()

    plt.plot(ldvm_instance.time[:ldvm_instance.i_step+1],cl_history,'r-',label='my LDVM',markersize=2)
    plt.plot(data[:,0],cl_lit,'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('lift')
    plt.legend()
    plt.savefig('lift.png', dpi=300)
    

    plt.figure()
    plt.plot(ldvm_instance.time[:ldvm_instance.i_step+1],cd_history,'r-',label='my LDVM',markersize=2)
    plt.plot(data[:,0],data[:,9],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('drag')
    plt.legend()
    plt.savefig('drag.png', dpi=300)
    plt.figure()
    plt.plot(ldvm_instance.time[:ldvm_instance.i_step+1],cm_history,'r-',label='my LDVM',markersize=2)
    plt.plot(data[:,0],data[:,10],'b--',label='literature',markersize=2)
    plt.xlabel('time')
    plt.ylabel('moment')
    
    plt.legend()
    plt.savefig('moment.png', dpi=300)
    
    plt.show()











