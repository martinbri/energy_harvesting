import numpy as np

import matplotlib.pyplot as plt
import os
def generate_naca0015(num_points=500,chord = 1.0):
    """
    Generate the coordinates for a NACA0015 airfoil.
    
    Parameters:
        num_points (int): Number of points to generate along the chord line.
        
    Returns:
        tuple: x and y coordinates of the airfoil.
    """
    # Define the chord length
    

    # Generate x-coordinates along the chord
    x = np.linspace(0, 1, num_points)
    x = chord * (0.5 * (1 - np.cos(np.pi * x)))  # Cosine spacing for higher density near edges

    # Maximum thickness as a fraction of the chord
    t = 0.15

    # Thickness distribution formula for a symmetric airfoil
    y_t = 5 * t * (
        0.2969 * np.sqrt(x/chord) -
        0.1260 * (x/chord) -
        0.3516 * (x/chord)**2 +
        0.2843 * (x/chord)**3 -
        0.1015 * (x/chord)**4
    )

    # Upper and lower surfaces
    x_upper = x
    y_upper = y_t
    x_lower = x
    y_lower = -y_t

    # Combine upper and lower surfaces
    x_coords = np.concatenate([x_upper[::-1], x_lower])
    y_coords = np.concatenate([y_upper[::-1], y_lower])
    
    dat=np.column_stack((x_coords, y_coords))
    
    np.savetxt("naca0015_airfoil.dat", dat)

    return x_coords, y_coords

def plot_airfoil(x, y):
    """
    Plot the airfoil shape.
    
    Parameters:
        x (array): x-coordinates of the airfoil.
        y (array): y-coordinates of the airfoil.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label="NACA0015 Airfoil")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("NACA0015 Airfoil")
    plt.legend()
    plt.grid(True)
    plt.savefig("naca0015_airfoil.png", dpi=300)
    plt.show()


def compare_data_base(data_python_folder,data_fortran_folder=[],k=0.05):
    """
    Compare two datasets and print the differences.
    
    Parameters:
        data_python (array): Data from Python.
        data_fortran (array): Data from Fortran.
    """
    data_python_path = [f"{data_python_folder}/{file}" for file in os.listdir(data_python_folder) if file.endswith(".dat")]
    data_fortran_path = [f"{data_fortran_folder}/{file}" for file in os.listdir(data_fortran_folder) if file.endswith(".dat")]


    data_python_path = [path for path in data_python_path if f'k_{k}' in path]
    data_fortran_path = [path for path in data_fortran_path if f'k_{k}' in path]
    
    # Sort data_python_path by the "ppp_XXX" value in the filename
    
    data_python_path.sort(key=lambda path: int(path.split("ppp_")[1].split(".")[0]))
    data_fortran_path.sort(key=lambda path: int(path.split("ppp_")[1].split(".")[0]))
    
    print(data_python_path)
    print(data_fortran_path)
    fig,ax=plt.subplots(3,1,figsize=(4, 9),tight_layout=True,sharex=True)
    fig_,ax_=plt.subplots(1,figsize=(4, 3),tight_layout=True,sharex=True)
    
    linestyles=['-.', '--', '-']
    colors=['r', 'g', 'b']
    
    for (el,el_,line,i,col) in zip(data_python_path, data_fortran_path,linestyles,[0,1,2], colors):
        if not el.endswith(".dat"):
            raise ValueError("Data files should be in .dat format")
        data_python = np.loadtxt(el)
        data_fortran = np.loadtxt(el_)
        print(el)
        ax[i].plot(data_python[:, 0], data_python[:, 3], label=f"Python ppp { int(el.split('ppp_')[1].split('.')[0])}",color=(1,0,0,1), linestyle=line)
        ax[i].plot(data_fortran[:, 0], data_fortran[:, 9], label=f"Fortran ppp { int(el_.split('ppp_')[1].split('.')[0])}",color=(0,0,1,1), linestyle=line)
        print(f"Comparing {os.path.basename(el)} and {os.path.basename(el_)}")
        #ax[i].legend()
        ax[i].set_xlabel("time")
        ax[i].set_ylabel(r"$C_d$")
        ax_.plot(data_python[:, 0], data_python[:, 3], label=f"Python ppp { int(el.split('ppp_')[1].split('.')[0])}",color=col,)
        #ax_.legend()
        ax_.set_xlabel("time")
        ax_.set_ylabel(r"$C_d$")
        
        
        
    fig.savefig("{}/data_base_comparison_k_{}.png".format(data_python_folder,k), dpi=300)
    
    fig_.savefig("{}/data_base_comparison_k_{}_all_ppp.png".format(data_python_folder,k), dpi=300)
if __name__ == "__main__":
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=0.05)
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=0.1)
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=0.2)
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=0.5)
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=1)
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=2)
    
    compare_data_base("data_base","../LDVM_v2_original.5/data_base",k=5)
    pass
