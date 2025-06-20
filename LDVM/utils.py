import numpy as np

import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    x, y = generate_naca0015()
    plot_airfoil(x, y)
    