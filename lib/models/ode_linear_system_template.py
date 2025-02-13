import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the matrix form of the ODE system
# the row of the array represents the derivative (coefficient)
# of the state variables and the column represents the state variables

#The B represents the forcing function, just a constant addition

def system(t, z):
    # State variables
    A = np.array([
        [0, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [0, 0, -1, 0]
    ])
    
    # Forcing function (e.g., external input)
    B = np.array([0, 0, 0, np.sin(t)])
    
    return A @ z + B  # Matrix-vector multiplication

# Initial conditions: x(0) = 1, dx/dt(0) = 0, y(0) = 0, dy/dt(0) = 1
z0 = [1, 0, 0, 1]

# Time span
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

# Solve the system
sol = solve_ivp(system, t_span, z0, t_eval=t_eval)

# Plot the solution
plt.plot(sol.t, sol.y[0], label="x(t)")
plt.plot(sol.t, sol.y[2], label="y(t)")
plt.xlabel("t")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()