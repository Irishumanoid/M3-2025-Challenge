import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equation
def dydt(y, t):
    k = 0.3
    dydt = -k * y
    return dydt

# Initial condition
y0 = 5

# Time points where solution is computed
t = np.linspace(0, 20, 100)

# Solve ODE
y = odeint(model, y0, t)

# Plot results
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Solution of dy/dt = -0.3y')
plt.show()
