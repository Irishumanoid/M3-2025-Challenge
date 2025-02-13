import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
alpha, beta, gamma, delta = 0.1, 0.02, 0.3, 0.01

vars = 2
# Define system of ODEs
def lotka_volterra(t, z):
    X = [z[0], z[1]]
    dXdt = [alpha * X[0] - beta * X[0] * X[1], delta * X[0] * X[1] - gamma * X[1]]
    print(dXdt)
    return dXdt

# Time range and initial conditions
t_span = (0, 200)
t_eval = np.linspace(0, 200, 1000)
initial_conditions = [40, 9]  # Initial prey and predator populations

# Solve system
sol = solve_ivp(lotka_volterra, t_span, initial_conditions, t_eval=t_eval)

# Plot results
plt.plot(sol.t, sol.y[0], label="Prey (x)")
plt.plot(sol.t, sol.y[1], label="Predator (y)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.show()