import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# SEIR model differential equations.
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Parameters
beta = 0.3    # Transmission rate
sigma = 1/5.2 # Rate at which exposed individuals become infectious (1/incubation period)
gamma = 1/14  # Recovery rate (1/infectious period)

# Initial conditions: S, E, I, R
S0 = 0.99      # Fraction of susceptible individuals
E0 = 0.01      # Fraction of exposed individuals
I0 = 0.0       # Fraction of infected individuals
R0 = 0.0       # Fraction of recovered individuals

# Time grid (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
initial_conditions = [S0, E0, I0, R0]

# Solve ODE
solution = odeint(seir_model, initial_conditions, t, args=(beta, sigma, gamma))

# Extract results
S, E, I, R = solution.T

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible")
plt.plot(t, E, label="Exposed")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of population")
plt.title("SEIR Model")
plt.legend()
plt.grid()
plt.show()