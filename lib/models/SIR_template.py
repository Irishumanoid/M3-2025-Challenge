import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the SIR model differential equations
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parameters
beta = 0.3  # infection rate
gamma = 0.1  # recovery rate
S0 = 0.99  # initial susceptible fraction
I0 = 0.01  # initial infected fraction
R0 = 0.0  # initial recovered fraction

# Time vector
t_span = (0, 160)  # from day 0 to day 160
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial conditions vector
y0 = [S0, I0, R0]

# Solve the ODEs
solution = solve_ivp(sir_model, t_span, y0, t_eval=t_eval, args=(beta, gamma))

# Extract the results
S, I, R = solution.y

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(solution.t, S, label='Susceptible', color='blue')
plt.plot(solution.t, I, label='Infected', color='red')
plt.plot(solution.t, R, label='Recovered', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Fraction of population')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()
