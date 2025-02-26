import numpy as np
from scipy.optimize import minimize

# Objective function
def objective(x):
    y = (x[0] - 1)**2 + (x[1] - 2.5)**2
    return y

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
    {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2}
]

# Bounds for variables: x0 >= 0, x1 >= 0
bounds = [
    (0, None), 
    (0, None)
]

# Initial guess
x0 = np.array([2, 0])

# Solve the problem
# If SLSQP doesn't work, try COBYLA or trust-constr
result = minimize(objective, 
                  x0, 
                  method='SLSQP', 
                  bounds=bounds, 
                  constraints=constraints)

# Print results
print("Optimal solution:", result.x)
print("Objective function value:", result.fun)
