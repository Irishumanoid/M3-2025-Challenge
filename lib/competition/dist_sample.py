import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mean, num_points, std=10):
    lower, upper = 0, np.inf  
    a, b = (lower - mean) / std, (upper - mean) / std 
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=num_points)

data = truncated_normal(6, 10)
print(data)
