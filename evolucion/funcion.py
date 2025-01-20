import numpy as np

def function_objetivo(x):
    return 0.1 * x * np.log(1 + np.abs(x)) * (np.cos(x) ** 2)