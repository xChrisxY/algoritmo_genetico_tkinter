import numpy as np

def function_objetivo(x):
    return np.log(np.abs(x**3)) * np.cos(x) * np.sin(x)