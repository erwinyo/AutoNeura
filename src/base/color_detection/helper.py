import numpy as np

def calculate_manhattan_distance(vector1, vector2):
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(np.abs(vector1 - vector2))

def calculate_chebyshev_distance(vector1, vector2):
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.max(np.abs(vector1 - vector2))

def calculate_minkowski_distance(vector1, vector2, power_parameter):
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    if power_parameter <= 0:
        raise ValueError("power_parameter must be a positive value.")
    return np.sum(np.abs(vector1 - vector2) ** power_parameter) ** (1 / power_parameter)

def calculate_euclidean_distance(array1, array2, axis=None):
    if len(array1.shape) != len(array2.shape) and any(d != 1 for d in np.subtract(array1.shape, array2.shape) if d is not None):
        raise ValueError("Input arrays must have compatible shapes, except for the specified axis.")
    return np.linalg.norm(array1 - array2, axis=axis)