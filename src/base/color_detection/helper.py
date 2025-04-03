import numpy as np

from base.config import (
    logger
)

def calculate_manhattan_distance(vector1, vector2):
    logger.info("Calculating Manhattan distance.")
    logger.debug(f"Shape of vector1: {vector1.shape}")
    logger.debug(f"Shape of vector2: {vector2.shape}")

    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(np.abs(vector1 - vector2))

def calculate_chebyshev_distance(vector1, vector2):
    logger.info("Calculating Chebyshev distance.")
    logger.debug(f"Shape of vector1: {vector1.shape}")
    logger.debug(f"Shape of vector2: {vector2.shape}")

    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.max(np.abs(vector1 - vector2))

def calculate_minkowski_distance(vector1, vector2, power_parameter):
    logger.info("Calculating Minkowski distance.")
    logger.debug(f"Shape of vector1: {vector1.shape}")
    logger.debug(f"Shape of vector2: {vector2.shape}")
    logger.debug(f"Power parameter: {power_parameter}")

    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")
    if power_parameter <= 0:
        raise ValueError("power_parameter must be a positive value.")
    return np.sum(np.abs(vector1 - vector2) ** power_parameter) ** (1 / power_parameter)

def calculate_euclidean_distance(array1, array2, axis=None):
    logger.info("Calculating Euclidean distance.")
    logger.debug(f"Shape of array1: {array1.shape}")
    logger.debug(f"Shape of array2: {array2.shape}")
    logger.debug(f"Axis: {axis}")

    if len(array1.shape) != len(array2.shape) and any(d != 1 for d in np.subtract(array1.shape, array2.shape) if d is not None):
        raise ValueError("Input arrays must have compatible shapes, except for the specified axis.")
    return np.linalg.norm(array1 - array2, axis=axis)