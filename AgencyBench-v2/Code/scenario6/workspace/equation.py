import numpy as np


def equation(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Acceleration in Nonl-linear Harmonic Oscillator

    Args:
        x: A numpy array representing observations of Position at time t.
        t: A numpy array representing observations of Time.
        v: A numpy array representing observations of Velocity at time t.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Acceleration in Nonl-linear Harmonic Oscillator as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * x + params[1] * t + params[2] * v + params[3]
    return output