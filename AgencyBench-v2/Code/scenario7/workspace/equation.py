import numpy as np

def equation(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Population growth rate

    Args:
        t: A numpy array representing observations of Time.
        P: A numpy array representing observations of Population at time t.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Population growth rate as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * t + params[1] * P + params[2]
    return output