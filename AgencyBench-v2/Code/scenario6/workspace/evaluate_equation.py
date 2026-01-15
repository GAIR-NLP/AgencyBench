import numpy as np
import pandas as pd

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

from equation import equation


def evaluate(data: list) -> float:
    # Load data observations
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    X = inputs
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return loss

if __name__ == '__main__':
    # Read data from train.parquet and pass it to evaluate function
    file_path = 'equation.parquet'
    df = pd.read_parquet(file_path)
    training_set_data = df.iloc[0]['training_set'].tolist()
    
    score = evaluate(training_set_data)
    
    if score is not None:
        print(score)