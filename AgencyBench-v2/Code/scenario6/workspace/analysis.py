import pandas as pd

file_path = 'equation.parquet'
df = pd.read_parquet(file_path)
training_set_data = df.iloc[0]['training_set'].tolist()
print(training_set_data[:5])

