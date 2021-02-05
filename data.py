import pandas as pd
import numpy as np
import os


data_dir = '/home/fssv2/dirty_mnist_dataset'

list_file = os.listdir(data_dir)

print(list_file)
csv_path = [file for file in list_file if file.endswith('.csv')]
print(csv_path)
csv_path = os.path.join(data_dir, csv_path[0])

print(csv_path)

a = pd.read_csv(csv_path)
print(a.iloc[0])
print(a.iloc[0][1:].values)
print(len(a.iloc[0][1:].values))
print(a.iloc[[0,1]])
b = a.iloc[[0, 1]]
b = b.iloc[0, 1:].values
print(b)
# print(str(b).zfill(5))