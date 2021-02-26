import numpy as np
import pandas as pd

a = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
b = np.array([[2, 2, 2, 2], [2, 2, 2, 2]])
a = np.expand_dims(a, axis=-1)
b = np.expand_dims(b, axis=-1)
c = []
c.append(a)
c.append(b)
print(c)
array = np.concatenate(c, axis=2)
print(array)
mean = array.mean(axis=2)
print(mean)

mean = (mean > 1) * 1
print(mean)

model_name = 'Resnet50_01'
csv_file_path = './' + model_name + '.csv'
print(csv_file_path)

import os
basename = os.path.basename('./djfkjf/dlksajflk/dslkafjlk/tst.cvs')
print(basename)