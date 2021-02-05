import numpy as np


a = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([[3, 3, 3], [4, 4, 4]])
print(a)
print(b)

array_list = []
array_list.append(a)
array_list.append(b)
c = np.concatenate(array_list, axis=1)
print(c)
# a = a[..., np.newaxis]
# print(a)


