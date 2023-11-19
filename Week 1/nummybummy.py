import numpy as np

my_array = np.random.randn(25)

for i in range(len(my_array)):
    if my_array[i] > 0.09:
        my_array[i] *= my_array[i]
    else:
        my_array[i] = 42

my_array = my_array.reshape((5,5))

print(my_array[:,3])