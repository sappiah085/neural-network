import pandas as pd
import numpy as np

data_array = np.array(pd.read_csv('mnist_test.csv'))
Y = data_array[:, 0].reshape(data_array.shape[0], -1)
X = data_array[:, 1:]/255
