from prime import forward_pass, get_predictions, get_accuracy
from data import OpenWeights
import numpy as np
import pandas as pd
W1, b1, W2, b2 = OpenWeights()
X = pd.read_csv('test.csv')
X = np.array(X)
for i in range(len(X)):
  A2 = forward_pass( X[i] ,W1, b1, W2, b2)
  predictions = get_predictions(A2)

  print(f"confidence: {A2}  prediction: {predictions}\n")
