from prime import forward_pass, get_predictions
from data import OpenWeights
import numpy as np
import pandas as pd
from tryF import X
from matplotlib import pyplot as plt
W1, b1, W2, b2 = OpenWeights()

for i in range(len(X)):
  A2 = forward_pass( X[i] ,W1, b1, W2, b2)
  predictions = get_predictions(A2)
  plt.imshow(X[i].reshape(28,28), cmap='gray')
  plt.axis('off')
  print(f"confidence: {np.max(A2, 1)[0] * 100}%  prediction: {predictions[0]}\n")
  plt.show()
