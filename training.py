import numpy as np
import matplotlib.pyplot as plt
from prime import forward_pass, one_hot_values, init_params, activation1, activation2, update_params, get_accuracy, get_predictions, lossFunc
from data import saveWeight
from tryF import data_array, X, Y
m, n = data_array.shape
NEURONS_NEEDED = 50

def gradient_descent(X, Y, alpha, iterations ,m):
    Y_classes = len(np.unique(Y))
    losses = []
    accuracies = []
    W1, b1, W2, b2 = init_params(n-1, NEURONS_NEEDED, Y_classes)
    for i in range(iterations):
        forward_pass(X, W1, b1, W2, b2)
        one_hot_y = one_hot_values(Y)
        # Backward pass
        dw2, db2, dZ2 = activation2.back_pass(activation1.output, activation2.output, one_hot_y, m)
        dw1, db1, dZ1 = activation1.back_pass(W2, dZ2, activation1.output, X, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dw1, db1, dw2, db2, alpha)
        loss = lossFunc(activation2.output, one_hot_y)
        predictions = get_predictions(activation2.output)
        accuracy = get_accuracy(predictions, Y)
        losses.append(loss)
        accuracies.append(accuracy)
    return W1, b1, W2, b2, losses, accuracies

W1, b1, W2, b2, losses, accuracies = gradient_descent(X, Y, 0.1, 1500, m)
saveWeight(W1,b1,W2,b2)


plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(losses, label="Loss", color="red")
plt.title("Loss Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(accuracies, label="Accuracy", color="blue")
plt.title("Accuracy Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
