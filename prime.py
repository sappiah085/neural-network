import numpy as np

np.random.seed(0)

def init_params(X, neurons, Y):
    W1 = np.random.randn(X, neurons) * 0.1
    b1 = np.zeros((1, neurons))
    W2 = np.random.randn(neurons, Y) * 0.1
    b2 = np.zeros((1, Y))
    return W1, b1, W2, b2

class Layer:
    def forward_pass(self, X, w, b):
        self.output = np.dot(X, w) + b

class ReLu:
    def forward_pass(self, X):
        self.output = np.maximum(0, X)
    def back_pass(self, w2, dZ2, Output, X ,m):
        def DReLU(O):
            return O > 0
        dZ1 = dZ2.dot(w2.T) * DReLU(Output)
        dw1 = 1 / m * X.T.dot(dZ1)
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
        return dw1, db1, dZ1

class Softmax:
    def forward_pass(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def back_pass(self, A1, A2, one_hot_y, m):
        dZ2 = A2 - one_hot_y
        dW2 = 1 / m * A1.T.dot(dZ2)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
        return dW2, db2, dZ2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=1)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y.flatten())

def one_hot_values(Y):
    Y = Y.flatten()
    Y = Y.astype(int)
    one_hot_values_y = np.zeros((Y.size, Y.max() + 1))
    one_hot_values_y[np.arange(Y.size), Y] = 1
    return one_hot_values_y

def lossFunc(output, one_hot_y):
    clipped_probs = np.clip(output, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(one_hot_y * np.log(clipped_probs), axis=1))

      # The last column (labels)
layer1 = Layer()
activation1 = ReLu()
layer2 = Layer()
activation2 = Softmax()

def forward_pass(X, W1, b1, W2, b2):
    layer1.forward_pass(X, W1, b1)
    activation1.forward_pass(layer1.output)
    layer2.forward_pass(activation1.output, W2, b2)
    activation2.forward_pass(layer2.output)
    return activation2.output



def make_predictions(X, W1, b1, W2, b2):
    A2 = forward_pass(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions



