import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class MLP:
    """
    Feedforward Multi-Layer Perceptron with 1 variable-neuron hidden layer and 1-neuron output layer. Stochastic Gradient Descent (SGD) is used in training.
    """

    def __init__(self, size=16, lr = 0.5, verbose = 0):
        self.size = size
        self.lr = lr
        self.verbose = verbose
        pass

    def encode(self, y):
        """
        One-hot encode labels

        Parameters:
        ---
        y: array-like
            Labels.
        """

        return np.identity(self.k)[y]

    def predict(self, X):
        """
        Predict for a testing data X

        Parameters
        ---
        X: array-like
            Testing samples.
        """

        return [np.argmax(sigmoid(np.dot(sigmoid(np.dot(x, self.weights[0])),self.weights[1]))) for x in X]

    def train(self, X, Y, epochs=300):
        """
        Train perceptron using stochastic gradient descent (SGD).

        Parameters
        ---
        X: array-like
            Training samples
        
        Y: array-like
            Training labels

        epochs: int
            Number of epochs to train on
        """

        self.classes = np.unique(Y)
        self.k = len(self.classes)

        self.weights = [
            np.random.rand(X.shape[1], self.size),
            np.random.rand(self.size, self.k)
        ]

        history_l = []

        for t in range(epochs):

            loss = []

            for (x, y) in zip(X, self.encode(Y)):

                h = sigmoid(np.dot(x, self.weights[0]))
                output = sigmoid(np.dot(h,self.weights[1]))

                loss.append(np.mean((output - y)**2))
                
                delta = (output - y) * output * (1 - output)

                self.weights[1] -= self.lr * (h * delta.reshape(-1, 1)).T

                gamma = np.dot(delta,self.weights[1].T) * (h * (1 - h))

                self.weights[0] -= self.lr * (x.reshape(-1,1) * gamma)

            history_l.append(np.sum(loss))

            if self.verbose == 1:
                print(f'Iteration {t}: Loss: {np.sum(loss)}')
        
        return history_l
