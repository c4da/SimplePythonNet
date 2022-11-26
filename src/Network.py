import random

import numpy as np

"""
net = Network([2, 3, 1]) constructs a simple network with 2 inputs ie. 2 neurons in the input layer
, 3 neurons in the hidden layer and 1 output

Weights matrix has swapped indexes from i, j to j, i so that activation function is simplier:
a' = sigma(w*a + b)
a is a vector of an activations of the second layer neurons, sigma is applied elementwise to every entry
in the vector (w*a + b)
"""


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.random.randn generates a rand value with mean 0 and std 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedForward(self, a: np.array) -> np.array:
        """
        Returns the output of the network if "a" is input.
        :param a:
        :return:
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, trainingData : np.array, epochs : int, miniBatchSize : int, learningRate : float, testData : np.array = None):
        if testData:
            n_test = len(testData)
            n = len(trainingData)
            for j in range(epochs):
                random.shuffle(trainingData)
                mini_batches = [trainingData[k:k + miniBatchSize] for k in range(0, n, miniBatchSize)]
                for miniBatch in mini_batches:
                    self.updateMiniBatch(miniBatch, learningRate)

                if testData:
                    print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), n_test))
                else:
                    print("Epoch {0} complete".format(j))

    def updateMiniBatch(self, miniBatch : np.array, eta : float):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.backProp(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(miniBatch)) * nw for w, nw in zip(self.weights, nabla_w)]



def sigmoid(z: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-z))

def main():
    net = Network([2, 3, 1])


if __name__ == "__main__":
    main()
