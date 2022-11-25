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


def sigmoid(z: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-z))

def main():
    net = Network([2, 3, 1])


if __name__ == "__main__":
    main()
