import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.Layers = []
        self.Losses = None
        self.Losses_prime = None

    def add(self, layer):
        self.Layers.append(layer)

    def predict(self, inputs):
        result = []
        outputs = inputs
        for NetworkLayer in self.Layers:
            outputs = NetworkLayer.forward(outputs)
        result.append(outputs)
        return np.round(result)

    def use(self, mse, mse_prime):
        self.Losses = mse
        self.Losses_prime = mse_prime

    def fit(self, x_train, y_train, epochs, learning_rate):
        outputs = x_train
        print(outputs.shape)
        for epoch in range(epochs):
            error = 0
            for NetworkLayer in self.Layers:
                outputs = NetworkLayer.forward(outputs)

            error += self.Losses(outputs, y_train)
            if epoch % 5000 == 0:
                print(error)

            err = self.Losses_prime(outputs, y_train)
            for NetworkLayer in reversed(self.Layers):
                err = NetworkLayer.backward(err, learning_rate)
