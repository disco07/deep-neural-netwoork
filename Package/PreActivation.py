import numpy as np

from Package.NetworkLayer import NetworkLayer


class PreActivation(NetworkLayer):
    def __init__(self, number_inputs, number_outputs):
        super().__init__()
        self.weights = np.random.randn(number_inputs, number_outputs)
        self.bias = 0

    def forward(self, inputs):
        self.inputs = inputs
        print(f"donne {self.inputs.shape}")
        self.outputs = np.dot(self.inputs, self.weights) + self.bias
        return self.outputs

    def backward(self, outputs_error, learning_rate):
        weights_error = np.dot(self.inputs.T, outputs_error)
        input_error = np.dot(outputs_error, self.weights.T)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * outputs_error
        return input_error
