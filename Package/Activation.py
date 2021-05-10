from Package.NetworkLayer import NetworkLayer


class Activation(NetworkLayer):
    def __init__(self, activation_function, activation_function_prime):
        super().__init__()
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_function(self.inputs)
        return self.outputs

    def backward(self, outputs_error, learning_rate):
        return self.activation_function_prime(outputs_error)