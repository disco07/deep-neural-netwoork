class NetworkLayer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, outputs_error, learning_rate):
        raise NotImplementedError
