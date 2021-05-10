import numpy as np
import matplotlib.pyplot as plt
from Package.PreActivation import PreActivation
from Package.Activation import Activation
from Package.NeuralNetwork import NeuralNetwork
from Package.ErrorFunction import mse, mse_prime
from Package.ActivationFunction import tanh, tanh_prime

sick = (np.random.randn(100, 2) + np.array([-2, -2]))
health = (np.random.randn(100, 2) + np.array([2, 2]))
x_train = np.concatenate((sick, health), axis=0)
y_train = np.concatenate(((np.zeros(100)), np.ones(100)), axis=0).reshape((200, 1))

# fig, axs = plt.subplots(figsize=(8, 6))
# axs.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
# plt.show()

network = NeuralNetwork()
network.add(PreActivation(2, 3))
network.add(Activation(tanh, tanh_prime))
network.add(PreActivation(3, 1))
network.add(Activation(tanh, tanh_prime))

out = network.predict(x_train)
print("Precision :", np.mean(out == y_train))

network.use(mse, mse_prime)
network.fit(x_train, y_train, 30000, 0.01)

out = network.predict(x_train)
print("Precision :", np.mean(out == y_train))