import numpy as np
import matplotlib.pyplot as plt


def sigma(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


alpha = 0.5  # learning rate
N = 5000  # number of iterations
X_all = np.identity(8)  # all input
Y_all = X_all  # all output
O = np.zeros(8)  # network output vector
SSE = np.zeros((8, N))  # sum of squared errors
W_plot = np.zeros(N)

figure_1 = plt.figure().add_subplot(111)
figure_2 = plt.figure().add_subplot(111)

# inputs
for i in range(8):
    print('input number: ' + str(i))
    X = X_all[i, :]
    Y = Y_all[i, :]

    # initialize parameters
    O1 = np.zeros(8)  # output of input layer
    O2 = np.zeros(3)  # output of hidden layer
    O3 = np.zeros(8)  # output of last layer
    W_A = np.random.randn(8, 3)  # weights between input and hidden layer
    W_B = np.random.randn(3, 8)  # weights between hidden layer and last layer

    # the network
    for n in range(N):

        W_plot[n] = W_A[i, 0]

        # the first layer
        O1 = X

        # the hidden layer
        for i in range(3):
            O2[i] = sigma(np.dot(W_A[:, i], O1))

        # the output layer
        for i in range(8):
            O3[i] = sigma(np.dot(W_B[:, i], O2))

        # network output
        O = O3

        # calculate sse for network
        SSE[i, n] = 0.5 * sum(pow((Y - O), 2))

        # calculating delta and delta_h
        delta = O * (1 - O) * (Y - O)
        delta_h = O2 * (1 - O2) * np.dot(W_B, delta)

        # updating the weights
        W_A += alpha * np.dot(np.reshape(X, (8, 1)), np.reshape(delta_h, (1, 3)))
        W_B += alpha * np.dot(np.reshape(O2, (3, 1)), np.reshape(delta, (1, 8)))
    figure_2.plot(W_plot)
    print('network input: \n', X)
    print('desired output: \n', Y)
    print('network output: \n', O)
    print('hidden values: \n', O2, '\n')
    figure_1.plot(SSE[i, :])

figure_1.title.set_text('sum of squared error of the network for the 8 inputs')
figure_1.set_xlabel('number of iterations')
figure_1.set_ylabel('SSE')

figure_2.title.set_text('weights from inputs to one hidden neuron')
figure_2.set_xlabel('number of iterations')
figure_2.set_ylabel('weights value')

plt.show()

