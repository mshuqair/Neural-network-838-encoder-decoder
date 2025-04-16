import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


# Parameters
alpha = 0.5              # Learning rate
N = 5000                 # Number of iterations
X_all = np.identity(8)   # Input vectors
Y_all = X_all            # Target output (identity mapping)

O = np.zeros(8)          # Final output vector
SSE = np.zeros((8, N))   # Sum of squared errors for each input over time
W_plot = np.zeros(N)     # Tracking one weight for visualization

# Plot setup
sns.set_theme(style='darkgrid')
figure_sse = plt.figure().add_subplot(111)
figure_weights = plt.figure().add_subplot(111)

# Loop through each input vector
for sample_idx in range(8):
    print(f'Input number: {sample_idx}')
    
    X = X_all[sample_idx, :]
    Y = Y_all[sample_idx, :]

    # Initialize outputs
    O1 = np.zeros(8)      # Input layer
    O2 = np.zeros(3)      # Hidden layer
    O3 = np.zeros(8)      # Output layer

    # Initialize weights
    W_A = np.random.randn(8, 3)  # Input to hidden
    W_B = np.random.randn(3, 8)  # Hidden to output

    # Training loop
    for n in range(N):
        W_plot[n] = W_A[sample_idx, 0]

        # Forward pass
        O1 = X
        for j in range(3):
            O2[j] = sigmoid(np.dot(W_A[:, j], O1))
        for k in range(8):
            O3[k] = sigmoid(np.dot(W_B[:, k], O2))

        O = O3

        # Error calculation
        SSE[sample_idx, n] = 0.5 * np.sum((Y - O) ** 2)

        # Backpropagation
        delta = O * (1 - O) * (Y - O)
        delta_h = O2 * (1 - O2) * np.dot(W_B, delta)

        # Weight updates
        W_A += alpha * np.outer(X, delta_h)
        W_B += alpha * np.outer(O2, delta)

    # Plotting and output
    figure_weights.plot(W_plot)
    print('Network input:\n', X)
    print('Desired output:\n', Y)
    print('Network output:\n', O)
    print('Hidden layer values:\n', O2, '\n')

    figure_sse.plot(SSE[sample_idx, :])

# Final plots
figure_sse.set_title('Sum of Squared Error of the Network for the 8 Inputs')
figure_sse.set_xlabel('Number of Iterations')
figure_sse.set_ylabel('SSE')

figure_weights.set_title('Weights from Inputs to One Hidden Neuron')
figure_weights.set_xlabel('Number of Iterations')
figure_weights.set_ylabel('Weight Value')

plt.show()
