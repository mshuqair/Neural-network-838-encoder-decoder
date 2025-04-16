import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


def plot_curve(accuracy_train, loss_train):
    """Plot training accuracy and loss curves."""
    sns.set_theme(style='darkgrid')
    epochs = np.arange(len(loss_train))

    plt.figure(figsize=(10, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_train, label='Accuracy', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_train, label='Loss (MSE)', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss')

    plt.tight_layout()
    plt.show()


# -------------------- Main Code --------------------

# Hyperparameters
alpha = 0.1           # Learning rate
epochs = 1000         # Number of training epochs
X = np.identity(8)    # Inputs
y = X                 # Targets (identity mapping)

# Optimizer
optimizer = Adam(learning_rate=alpha)

# Model definition
model = Sequential([
    Dense(units=3, input_dim=8, activation='relu'),
    Dense(units=8, activation='sigmoid')
])

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(x=X, y=y, batch_size=8, epochs=epochs, verbose=1)

# Plot training curves
acc_curve = np.array(history.history['accuracy'])
loss_curve = np.array(history.history['loss'])
plot_curve(acc_curve, loss_curve)

# Predict and compare output
y_pred = np.around(model.predict(x=X), decimals=2)
print('Desired output:\n', y)
print('Network output:\n', y_pred)
