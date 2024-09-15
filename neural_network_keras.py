import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


def plot_curve(accuracy_train, loss_train):
    sns.set_theme(style='darkgrid')
    epochs = np.arange(loss_train.shape[0])
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_train)
    plt.xlabel('Epoch#')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_train)
    plt.xlabel('Epoch#')
    plt.ylabel('MSE')
    plt.title('Training loss')
    plt.show()


# Main Code
alpha = 0.1  # learning rate
N = 1000  # number of epochs
X = np.identity(8)  # all input
y = X  # all output

opt = Adam(learning_rate=alpha)

model = Sequential()
model.add(Dense(input_dim=8, units=3, activation='relu'))
model.add(Dense(units=8, activation='sigmoid'))
model.compile(optimizer=opt, loss='mse', metrics='accuracy')
model.summary()

history = model.fit(x=X, y=y, batch_size=8, epochs=N, verbose=1)

acc_curve = np.array(history.history['accuracy'])
loss_curve = np.array(history.history['loss'])
plot_curve(acc_curve, loss_curve)

y_predicted = np.around(model.predict(x=X), decimals=2)
print('Desired output: \n', y)
print('Network output: \n', y_predicted)
