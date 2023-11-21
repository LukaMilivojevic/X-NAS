import numpy as np
import matplotlib.pyplot as plt
from tensorflow import nn
from tensorflow.keras.utils import normalize
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense         
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
from tensorflow import keras
from os import path


def show(img):
    """
    Ova funkcija sluzi da prikaze odredjenu sliku u
    grayscale formatu. U ovom programu to ce biti slike
    iz trening seta ili test seta.
    """
    fig = plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.show()


def local_file_name(folder: str, file: str) -> str:
    return path.join(path.dirname(path.abspath(__file__)), folder, file)



mnist = datasets.mnist  # u ovoj i sledecoj liniji vrsi se ucitavanje MNIST dataset-a
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # sastoji se iz 70000 slika rucno pisanih cifara velicine 28x28
#show(x_train[0])
x_train = normalize(x_train, axis=1)  # skaliranje vrednosti piksela od vrednosti u opsegu 0-255 na vrednosti opsega 0-1
x_test = normalize(x_test, axis=1)
#show(x_train[0])

model = Sequential()  # kreiranje mreze
model.add(Flatten())
model.add(Dense(128, activation=nn.relu))
#model.add(Dense(0, activation=nn.relu))
model.add(Dense(128, activation=nn.relu))
model.add(Dense(10, activation=nn.softmax))

optimizer = keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.00, nesterov=False, name="SGD"
)
#optimizer="adam",
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=3)  # treniranje mreze
loss, accuracy = model.evaluate(x_test, y_test)  # testiranje mreze
print(f"LOSS:{loss}, ACC:{accuracy}")
#model.save(local_file_name("testfolder", "cifre2.model"))  # cuvanje modela
#new_model = load_model(local_file_name("testfolder", "cifre2.model"))
