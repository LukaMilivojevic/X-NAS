import numpy as np
import random
from tensorflow import keras
from tensorflow import nn


def create_ind(max_layers: int = 10, max_neurons: int = 64) -> np.ndarray:
    """
    Generisanje jednog hromozoma koji predstavlja mrezu.

    Parametri:
        max_layers: najveci moguc broj slojeva u mrezi
        max_neurons: najveci moguc broj neurona u sloju 
    """
    chromosome = []
    chromosome.append(round(random.uniform(0.001, 0.1), 3))  # learning rate
    chromosome.append(round(random.uniform(0.8, 1), 2))  # momentum
    number_of_layers = random.randint(2, max_layers) 
    chromosome.append(number_of_layers)
    for _ in range(number_of_layers):
        number_of_neurons = random.randint(1, max_neurons) 
        chromosome.append(number_of_neurons)
    while len(chromosome) < max_layers+3:
        chromosome.append(0)
        #for __ in range(number_of_neurons):
        #    chromosome.append((round(random.uniform(0, 1), 2), round(random.uniform(0, 1), 2)))
    return np.array(chromosome, dtype=object)


def decode(chromosome: np.ndarray):
    """
    Funkcija koja dekodira hromozom u neuronsku mrezu.

    Parametri:
        chromosome: hromozom u kome je kodirana mreza
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=chromosome[0], momentum=chromosome[1], nesterov=False, name="SGD"
    )
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())  # prvi sloj je flatten koji transformise sliku iz matrice u 1D niz
    for layer in chromosome[3:]:
        if layer == 0:
            break
        model.add(keras.layers.Dense(layer, activation=nn.relu))
    model.add(keras.layers.Dense(10, activation=nn.softmax))  # poslednji sloj mora imati fiksan broj neurona koji je jednak broju klasa koje se predictuju
    return model, optimizer


if __name__ == "__main__":
    mnist = keras.datasets.mnist  # u ovoj i sledecoj liniji vrsi se ucitavanje MNIST dataset-a
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # sastoji se iz 70000 slika rucno pisanih cifara velicine 28x28
    x_train = keras.utils.normalize(x_train, axis=1)  # skaliranje vrednosti piksela od vrednosti u opsegu 0-255 na vrednosti opsega 0-1
    x_test = keras.utils.normalize(x_test, axis=1)
    ind = create_ind()
    print(ind)
    model, optimizer = decode(ind)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # odredjivanje optimizatora, loss funkcije i metrike
    )
    model.fit(x_train, y_train, epochs=3) 
    model.summary()
    loss, accuracy = model.evaluate(x_test, y_test)  # testiranje mreze
    print(f"LOSS:{loss}, ACC:{accuracy}")
