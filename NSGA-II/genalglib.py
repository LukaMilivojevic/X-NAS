import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set
from tensorflow import keras
from network.coding import decode
from network.coding import create_ind
from os import path


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)


def local_file_name(folder: str, file: str) -> str:
    return path.join(path.dirname(path.abspath(__file__)), folder, file)


def create_solutions(chrom_len: int = 16) -> np.ndarray:
    """
    Ova funkcija generise resenja u vidu numpy nizova nula i jedinica
    koja se koriste pri izradi i testiranju NSGA-II algoritma.
 
    Parametri:
        chrom_len: opisuje duzinu hromozoma
    """
    solutions = np.zeros((2, chrom_len))
    for solution in solutions:
        solution[: chrom_len // 2] = 1
        np.random.shuffle(solution)
    return solutions


def calculate_fitness_one_shit(
    solutions: np.ndarray, individual: np.ndarray, chrom_len: int = 16
) -> List[int]:
    """
    Ova funkcija izracunava fitness jedinke po sledecem principu:
    za svaki bit koji se nalazi na istom mestu kao u resenju, fitness
    se smanjuje za 1 (u pocetku je fitness jednak duzini resenja). 
    Dakle, potrebno je minimizovati bit-distance u odnosu na 2 razlicita resenja.

    Parametri:
        solutions: niz resenja
        individual: jedinka iz populacije
        chrom_len: duzina hromozoma
    """
    fitness_list = []
    for solution in solutions:
        fitness = chrom_len
        num_of_equal = individual == solution
        fitness_list.append(fitness - num_of_equal.sum())
    return fitness_list


def calculate_fitness_experimental(individual: np.ndarray, chrom_len: int = 16):
    mul = 2
    decoded = 0
    for digit in individual:
        decoded += mul*digit
        mul *= 2
    decoded = -1000+2000*decoded/2**15
    fitness_list = []
    fitness_list.append(decoded**2)
    fitness_list.append((decoded-2)**2)
    return fitness_list


def calculate_fitness_one(
    model: np.ndarray,
    optimizer,
    number_of_neurons: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    index: int,
    gen_num: int,
) -> List[float]:
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=3)
    loss = model.evaluate(x_test, y_test)[0]
    model.save(local_file_name(f"generation{gen_num}", f"individual{index}"))
    return loss, number_of_neurons


def calculate_fitness_all(
    population: np.ndarray, solutions: np.ndarray, gen_num: int
) -> List[List[int]]:
    """
    Funkcija koja izracunava fitness za svaku jedinku iz populacije.

    Parametri:
        population: niz jedinki
        solutions: niz resenja
        gen_num: redni broj generacije
    """
    fitness_list = []
    for index, individual in enumerate(population):
        model, optimizer = decode(individual)
        fitness_list.append(calculate_fitness_one(model, optimizer, sum(individual[3:]), x_train, y_train, x_test, y_test, index, gen_num))
    return fitness_list


def populate(population_size: int) -> np.ndarray:
    """
    Kreiranje populacije jedinki.

    Parametri:
        population_size: velicina populacije
    """
    population = []
    for _ in range(population_size):
        population.append(create_ind())
    return np.array(population)


def plot_param(param_list: List[int], name):
    """
    Funkcija koja plottuje jedinke ili njihov fitnes u koordinatnom sistemu.

    Parametri:
        param_list: lista koja sadrzi jedinke ili fitness istih
        name: ime fajla u koji se cuva grafik
    """
    plt.figure()
    for elem in param_list:
        plt.scatter(elem[0], elem[1])
    plt.savefig(f'{name}.png')


def find_pareto(population: np.ndarray, fitness_list: List[List[int]]) -> Set[int]:
    """
    Ova funkcija pronalazi pareto front jedinki poredjenjem njihovog fitness. 
    Za sada pronalazi samo prvi front, uskoro cu dodati i ostatak koda da pronadje sve frontove.

    Parametri:
        population: lista jedinki
        fitness_list: lista fitnessa jedinki
    """
    pop_size = population.shape[0]
    list_of_sp = []
    num_dominated_by = []
    for i, first_ind in enumerate(fitness_list):
        num = 0
        sp = []
        for j, second_ind in enumerate(fitness_list):
            if (first_ind[0] <= second_ind[0] or first_ind[1] <= second_ind[1]) and (first_ind[0] < second_ind[0] or first_ind[1] < second_ind[1]):
                sp.append(j)
            elif first_ind[0] == second_ind[0] and first_ind[1] == second_ind[1]:
                continue
            else:
                num += 1
               
        num_dominated_by.append(num)
        list_of_sp.append(sp)
    #front = set()
    front = []
    for index, elem in enumerate(num_dominated_by):
        if elem == 0 and len(front) < round(pop_size/2):
            #front.add(index)
            front.append(index)

    while len(front) != round(pop_size/2):
        for index in range(pop_size):
            for elem in front:
                if index in list_of_sp[elem] and num_dominated_by[index] > 0:
                    num_dominated_by[index] -= 1
        #front_pom = set()
        front_pom = []
        for index, elem in enumerate(num_dominated_by):
            if elem == 0 and not(index in front):
                #front_pom.add(index)
                front_pom.append(index)

        if len(front)+len(front_pom) > round(pop_size/2) and round(pop_size/2)-len(front) != 0:
            front_pom = crowding_distance(front_pom, round(pop_size/2)-len(front), fitness_list)

        for elem in front_pom:
            #front.add(elem)
            front.append(elem)
    front = [i for i in front]
    front = np.array(front)
    return front


def crowding_distance_old(front: Set[int], num_to_pass: int):
    """
    Sortiranje jedinki po crowding-distance pre dodavanja jedinki u
    sledecu generaciju. Za pocetak se uzima prva i poslednja jedinka iz fronta
    pa se redom dodaju ostale izmedju.

    Parametri:
    front: poslednji izabrani front iz kog treba izdvojiti jedinke
    num_to_pass: broj jedinki koje treba da prodju u sledecu generaciju 
    """
    next_front = set()
    list_front = [elem for elem in front]
    next_front.add(list_front[0])
    if num_to_pass > 1:
        next_front.add(list_front[-1])
    index = 0
    num_to_pass = round(num_to_pass)
    while len(next_front) != num_to_pass:
        next_front.add(list_front[index])
        index += 1
    return next_front


def crowding_distance(front: Set[int], num_to_pass: int, fitness_list: List[List[int]]):
    """
    Sortiranje jedinki po crowding-distance pre dodavanja jedinki u
    sledecu generaciju. Za pocetak se uzima prva i poslednja jedinka iz fronta
    pa se redom dodaju ostale izmedju.

    Parametri:
    front: poslednji izabrani front iz kog treba izdvojiti jedinke
    num_to_pass: broj jedinki koje treba da prodju u sledecu generaciju 
    """
    parameters = [fitness_list[index] for index in front]
    first_param = [elem[0] for elem in parameters]
    numbering = [index for index in range(0, len(front))]
    first_merged = list(zip(first_param, numbering))
    first_merged = sorted(first_merged)
    first_merged = [(elem[0]/(first_merged[-1][0]-first_merged[0][0]), elem[1]) for elem in first_merged]

    second_param = [elem[1] for elem in parameters]
    second_merged = list(zip(second_param, numbering))
    second_merged = sorted(second_merged)
    second_merged = [(elem[0]/(second_merged[-1][0]-second_merged[0][0]), elem[1]) for elem in second_merged]

    crowding_distance_list = np.zeros(len(front))
    crowding_distance_list[first_merged[0][1]] = float("inf")
    crowding_distance_list[first_merged[-1][1]] = float("inf")
    crowding_distance_list[first_merged[0][1]] = float("inf")
    crowding_distance_list[second_merged[-1][1]] = float("inf")

    for index, individual in enumerate(first_merged[1:-1], 1):
        crowding_distance_list[individual[1]] += (first_merged[index+1][0]-first_merged[index-1][0])

    for index, individual in enumerate(second_merged[1:-1], 1):
        crowding_distance_list[individual[1]] += (second_merged[index+1][0]-second_merged[index-1][0])
    
    crowding_distance_list = [elem for elem in crowding_distance_list]
    indexed_crowding_distance_list = [(value, index) for index, value in enumerate(crowding_distance_list)]
    indexed_crowding_distance_list = sorted(indexed_crowding_distance_list)
    print(indexed_crowding_distance_list)
    indexes = []
    for elem in indexed_crowding_distance_list[-1:-1-num_to_pass:-1]:
        indexes.append(elem[1])
    
    end_list = [front[index] for index in indexes]
    return end_list


if __name__ == "__main__":
    fitness_list = [[12, 0],
                [11.5, 0.5],
                [11, 1],
                [10.8, 1.2],
                [10.5, 1.5],
                [10.3, 1.8],
                [9.5, 2],
                [9, 2.5],
                [7, 3],
                [5, 4],
                [2.5, 6],
                [2, 10],
                [1.5, 11],
                [1, 11.5],
                [0.8, 11.7],
                [0, 12]]

    example_front = [0,1,2,3,4,5,6,7,8]
    test_array = np.array([fitness_list[elem] for elem in example_front])
    x = test_array[:, 0]
    y = test_array[:, 1]
    plt.xlabel('Objective A')
    plt.ylabel('Objective B')
    plt.scatter(x, y)
    plt.show()

    x = []
    y = []
    for elem in crowding_distance(example_front, 4, test_array):
        print("ELEMENT ", elem)
        x.append(test_array[elem][0])
        y.append(test_array[elem][1])

    plt.xlabel('Objective A')
    plt.ylabel('Objective B')

    plt.scatter(x, y)
    plt.show()
