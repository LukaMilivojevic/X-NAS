from genalglib import create_solutions
from genalglib import populate
from genalglib import find_pareto
from genalglib import calculate_fitness_all
from genalglib import plot_param
from typing import Tuple
from os import path
import numpy as np
import random


def crossover(parent_1: np.ndarray, parent_2: np.ndarray, max_neurons: int = 64) -> Tuple[np.ndarray]:
    """
    Dobijanje dve nove jedinke ombinovanjem parent jedinki iz trenutne populacije.

    Parametri:
        parent_n: parent jedinke cijim se crossoverom dobijaju nove
    """

    counter_1 = 0
    for elem in parent_1:
        if elem == 0:
            break
        counter_1 += 1
    
    counter_2 = 0
    for elem in parent_2:
        if elem == 0:
            break
        counter_2 += 1
    
    chrom_len = min(counter_1, counter_2)
    crossover_point = random.randint(0, chrom_len-1)
    child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    number_of_layers = child_1[2]
    for index in range(1, number_of_layers+1):
        if child_1[index+2] == 0:
            child_1[index+2] = random.randint(1, max_neurons)
    
    child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    number_of_layers = child_2[2]
    for index in range(1, number_of_layers+1):
        if child_2[index+2] == 0:
            child_2[index+2] = random.randint(1, max_neurons)

    return child_1, child_2


def breed(population: np.ndarray, population_size: int) -> np.ndarray:
    """
    Razmnozavanje jedinki crossoverom

    Parametri:
        population: populacija jedinki
    """
    current_population_size = population.shape[0]
    print("SIZE1 ", current_population_size)
    while population.shape[0] <= population_size:
        new_population = []
        parent_1 = population[random.randint(0, current_population_size-1)]
        parent_2 = population[random.randint(0, current_population_size-1)]
        child_1, child_2 = crossover(parent_1, parent_2)
        new_population.append(child_1)
        if population.shape[0] < population_size-2:
            new_population.append(child_2)
        population = np.vstack((population, np.array(new_population)))

    return population


def mutation(population: np.ndarray, probability: float, max_layers: int, max_neurons: int):
    """
    U svakoj generaciji postoji verovatnoca da ce neka jedinka
    nasumicno mutirati. U ovom slucaju mutacija znaci invertovanje
    svakog bita hromozoma iz 0 u 1.
    
    Parametri:
        population: populacija jedinki
        probability: verovatnoca za mutiranje
        chrom_len: duzina hromozoma
    """
    for individual in population:
        counter = 0
        for elem in individual:
            if elem == 0:
                break
            counter += 1
        for index in range(counter):
            if np.random.random_sample() <= probability:
                if index > 2:
                    individual[index] = random.randint(1, max_neurons)
                elif index == 2:
                    old_layers = individual[index]
                    new_layers = random.randint(2, max_layers)
                    individual[index] = new_layers
                    if new_layers > old_layers:
                        for i in range(old_layers+2, new_layers+3):
                            individual[i] = random.randint(1, max_neurons)
                elif index == 0:
                    individual[index] = round(random.uniform(0.001, 0.1), 3)
                elif index == 1:
                    individual[index] = round(random.uniform(0.8, 1), 2)


def local_file_name(folder: str, file: str) -> str:
    return path.join(path.dirname(path.abspath(__file__)), folder, file)


def main(number_of_generations: int, population_size: int = 100):
    """
    Glavni deo algoritma

    Parametri:
        number_of_generations: zeljeni broj generacija
        population_size: velicina populacije
    """
    solutions = create_solutions()
    population = populate(population_size)
    for generation in range(number_of_generations):
        print(f'Generation (out of {number_of_generations}): {generation}')
        fitness = calculate_fitness_all(population, solutions, generation)
        plot_param(fitness, f"generacija{generation}")
        print("POPULACIJA: ", population)
        selected = find_pareto(population, fitness)
        with open(local_file_name(f"generation{generation}", "pareto"), "wb") as file:
            np.savetxt(file, selected, delimiter=',')
        population = np.array([population[i] for i in selected])
        population = breed(population, population_size)
        mutation(population, 0.01, 10, 64)
    print(fitness)
    plot_param(fitness, "poslednja")


if __name__ == "__main__":
    generations_num = 50
    population_size = 50
    main(generations_num, population_size)
