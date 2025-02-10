# Lab 2 - Travelling Salesman Problem (TSP)

## Imports


```python
import functools
import pandas as pd
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass

```

## General constants


```python
PATH = "cities/"
INSTANCES = [
    "vanuatu.csv",
    "italy.csv",
    "russia.csv",
    "us.csv",
    "china.csv"
]
BEST_RESULTS = [
    -1_345.54,
    -4_172.76,
    -32_722.5,
    -39_016.62,
    None
] 
```

## Classes


```python
class City:
    
    @staticmethod
    def distance(start, end):
        return geopy.distance.geodesic(
            (start.lat, start.lon), (end.lat, end.lon)
        ).km
    
    def __init__(self, name, lat, lon):
        self.name: str = name
        self.lat: float | np.float64 = lat
        self.lon: float | np.float64 = lon
        
    def __repr__(self):
        return f"{self.name}"
    
    def __str__(self):
        return f"{self.name} ({self.lat}°, {self.lon}°)"
```


```python
@dataclass
class Individual:
    genome: np.ndarray
    fitness: np.float64 | float = None
```

## Helper functions


```python
def distance_matrix(coordinates: list) -> np.ndarray:
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(i+1):
            dist_matrix[i, j] = dist_matrix[j, i] = City.distance(coordinates[i], coordinates[j]) if i != j else 0
          
    return dist_matrix


def counter(fn):
    """Simple decorator for counting number of calls"""

    @functools.wraps(fn)
    def helper(*args, **kargs):
        helper.calls += 1
        return fn(*args, **kargs)

    helper.calls = 0
    return helper


@counter
def cost(solution: np.ndarray, dist_matrix: np.ndarray) -> np.float64 | float:
    """Cost of a cycle"""
    return np.sum(
        np.array([
            dist_matrix[start, end] for (start, end) in zip(solution[:-1], solution[1:])
        ])
    )
    
    
def fitness(individual, dist_matrix) -> np.float64 | float:
    solution = np.array(individual.genome.tolist() + [individual.genome[0]])
    return -cost(solution, dist_matrix)
```

#### Parent selection


```python
def parent_selection(population) -> Individual:
    BUCKET_SIZE = 10
    candidates = sorted(np.random.choice(population, BUCKET_SIZE), key=lambda e: e.fitness, reverse=True)
    return candidates[0]
```

#### Crossover


```python
def cycle_xover(p1: Individual, p2: Individual) -> Individual:
    
    num_cities = p1.genome.size
    genome = p1.genome.copy()
    
    a, b = np.random.randint(num_cities-1), np.random.randint(num_cities-1)
    l1, l2 = min(a, b), max(a, b)
    segment = p1.genome[l1:l2+1]
    
    genome[l1:l2+1] = p1.genome[l1:l2+1]
    others = p2.genome[~np.isin(p2.genome, segment)]
    
    assert (len(others) - l1) == (len(genome) - (l2+1))
    
    genome[:l1] = others[:l1]
    genome[l2+1:] = others[l1:]
    
    return Individual(genome)


def inver_over_xover(p1: Individual, p2: Individual) -> Individual:
    """INVER-OVER crossover"""
    genome1 = p1.genome.copy()
    num_cities = p1.genome.size
    genome = np.zeros(num_cities, dtype=np.int16)
    
    p1_start = np.random.randint(num_cities-1)
    genome1 = np.roll(genome1, -p1_start)
    
    p2_start = p2.genome.tolist().index(genome1[0])
    p1_end = genome1.tolist().index(p2.genome[(p2_start+1) % num_cities])
    
    genome[0], genome[1] = genome1[0], genome1[p1_end]
    
    genome[2:p1_end+1], genome[p1_end+1:] = genome1[1:p1_end][::-1], genome1[p1_end+1:]
    genome = np.roll(genome, p1_start)
    
    return Individual(genome)
```

#### Mutation


```python
def scramble_mutation(p: Individual) -> Individual:
    SIGMA = 0.5
    genome = p.genome.copy()
    mask = np.random.random(len(genome)) < SIGMA
    genome[mask] = np.random.permutation(genome[mask])
    return Individual(genome)


def inversion_mutation(p: Individual) -> Individual:
    a, b = np.random.randint(0, p.genome.size-1), np.random.randint(0, p.genome.size-1)
    l1, l2 = min(a, b), max(a, b)
    
    genome = p.genome.copy()
    genome = np.roll(genome, -l1)
    genome[:l2-l1+1] = genome[:l2-l1+1][::-1]
    
    return Individual(genome)


def swap_mutation(p: Individual) -> Individual:
    l1, l2 = map(np.random.randint, [0,0], [p.genome.size, p.genome.size])
    genome = p.genome.copy()
    genome[l1], genome[l2] = genome[l2], genome[l1]
    return Individual(genome)
```

#### Operators selection


```python
xover = inver_over_xover
mutation = inversion_mutation
```

## Greedy


```python
def greedy_solve(coordinates, dist_matrix, city, rnd):
    """Greedy algorithm with random initialization: sub-optimal"""
    temp = dist_matrix.copy()
    
    num_cities = len(coordinates)
    visited = np.full(num_cities, False)
    
    solution = -np.ones(num_cities+1, dtype=np.int16)
    solution[0], visited[0] = city, True
    for step in range(num_cities-1):
        temp[:, city] = np.inf
        
        selected = 0
        sorted_indexes = np.argsort(temp[city])
        while rnd and np.random.rand() < 0.2 and selected < num_cities - step - 2:
            selected += 1
            
        city = sorted_indexes[selected]
            
        solution[step+1] = city
        visited[city] = True
        
    solution[-1] = solution[0]
    #print(solution[:-1])
    assert set(solution[:-1]) == set(range(num_cities))
    
    return solution, -cost(solution, dist_matrix)
```


```python
def greedy_init(coordinates, dist_matrix, rnd=False):
    
    best_fitness, best_sol = -np.inf, None
    solutions =  []
    for start in tqdm(range(coordinates.size)):
        sol, fitness_sol = greedy_solve(coordinates, dist_matrix, start, rnd)
        solutions.append(sol[:-1])
        
        if fitness_sol > best_fitness:
            best_fitness, best_sol = fitness_sol, sol
            
    assert best_sol is not None
            
    return solutions, best_fitness, best_sol
```

## Evolutionary


```python
def single_mutation(p: Individual):
    from_pos = np.random.randint(p.genome.size)
    to_pos = np.random.randint(p.genome.size)
    genome = p.genome.copy()
    genome[to_pos], genome[from_pos] = p.genome[from_pos], p.genome[to_pos]
    return Individual(genome)
```


```python
def evolutionary_solve(coordinates, dist_matrix: np.ndarray, start_population, pop_size, max_generations):
    
    OFFSPRING_SIZE = int(2*pop_size / 3)
    
    start_pop_len = len(start_population)
    num_cities = len(coordinates)
    population = [Individual(start_individual) for start_individual in start_population]
    
    for i in population:
        i.fitness = fitness(i, dist_matrix)
    
    assert len(population) == start_pop_len and start_pop_len >= 0
        
    # Discard some individuals if too many wrt start ones
    if start_pop_len > pop_size:
        population = np.random.choice(population, size=pop_size, replace=False).tolist()
        np.random.shuffle(population)
        
    # Extend population if needed more individuals than start population
    elif start_pop_len < pop_size:
        population.extend(
            [Individual(np.random.permutation(num_cities)) for _ in range(pop_size - start_pop_len)]
        )
        
        # Compute fitness for new individuals
        for i in population[start_pop_len:]:
            i.fitness = fitness(i, dist_matrix)
    
    champions = [max(population, key=lambda i: i.fitness).fitness]
    
    for _ in tqdm(range(max_generations)):
        offspring = []
        for _ in range(OFFSPRING_SIZE):
            if np.random.random() < 0.1:
                p = parent_selection(population)
                o = mutation(p)
            else:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                o = xover(p1, p2)
            
            offspring.append(o)
        
        for i in offspring:
            i.fitness = fitness(i, dist_matrix)
            
        population.extend(offspring)
        
        # Elitism + generational model
        # population.sort(key=lambda i: i.fitness, reverse=True)
        # population = population[:RETAIN_SIZE_ELITIST]
        
        # Survivor selection
        population.sort(key=lambda i: i.fitness, reverse=True)
        population = population[:pop_size]
        
        champions.append(population[0].fitness)
    
    return population[0].genome, population[0].fitness, champions
```

## Solver


```python
MAX_GENERATIONS = [100, 1000, 2000, 2000, 5000]
POPULATION_SIZES = [100, 100, 200, 200, 200]
```


```python
def solve(PATH, INSTANCES):
    for (INSTANCE, BEST_RESULT, MAX_GEN, POP_SIZE) in list(zip(INSTANCES, BEST_RESULTS, MAX_GENERATIONS, POPULATION_SIZES)):
        
        print(f"Instance {INSTANCE}")
        
        cities = pd.read_csv(f"{PATH}{INSTANCE}", header=None, names=["name", "lat", "lon"])
        
        coordinates = np.array([City(city.name, city.lat, city.lon) for city in cities.itertuples()])
        dist_matrix = distance_matrix(coordinates)
        
        _, fitness_greedy, _ = greedy_init(coordinates, dist_matrix, rnd=False)
        greedy_solutions, _, _ = greedy_init(coordinates, dist_matrix, rnd=True)
        calls_greedy = len(coordinates)
        
        _, fitness_ea, champions = evolutionary_solve(coordinates, dist_matrix, greedy_solutions, POP_SIZE, MAX_GEN)
        best_ea = max(champions)
        
        plt.figure(figsize=(14,8))
        plt.plot(champions, color="blue")
        plt.scatter(range(len(champions)), champions, marker=".", color="blue")
        plt.hlines(fitness_greedy, xmin=0, xmax=len(champions), linestyles="-", color="red")
        if BEST_RESULT is not None:
            plt.hlines(BEST_RESULT, xmin=0, xmax=len(champions), linestyles="-", color="darkgreen")
        plt.show()
        
        print(f"Greedy solution: {fitness_greedy:.3f}\nCost calls: {calls_greedy}")
        print(f"EA solution: {fitness_ea:.3f}")
        print(f"Best solution: {F'{BEST_RESULT:.3f}' if BEST_RESULT is not None else '-'}")
        print(f"Number of steps: {champions.index(best_ea)}")
```


```python
solve(PATH, INSTANCES)
```
