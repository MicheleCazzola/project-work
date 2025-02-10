# Lab 2 - Travel Salesman Problem (TSP)

## Repository overview
The solution for this problem is implemented in the notebook [tsp.ipynb](tsp.ipynb).

The folder [cities](cities/) contains the cities for each instance of the problem.

## Official solution
The solution is given in two different versions:
- (pure deterministic) greedy, where the salesman goes each time to the nearest city
- evolutionary, starting from random greedy: here, a GA approach is used, starting from a population produced by a greedy implementation, with some randomness

### Greedy
The greedy algorithm is based on an iteration over all the cities, where each one is visited at each iteration: starting from a given city, the closest one is selected to be visited at the next step.

However, this approach is very good at optimizing the distance between pair of cities in the forward trip, but not at minimizing the distance between the last city and the starting one. For this
reason, the base algorithm is executed iteratively on all cities in the problem space: in this way, the backward edge changes among executions and the penalty it introduced is in some way minimized
by picking the best starting city.

### Evolutionary
The evolutionary algorithm is implemented using a GA approach:
- representation: integer, one-time-items, where each gene is a city, represented by its index
- mutation: inversion mutation
- recombination: inver-over crossover, to preserve traits in the edges, since this TSP is modeled by a DAG, where only adjacent cities matter
- parent selection: fitness-proportional, by randomly selecting the best individual among 10 candidates
- survivor selection: deterministic, fitness-based
- population model: steady state, with offspring size equal to $\frac{2}{3}$ of population summarized

#### Initialization
The initialization plays a relevant role here, since it is observed that the final result has a strong dependency with the technique used.

Here a randomized greedy is used: it follows the same approach of the algorithm used in the [greedy solution](#greedy), but here the k-th nearest city
is chosen with a probability of $9 * {10}^{-(1+k)}$, where k starts from 0.
This operation is repeated starting from all cities, so the cardinality of the
population is equal to the number of cities.

Therefore, the individual of the first generation represent something slightly worse than possible local optima,
allowing the evolutionary algorithm to explore the state space without missing better solutions.

It is possible to have a number of initial greedy solutions different by the desired population size:
- if larger, they are randomly sampled without replacement and using a uniform distribution
- if smaller, the population is extended with random individuals 


### Evaluation
The evaluation is performed as follows:
- cost of the solution, in terms of distance covered by the Hamiltonian cycle, expressed in kilometers (km)
- number of calls to the cost function to find out the best solution: in the greedy approach it is equal to the number of cities, whereas it is smaller than the number of generations in the evolutionary approach

For each instance (except for Chinese one), a reference value is provided: it comes from Wolfram tool and it is expected to be nearly optimal (even if not absolute).

## Collaborations
The following parts:
- choice of initial population
- basic idea behind greedy strategy

have been done in collaboration with [Vincenzo Avantaggiato](https://github.com/VincenzoAvantaggiato). 

## Results
The results are summarized in the following table:

|Instance name  |Greedy cost    |Greedy calls|EA cost|EA calls|Best result|
|:-----:        |:--:           |:--: |:--:|:--: |:--:|
|Vanuatu        |1475.528       |8  |1345.545 |3   |1345.54|
|Italy          |4436.032       |46  |4263.110 |898  |4172.76|
|Russia         |40051.587      |167 |34216.317 |1317 |32722.5|
|US             |46244.333      |326|40047.079 |1997|39016.62|
|China          |62116.045      |726|55240.404 |4991|-|

**Notes**::
- the columns *Greedy/EA calls*: displays the number of generations necessary to the algorithm to find the (local) optimal solution;
- the columns *Greedy/EA cost*: displays the cost of the solution found, as absolute value (opposite of fitness, conceptually);
- the results are collected among several tries, only the best for each instance is reported.

## Observations
The evolutionary approach used is sub-optimal (wrt the Wolfram result), but it can be executed in few minutes at the maximum.
The greedy solution is worse than the evolutionary one, but its execution time is in the order of the seconds.

The instances **USA** and **China** could be significantly improved by increasing the number of generations, but it has been avoided due to limitations of computational resources.

### Particular patterns
In the instance **Vanuatu**, the optimal solution is reached after few steps, due to the reduced dimension of the state (and problem) space.
In the instance **Italy**, the techniques adopted are not able to improve the fitness after few hundreds generations, despite having reached a good solution. This happened quite often, but for some reason there are random configurations that allow to find better solutions.
