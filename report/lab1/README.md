# Lab 1 - Set cover problem

## Repository overview
The official notebook is [set-cover.ipynb](set-cover.ipynb), with a solution based on a single mutation tweak. A speed up has been performed to let instances run in about one minute each, at maximum.

The notebook [strategies.ipynb](strategies.ipynb) contains some scripts with other strategies tried. They are in general much slower than the official solution, but they provide slightly better results in some cases, depending on the instance and, sometimes, on the problem data.

## Official solution
The official solution has the following characteristics:
- **single mutation tweak**: a multiple mutation could be more suitable, but it would prevent to speed up the algorithm;
- **data structure** (called *covering* in the tweak function) keeping track of the number of collected sets each element is covered by: in this way, the algorithm is sped up since the computation of the cost is performed only in the universe dimension;
- **heuristic** to avoid the algorithm searching invalid solutions: given a negative tweak (a set is removed by the current solution), if an element results to be uncovered by the current solution, the modification is rolled back, blocking the path of the algorithm in that region of the fitness landscape;
- **random start**: since the cost is computed using the coverage of the universe by the sets collected in the current solution, the algorithm can start from a random solution, without any specific constraint.

It also contains a code snippet with a **greedy optimization** algorithm, which is able to find an approximate solution of the problem, with a factor proportional to *log(n)* with respect to the optimal one, where *n* is the size of the universe. It is possible to check the results by simply running the notebook.

## Collaborations
The following parts:
- tweak function
- snippet of code for plotting history

have been done in collaboration with [Vincenzo Avantaggiato](https://github.com/VincenzoAvantaggiato). 

## Results
The results are summarized in the following table:

|Instance|Universe size|Number of sets|Density|Number of steps|Cost|
|:--:|:--:  |:--: |:--:|:--: |:--:      |
|1   |100   |10   |0.2 |20   |280.70    |
|2   |1000  |100  |0.2 |115  |7877.12   |
|3   |10000 |1000 |0.2 |4889 |127624.80 |
|4   |100000|10000|0.1 |51355|1939231.02|
|5   |100000|10000|0.2 |59199|2155944.29|
|6   |100000|10000|0.3 |62586|2184591.56|

**Notes**: the columns:
-  *Number of steps*: displays the number of steps necessary to the algorithm to find the (local) optimal solution;
-  *Cost*: displays the cost of the solution found, as absolute value (opposite of fitness, conceptually).