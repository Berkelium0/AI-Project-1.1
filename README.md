# Repository for ss24.1.1/team776

This is the repository for you solution. You can modify this README file any way you see fit.

**Topic:** SS24 Assignment 1.1: Find Train Connections

# Train Route Optimization

## Dependencies

This project is written in Python 3.11. While it may work on older versions, this has not been tested. The project
relies on the following external libraries:

- `heapq` for priority queue implementation.
- `pandas` for data manipulation and CSV reading.
- `networkx` for graph construction and manipulation.
- `datetime` for time parsing and handling.
- `matplotlib` for graph visualization.

Install the necessary libraries using pip:

```bash
pip install pandas networkx matplotlib
```

## Repository Structure

- `schedule.csv` and `mini-schedule.csv`: CSV files with the train schedule information.
- `example-problems.csv` and `example-solutions.csv`: Example data files used for debugging and validation purposes.
- `my-example-solutions.csv`: Output file storing the generated answers. This is used to compare the generated solutions
  with the example solutions.
- `problems.csv`: Contains the problem data that needs to be solved, consisting of 120 problems.
- `solutions.csv`: Contains the generated answers for the problems.
- `main.py`: The primary script that reads the problems, solves them, and writes the answers to a file.
- `Graphs`: File with two plots of the graphs drawn from the schedules.

## How to Run

To run this code with other data, modify the paths and filenames in the `main.py` script accordingly. You can edit the
loop at the bottom of the script to specify the dataset you want to process.

## The Problem

The task is to find optimal train routes between stations based on various cost functions. The routes are constructed
from train schedules, and the cost can be defined in terms of the number of stops, travel time, price or arrival time.
The problem dataset includes different schedules and cost functions to optimize for.

### Problem Modes and Challenges

The problem data is divided into four main tasks, each addressing a different challenge. Here's a brief overview:

| Problem | Challenges   |
|---------|--------------|
| A       | stops        |
| B       | travel time  |
| C       | price        |
| D       | arrival time |

## My Approach

Before focusing on specific challenges, I first created the skeleton of the solver code:

1. **Read from CSV**: I started by reading the problem data and train schedules from CSV files using `pandas`.
2. **Construct Graphs**: I constructed directed graphs using `networkx`, where nodes represent stations and edges
   represent costs. Each edge carries relevant information such as travel time, distance, and train number.
3. **Visualize the Graph**: Optionally, I checked the structure of the graph by visualizing it with `matplotlib` to
   ensure correctness.
4. **Optimal Path Finder**: I created a pathfinder function that takes the required information from the
   problem statement and adjusts the weight function accordingly.
5. **Dijkstra Function**: The pathfinder calls the Dijkstra function, implemented to handle different cost functions.
6. **Format and Write Results**: After finding the path and costs, the results are formatted as required and written to
   a CSV file.

### Problem A

This problem asks to find the path with the fewest stops. Each edge's cost is set to 1 since each node (station) is
equally weighted in terms of stops.

### Problem B

For the travel time, the cost is the time it takes to travel from one station to another. The Dijkstra function handles
this by using travel time as the edge weight.

### Problem C

This problem is more complex. It requires tracking the price with the following condition: from the time of boarding a
train, the cost function increases by 1 until the 10th station, after which it stops increasing. This simulates
switching from a stop-based ticket to a train-based ticket.

### Problem D

This is the most challenging part of the homework. The goal is to minimize the arrival time. The cost is set to the
arrival time, and Dijkstra's algorithm tries to find the earliest arrival. This part still has issues and is not fully
implemented. The challenge lies in accurately adjusting for overnight travels and ensuring the correct sequence of train
departures and arrivals.


## Code Breakdown

### Main Function

The `main` function orchestrates the workflow:

- Reads the problem data.
- Constructs the train route graphs.
- Optionally visualizes the graphs.
- Applies the optimal pathfinding algorithm for each problem.
- Saves the solutions to a CSV file.

### Functions

- `parse_time`: Parses time strings into `datetime` objects.
- `construct_graph`: Builds the graph from train schedule CSV files.
- `visualize_graph`: Visualizes the graph using `matplotlib`.
- `format_datetime`: Formats `datetime` objects for output.
- `dijkstra`: Implements Dijkstra's algorithm to find the shortest path in the graph.
- `path_writer`: Converts the path found by Dijkstra's algorithm into a readable format.
- `optimal_path_finder`: Finds the optimal path for a given problem row.
- `format_solution`: Formats the solution for CSV output.
