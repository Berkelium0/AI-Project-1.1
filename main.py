import heapq

import pandas as pd
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt

PROBLEMS = "example-problems.csv"
mini_schedule_graph = None
schedule_graph = None


def parse_time(time_str):
    time_str = time_str.strip("'")
    return datetime.strptime(time_str, "%H:%M:%S")


def construct_graph(file_path):
    graph = nx.MultiDiGraph()
    previous_node = None
    for chunk in pd.read_csv(file_path, chunksize=10000):
        for _, row in chunk.iterrows():
            train_no = row["Train No."].strip()
            islno = row["islno"]
            station_code = row["station Code"].strip()
            arrival_time = parse_time(row["Arrival time"])
            departure_time = parse_time(row["Departure time"])
            distance = row["Distance"]

            train_info = {
                "train_no": train_no,
                "islno": islno,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "distance": distance,
                "station_code": station_code,
            }

            if station_code in graph:
                graph.nodes[station_code]['trains'].append(train_info)
            else:
                # Add the node with the train information list
                graph.add_node(station_code, trains=[train_info])

            if previous_node:
                if train_no == previous_node["train_no"]:

                    traveltime = (arrival_time - previous_node["departure_time"]).total_seconds()
                    if traveltime < 0:  # handle next day arrival
                        traveltime += 24 * 3600

                    graph.add_edge(previous_node["station_code"],
                                   station_code,
                                   stops=1,
                                   traveltime=int(traveltime),
                                   train_info=previous_node)

            previous_node = train_info
    return graph


def visualize_graph(graph):
    print("visualizing")
    pos = nx.spring_layout(graph, k=20)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold',
            edge_color='gray')
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red')
    plt.title('Train Connection Graph')
    plt.show()


def dijkstra(graph, start, end, weight='weight'):
    # Clear visited tags before running the algorithm
    for u, v, key in graph.edges(keys=True):
        if 'visited' in graph[u][v][key]:
            del graph[u][v][key]['visited']

    # Priority queue to hold nodes to explore
    queue = []
    # Start with the start node with a cost of 0
    heapq.heappush(queue, (0, start, None))
    # Dictionary to hold the cost to reach each node
    costs = {node: float('inf') for node in graph.nodes}
    costs[start] = 0
    # Dictionary to hold the path to each node
    predecessors = {node: None for node in graph.nodes}

    while queue:
        # Get the node with the smallest cost
        current_cost, current_node, edge = heapq.heappop(queue)

        # If we reached the end node, we're done
        if current_node == end:
            break

        # Explore the neighbors
        for neighbor in graph.successors(current_node):
            for key, data in graph[current_node][neighbor].items():
                weight_value = data.get(weight, 1)  # Default weight is 1 if not provided
                new_cost = current_cost + weight_value

                # If we found a cheaper path to the neighbor, update its cost and path
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    predecessors[neighbor] = (current_node, key)
                    heapq.heappush(queue, (new_cost, neighbor, (current_node, neighbor, key)))

    # Reconstruct the path from end to start using the predecessors dictionary
    path = []
    current = end
    while predecessors[current] is not None:
        prev, edge_key = predecessors[current]
        path.insert(0, (prev, current, edge_key))
        current = prev

    # Tag the edges as visited
    for u, v, key in path:
        if 'visited' not in graph[u][v][key]:
            graph[u][v][key]['visited'] = True

    return path, costs[end]


def path_writer(path, graph):
    edges = []

    for u, v, key in path:
        if 'visited' in graph[u][v][key]:
            edges.append((u, v, key, graph[u][v][key]))
    print(edges)

    sol = ""
    prev_edge = None
    for stop in edges:
        if prev_edge is None:
            sol = sol + f'{stop[3]["train_info"]["train_no"]} : {stop[3]["train_info"]["islno"]}'
        elif prev_edge["train_info"]["train_no"] != stop[3]["train_info"]["train_no"]:
            sol = sol + (f' -> {prev_edge["train_info"]["islno"] + 1} ; {stop[3]["train_info"]["train_no"]} : '
                         f'{stop[3]["train_info"]["islno"]}')
        prev_edge = stop[3]
    sol = sol + f' -> {edges[-1][3]["train_info"]["islno"] + 1}'
    print(sol)

    return sol


def optimal_path_finder(row):
    global mini_schedule_graph
    global schedule_graph
    print(row)
    from_station = row["FromStation"]
    to_station = row["ToStation"]
    cost_function = row["CostFunction"]

    graph = mini_schedule_graph if row["Schedule"] == "mini-schedule.csv" else schedule_graph

    if cost_function == "stops":
        weight = 'stops'
    elif cost_function == "traveltime":
        weight = 'traveltime'
    elif cost_function == "price":
        weight = 'price'
    else:
        weight = 'arrivaltime'

    path, cost = dijkstra(graph, from_station, to_station, weight=weight)

    return {
        "ProblemNo": row["ProblemNo"],
        "Connection": path_writer(path, graph) if path else "No path",
        "Cost": cost
    }


def format_solution(solution):
    problem_no = solution['ProblemNo']
    connections = solution['Connection']
    connections_str = " ; ".join(map(str, connections)) if isinstance(connections, list) else connections
    cost = solution['Cost']
    return problem_no, connections_str, cost


def main():
    global mini_schedule_graph
    global schedule_graph
    problem_df = pd.read_csv(PROBLEMS)

    # Construct graphs
    mini_schedule_graph = construct_graph("mini-schedule.csv")
    schedule_graph = construct_graph("schedule.csv")

    # Visualize the Graphs
    # visualize_graph(mini_schedule_graph)
    # visualize_graph(schedule_graph)
    # print("All stations and their train information:")
    # for node, data in mini_schedule_graph.nodes(data=True):
    #     print(f"Station: {node}, Attributes: {data}")

    # Get optimal path
    # solutions = problem_df.apply(optimal_path_finder, axis=1)

    solutions = problem_df.iloc[:40].apply(optimal_path_finder, axis=1)

    # Save solutions to CSV
    formatted_solutions = [format_solution(sol) for sol in solutions]
    solutions_df = pd.DataFrame(formatted_solutions, columns=['ProblemNo', 'Connection', 'Cost'])
    solutions_df.to_csv("my-example-solutions.csv", index=False)


if __name__ == "__main__":
    main()
