import heapq
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

PROBLEMS = "problems.csv"
mini_schedule_graph = None
schedule_graph = None


# Function to parse time strings into datetime objects
def parse_time(time_str, default_year=1970):
    time_str = time_str.strip("'")  # Remove any single quotes around the time string
    try:
        # Try to parse the full datetime string (including the year)
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # If year is missing, assume the default year
        try:
            time_with_default_year = f"{default_year}-01-01 {time_str}"
            return datetime.strptime(time_with_default_year, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Invalid time format. Expected 'HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'")


# Function to construct a graph from a CSV file
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

            # Add or update the node with train information
            if station_code in graph:
                graph.nodes[station_code]['trains'].append(train_info)
            else:
                graph.add_node(station_code, trains=[train_info])

            # Add edges between consecutive stations of the same train
            if previous_node:
                if train_no == previous_node["train_no"]:
                    traveltime = (arrival_time - previous_node["departure_time"]).total_seconds()
                    if traveltime < 0:  # handle next day arrival
                        traveltime += 24 * 3600

                    graph.add_edge(previous_node["station_code"],
                                   station_code,
                                   stops=1,
                                   traveltime=int(traveltime),
                                   price=1,
                                   train_info=previous_node)

            previous_node = train_info
    return graph


# Function to visualize the graph
def visualize_graph(graph):
    print("visualizing")
    pos = nx.kamada_kawai_layout(graph)
    plt.figure(figsize=(25, 25))
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold',
            edge_color='gray')
    labels = nx.get_edge_attributes(graph, 'price')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red')
    plt.title('Train Connection Graph')
    plt.show()


# Function to format a datetime object into a specific string format
def format_datetime(dt):
    base_date = datetime(1970, 1, 1)
    delta = dt - base_date
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_time = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_time


# Dijkstra's algorithm to find the shortest path
def dijkstra(graph, start, end, weight):
    # Clear visited tags before running the algorithm
    for u, v, key in graph.edges(keys=True):
        if 'visited' in graph[u][v][key]:
            del graph[u][v][key]['visited']

    # Parse the departure time
    if 'arrivaltime' in weight:
        departure_time = datetime.strptime(weight.split(" ")[1], "%H:%M:%S")
    else:
        departure_time = None

    # Priority queue to hold nodes to explore
    queue = []
    heapq.heappush(queue, (0, start, None, (None, 0), departure_time))

    # Dictionary to hold the cost to reach each node
    costs = {node: float('inf') for node in graph.nodes}
    costs[start] = 0

    # Dictionary to hold the path to each node
    predecessors = {node: None for node in graph.nodes}

    while queue:
        current_cost, current_node, current_edge, prev_train, current_time = heapq.heappop(queue)

        # If we reached the end node, we're done
        if current_node == end:
            break

        # Explore the neighbors
        for neighbor in graph.successors(current_node):
            for key, data in graph[current_node][neighbor].items():
                train_info = data["train_info"]
                train_no = train_info['train_no']
                train_departure_time = train_info['departure_time']

                if 'arrivaltime' in weight:
                    if train_departure_time < current_time:
                        train_departure_time += timedelta(days=1)  # Adjust for overnight travel

                    travel_duration = data['traveltime']
                    new_time = train_departure_time + timedelta(seconds=travel_duration)
                    # Check if changing trains requires additional time (e.g., 15 minutes)
                    if current_edge and current_edge[2] != key:
                        train_change_duration = timedelta(minutes=15)
                        new_time += train_change_duration
                else:
                    new_time = current_time

                # Calculate the cost based on the weight function
                if weight == 'price':
                    weight_value = data.get('price', 1)
                    if prev_train[0] is not None:
                        if train_no == prev_train[0]:
                            if prev_train[1] >= 10:
                                new_cost = current_cost  # No additional cost after 10th stop
                            else:
                                new_cost = current_cost + weight_value

                            prev_train = (train_no, prev_train[1] + 1)  # Increment stop count
                        else:
                            new_cost = current_cost + weight_value
                            prev_train = (train_no, 1)  # Reset stop count for new train
                    else:
                        new_cost = current_cost + weight_value
                        prev_train = (train_no, 1)
                elif 'arrivaltime' in weight:
                    new_cost = new_time.timestamp()
                else:
                    weight_value = data.get(weight, 1)
                    new_cost = current_cost + weight_value

                # If we found a cheaper path to the neighbor, update its cost and path
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    predecessors[neighbor] = (current_node, key)
                    heapq.heappush(queue, (new_cost, neighbor, (current_node, neighbor, key), prev_train, new_time))

    # Reconstruct the path from end to start using the predecessors dictionary
    path = []
    current = end
    i = 0
    while predecessors[current] is not None and i < 100:
        prev, edge_key = predecessors[current]
        path.insert(0, (prev, current, edge_key))
        current = prev
        i += 1

    # Tag the edges as visited
    for u, v, key in path:
        graph[u][v][key]['visited'] = True
    if 'arrivaltime' in weight:
        return path, format_datetime(datetime.fromtimestamp(costs[end]))
    else:
        return path, costs[end]


# Function to write the path into a readable format
def path_writer(path, graph):
    edges = []

    for u, v, key in path:
        if 'visited' in graph[u][v][key]:
            edges.append((u, v, key, graph[u][v][key]))

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

    return sol


# Function to find the optimal path for a given row in the problems CSV
def optimal_path_finder(row):
    global mini_schedule_graph
    global schedule_graph

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
        weight = cost_function

    path, cost = dijkstra(graph, from_station, to_station, weight=weight)

    return {
        "ProblemNo": row["ProblemNo"],
        "Connection": path_writer(path, graph) if path else "No path",
        "Cost": cost
    }


# Function to format the solution for output
def format_solution(solution):
    problem_no = solution['ProblemNo']
    connections = solution['Connection']
    connections_str = " ; ".join(map(str, connections)) if isinstance(connections, list) else connections
    cost = solution['Cost']
    return problem_no, connections_str, cost


# Main function to execute the workflow
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

    # Get optimal path
    solutions = problem_df.apply(optimal_path_finder, axis=1)
    # solutions = problem_df.iloc[61:62].apply(optimal_path_finder, axis=1)

    # Save solutions to CSV
    formatted_solutions = [format_solution(sol) for sol in solutions]
    solutions_df = pd.DataFrame(formatted_solutions, columns=['ProblemNo', 'Connection', 'Cost'])
    solutions_df.to_csv("my-solutions.csv", index=False)


if __name__ == "__main__":
    main()
