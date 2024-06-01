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
    graph = nx.DiGraph()
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

            # travel_time = (arrival_time - departure_time).total_seconds()
            # if travel_time < 0:  # handle next day arrival
            #     travel_time += 24 * 3600

            if station_code in graph:
                graph.nodes[station_code]['trains'].append(train_info)
            else:
                # Add the node with the train information list
                graph.add_node(station_code, trains=[train_info])

            if previous_node:
                if train_no == previous_node["train_no"]:
                    graph.add_edge(previous_node["station_code"], station_code, stops=1)

            previous_node = train_info
    return graph


def optimal_path_finder(row):
    global mini_schedule_graph
    global schedule_graph
    print(row)
    from_station = row["FromStation"]
    to_station = row["ToStation"]
    cost_function = row["CostFunction"]

    graph = mini_schedule_graph if row["Schedule"] == "mini-schedule.csv" else schedule_graph

    if cost_function == "stops":
        weight = 'stops'  # Add stops calculation to edges if necessary
    elif cost_function == "traveltime":
        weight = 'weight'
    elif cost_function == "price":
        weight = 'price'  # Add price calculation to edges if necessary
    else:
        weight = 'arrivaltime'  # Handle arrival time cost calculation

    path, cost = dijkstra(graph, from_station, to_station, weight=weight)

    return {
        "ProblemNo": row["ProblemNo"],
        "Connection": path if path else "No path",
        "Cost": cost
    }


def dijkstra(graph, from_station, to_station, weight='weight'):
    try:
        path = nx.dijkstra_path(graph, from_station, to_station, weight=weight)
        cost = nx.dijkstra_path_length(graph, from_station, to_station, weight=weight)
        return path, cost
    except nx.NetworkXNoPath:
        return None, float('inf')


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


def format_solution(solution):
    problem_no = solution['ProblemNo']
    connections = solution['Connection']
    connections_str = " ; ".join(map(str, connections)) if isinstance(connections, list) else connections
    cost = solution['Cost']
    return problem_no, connections_str, cost


# Sample usage
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

    solutions = problem_df.iloc[:20].apply(optimal_path_finder, axis=1)

    # Save solutions to CSV
    print(type(solutions))
    formatted_solutions = [format_solution(sol) for sol in solutions]
    solutions_df = pd.DataFrame(formatted_solutions, columns=['ProblemNo', 'Connection', 'Cost'])
    solutions_df.to_csv("my-example-solutions.csv", index=False)


if __name__ == "__main__":
    main()
