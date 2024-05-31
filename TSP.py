import time
from collections import deque
import heapq


# Function to calculate the total distance of a given path
def calculate_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    # Add distance from last city back to the first city
    total_distance += distance_matrix[path[-1]][path[0]]
    return total_distance


# BFS TSP solver
def tsp_bfs(distance_matrix):
    n = len(distance_matrix)
    start_city = 0
    queue = deque([(start_city, [start_city], 0)])  # (current_city, path, current_distance)
    min_distance = float('inf')
    best_path = None

    while queue:
        current_city, path, current_distance = queue.popleft()

        if len(path) == n:
            # Complete the cycle by returning to the start city
            total_distance = current_distance + distance_matrix[current_city][start_city]
            if total_distance < min_distance:
                min_distance = total_distance
                best_path = path + [start_city]
            continue

        for next_city in range(n):
            if next_city not in path:
                new_distance = current_distance + distance_matrix[current_city][next_city]
                queue.append((next_city, path + [next_city], new_distance))

    return best_path, min_distance


def tsp_ucs(distance_matrix):
    n = len(distance_matrix)
    start_city = 0
    priority_queue = [(0, start_city, [start_city])]
    min_distance = float('inf')
    best_path = None

    while priority_queue:
        current_distance, current_city, path = heapq.heappop(priority_queue)

        if len(path) == n:
            total_distance = current_distance + distance_matrix[current_city][start_city]
            if total_distance < min_distance:
                min_distance = total_distance
                best_path = path + [start_city]
            continue

        for next_city in range(n):
            if next_city not in path:
                new_distance = current_distance + distance_matrix[current_city][next_city]
                heapq.heappush(priority_queue, (new_distance, next_city, path + [next_city]))

    return best_path, min_distance

# Define a State class to represent the state of the search
class State:
    def __init__(self, current_city, visited, cost, path):
        self.current_city = current_city
        self.visited = visited
        self.cost = cost
        self.path = path


    def __lt__(self, other):
        return self.cost < other.cost


# A* search algorithm for solving the Traveling Salesman Problem
def tsp_a_star(graph, start):
    n = len(graph)
    pq = []
    initial_state = State(start, set([start]), 0, [start])
    heapq.heappush(pq, (0, initial_state))

    while pq:
        _, current_state = heapq.heappop(pq)

        if len(current_state.visited) == n:

            return current_state.cost + graph[current_state.current_city][start], current_state.path + [start]

        for next_city in range(n):
            if next_city not in current_state.visited:
                new_visited = current_state.visited.copy()
                new_visited.add(next_city)
                new_cost = current_state.cost + graph[current_state.current_city][next_city]
                new_path = current_state.path + [next_city]
                heuristic_cost = mst_heuristic(graph, new_visited, next_city, start)
                total_cost = new_cost + heuristic_cost
                new_state = State(next_city, new_visited, new_cost, new_path)
                heapq.heappush(pq, (total_cost, new_state))


# Heuristic function to estimate the remaining cost to reach the goal
def mst_heuristic(graph, visited, current_city, start):
    unvisited = set(range(len(graph))) - visited
    if not unvisited:
        return graph[current_city][start]
    mst_cost = 0
    return mst_cost + min(graph[current_city][city] for city in unvisited)
# Example usage
distance_matrix =  adjacency_matrix = [
    [0, 70, 87, 6, 13, 82, 35, 55],
    [70, 0, 2, 62, 29, 32, 77, 56],
    [87, 2, 0, 83, 12, 2, 10, 20],
    [6, 62, 83, 0, 86, 39, 96, 8],
    [13, 29, 12, 86, 0, 24, 61, 13],
    [82, 32, 2, 39, 24, 0, 95, 77],
    [35, 77, 10, 96, 61, 95, 0, 1],
    [55, 56, 20, 8, 13, 77, 1, 0]
]

start_time=time.time()
best_path, min_distance = tsp_bfs(distance_matrix)
end_time = time.time()
print(f"Best path: {best_path}")
print(f"Minimum distance: {min_distance}")
print(f"Time taken: {(end_time - start_time)*1000}ms")

start_time = time.time()
best_path, min_distance = tsp_ucs(distance_matrix)
end_time = time.time()
print(f"Best path: {best_path}")
print(f"Minimum distance: {min_distance}")
print(f"Time taken: {(end_time - start_time)*1000}ms")

start_time = time.time()
min_distance,best_path = tsp_a_star(distance_matrix,0)
end_time = time.time()
print(f"Best path: {best_path}")
print(f"Minimum distance: {min_distance}")
print(f"Time taken: {(end_time - start_time)*1000}ms")


