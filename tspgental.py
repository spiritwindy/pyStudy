import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import permutations

# 生成随机城市坐标
def generate_cities(n):
    return np.random.rand(n, 2) * 100

# 计算路径距离
def path_distance(path, cities):
    return sum(np.linalg.norm(cities[path[i]] - cities[path[i+1]]) for i in range(len(path) - 1)) + np.linalg.norm(cities[path[-1]] - cities[path[0]])

# 初始化种群
def initial_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

# 选择函数（锦标赛选择）
def selection(population, cities):
    tournament = random.sample(population, 5)
    return min(tournament, key=lambda path: path_distance(path, cities))

# 交叉（部分映射交叉 PMX）
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}
    for i in range(start, end):
        while parent2[i] in mapping:
            parent2[i] = mapping[parent2[i]]
    
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    
    return child

# 变异（交换变异）
def mutate(path, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# 遗传算法求解 TSP
def genetic_algorithm(cities, pop_size=100, generations=500, mutation_rate=0.1):
    population = initial_population(pop_size, len(cities))
    
    for _ in range(generations):
        new_population = []
        plot_path(cities, population[0])
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, cities), selection(population, cities)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        population = new_population
    
    best_path = min(population, key=lambda path: path_distance(path, cities))
    return best_path

# 可视化路径
def plot_path(cities, path):
    ordered_cities = np.array([cities[i] for i in path] + [cities[path[0]]])
    plt.figure(figsize=(8, 6))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'bo-')
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    plt.title("Genetic Algorithm TSP Solution")
    plt.show()

# 运行示例
num_cities = 20
cities = generate_cities(num_cities)
best_path = genetic_algorithm(cities)
plot_path(cities, best_path)
