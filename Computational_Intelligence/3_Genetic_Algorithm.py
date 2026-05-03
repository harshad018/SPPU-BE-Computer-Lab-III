import random

# Genetic Algorithm to maximize the function f(x) = x^2 where x is an integer between 0 and 31

POPULATION_SIZE = 10
GENES_LENGTH = 5  # 5 bits can represent 0 to 31
GENERATIONS = 20
MUTATION_RATE = 0.1

def fitness(individual):
    # Convert binary list to integer
    x = int("".join(str(gene) for gene in individual), 2)
    return x ** 2

def create_population():
    return [[random.randint(0, 1) for _ in range(GENES_LENGTH)] for _ in range(POPULATION_SIZE)]

def selection(population):
    # Roulette wheel selection
    total_fitness = sum(fitness(ind) for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += fitness(ind)
        if current > pick:
            return ind
    return population[-1]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, GENES_LENGTH - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual):
    for i in range(GENES_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

def main():
    print("--- Genetic Algorithm (Maximize f(x) = x^2) ---")
    population = create_population()
    
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        best_individual = population[0]
        best_fitness = fitness(best_individual)
        best_x = int("".join(str(gene) for gene in best_individual), 2)
        
        print(f"Generation {generation}: Best x = {best_x}, Fitness = {best_fitness}")
        
        new_population = [best_individual]  # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
            
        population = new_population[:POPULATION_SIZE]

    # Final result
    population = sorted(population, key=lambda x: fitness(x), reverse=True)
    best_individual = population[0]
    best_x = int("".join(str(gene) for gene in best_individual), 2)
    print(f"\nOptimal Solution Found: x = {best_x}, Maximum Value = {fitness(best_individual)}")

if __name__ == "__main__":
    main()
