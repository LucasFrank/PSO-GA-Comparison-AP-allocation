import random

class GeneticAlgorithm:

    def __init__(self, population_size, boundaries, boundaries_type, mutation_rate = 0.1): # matrix 2 rows(upper and lower bound) - N columns - number of parameters
        self.population_size = population_size
        self.population = []
        self.mutation_rate = mutation_rate
        self.best_individual = None

        for _ in range(self.population_size):
            new_dna = DNA(boundaries, boundaries_type)
            self.population.append(new_dna)

    def get_population(self):
        return self.population

    def print_population(self):
        for i in range(len(self.population)):
            print("Individual {}: Parameters:({},{}) Fitness: {}".format(i+1, self.population[i].get_parameters()[0], self.population[i].get_parameters()[1], self.population[i].get_fitness()))

    def generate(self):
        for i in range(len(self.population)):
            partner_1 = self.population[i]
            partner_2 = self.get_random_partner(i)
            child = partner_1.crossover(partner_2)
            child.mutate(self.mutation_rate)
            self.population[i] = child

    def get_random_partner(self, index):
        random_pick = random.randrange(0, self.population_size)
        while random_pick == index:
            random_pick = random.randrange(0, self.population_size)
        return self.population[random_pick]

    def evaluate(self, i, fitness_value):
        i.set_fitness(fitness_value)

        if(self.best_individual == None or self.best_individual.get_fitness() < i.get_fitness()):
            self.best_individual = i

    def print_best_solution(self):
        print("Best Individual: Parameters:({}, {}) Fitness: {}".format(self.best_individual.get_parameters()[0], self.best_individual.get_parameters()[1], self.best_individual.get_fitness()))

class DNA:

    def __init__(self, boundaries, boundaries_type):
        self.fitness = 0;
        self.number_of_parameters = len(boundaries[0])
        self.genes = [None] * self.number_of_parameters
        self.boundaries = boundaries
        self.boundaries_type = boundaries_type

        for i in range(self.number_of_parameters):
            if self.boundaries_type[i]: # if true, param is a integer, otherwise is a float
                self.genes[i] = random.randrange(self.boundaries[0, i], self.boundaries[1, i])
            else:
                self.genes[i] = random.uniform(self.boundaries[0, i], self.boundaries[1, i])

    def crossover(self, partner):

        new_child = DNA(self.boundaries, self.boundaries_type)

        random_cross = True if random.uniform(0, 1) < 0.5 else False

        for i in range(self.number_of_parameters):
            if(random_cross):
                if(i < self.number_of_parameters/2):
                    new_child.genes[i] = self.genes[i]
                else:
                    new_child.genes[i] = partner.genes[i]
            else:
                if(i < self.number_of_parameters/2):
                    new_child.genes[i] = partner.genes[i]
                else:
                    new_child.genes[i] = self.genes[i]

        return new_child

    def mutate(self, mutation_rate):

        for i in range(self.number_of_parameters):
            if(random.uniform(0, 1) < mutation_rate):
                self.genes[i] = self.get_random_value(i)

    def get_random_value(self, index):
        value = None

        if self.boundaries_type[index]:
            value = random.randrange(self.boundaries[0, index], self.boundaries[1, index])
        else:
            value = random.uniform(self.boundaries[0, index], self.boundaries[1, index])

        return value

    def set_fitness(self, fitness_value):
        self.fitness = fitness_value

    def get_fitness(self):
        return self.fitness

    def get_parameters(self):
        return self.genes
