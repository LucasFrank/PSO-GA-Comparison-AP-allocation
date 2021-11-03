import random

class PSO:

    def __init__(self, population_size, boundaries, boundaries_type, w = 0.7, c1 = 2, c2 = 2):
        self.population_size = population_size
        self.population = []
        self.g_best_particle = None
        self.best_particle_swap = 0
        #PSO Coefficients Variable
        self.w = w
        self.c1 = c1
        self.c2 = c2
        #Initializing Position Min and Max
        self.position_min = boundaries[0,:]
        self.position_max = boundaries[1,:]
        self.boundaries_type = boundaries_type
        #Initializing Velocity Min and Max
        self.velocity_min = []
        self.velocity_max = []
        for index in range(len(self.position_min)):
            self.velocity_min.append(-0.1 * (self.position_max[index] - self.position_min[index]))
            self.velocity_max.append(0.1 * (self.position_max[index] - self.position_min[index]))

        for i in range(self.population_size):
            self.initialize_particle(i)


    def initialize_particle(self, index):
        pos = []
        for i in range(len(self.position_min)):
            if self.boundaries_type[i]:
                aux = random.randrange(self.position_min[i], self.position_max[i]) # 0
            else:
                aux = random.uniform(self.position_min[i], self.position_max[i]) # 2
            pos.append(aux)

        vel = []
        for i in range(len(self.velocity_min)):
            if self.boundaries_type[i]:
                aux = random.randrange(self.velocity_min[i], self.velocity_max[i]) # 0
            else:
                aux = random.uniform(self.velocity_min[i], self.velocity_max[i]) # 2
            vel.append(aux)

        self.population.append(Particle(pos, vel, index))

    def insert_particle_fitness(self, particle, fitness_value):
        if particle.p_best_fitness == None or particle.p_best_fitness < fitness_value:
            particle.set_pbest(particle.get_position(), fitness_value)

            if self.g_best_particle == None or self.g_best_particle.p_best_fitness < fitness_value:
                self.set_gbest_particle(particle)
                self.best_particle_swap += 1

    def get_position(self, index):
        return self.population[index].position

    def get_population(self):
        return self.population

    def get_population_size():
        return self.population_size

    def get_gbest_particle():
        return self.g_best_particle

    def get_position_min(self):
        return self.position_min

    def get_position_max(self):
        return self.position_max

    def get_velocity_min(self):
        return self.velocity_min

    def get_velocity_max(self):
        return self.velocity_max

    def set_gbest_particle(self, new_gbest):
        self.g_best_particle = new_gbest

    def get_best_particle_swap(self):
        return self.best_particle_swap

    def print_global_best_particle(self):
        print('-----------------------------------------------')
        print('Best Particle: {}'.format(self.g_best_particle.id))
        print('Best Position:' + str(self.g_best_particle.p_best_pos))
        print('Best Fitness: {}'.format(self.g_best_particle.p_best_fitness))
        print('-----------------------------------------------')

    def print_position(self):
        print('Position Min: ' + str(self.position_min))
        print('Position Max: ' + str(self.position_max))

    def print_velocity(self):
        print('Velocity Min: ' + str(self.velocity_min))
        print('Velocity Max: ' + str(self.velocity_max))

    def calculate_position_velocity(self, i):
        i.calculate_velocity(self.g_best_particle, self.velocity_min, self.velocity_max, self.w, self.c1, self.c2)
        i.calculate_position(self.position_min, self.position_max, self.boundaries_type)

class Particle():

    def __init__(self, pos, vel, index):
        self.num_of_param = len(pos)
        self.id = index

        self.p_best_pos = pos
        self.p_best_fitness = None

        self.position = pos
        self.velocity = vel

    def calculate_position(self, position_min, position_max, boundaries_type):
        for i in range(self.num_of_param):
            self.position[i] = int(self.position[i] + self.velocity[i]) if boundaries_type[i] else self.position[i] + self.velocity[i]
            self.position[i] = max(self.position[i], position_min[i])
            self.position[i] = min(self.position[i], position_max[i])

    def calculate_velocity(self, gbest, velocity_min, velocity_max, w, c1, c2):
        for i in range(self.num_of_param):
            self.velocity[i] = w * self.velocity[i] + c1 * random.uniform(0,1) * (self.p_best_pos[i] - self.position[i]) + c2 * random.uniform(0,1) * (gbest.p_best_pos[i] - self.position[i])
            self.velocity[i] = max(self.velocity[i], velocity_min[i])
            self.velocity[i] = min(self.velocity[i], velocity_max[i])

    def get_position(self):
        return self.position

    def get_index(self):
        return self.id

    def set_pbest(self, pos, fitness_value):
        self.p_best_pos = pos.copy()
        self.p_best_fitness = fitness_value
