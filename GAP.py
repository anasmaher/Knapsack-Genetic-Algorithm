import random
import matplotlib.pyplot as plt

# Population generation
def generatePopulation(size):
    population = []
    for i in range(size):
        genes = [0, 1]
        chromosome = []
        for j in range(len(items)):
            chromosome.append(random.choice(genes))
        population.append(chromosome)
        
    return population

# Fitness calculation
def getFitness(chromosome):
    totalWeight = 0
    totalValue = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            totalWeight += items[i][0]
            totalValue += items[i][1]
            
    if totalWeight > maxWeight:
        return 0
    else:
        return totalValue

# Selection
def selectChromosomes(population):
    fitnessValues = []
    for ch in population:
        fitnessValues.append(getFitness(ch))
    
    parent1 = random.choices(population, k=1)[0]
    parent2 = random.choices(population, k=1)[0]
    
    return parent1, parent2

# Crossoever
def crossover(parent1, parent2):
    cPoint = random.randint(0, len(items) - 1)
    child1 = parent1[0:cPoint] + parent2[cPoint:]
    child2 = parent2[0:cPoint] + parent1[cPoint:]
    
    return child1, child2

# Mutation
def mutate(chromosome):
    mPoint = random.randint(0, len(items) - 1)
    chromosome[mPoint] = not chromosome[mPoint]
    return chromosome

# Finding best chromosome from the population
def getBestChromosome(population):
    fitnessValues = []
    for i in population:
        fitnessValues.append(getFitness(i))

    maxValue = max(fitnessValues)
    maxValIndex = fitnessValues.index(maxValue)
    
    return population[maxValIndex]


def replacement(child1, child2, population):
    ch1F = getFitness(child1)
    ch2F = getFitness(child2)
    
    # Find worst two chromosomes in the population
    worstFirst = min(getFitness(ch) for ch in population)
    worstFirstInd = population.index(min(population, key=getFitness))
    population[worstFirstInd] = child1
    
    worstSecond = min(getFitness(ch) for ch in population)
    worstSecondInd = population.index(min(population, key=getFitness))
    population[worstSecondInd] = child2

def display(bestFitnessValues):
    generationss = range(1, len(bestFitnessValues) + 1)
    plt.plot(generationss, bestFitnessValues)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('Generations Progress')
    plt.show()
    
    print(bestFitnessValues)
    
# Problem input
items = [
    [1, 2],
    [2, 4],
    [3, 4],
    [4, 5],
    [5, 7],
    [6, 9],
    [7, 10],
    [8, 12],
    [9, 13],
    [10, 15]
]

# Parameters
maxWeight = 10
populationSize = 100
mutationProbability = 0.3
generations = 300

def main():
    population = generatePopulation(populationSize)
    bestFitnessValues = []
    
    for _ in range(generations):
        # Select two chromosomes for crossover
        parent1, parent2 = selectChromosomes(population)

        # Perform crossover to generate two new chromosomes
        child1, child2 = crossover(parent1, parent2)

        # Perform mutation on the two new chromosomes
        if random.uniform(0, 1) < mutationProbability:
            child1 = mutate(child1)
        if random.uniform(0, 1) < mutationProbability:
            child2 = mutate(child2)

        replacement(child1, child2, population)
        
        bestChromosome = getBestChromosome(population)
        bestFitness = getFitness(bestChromosome)
        bestFitnessValues.append(bestFitness) 
        
    # Plot the best fitness value for each generation
    display(bestFitnessValues)
    
main()