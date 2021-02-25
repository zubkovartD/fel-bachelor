## Intro into genetic algorithm

import math, cmath
import numpy as np
import scipy.io as sio
from deap import creator
from deap import tools
from deap import base
import random
import matplotlib.pyplot as plt
import seaborn as sns

import elitism

# %% Import measured data

meas_data = sio.loadmat('VeryGoodMeasurement_measData.mat')

f = meas_data['f']
U = meas_data['U']
I = meas_data['I']
V = meas_data['V']

w = 2 * math.pi * f
jw = 1j * w

# %% Calculations using measured data

Z_tot = U / I

# %% Example of an individual with known parameters

### These are chromosomes:

# f_res = 140  # Hz
# R_e = 3.54  # Ohm
# L_e = 1.45e-4  # H
# Bl = 2.49  # N./A
# M_ms = 2.8e-3  # kg
# C_ms = 4.65e-4  # m/N
# K_ms = 1 / C_ms
# R_ms = 0.65  # kg/s

### This is an individual containing all chromosomes:

# individual = [R_e, L_e, Bl, M_ms, C_ms, R_ms]

# %% Calculating fitness function (equation 4)
#
# E_1 = (abs(U / I - (R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms))) ** 2).mean()
#
# E_2 = (abs(V / I - Bl / (jw * M_ms + K_ms / jw + R_ms)) ** 2).mean()

# E_tot = E_1 + E_2

# %% Now calculate simulated Z_tot using founded values (equation 2)

# Z_tot_sim = R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms)

# problem constants:
DIMENSIONS = 6  # number of dimensions
BOUND_LOW, BOUND_UP = 0, 10.0  # boundaries for all dimensions

# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
IND_SIZE = 6
creator.create("Individual", list, fitness=creator.FitnessMin)


# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]


# create an operator that randomly returns a float in the desired range and dimension:
toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def myFunc(individual):
    R_e = individual[0]
    L_e = individual[1] * 1e-4
    Bl = individual[2]
    M_ms = individual[3] * 1e-3
    K_ms = individual[4] * 1e3
    R_ms = individual[5] * 1e-1
    f = ((abs(U / I - (R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms))) ** 2).mean()) + ((abs(V / I - Bl / (jw * M_ms + K_ms / jw + R_ms)) ** 2).mean())
    return f,  # return a tuple


toolbox.register("evaluate", myFunc)


# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)

#1/(math.pi*2*math.sqrt(K_ms/M_ms)) — fres
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.show()

    R_e = best[0]
    L_e = best[1] * 1e-4
    Bl = best[2]
    M_ms = best[3] * 1e-3
    K_ms = best[4] * 1e3
    R_ms = best[5] * 1e-1
    Z_tot_sim = R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms)

    fig, ax1 = plt.subplots()

    plt.title('Input electrical impedance')
    plt.xscale('log')
    plt.grid()

    color = 'tab:blue'
    ax1.set_xlabel('Frequency, Hz')
    ax1.set_ylabel('Magnitude', color=color)
    ax1.plot(f, abs(Z_tot), color=color, label="Measured")
    ax1.plot(f, abs(Z_tot_sim), color=color, label="Simulated ", linestyle='dashed')
    ax1.legend()
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Phase, rad', color=color)  # we already handled the x-label with ax1
    ax2.plot(f, np.angle(Z_tot), color=color)
    ax2.plot(f, np.angle(Z_tot_sim), color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.show()

if __name__ == "__main__":
    main()

# # %% Plotting
#
# fig, ax1 = plt.subplots()
#
# plt.title('Input electrical impedance')
# plt.xscale('log')
# plt.grid()
#
# color = 'tab:blue'
# ax1.set_xlabel('Frequency, Hz')
# ax1.set_ylabel('Magnitude', color=color)
# ax1.plot(f, abs(Z_tot), color=color, label="Measured")
# ax1.plot(f, abs(Z_tot_sim), color=color, label="Simulated ", linestyle='dashed')
# ax1.legend()
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:red'
# ax2.set_ylabel('Phase, rad', color=color)  # we already handled the x-label with ax1
# ax2.plot(f, np.angle(Z_tot), color=color)
# ax2.plot(f, np.angle(Z_tot_sim), color=color, linestyle='dashed')
# ax2.tick_params(axis='y', labelcolor=color)
#
# plt.show()
