#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:42:04 2017

@author: kelvin
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# 'size, m2'
X = np.array([42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70])
# 'price, $1000'
Y = np.array([41, 46, 44, 49, 48, 56, 51, 52, 59, 58, 66, 61, 68, 66, 71])

def plot_result(X, Y, model):
    b = model[0]
    w = model[1]
    t = np.arange(X.min(), X.max(), 0.01)
    plt.xlabel('size, m2')
    plt.ylabel('price, $1000')
    plt.plot(X,Y, 'bo',
             t, w*t+b , 'k'
             )
    plt.show()
    return 0

my_model = [0, 1] # [b, k]

def linear_model(X, model):
    b = model[0]
    w = model[1]
    y_pred = w*X+b
    return y_pred

def mean_square_error(X, Y, model): # loss function
    J = 0
    m = len(Y)
    for i, y in enumerate(Y):
        J += (1/(2*m))*(linear_model(X[i], model)-Y[i])**2
    return J

def generate_population(p, w_size):
    population = []
    for i in range(p):
        model = []
        for j in range(w_size + 1): # +1 for b (bias term)
            model.append(2 * random.random() - 1)  # random initialization from -1 to 1 for b and w
        population.append(model)
    return np.array(population)

def mutation(genom, t=0.5, m=0.025):
    mutant = []
    for gen in genom:
        if random.random() <= t:
            gen += m*(2*random.random() -1)
        mutant.append(gen)
    return mutant

def selection(offspring, population):
    offspring.sort()
    population = [kid[1] for kid in offspring[:len(population)]]
    return population

def evolution(population, X_in, Y, number_of_generations, children):
    for i in range(number_of_generations):
        offspring = []
        for genom in population:
            for j in range(children):
                child = mutation(genom)
                child_loss = mean_square_error(X, Y, child)
                offspring.append([child_loss, child])
            population = selection(offspring, population)
            print(offspring[0][0])
            plot_result(X_in, Y, population[0])
    return population


population = generate_population(3, 1)
population = evolution(population, X, Y, 100, 3)
