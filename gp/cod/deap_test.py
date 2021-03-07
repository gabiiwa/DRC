#%%
#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os
from numpy import random
from random import sample 

from sklearn.model_selection import train_test_split

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


from read_data import *
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
#%%
X_START = -4
X_END = 4
X_STEP = 0.5

def unknown(x):
    
    return 1.3*x + 1.9*x**2 - 4.2*x**3 + 5.0

X = np.array([x for x in np.arange(X_START, X_END, X_STEP)])

def sample(inputs):
    return np.array([unknown(inp) + random.normal(5.) for inp in inputs])

# observations

Y = sample(X)

data = list(zip(X, Y))

#%%
IND_SIZE = 5
NGEN = 100

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, typecode="d", fitness=creator.FitnessMin, strategy=None)

creator.create("Strategy", list, typecode="d")

#%%
def generateES(ind_cls, strg_cls, size):
    ind = ind_cls(random.normal() for _ in range(size))
    ind.strategy = strg_cls(random.normal() for _ in range(size))
    return ind

toolbox = base.Toolbox()

# generation functions
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#%%
def pred(ind, x):
    
    y_ = 0.0
    
    for i in range(1,IND_SIZE):
        y_ += ind[i-1]*x**i
    
    y_ += ind[IND_SIZE-1]
       
    return y_

def fitness(ind, data):
    
    mse = 0.0
    
    for x, y in data:
        
        y_ = pred(ind, x)
        mse += (y - y_)**2
        
    return mse/len(data),

toolbox.register("evaluate", fitness, data=data)