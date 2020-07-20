#!/usr/bin/python
# -*- coding: utf-8 -*
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

import numpy as np
import pylab as pl
import matplotlib.pyplot as pl
import pandas as pd
import os
import sys
import re
from scipy import stats
#%% 
def transforma_estagio(valor):
    if   valor == 'Estágio 1 - >= 90 ml':  return 1
    elif valor == 'Estágio 2 - 60-89 ml':  return 2
    elif valor == 'Estágio 3a - 45-59 ml': return 3
    elif valor == 'Estágio 3b - 30-44 ml': return 4
    elif valor == 'Estágio 4 - 15-29 ml':  return 5
    elif valor == 'Estágio 5 - < 15 ml':   return 6

def transforma_raca(valor):
    if valor == 'Branca': return 0
    elif valor == 'Preta': return 1
    elif valor == 'Parda': return 2
    elif valor == 'Amarela': return 3
    elif valor == 'Indigena': return 4
    
    
def read_data_cahora_bassa_sequence(
            filename='./data/Banco35exames.csv',
        ):
    #%%
    filename='./data/Banco35exames.csv'
    df= pd.read_csv(filename,  delimiter=',')
    
    df['ESTAGIOI - BIN'] = df['ESTAGIOI - EQ'].map(transforma_estagio)
    df['ESTAGIOF - BIN'] = df['ESTAGIOF - EQ'].map(transforma_estagio)
    
    df['Raça'] = df['Raça'].map(transforma_raca)
    
    df['Codsexo'] = df['Codsexo'].replace('Masculino', 0)
    df['Codsexo'] = df['Codsexo'].replace('Feminino', 1)
    #%% 


#%% 
    