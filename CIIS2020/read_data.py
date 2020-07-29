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
    if   valor == 'Estágio 1 - >= 90 ml':  return '0'#1
    elif valor == 'Estágio 2 - 60-89 ml':  return '0'#2
    elif valor == 'Estágio 3a - 45-59 ml': return '1'#3
    elif valor == 'Estágio 3b - 30-44 ml': return '2'#4
    elif valor == 'Estágio 4 - 15-29 ml':  return '3'#5
    elif valor == 'Estágio 5 - < 15 ml':   return '4'#6

def transforma_raca(valor):
    if valor == 'Branca': return 0
    elif valor == 'Preta': return 1
    elif valor == 'Parda': return 2
    elif valor == 'Amarela': return 3
    elif valor == 'Indigena': return 4
    
    
def read_data_drc_25(
            filename='./data/Banco25exames.csv',
        ):
    #%%
    filename='./data/Banco25exames.csv'
    df= pd.read_csv(filename,  delimiter=',')
    
    df['ESTAGIOI - BIN'] = df['ESTAGIOI - EQ'].map(transforma_estagio)
    df['ESTAGIOF - BIN'] = df['ESTAGIOF - EQ'].map(transforma_estagio)
    
    df['Raça'] = df['Raça'].map(transforma_raca)
    
    df['Codsexo'] = df['Codsexo'].replace('Masculino', 0)
    df['Codsexo'] = df['Codsexo'].replace('Feminino', 1)
    
    target_names = ['ESTAGIOF - BIN']
    y = df[target_names]

    feature_names=[
         'Idade',
         'Raça',
         'Codsexo',
         'PAS_inicial',
         'PAS_final',
         'PAD_inicial',
         'PAD_final',
         'pesoi',
         'pesof',
         'HemoglobinaI',
         'ColesterolTotalI',
         'GlicemiadeJejumI',
         'TrigliceridesI',
         'PotassioI',
         'ColesterolHDLI',
         'UreiaI',
         'TSHI',
         'AcidoUricoI',
         'HemoglobinaGlicadaI',
         'TGPI',
         'GlicemiadeJejumF',
         'ColesterolTotalF',
         'TrigliceridesF',
         'ColesterolHDLF',
         'HemoglobinaF',
         'SodioSericoI',
         'PotassioF',
         'CKI',
         'CalcioTotalI',
         'VITAMINADI',
         'HemoglobinaGlicadaF',
         'ColesterolLDLI',
         'UreiaF',
         'Proteinuria24hsI',
         'TGPF',
         ]
    
    df=df[feature_names]
    
    idx=[ True,  True,  True, False, False, False, False,  True,  True,
       False, False,  True, False,  True,  True, False,  True,  True,
        True, False,  True, False, False,  True,  True,  True,  True,
        True,  True,  True,  True, False, False,  True,  True, False,
        True,  True, False,  True, False, False,  True,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True, False,  True,  True,
       False,  True,  True, False, False, False,  True, False, False,
        True, False,  True, False, False,  True,  True, False,  True,
        True, False,  True,  True,  True,  True, False, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True, False,  True,  True,
        True,  True, False, False,  True,  True, False, False, False,
       False,  True,  True,  True,  True, False, False,  True,  True,
       False,  True,  True, False,  True,  True,  True,  True,  True,
        True, False,  True,  True,  True, False,  True,  True,  True,
        True, False, False, False,  True, False,  True, False,  True,
       False,  True,  True, False,  True, False,  True, False,  True,
        True,  True,  True,  True, False,  True, False, False, False,
        True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True, False, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True,  True, False,  True,
        True,  True, False, False,  True,  True, False,  True,  True,
       False,  True,  True,  True,  True,  True,  True,  True, False,
       False,  True,  True,  True,  True,  True, False,  True, False,
        True, False,  True,  True, False,  True, False, False,  True,
       False,  True, False,  True, False, False,  True,  True,  True,
       False,  True, False,  True,  True, False,  True,  True,    True,
        True,  True, False, False,  True, False,  True,  True,    True,
        True, False,  True,  True,  True, False,  True, False,    True,
        True, False,  True,  True,  True,  True, False,  True,    True,
        True,  True,  True,  True,  True,  True,  True, False,    True,
        True, False,  True,  True,  True,  True,  True,  True,  True,
       False, False,  True,  True, False,  True, False, False, False,
       False, False,  True,  True,  True, False,  True, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False, False, False, False, False,  True, False,  True,
        True, False,  True,  True,  True,  True,  True,  True, False,
        True,  True, False,  True,  True, False,  True,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
        True, False,  True,  True, False, False,  True,  True,  True,
        True, False, False,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True, False,  True, False,
        True,  True,  True,  True, False, False, False,  True,  True,
       False,  True, False, False,  True, False,  True,  True]
    
    idn = [not i for i in idx]
    X_train, y_train = df[idx], y[idx]
    X_test, y_test  = df[idn], y[idn]
    data_description = ['X_'+str(i+1) for i in range(df.shape[1])]
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'classification',
      'name'            : 'DRC 35 features',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.values.T,
      'y_test'          : y_test.values.T,      
      'targets'         : feature_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "https://www.sciencedirect.com/science/article/pii/S2352914818302387",      
      'normalize'       : 'MinMax',
      }
    #%% 
    return dataset


#%% 
  
def read_data_drc_35(
            filename='./data/Banco35exames.csv',
        ):
    #%%
    #filename='./data/Banco35exames.csv'
    filename='C:/Users/jpsco/Documents/Professor/Doutorado/PGMC/BD2020/DRC/CIIS2020/data/Banco25exames.csv'
    df= pd.read_csv(filename,  delimiter=',')
    
    df['ESTAGIOI - BIN'] = df['ESTAGIOI - EQ'].map(transforma_estagio)
    df['ESTAGIOF - BIN'] = df['ESTAGIOF - EQ'].map(transforma_estagio)
    
    df['Raça'] = df['Raça'].map(transforma_raca)
    
    df['Codsexo'] = df['Codsexo'].replace('Masculino', 0)
    df['Codsexo'] = df['Codsexo'].replace('Feminino', 1)
    
    target_names = ['ESTAGIOF - BIN']
    y = df[target_names]

    feature_names=[
         'Idade',
         'Raça',
         'Codsexo',
         'PAS_inicial',
         'PAS_final',
         'PAD_inicial',
         'PAD_final',
         'pesoi',
         'pesof',
         'HemoglobinaI',
         'ColesterolTotalI',
         'GlicemiadeJejumI',
         'TrigliceridesI',
         'PotassioI',
         'ColesterolHDLI',
         'UreiaI',
         'TSHI',
         'AcidoUricoI',
         'HemoglobinaGlicadaI',
         'TGPI',
         'GlicemiadeJejumF',
         'ColesterolTotalF',
         'TrigliceridesF',
         'ColesterolHDLF',
         'HemoglobinaF',
         'SodioSericoI',
         'PotassioF',
         'CKI',
         'CalcioTotalI',
         'VITAMINADI',
         'HemoglobinaGlicadaF',
         'ColesterolLDLI',
         'UreiaF',
         'Proteinuria24hsI',
         'TGPF',
         ]
    # feature_names=['UreiaF', 'Idade', 'UreiaI', 'VITAMINADI', 'PAD_inicial',
    #    'GlicemiadeJejumI', 'TrigliceridesI', 'Proteinuria24hsI', 'pesoi',
    #    'TSHI']
    
    df=df[feature_names]
    
    idx=[ True,  True,  True, False, False, False, False,  True,  True,
       False, False,  True, False,  True,  True, False,  True,  True,
        True, False,  True, False, False,  True,  True,  True,  True,
        True,  True,  True,  True, False, False,  True,  True, False,
        True,  True, False,  True, False, False,  True,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True, False,  True,  True,
       False,  True,  True, False, False, False,  True, False, False,
        True, False,  True, False, False,  True,  True, False,  True,
        True, False,  True,  True,  True,  True, False, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True, False,  True,  True,
        True,  True, False, False,  True,  True, False, False, False,
       False,  True,  True,  True,  True, False, False,  True,  True,
       False,  True,  True, False,  True,  True,  True,  True,  True,
        True, False,  True,  True,  True, False,  True,  True,  True,
        True, False, False, False,  True, False,  True, False,  True,
       False,  True,  True, False,  True, False,  True, False,  True,
        True,  True,  True,  True, False,  True, False, False, False,
        True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True, False, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True,  True, False,  True,
        True,  True, False, False,  True,  True, False,  True,  True,
       False,  True,  True,  True,  True,  True,  True,  True, False,
       False,  True,  True,  True,  True,  True, False,  True, False,
        True, False,  True,  True, False,  True, False, False,  True,
       False,  True, False,  True, False, False,  True,  True,  True,
       False,  True, False,  True,  True, False,  True,  True,    True,
        True,  True, False, False,  True, False,  True,  True,    True,
        True, False,  True,  True,  True, False,  True, False,    True,
        True, False,  True,  True,  True,  True, False,  True,    True,
        True,  True,  True,  True,  True,  True,  True, False,    True,
        True, False,  True,  True,  True,  True,  True,  True,  True,
       False, False,  True,  True, False,  True, False, False, False,
       False, False,  True,  True,  True, False,  True, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False, False, False, False, False,  True, False,  True,
        True, False,  True,  True,  True,  True,  True,  True, False,
        True,  True, False,  True,  True, False,  True,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
        True, False,  True,  True, False, False,  True,  True,  True,
        True, False, False,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True, False,  True, False,
        True,  True,  True,  True, False, False, False,  True,  True,
       False,  True, False, False,  True, False,  True,  True]
    
    idn = [not i for i in idx]
    X_train, y_train = df[idx], y[idx]
    X_test, y_test  = df[idn], y[idn]
    data_description = ['X_'+str(i+1) for i in range(df.shape[1])]
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'classification',
      'name'            : 'DRC 35 features',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.values.T,
      'y_test'          : y_test.values.T,      
      'targets'         : feature_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "https://www.sciencedirect.com/science/article/pii/S2352914818302387",      
      'normalize'       : 'MinMax',
      }
    #%% 
    return dataset


#%% 
#%%-----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [
                read_data_drc_35(),                
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
