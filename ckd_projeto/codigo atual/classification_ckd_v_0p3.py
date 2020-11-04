#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
#import pygmo as pg
import pylab as pl
import six
import sys
sys.modules['sklearn.externals.six'] = six

from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     RandomizedSearchCV,
                                     cross_val_predict,train_test_split)
from sklearn.metrics import r2_score, mean_squared_error
import os
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
import keras.layers

#%%----------------------------------------------------------------------------
def RMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y -  y_pred    
    return np.sqrt(np.mean(np.power(error, 2)))

def MAPE(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100
  #return RMSE(y, y_pred)
  
#%%----------------------------------------------------------------------------

#pd.options.display.float_format = '{:20,.3f}'.format
pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings('ignore')

import sys, getopt
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

#print ("This is the name of the script: ", program_name)
#print ("Number of arguments: ", len(arguments))
#print ("The arguments are: " , arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0,1

#%%----------------------------------------------------------------------------   
def build_model(train_dataset):
  model = Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset)]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

#%%----------------------------------------------------------------------------   
basename='evo_ml_'

from read_data import *
datasets = [
            read_data_cenario('cenario1.csv'),
            read_data_cenario('cenario2.csv'),
            read_data_cenario('cenario3.csv'),
            read_data_cenario('cenario4.csv')
          ]
#%%----------------------------------------------------------------------------   
pd.options.display.float_format = '{:.3f}'.format
from scipy.stats import uniform, randint
from drc_cenarios import cenario1,cenario2,cenario3,cenario4

cenarios = [cenario1,cenario2,cenario3,cenario4] #keys
pop_size    = 20
max_iter    = 40
n_splits    = 3
scoring     = 'neg_mean_squared_error'
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run+100
    
    
    
    
    estimators=[
        # #
        # # XGB
        # #
        # (
        #  #
        #  # acronym
        #  #
        #  'XGB',
        #  #
        #  # distributions
        #  #
        #  dict(  n_estimators=randint(low=1, high=1e3),
        #         max_depth=randint(low=1, high=10),
        #         learning_rate=uniform(loc=0, scale=1),
        #         gamma=uniform(loc=0, scale=1),
        #         reg_alpha=uniform(loc=0, scale=1),
        #         reg_lambda=uniform(loc=0, scale=1),
        #         #degree=uniform(loc=1, scale=5),
        #                            ),
        #  #
        #  # estimator
        #  #
        #  XGBClassifier(random_state=random_seed)
        #  ),
        #
        # LinearRegression
        #
        # (
        # #
        # # acronym
        # #
        # 'LR',
        # #
        # # distributions
        # #
        # dict( 
        #      C=uniform(loc=1, scale=1e3),
        #      l1_ratio=uniform(loc=0, scale=1),
        #     ),
        # #
        # # estimator
        # #
        # LogisticRegression(random_state=random_seed)
        # ),
        
        # (
        # #
        # # acronym
        # #
        # 'SVC',
        # #
        # # distributions
        # #
        # dict( 
        #       C=uniform(loc=1, scale=1e3),
        #       gamma=uniform(loc=0.001, scale=100),
        #     ),
        # #
        # # estimator
        # #
        # SVC(kernel='rbf', max_iter=1000, random_state=random_seed)
        # ),
        
        # (
        # #
        # # acronym
        # #
        # 'KNN',
        # #
        # # distributions
        # #
        # dict( 
        #      n_neighbors=randint(low=1, high=15),
        #      p=randint(low=1, high=3),
        #     ),
        # #
        # # estimator
        # #
        # KNeighborsClassifier()
        # ),
        
        #(
        # #
        # # acronym
        # #
#         'MLP',
#         #
#         # distributions
#         #
#         dict( 
#              hidden_layer_sizes=randint(low=1, high=100),
#             ),
#         #
#         # estimator
#         #
#         MLPClassifier(activation='relu', random_state=random_seed)
#         ),
#        
        # (
        # #
        # # acronym
        # #
        # 'ELM',
        # #
        # # distributions
        # #
        # dict( 
        #      n_hidden=randint(low=10, high=300),
        #     ),
        # #
        # # estimator
        # #
        # ELMClassifier(activation_func='identity', random_state=random_seed)
        # ),
        
#        (
#        #
#        # acronym
#        #
#        'GPC',
#        #
#        # distributions
#        #
#        dict( 
#              
#            ),
#        #
#        # estimator
#        #
#        GaussianProcessClassifier(optimizer='fmin_l_bfgs_b', random_state=random_seed)
#        ),
        
        ]

    cont=0 
    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir  '+path)
        
        #for (target,y_train,y_test) in zip(dataset['target_names'], dataset['y_train'], dataset['y_test']):                        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            target                          = dataset['target_names'][tk]
            y      , y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            dataset_name, X      , X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            np.random.seed(random_seed)

            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'            
            
            print(s)     
            
            X = pd.DataFrame(X).fillna(0).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
            
            
            for (acronym, distribution, classifier) in  estimators:
                print(acronym, distribution, classifier)        
                clf = RandomizedSearchCV(estimator=classifier, 
                                     param_distributions=distribution, 
                                     n_iter=1,
                                     n_jobs=1,
                                     scoring=scoring,
                                     cv=3,
                                     verbose=1,
                                     random_state=random_seed)
                
                
                model = build_model(cenarios[cont])
                model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
                predictions = model.predict_classes(X_test)
                


                model.summary()
                cont+=1
                # clf.fit(X_train, y_train)                  
                # y_pred = clf.predict(X_test)              
                # columns = [str(i) for i in np.unique(y_test)]
                #plot_confusion_matrix_from_data(y_test, y_pred, columns, figsize=[4, 4],)
                #plot_confusion_matrix(y_test, y_pred, columns, figsize=[4, 4],)
                
                # print(acronym, classification_report(y_test, y_pred))
#%%----------------------------------------------------------------------------   

            # from xgboost import plot_importance
            # model = clf.best_estimator_
            # plot_importance(model)

#%%----------------------------------------------------------------------------   