  
#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
#import pygmo as pg
import pylab as pl

from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     RandomizedSearchCV,
                                     cross_val_predict,train_test_split)
from sklearn.metrics import r2_score, mean_squared_error
#from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, median_absolute_error
#from sklearn.metrics.classification import accuracy_score, f1_score, precision_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.linear_model import ElasticNet, Ridge, PassiveAggressiveRegressor, LogisticRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import  XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


#import re  
import os
#from sklearn.gaussian_process import GaussianProcess
#from catboost import Pool, CatBoostRegressor
#from pyearth import Earth as MARS
#from sklearn.ensemble import StackingRegressor
#from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import HistGradientBoostingRegressor
#from sklearn.kernel_approximation import RBFSampler,SkewedChi2Sampler

#from util.ELM import  ELMRegressor, ELMRegressor
#from util.MLP import MLPRegressor as MLPR
#from util.RBFNN import RBFNNRegressor, RBFNN
#from util.LSSVR import LSSVR

from scipy import stats

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
#from utils.confusion_matrix_pretty_print import plot_confusion_matrix_from_data
#from ds_utils.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

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

pop_size    = 20
max_iter    = 40
n_splits    = 3
scoring     = 'neg_mean_squared_error'
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run+100
    
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
            
            distributions = dict(n_estimators=randint(low=1, high=1e3),
                                 max_depth=randint(low=1, high=10),
                                 learning_rate=uniform(loc=0, scale=1),
                                 gamma=uniform(loc=0, scale=1),
                                 reg_alpha=uniform(loc=0, scale=1),
                                 reg_lambda=uniform(loc=0, scale=1),
                                 #degree=uniform(loc=1, scale=5),
                               )
            
            clf = RandomizedSearchCV(estimator=XGBClassifier(random_state=random_seed), 
                                     param_distributions=distributions, 
                                     n_iter=1,
                                     n_jobs=1,
                                     cv=3,
                                     scoring=scoring,
                                     verbose=2,
                                     random_state=random_seed)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
           
            clf.fit(X_train, y_train)  
            
            y_pred = clf.predict(X_test)
           
            columns = [str(i) for i in np.unique(y_test)]
            #plot_confusion_matrix_from_data(y_test, y_pred, columns, figsize=[4, 4],)
            #plot_confusion_matrix(y_test, y_pred, columns, figsize=[4, 4],)
            
            print(classification_report(y_test, y_pred))
#%%----------------------------------------------------------------------------   

            from xgboost import plot_importance
            model = clf.best_estimator_
            plot_importance(model)

#%%----------------------------------------------------------------------------   