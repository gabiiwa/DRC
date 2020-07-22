#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import os

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier


from read_data import *
from sklearn import metrics
from utils.confusion_matrix_pretty_print import plot_confusion_matrix_from_data
from sklearn.metrics import classification_report


#%%
pd.options.display.float_format = '{:.3f}'.format
datasets = [
            read_data_drc_35(),
           ]

random_seed=0
for dataset in datasets:
    task             = dataset['task'            ]
    dataset_name     = dataset['name'            ]
    feature_names    = dataset['feature_names'   ]
    target_names     = dataset['target_names'    ]
    n_samples        = dataset['n_samples'       ]
    n_features       = dataset['n_features'      ]
    X_train          = dataset['X_train'         ]
    X_test           = dataset['X_test'          ]
    y_train          = dataset['y_train'         ]
    y_test           = dataset['y_test'          ]
    targets          = dataset['targets'         ]
    true_labels      = dataset['true_labels'     ]
    predicted_labels = dataset['predicted_labels']
    descriptions     = dataset['descriptions'    ]
    items            = dataset['items'           ]
    reference        = dataset['reference'       ]
    normalize        = dataset['normalize'       ]
    n_samples_train  = len(y_train)
    
    #%%
    dr='drc_'+dataset_name.replace(' ','_').replace("'","").lower()
    path='./pkl_'+dr+'/'
    os.system('mkdir  '+path)
          
    for n, target in enumerate(target_names):
        y_train = dataset['y_train'][n]#.reshape(-1,1)
        y_test  = dataset['y_test' ][n]#.reshape(-1,1)
        n_samples_test                  = len(y_test)
    
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
        
        
        mod=ExtraTreesClassifier(n_estimators=5000)
        #mod=XGBClassifier(n_estimators=2000)
        #mod =SVC(kernel='sigmoid', C=100,)        

        mod.fit(X_train, y_train)

        y_pred = mod.predict(X_test)

        columns = [str(i) for i in np.unique(y_test)]
        plot_confusion_matrix_from_data(y_test, y_pred, columns, figsize=[4, 4],)
        
        print(classification_report(y_test, y_pred))
        
        
#%%