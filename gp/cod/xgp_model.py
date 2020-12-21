#%%
#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os

import xgp

from read_data import *
from sklearn.model_selection import train_test_split
from gplearn.functions import make_function
from sklearn.preprocessing import LabelBinarizer
#%%
def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)

#%%
pd.options.display.float_format = '{:.3f}'.format
# datasets = [
#             #read_data_ldc_tayfur( case = 0 ),
#             #read_data_tran2019(),
#             #read_data_ldc_vijay(),
#             #read_data_iraq_sequence(look_back=5, kind='ml'),
#             #read_data_cahora_bassa_sequence(look_back=1, look_forward=1, kind='ml', unit='month'),
#             #read_data_cahora_bassa_sequence(look_back=2, look_forward=1, kind='ml', unit='month'),
#             read_data_cahora_bassa_sequence(look_back=1, look_forward=1, kind='ml', unit='month'),
#             #read_data_british_columbia_daily_sequence(look_back=10, look_forward=1, kind='ml', roll=False,),
#             #read_data_british_columbia_daily_sequence(look_back=10, look_forward=1, kind='ml', roll=True,),
#             #read_data_cahora_bassa_sequence(look_back=6, look_forward=1, kind='ml', unit='month'),
#             #read_data_cahora_bassa_sequence(look_back=9, look_forward=1, kind='ml', unit='month'),
#             #read_data_cahora_bassa_sequence(look_back=21, look_forward=7, kind='ml', unit='day'),
#            ]
datasets = [
            read_data_cenario('cenario1.csv'),
            # read_data_cenario('cenario2.csv'),
            # read_data_cenario('cenario3.csv'),
            # read_data_cenario('cenario4.csv')
           ]

random_seed=0
for dataset in datasets:
    task             = dataset['task'            ]
    dataset_name     = dataset['name'            ]
    feature_names    = dataset['feature_names'   ]
    target_names     = dataset['target_names'    ]
    n_samples        = dataset['n_samples'       ]
    n_features       = dataset['n_features'      ]
    X                = dataset['X_train'         ]
    # X_test           = dataset['X_test'          ]
    y                = dataset['y_train'         ][0]
    # y_test           = dataset['y_test'          ]
    targets          = dataset['targets'         ]
    true_labels      = dataset['true_labels'     ]
    predicted_labels = dataset['predicted_labels']
    descriptions     = dataset['descriptions'    ]
    items            = dataset['items'           ]
    reference        = dataset['reference'       ]
    normalize        = dataset['normalize'       ]
    # n_samples_train  = len(y_train)

    #%%
    dr='gp_'+dataset_name.replace(' ','_').replace("'","").lower()
    path='./pkl_'+dr+'/'
    os.system('mkdir  '+path)
          
    for n, target in enumerate(target_names):
        # y_train = dataset['y_train'][n]#.reshape(-1,1)
        # y_test  = dataset['y_test' ][n]#.reshape(-1,1)
        # n_samples_test                  = len(y_test)
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.30,random_state=42)
        
        n_samples_train = len(y_train)
        n_samples_test = len(y_test)
    
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
               
        function_set = ['add', 'sub', 'mul', 'div',
                    'sqrt', 'abs', 'neg', 'inv',
                    #'log',
                    #'sin', 'cos', 'tan', 
                    'max', 'min',
                    #logical,
                    ]
                    
        est_gp = xgp.XGPClassifier(
                                    flavor='boosting',
                                    # loss_metric='mse',
                                    loss_metric='logloss',
                                    #funcs='add,sub,mul,div,cos,sin,log,exp,max,min,pow',
                                    funcs='add,sub,mul,div,cos,min,max',
                                    n_individuals=1000,
                                    n_generations=20,
                                    parsimony_coefficient=0.01,
                                    n_rounds=8,
                                    const_max=50, const_min=-50,
                                    random_state=42,
                                )

        est_gp.fit(X_train, y_train.squeeze(), verbose=True)

        clf=est_gp
        #%%
        y_pred = clf.predict(X_test)
        rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
        r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
        print(rmse, r2,r)

        pl.figure(figsize=(16,4)); 
        #pl.plot([a for a in y_train]+[None for a in y_test]); 
        pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
        pl.plot([None for a in y_train]+[a for a in y_pred], 'b-o', label='Predicted');
        pl.legend(); pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))
        pl.show()

        # pl.figure(figsize=(6,6)); 
        # pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
        # pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
        # pl.show()

#%%-----------------------------------------------------------------------------
