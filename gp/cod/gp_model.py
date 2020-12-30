#%%
#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os

from sklearn.model_selection import train_test_split

from gplearn.genetic import SymbolicClassifier
from gplearn.genetic import SymbolicRegressor
#import graphviz 
from IPython.display import Image
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

from read_data import *
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
#%%

def auc_score(y_true, y_pred, sample_weight=None):
    fpr  = metrics.accuracy_score(y_true, y_pred, )
    #fpr  = metrics.f1_score(y_true, y_pred, average='micro')
    return fpr

auc_scores = make_fitness(function=auc_score, greater_is_better=True)


def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)


logical = make_function(function=_logical,
                        name='logical',
                        arity=4)

#%%
from sklearn.datasets import load_breast_cancer

pd.options.display.float_format = '{:.3f}'.format
datasets = [
            read_data_cenario('cenario1.csv'),
            # read_data_cenario('cenario2.csv'),
            # read_data_cenario('cenario3.csv'),
            # read_data_cenario('cenario4.csv')
           ]
from sklearn.model_selection import train_test_split

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
    
    X=pd.DataFrame(X).fillna(0)
    
    dr='gp_'+dataset_name.replace(' ','_').replace("'","").lower()
    path='./pkl_'+dr+'/'
    os.system('mkdir  '+path)
    #%%      
    for n, target in enumerate(target_names):
        # y = dataset['y_train'][n]#.reshape(-1,1)
        # # y_test  = dataset['y_test' ][n]#.reshape(-1,1)
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
                    #'sqrt', 
                    'abs', 'neg', 'inv',
                    'log',
                    #'sin', 'cos', 'tan', 
                    'max', 'min',
                    logical,
                    ]
                    
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_train)
        

        est_gp =SymbolicClassifier(population_size=100, generations=20, stopping_criteria=0.01, 
                const_range=(-1e3, 1e3),
                function_set=function_set, 
                transformer='sigmoid', metric='log loss', parsimony_coefficient=0.001, p_crossover=0.8, 
                p_subtree_mutation=0.05, p_hoist_mutation=0.05, p_point_mutation=0.05, p_point_replace=0.05, 
                max_samples=1.0, feature_names=feature_names, warm_start=False, low_memory=False, n_jobs=-1
                , random_state=random_seed) #verbose=1 
        
        
        est_gp =SymbolicClassifier(population_size=100, 
                                   generations=50, 
                                   stopping_criteria=0.001, 
                                   const_range=(-1e3, 1e3),
                                   function_set=function_set, 
                                   transformer='sigmoid', 
                                   metric='log loss', 
                                   #metric=auc_scores,
                                   parsimony_coefficient=0.001, 
                                   p_crossover=0.8, 
                                   p_subtree_mutation=0.05, 
                                   p_hoist_mutation=0.05, 
                                   p_point_mutation=0.05, 
                                   p_point_replace=0.05, 
                                   max_samples=1.0, 
                                   feature_names=feature_names, 
                                   warm_start=False, 
                                   low_memory=False, 
                                   n_jobs=-1,
                                   random_state=random_seed,
                                   verbose=1,
                                   )
#        
       
        # X_train, X_test = random_state=42
        
        y_train_ =  np.squeeze(lb.transform(y_train))
      

        for i in range(y_train_.shape[1]):
            
            est_gp.fit(X_train,y_train_[:,i])#, verbose=True
            print(est_gp._program)
            
            # dot_data = est_gp._program.export_graphviz()
            # graph = graphviz.Source(dot_data)
            # Image(graph)
            # fn=(dataset_name+'__'+target).lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
            # graph.filename=fn; graph.render()
        
            clf=est_gp
            #%%
            y_pred = clf.predict(X_test)
            print(metrics.classification_report(y_test, y_pred, ))
            # rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
            # r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
            # print(rmse, r2,r)
    
            # pl.figure(figsize=(16,4)); 
            # # #pl.plot([a for a in y_train]+[None for a in y_test]); 
            # pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
            # # # pl.plot([None for a in y_train]+[a for a in y_pred], 'b-', label='Predicted');
            # # # pl.legend(); pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))
            # pl.show()
    
            # pl.figure(figsize=(6,6)); 
            # pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
            # pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
            # pl.show()

#%%-----------------------------------------------------------------------------

