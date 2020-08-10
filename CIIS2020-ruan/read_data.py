#!/usr/bin/python
# -*- coding: utf-8 -*
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

#-------------------------------------------------------------------------------
def atr_classes_def():
    
    #  Index(['BindingDB', 'Host', 'Guest', 'pH', 'Temp', 'SlogP', 'SMR', 'LabuteASA',
    #       'TPSA', 'AMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds',
    #       'NumAtoms', 'Formal Charge', 'SlogP (#1)', 'SMR (#1)', 'LabuteASA (#1)',
    #       'TPSA (#1)', 'AMW (#1)', 'NumLipinskiHBA (#1)', 'NumLipinskiHBD (#1)',
    #       'NumRotatableBonds (#1)', 'NumAtoms (#1)', 'delta_g0'],
    #      dtype='object')
  
    ## ATRIBUTOS DO MEIO - ENVIRONMENT     
    col_env = ['pH', 'Temp']
    
    ## ATRIBUTOS DO LIGANTE - LIGANT
    col_lig = ['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumAtoms', 'Formal Charge']
    
    ## ATRIBUTOS DO HOSPEDEIRO - HOST
    col_host = [i + ' (#1)' for i in col_lig]
    col_host = col_host[:-1]
    
    ## DICT DE ATRIBUTOS
    opt_sel_col = {'col_env': col_env,
                   'col_host': col_host,
                   'col_lig': col_lig,
                   'all_atr': col_env + col_lig + col_host}   
      
    ## ATRIBUTOS IDS
    col_ids = ['BindingDB', 'Host', 'Guest']
    
    return (opt_sel_col, col_ids)


def sep_data_base(X, y, groups, test_size=0.2, save_train_test = False, df=None):
    
    
    dt = np.vstack((X.T,y[0])).T
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    #train_index, test_index = tuple(sss.split(dt, groups))[0]

    train_index=np.array([164, 107,   2, 74,  84, 127,  35, 210,  81, 274,  52, 150, 227,
       219, 118,  27, 149,  92,  45, 278,  70, 271,  37,  93, 117, 209,
       228, 259,  17, 139,  30, 121,   4, 191, 205, 269,  78, 252, 172,
       182,  57, 100, 163, 173, 145,  49,  14,   5,  22, 204, 238,   8,
       108, 279,  85, 105,  12, 201, 231, 244, 235, 147,  53, 237,  59,
       273, 132,  15, 175,  63, 171,  20, 176, 222, 140,  25, 165, 104,
       261, 251, 180, 167, 221, 258, 247,   0, 193,  83,  50, 184,  42,
        62, 263, 229, 195,  87,  88,  66,   3,  39,  67, 169,  97,   1,
       114,  55, 111, 276, 220, 106, 103, 211, 168,  89, 162,   9, 249,
        98,  54, 203,  61, 246, 116,  86, 270,  56, 136, 225, 179,  69,
       154, 245,  24, 178,  38,  47, 254,  65, 243,  29, 159, 156, 128,
       214, 123, 192, 143, 250,  26, 151, 224,  11, 186, 177, 233, 241,
       272, 166, 200, 194, 257, 137,  28, 215, 189,  99,  41, 155,  16,
       265, 185,  76, 242, 153, 240, 133, 226, 198,  51, 248, 129, 275,
       148, 112,  73,  94, 239, 126,  10,  18, 161,  60, 217,  95, 255,
        34, 141, 208, 267,  44, 190, 256,   7, 134,  68, 102,  96, 234,
        43, 188, 174,  58, 218,  236, 207,  6, 152, 268,  266, 109, 119,
       144, 264, 223])
    
    test_index=np.array([232, 23, 113, 197,  213, 230, 91, 130,  33,  48,  71, 277, 138,
        40, 170, 160,  32, 260, 101, 196,  75, 262, 122, 125,  77,  72,
        90,  13,  80, 115, 187, 199, 212, 135, 158, 183,  36,  79, 142,
       253,  31, 131, 110, 202, 206,  19,   46, 120, 216,  82, 157,  21,
       124,  64, 146, 181])

    
    X = dt.T[:-1].T
    y = dt.T[-1].T
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    if save_train_test:
        
        aux_list = list()
        for i in df['Unnamed: 0']:
            #print(i)
            if i in train_index:
                aux_list.append(1)
            elif i in test_index:
                aux_list.append(0)
            else:
                print('shit!')
        
        df['isInTrainSet'] = aux_list
        
        df.to_csv('./data/data_hostguest_rmc/all_CDs_train_test.csv')
        
        #print(df.head(20))
        
    
    return (X_train, X_test, y_train, y_test)
    



#-------------------------------------------------------------------------------
def read_data_host_guest(filename='./data/data_hostguest_rmc/all_CDs.csv'):
    data = pd.read_csv(filename,sep=',')
  
    opt_sel_col, col_ids = atr_classes_def()
    
    ########################################################################### READ DATA
    X                     = data.drop(['Unnamed: 0', 'BindingDB', 'Host', 'Guest','delta_g0'], axis=1).values
    y                     = data[['delta_g0',]].T.values
    ids_values            = data[col_ids] 
    n_samples, n_features = X.shape
    
    X_train, X_test, y_train, y_test = sep_data_base(X, y, data['Host'], test_size=0.2, save_train_test = True, df=data)

      
    regression_data = {
    # Definição do Problema
    'name'                 : 'All CDs HostGuest problem',
    #
    # Conjunto de treino 
    'X_train'              : X_train, #X[:n_train],
    'y_train'              : y_train.reshape(1,-1), #y[0,:n_train].reshape(1,-1),
    #
    # Conjunto de teste
    'X_test'               : X_test, #X[n_train:],
    'y_test'               : y_test.reshape(1,-1), #y[0,n_train:].reshape(1,-1),
    #
    # Valores objetivo
    'target_names'         : ['delta_g0',],
    'targets'              : ['delta_g0',],
    #
    # Dados gerais
    'n_samples'            : n_samples, 'n_features':n_features,
    'reference'            : "Gilson, Michael K., et al. 'BindingDB in 2015: a public database for medicinal chemistry, computational chemistry and systems pharmacology.' Nucleic acids research 44.D1 (2016): D1045-D1053.",
    'url'                  : None,
    'true_labels'          : np.arange(n_samples),
    'predicted_labels'     : None,
    'task'                 : 'regression',
    'descriptions'         : np.arange(n_samples),
    'items'                : None,
    'feature_names'        : np.array(opt_sel_col['all_atr']),
    'dict_feature_names'   : opt_sel_col,
    'col_ids_name'         : col_ids,
    'ids_values'           : ids_values,    
    'normalize'            : 'None',
    }

    return regression_data
#-------------------------------------------------------------------------------
def read_data_dutos_csv(filename='./data/data_dutos/dados_simulacao.csv'):
  data = pd.read_csv(filename,header=1)
  data = np.array(data)
  data, y = data[:,0:6], data[:,6:8]
  n_samples, n_features =data.shape
  regression_data=  {
    'name':'Dented Pipes',
    'points':data,
    'n_samples':n_samples, 'n_features':n_features,
    'targets':np.array([y[:,0],y[:,1]]),#y[:,1],
    'reference':"",
    'true_labels':np.arange(n_samples),
    'predicted_labels':None,
    'task':'regression',
    'descriptions':np.arange(n_samples),
    'items':None,
    'feature_names':['F/D','sq/sy','D/t','d/D','l','L'], #[str(i) for i in np.arange(n_features)],
    'target_names':['EQV','EML'],      
      'normalize': 'None',
  }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_bogas(filename='./data/data_bogas/bogas_2013.xls'):  
  
  #filename="bogas_2013.xls"
  aux = pd.read_excel(filename)
  
  aux = aux.query("Type == ['A', 'B', ]")
  
  #A = aux
  #n_row,n_col = A.shape
  #col_aux=['1d','2d','3d','7d','14d','28d','56d','90d']
  #days = [1,2,3,7,14,28,56,90]
  
  #col_inputs	=  A.columns[::-1][1::]
  #col_output_1	= A.columns[::-1][0]#'Concrete compressive strength(MPa, megapascals)'
  
  col_output_1 = ['f_cm (MPa)']
  col_inputs = [
    'Cement (kg/m3)', 
    'Coarse aggregate (m3/m3)', 
    'Days', 
    'Effective w/c (L/m3)', 
    'Slump (mm)', 
    'Sp/c (%)',  
    'f_m (MPa)',
    #'f_LWA (MPa)',
  ]
  
  A = aux[col_inputs+col_output_1]
  cols_to_consider=list(col_inputs)
  grouped = A.groupby(cols_to_consider)
  index = [gp_keys[0] for gp_keys in grouped.groups.values()]
  unique_df = A.reindex(index)
  
  # fitering duplicated experiments average Compressive Strength 
  # http://wesmckinney.com/blog/filtering-out-duplicate-dataframe-rows/
  #B=[]
  #for key in grouped: 
    #l =  list( key[1].mean() )
    #B.append(l)

  #B = pd.DataFrame(B)
  #B.columns=A.columns

  B=A.dropna()
  B.drop_duplicates(inplace=True) 
  X = B.drop(col_output_1, axis=1)
  y1	= B[col_output_1].values
  
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
    'name':'Bogas 2013 Experimental Dataset',
    'points':data,
    #'targets':np.array([y1]),
    'reference':"https://www.sciencedirect.com/science/article/pii/S0041624X12002739",
    'n_samples':n_samples, 'n_features':n_features,
    'targets': np.array([ y1.mean(axis=1), ]),
    'true_labels':None,
    'predicted_labels':None,
    'descriptions':data_description,
    'items':None,
    'task':'regression',
    'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
    'target_names':['LWAC f_cm (Mpa)',],
    'target_std':None,      
      'normalize': 'None',
  }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_yeh(filename="./data/data_chou/yeh.xlsx"):  
  
  #filename="slump_test.xlsx"
  aux = pd.read_excel(filename)
  A = aux
  n_row,n_col = A.shape
  col_inputs	=  A.columns[::-1][1::]
  col_output_1	= A.columns[::-1][0]#'Concrete compressive strength(MPa, megapascals)'

  cols_to_consider=list(col_inputs)
  grouped = A.groupby(cols_to_consider)
  index = [gp_keys[0] for gp_keys in grouped.groups.values()]
  unique_df = A.reindex(index)

  # fitering duplicated experiments average Compressive Strength 
  # http://wesmckinney.com/blog/filtering-out-duplicate-dataframe-rows/
  B=[]
  for key in grouped: 
    l =  list( key[1].mean() )
    B.append(l)

  B = pd.DataFrame(B)
  B.columns=A.columns

  B.drop_duplicates(inplace=True)  
  X = B.drop(col_output_1, axis=1)
  y1	= B[[col_output_1]].values
 
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
    'name':'Dataset 1',
    'points':data,
    #'targets':np.array([y1]),
    'n_samples':n_samples, 'n_features':n_features,
    'targets': np.array([ y1.mean(axis=1), ]),
    'true_labels':None,
    'predicted_labels':None,
    'descriptions':data_description,
    'task':'regression',
    'reference':"http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength",
    'items':None,
    'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
    'target_names':['Concrete Compressive Strength',],
    'target_std':None,      
      'normalize': 'None',
  }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_lim(filename="./data/data_chou/lim.xlsx"):  
  aux = pd.read_excel(filename)
  A = aux
  col_inputs	=  ['Slump (mm)', 'W/B (%)', 'W (kg/m3)', 's/a (%)','FA (%)','AE (kg/m3)','SP (kg/m3)']
  col_output_1	= "fc (MPa)"
  X  	= A[col_inputs]
  y1	= A[[col_output_1]].values
 
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Dataset 2',
      'points':data,
      #'targets':np.array([y1]),
      'n_samples':n_samples, 'n_features':n_features,
    'targets': np.array([ y1.mean(axis=1), ]),
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'task':'regression',
      'target_names':['Concrete Compressive Strength',],
      'reference1':"https://www.sciencedirect.com/science/article/pii/S0008884603002977",
      'reference':'https://www.sciencedirect.com/science/article/pii/S0950061814010708#b0285',
      'target_std':None,      
        'normalize': 'None',
    }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_siddique(filename="./data/data_chou/siddique.xlsx"):  
  aux = pd.read_excel(filename)
  A = aux
  col_inputs	=  ['Cement (kg/m3)','Fly ash (kg/m3)','Water/powder','SP dosage (%)','Sand (kg/m3)','Coarse Agg (kg/m3)',]
  col_output_1	= "Strength (MPa)"
  A = A.drop_duplicates(subset=col_inputs)
  X  	= A[col_inputs]
  y1	= A[[col_output_1]].values
 
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Dataset 3',
      'points':data,
      #'targets':np.array([y1]),
      'n_samples':n_samples, 'n_features':n_features,
    'targets': np.array([ y1.mean(axis=1), ]),
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'task':'regression',
      'reference':"https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength",
      'target_names':['Concrete Compressive Strength',],
      'target_std':None,      
      'normalize': 'None',
    }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_pala(filename="./data/data_chou/pala-2007.xlsx"):  
  
  #filename="slump_test.xlsx"
  aux = pd.read_excel(filename)
  A = aux
  col_inputs	=  ["FA (%)","SF (%)","TCM (kg/m3)","ssa (kg/m3)","ca (kg/m3)","W (lt/m3)","HRWRA (lt/m3)","Age (days)",]
  col_output_1	= 'fc (MPa)'
  #A = A[ (A[col_output_1]>20) & (A[col_output_1]<70) ]
  X  	= A[col_inputs]
  y1	= A[[col_output_1]].values
 
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Dataset 5',
      'points':data,
      #'targets':np.array([y1]),
     'n_samples':n_samples, 'n_features':n_features,
     'targets': np.array([ y1.mean(axis=1), ]),
      'true_labels':None,
      'predicted_labels':None,
      'reference':"https://www.sciencedirect.com/science/article/pii/S0950061805000942",
      'descriptions':data_description,
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'target_names':['Concrete Compressive Strength',],
      'target_std':None,
      'task':'regression',      
      'normalize': 'None',
    }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_slump(filename="./data/data_slump/slump_test.xlsx"):  
  
  #filename="slump_test.xlsx"
  aux = pd.read_excel(filename)
  A = aux
  col_inputs	= ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
  col_output_1	= 'SLUMP(cm)'
  col_output_1	= 'FLOW(cm)'
  
  A = A[ (A[col_output_1]>20) & (A[col_output_1]<70) ]
  
  X  	= A[col_inputs]
  y1	= A[[col_output_1]].values
 
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Slump Test',
      'points':data,
      #'targets':np.array([y1]),
     'n_samples':n_samples, 'n_features':n_features,
     'targets': np.array([ y1.mean(axis=1), ]),
      'true_labels':None,
      'reference':"https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test",
      'predicted_labels':None,
      'descriptions':data_description,
      'task':'regression',
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'target_names':['Slump',],
      'target_std':None,      
      'normalize': 'None',
    }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_geraldo_csv(filename):
  data = pd.read_csv(filename,header=1)
  data_description = [str(i) for i in data['CP'].values]
  list_items=[1,2,3,4,5,6]
  variable_names = np.array(data.columns[list_items])
  data = data[list_items]
  data = np.array(data)
  y = data[:,-1]
  data=data[:,:(data.shape[1] - 1)]
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Misturas Betuminosas',
      'points':data,
     'n_samples':n_samples, 'n_features':n_features,
     'targets':np.array([[i] for i in y]),
      'true_labels':None,
      'task':'regression',
      'predicted_labels':None,
      'descriptions':data_description,
      'reference':"",
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'target_names':['MR',],
      'target_std':None,      
      'normalize': 'None',
    }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_bituminous_marshall(filename='./data/data_bituminous/arquivo_dados_completos.xls', 
                                  n_var=4, experiment='Marshall'):  
  #filename="arquivo_dados_completos.xlsx"
  aux = pd.read_excel(filename)
  #aux = aux[aux['MISTURA']==3]
  
  A = aux[aux['E']== experiment]

  if n_var == 4:
    col_inputs	= ['Visc', 't', 'Va', 'T',]
  elif n_var == 6:
    col_inputs	= ['Visc', 't', 'Va', 'VMA', 'VFA',  'T']
  elif n_var == 9:
    col_inputs	= ['Visc', 't', 'Va', 'VMA', 'VFA', 'T', 'AG', 'GAF', 'FAF']
  elif n_var == 10:
    col_inputs	= [ 'MISTURA', 'Visc', 'IST', 't', 'Gmb', 'Gmm', 'Va', 'VMA', 'VFA',  'T',]
  elif n_var == 11:
    col_inputs	= [ 'MISTURA', 'Visc', 'IST', 't', 'Gmb', 'Gmm', 'Va', 'VMA', 'VFA',  'T', 'RTa']
  else:
    print('Please check the number of variables of the problem')
    exit()
      
  
  try:
    col_output_1 = ['MR1', 'MR2', 'MR3']
  except:
    col_output_1	= ['MR']

  # remove duplicates
  A = A[col_inputs+col_output_1]
  cols_to_consider=list(col_inputs)
  grouped = A.groupby(cols_to_consider)
  index = [gp_keys[0] for gp_keys in grouped.groups.values()]
  unique_df = A.reindex(index)

  # fitering duplicated experiments average Compressive Strength 
  # http://wesmckinney.com/blog/filtering-out-duplicate-dataframe-rows/
  B=[]
  for key in grouped: 
    l =  list( key[1].mean() )
    #if len(key[1])>1: print key[1]     
    B.append(l)

  B = pd.DataFrame(B)
  B.columns=A.columns
  #

  MR = B[col_output_1]
  MR = MR[MR>0]
  
  X  	= B[col_inputs]
  y1	= B[col_output_1].values
  #y1 	= scaler.fit_transform(y1)  
  #y2	= aux[col_output_2] 
  #y2 	= scaler.fit_transform(y2)

  vm= [1582.23146637,  3356.76755887,  5131.30365137,  6905.83974388,
        8680.37583638, 10454.91192889, 12229.44802139, 14003.98411389,
       15778.5202064 , 17553.0562989 , 19327.59239141]
  
  ym = y1.mean(axis=1)
  km=ym.copy()
  kl = [False for i in range(y1.shape[0])]
  for i in range(len(ym)):
    for j in range(1,len(vm)):
        if (vm[j-1]<ym[i]<=vm[j]):
            km[i]=j
  
  pl.plot(km, ym, 'ro')
     
  sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  for train_index, test_index in sss.split(X.values, km):
    print("TRAIN:", train_index, "TEST:", test_index)
    
  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Dense Bituminous Mixes - ' + experiment,
      'points':data,
      #'targets':np.array([y1]),
     'n_samples':n_samples, 'n_features':n_features,
     'targets': np.array([ MR.mean(axis=1).values, ]), #np.array([np.squeeze(y1)]),
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'task':'regression',
      'reference':"",
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'target_names':['Resilient Modulus',],
      'target_std':np.array([ MR.std(axis=1).values, ]),      
      'normalize': 'None',
    }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_bituminous_mixes(filename):
 
  aux = pd.read_excel(filename)
  aux = aux.dropna(axis=1) # remove colunas com NAN
  aux.drop_duplicates(take_last=True, inplace=True)

  
  col_inputs	= [ 'Visc', 'IST', 't', 'Gmb', 'Gmm', 'Va', 'VMA', 'VFA',  'T']
  col_inputs	= [ 'Visc', 't', 'Va', 'T']
  col_output_1	= ['MR']
  col_output_2	= ['RTa']

  #aux = aux[aux['MISTURA']==1]
  #col_inputs	= [ 'Visc', 't', 'Va', 'T']
  
  #aux = aux[aux['T']==25]
  #col_inputs	= [ 'Visc', 't', 'Va',]

  scaler = preprocessing.MinMaxScaler(feature_range=(0.15, 0.85))

  X  	= aux[col_inputs]
  
  y1	= aux[col_output_1].values/1e3
  #y1 	= scaler.fit_transform(y1)  
  #y2	= aux[col_output_2] 
  #y2 	= scaler.fit_transform(y2)

  names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  
  #xl = pd.ExcelFile(filename)
  #sheet_names=xl.sheet_names
  #dfs = {sheet_name: xl.parse(sheet_name,header=None) 
          #for sheet_name in xl.sheet_names}

  #df= dfs[sheet_names[0]]
  #data=df.values
  #names = np.array([str(i) for i in data[0]])
  #data = np.delete(data,0,axis=0)
  #y1 = np.array(data[:,-1],dtype=float)
  
  #list_items=[0,5,6,7,8,9,10,11,12,13,14]  # variavel 13 é o RT - resistenca a tração (somente a 25 °C)
  #list_items=[0,5,6,7,10,11,12,14]
  #list_items=[5,7,8,9,10,11,12,14]
  #list_items=[5,7,8,9,14]
  #data = data[:,list_items]
  #data = np.array(data,dtype=float)
  #data_description = ['var_'+str(i) for i in range(data.shape[1])]
  #variable_names = names[list_items]
  	
  #data=data[:,:(data.shape[1] - 1)]
  n_samples, n_features =data.shape
  regression_data=  {
      'name':'Dense Bituminous Mixes',
      'points':data,
      #'targets':np.array([y1]),
         'n_samples':n_samples, 'n_features':n_features,
         'targets':np.array([np.squeeze(y1)]),
      'true_labels':None,
      'predicted_labels':None,
      'task':'regression',
      'descriptions':data_description,
      'items':None,
      'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
      'reference':"",
      'target_names':['Resilient Modulus',],
      #'target_names':['Modulo de resiliencia',],      
      'normalize': 'None',
    }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_efficiency(filename='./data/data_efficiency//ENB2012_data.xlsx'):
 
  xl = pd.ExcelFile(filename)
  sheet_names=xl.sheet_names
  dfs = {sheet_name: xl.parse(sheet_name,header=None) 
          for sheet_name in xl.sheet_names}

  df= dfs[sheet_names[0]]
  data=df.values
  names = [str(i) for i in data[0]]
  data = np.delete(data,0,axis=0)
  y2 = np.array(data[:,-1],dtype=float)
  y1 = np.array(data[:,-2],dtype=float)
  
  
  data_description = ['X_'+str(i) for i in range(8)]
  variable_names = np.array([
    'Relative Compactness',
    'Surface Area',
    'Wall Area', 
    'Roof Area', 
    'Overall Height',
    'Orientation', 
    'Glazing Area',
    'Glazing Area Distribution ',
  ])
  target_names = np.array([
    'Heating Load', 
    'Cooling Load',
    ])
  data = np.array(data,dtype=float)
  data = data[:,0:8]
  n_samples, n_features =data.shape
  
  X_train, X_test, y_train, y_test = data, np.array([]), np.array([y1,y2]), np.array([[],[]])
  n_samples, n_features =data.shape
  dataset=  {
      'task':'regression',
      'name':'Energy Efficiency',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train,
      'X_test':X_test,
      'y_train':y_train,
      'y_test':y_test,      
      'points':data,
      'targets':np.array([y1,y2]),
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,      
      'normalize': 'None',
      'reference':"https://archive.ics.uci.edu/ml/datasets/energy+efficiency",
      }
  return dataset
#-------------------------------------------------------------------------------
def read_data_toy(filename=None):
  from sklearn.datasets import make_regression
  n_samples, n_features, n_informative, n_targets = 400, 100, 90, 2
  effective_rank, noise = n_informative, 1./n_features

  variable_names = ['X_'+str(i) for i in range(n_features)]
  target_names = ['Y_'+str(i) for i in range(n_targets)]
  X, y =  make_regression(n_samples, n_features, n_informative, n_targets, effective_rank, noise, random_state=0)
	  
  regression_data=  {
	'name':'Toy Problem',
	'points':X,
	'targets':[i for i in y.T],
	     'n_samples':n_samples, 'n_features':n_features,
         'true_labels':None,
	'predicted_labels':None,
      'reference':"",
	'descriptions':variable_names,
	'items':None,
      'task':'regression',
	'feature_names':variable_names,
	'target_names':target_names,      
      'normalize': 'None',
      }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_shamiri(filename='./data/data_shamiri/shamiri.csv'):
  
    data = pd.read_csv(filename)
    
    variable_names=[u'Water', u'Cement', u'Fine aggregate', u'Coarse aggregate',
       u'Superplasticizer', ]
    target_names=[ u'Compressive strength']
    X=data[variable_names].values
    y=[data[target_names].values.ravel()]
    n_samples, n_features = X.shape 
      
    #variable_names = ['X_'+str(i) for i in range(n_features)]
    #target_names = ['Y_'+str(i) for i in range(n_targets)]
    	  
    regression_data=  {
    	'name':'Shamiri',
    	'points':X,
    	'targets':y,
         'n_samples':n_samples, 'n_features':n_features,
	'true_labels':None,
    	'predicted_labels':None,
      'reference':"https://doi.org/10.1016/j.conbuildmat.2019.02.165",
    	'descriptions':variable_names,
    	'items':None,
      'task':'regression',
    	'feature_names':variable_names,
    	'target_names':target_names,      
      'normalize': 'None',
      }
    return regression_data
#-------------------------------------------------------------------------------
def read_data_cergy(path='./data/data_cergy/'):
    data = pd.read_csv(path + 'data_cergy_modificado.csv')
    
    target_var={'$f_c$ (MPa)':['R1', 'R2', 'R3','R4'], '$E$ (GPa)':['M1', 'M2', 'M3', 'M4']}
    
    for t in target_var: 
        for i in target_var[t]:
            data[i]=np.where(data[i]==0, None, data[i],)
            
            data[t] = data[target_var[t]].mean(axis=1, )
    
    X_=data.copy()        
    for t in target_var: 
        X_.drop(t, axis=1, inplace=True)
        for i in target_var[t]:
            X_.drop(i, axis=1, inplace=True)
    
    Y_ = data[target_var.keys()]
    #Y_['$E$ (GPa)'] /= 1e3

    n_features, n_targets = 4,2
    variable_names = ['X'+str(i) for i in range(n_features)]
    target_names = Y_.columns.values#['Y'+str(i) for i in range(n_targets)]
    n_samples, n_features =X_.shape 
    data_description = X_.columns.values
    
    targets=[ Y_.values.T[i] for i in range(Y_.values.shape[1])]
    
    regression_data=  {
      'name':'Cergy Lightweight Concrete',
      'points':X_,
      #'targets':np.array([E.mean(axis=1).values,Fc.mean(axis=1).values]),
      'targets':targets,
          'n_samples':n_samples, 'n_features':n_features,
 'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'feature_names':variable_names,
      'reference':"",
      'target_names':target_names,
      'task':'regression',      
      'normalize': 'None',
      #'target_std':np.array([E.std(axis=1).values,Fc.std(axis=1).values]),
    }
    return regression_data
#-------------------------------------------------------------------------------
def read_data_borgomano(filename='./data/data_borgomano/borgomano.csv'):
  
    data = pd.read_csv(filename)
    
    X=data.drop('Class', axis=1).values.copy()
    n_samples, n_features = X.shape 
    y=[data['Class'].values.copy()]
    target_names=[ u'Class']    
    variable_names=data.drop('Class', axis=1).columns
        	  
    regression_data=  {
    	'name':'Borgomano',
    	'points':X,
    	'targets':y,
         'n_samples':n_samples, 'n_features':n_features,
	'true_labels':['C'+str(i) for i in y[0]],
    	'predicted_labels':None,
      'reference':"doi.org/10.1016/j.jappgeo.2018.06.012",
    	'descriptions':variable_names,
    	'items':None,
      'task':'classification',
    	'feature_names':variable_names,
    	'target_names':target_names,      
      'normalize': 'None',
      }
    return regression_data
#-------------------------------------------------------------------------------
def read_data_xie_dgf(filename='./data/data_xie/1-s2.0-S0920410517308094-mmc1.xlsx'):
  
    xl = pd.ExcelFile(filename)
    le 	= preprocessing.LabelEncoder()

    for xx in xl.sheet_names[0:1]:
        df=xl.parse(xx)
        nam=xx.split()[0]
        print(nam,df.columns)
        variable_names=['GR', 'AC', 'DEN', 'CNL', 'LLD', 'LLS', 'CAL']
        X = df[variable_names].values.copy()
        le.fit(df['Type'])
        y=[le.transform(df['Type'])]
        true_labels=df['Type']
        n_samples, n_features = X.shape    
        target_names=[ u'Class']    
    
    
        	  
    regression_data=  {
    	'name':'DGF',
    	'points':X,
    	'targets':y,
    	'true_labels':true_labels,
    	'predicted_labels':None,
        'reference':"https://doi.org/10.1016/j.petrol.2017.10.028",
         'n_samples':n_samples, 'n_features':n_features,
	'descriptions':variable_names,
    	'items':None,
        'task':'classification',
    	'feature_names':variable_names,
    	'target_names':target_names,      
      'normalize': 'None',
      }
    return regression_data
#-------------------------------------------------------------------------------
def read_data_xie_hgf(filename='./data/data_xie/1-s2.0-S0920410517308094-mmc1.xlsx'):
  
    xl = pd.ExcelFile(filename)
    le 	= preprocessing.LabelEncoder()

    for xx in xl.sheet_names[1:2]:
        df=xl.parse(xx)
        nam=xx.split()[0]
        print(nam,df.columns)
        variable_names=['GR', 'AC', 'DEN', 'CNL', 'LLD', 'LLS', 'CAL']
        X = df[variable_names].values.copy()
        le.fit(df['Type'])
        y=[le.transform(df['Type'])]
        true_labels=df['Type']
        n_samples, n_features = X.shape    
        target_names=[ u'Class']    
    
    
        	  
    regression_data=  {
    	'name':'HGF',
    	'points':X,
    	'targets':y,
    	     'n_samples':n_samples, 'n_features':n_features,
'true_labels':true_labels,
    	'predicted_labels':None,
        'reference':"https://doi.org/10.1016/j.petrol.2017.10.028",
    	'descriptions':variable_names,
    	'items':None,
        'task':'classification',
    	'feature_names':variable_names,
    	'target_names':target_names,      
      'normalize': 'None',
      }
    return regression_data
#-------------------------------------------------------------------------------
def read_data_nguyen_02(filename='./data/data_nguyen/Dataset2.xlsx'):
  aux = pd.read_excel(filename)
  #aux = aux.query("Type == ['A', 'B', ]")
  col_outputs = ['Concrete compressive strength\n(MPa) ']
  X = aux.drop(col_outputs, axis=1)
  y= [aux[col_outputs].values.ravel()]
    
  col_inputs=names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
    'name':'Nguyen02',
    'dataset_description':'Nguyen: 1133 samples of high performance concrete',
    'points':X,
    #'targets':np.array([y1]),
    'reference':"http://doi.org/10.1016/j.conbuildmat.2018.05.201",
    'n_samples':n_samples, 'n_features':n_features,
    'targets': y,
    'true_labels':None,
    'predicted_labels':None,
    'descriptions':data_description,
    'items':None,
    'task':'regression',
    'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
    'target_names':col_outputs,
    'target_std':None,      
      'normalize': 'None',
  }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_nguyen_01(filename='./data/data_nguyen/Dataset1.xlsx'):
  aux = pd.read_excel(filename)
  #aux = aux.query("Type == ['A', 'B', ]")
  col_outputs = ['Comp-28D']
  X = aux.drop(col_outputs, axis=1)
  y= [aux[col_outputs].values.ravel()]
    
  col_inputs=names = np.array([str(i) for i in X.columns])
  data=X.values
  data_description = ['var_'+str(i) for i in range(X.shape[1])]
  variable_names = col_inputs
  n_samples, n_features =data.shape
  regression_data=  {
    'name':'Nguyen01',
    'dataset_description':'Nguyen: 1 contains 177 samples of foamed concrete',
    'points':X,
    #'targets':np.array([y1]),
    'reference':"doi.org/10.1111/mice.12422",
    'n_samples':n_samples, 'n_features':n_features,
    'targets': y,
    'true_labels':None,
    'predicted_labels':None,
    'descriptions':data_description,
    'items':None,
    'task':'regression',
    'feature_names':['X'+str(i+1) for i in range(data.shape[1])],
    'target_names':col_outputs,
    'target_std':None,      
      'normalize': 'None',
  }
  return regression_data 
#-------------------------------------------------------------------------------
def read_data_akyuncu(filename='./data/data_akyuncu/akyuncu_2018.csv'):
  
  data = pd.read_csv(filename,sep=',')
  X=data.drop(['No', 'Concrete Type',], axis=1).values
  y=data[['Slump (cm)']].values
  n_samples, n_features =X.shape
  
  train_index = range(X.shape[0])
  X_train, X_test, y_train, y_test = X[train_index], None, y[train_index], None
#    
#    dataset = {}
#    dataset['var_names'], dataset['target_names'] = X_train.columns, y_train.columns
#    dataset['name'] = f.split('.csv')[0].split('/')[-1]
#    dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = X_train.values, X_test.values, [y_train.values], [y_test.values]
#    dataset['n_samples'], dataset['n_features'] = X_train.shape
#    dataset['task'] = 'regression'

  regression_data=  {
    'name':'Tahiri Cooling and Heating Loads',    
    'points':X,
    'n_samples':n_samples, 'n_features':n_features,
    'targets':y,#np.array([y[:,0],y[:,1]]),
    'reference':"https://doi.org/10.1016/j.csite.2018.03.006",
    'true_labels':np.arange(n_samples),
    'predicted_labels':None,
    'task':'regression',
    'descriptions':np.arange(n_samples),
    'items':None,
    'feature_names':data.drop(['Cooling', 'Heating'], axis=1).columns,
    'target_names':['Cooling', 'Heating'],      
     'normalize': 'None',
  }
  return regression_data
#-------------------------------------------------------------------------------
def read_data_qsar_aquatic(filename='./data/data_qsar/qsar_aquatic_toxicity.csv'):
  
  data = pd.read_csv(filename,sep=';')
  X=data.drop(['LC50 [-LOG(mol/L)]',], axis=1)
  y=data[['LC50 [-LOG(mol/L)]']]
  y.columns=['LC50']
  n_samples, n_features =X.shape
  
  test_index=[int(4.95*(i+1)) for i in range(110)]
  train_index = range(X.shape[0])
  X_train, X_test, y_train, y_test = X.values[train_index], np.array([]), y.values[train_index], np.array([])
  target_names=y.columns
  variable_names=X.columns

  X_train, X_test, y_train, y_test = np.delete(X.values,test_index, axis=0),X.values[test_index],np.delete(y.values,test_index, axis=0)/1000.,y.values[test_index]/1000.

  dataset=  {
      'name':'QSAR aquatic toxicity',    
      'task':'regression',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train,
      'X_test':X_test,
      'y_train':[y_train],
      'y_test':[y_test],      
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':'https://archive.ics.uci.edu/ml/datasets/QSAR+aquatic+toxicity#',
      'items':None,
      'reference':"https://sci-hub.tw/10.1177/026119291404200106",      
      'normalize': 'None',
      }
  return dataset
#-------------------------------------------------------------------------------
def read_data_burkina_faso(filename):

    X=pd.read_excel(filename, header=1, index_col=0)
    
    train_date = X.index <= '31-12-2008'
    test_date  = X.index >  '31-12-2008'
    
    y = X.drop(X.columns[:-1], axis=1)
    X = X.drop(X.columns[-1], axis=1)    
    X_train, X_test, y_train, y_test = X[train_date], X[test_date], y[train_date], y[test_date] 
    
    n_samples, n_features = X_train.shape
    variable_names=np.array(X_train.columns)
    target_names=np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    dataset=  {
      'task':'regression',
      'name':'Solar Radiation '+filename.split('.xlsx')[0].split('/')[-1],
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train.values,
      'X_test':X_test.values,
      'y_train':np.array([y_train.values.ravel()]),
      'y_test':np.array([y_test.values.ravel()]),      
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"https://doi.org/10.3390/en12071365",      
      'normalize': 'MinMax',
      }   
    return dataset
#-------------------------------------------------------------------------------
def read_data_burkina_faso_boromo(filename='./data/data_burkina_faso/Boromo.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_dori(filename='./data/data_burkina_faso/Dori.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_gaoua(filename='./data/data_burkina_faso/Gaoua.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_po(filename='./data/data_burkina_faso/Po.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_bobo_dioulasso(filename='./data/data_burkina_faso/Bobo Dioulasso.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_bur_dedougou(filename='./data/data_burkina_faso/Bur Dedougou.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_fada_ngourma(filename="./data/data_burkina_faso/Fada N'gourma.xlsx"):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def read_data_burkina_faso_ouahigouy(filename='./data/data_burkina_faso/Ouahigouya.xlsx'):
    return(read_data_burkina_faso(filename=filename))    
#-------------------------------------------------------------------------------
def secSinceNoon(datTimStr):
    tt = pd.to_datetime(datTimStr).time()
    return tt.hour * 3600 + tt.minute * 60 + tt.second

def read_data_energy_appliances(filename='./data/data_appliances/energydata_complete.csv'):
    #%%
    filename='./data/data_appliances/energydata_complete.csv'
    X=pd.read_csv(filename)
    X['NSM'] = X['date'].apply(secSinceNoon)
    X['Week day'] = pd.DatetimeIndex(X['date']).dayofweek
    X['Week status'] = ((pd.DatetimeIndex(X['date']).dayofweek) // 5 == 0).astype(int)
    
    X.drop(['date'], axis=1, inplace=True)
    
    train_index     = 14803
    
    y = X[X.columns[:1]]
    X = X[X.columns[1:]]
    
    X_train, X_test, y_train, y_test = X[:train_index], X[train_index:], y[:train_index], y[train_index:] 
    
    n_samples, n_features = X_train.shape
    variable_names=np.array(X_train.columns)
    target_names=np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    dataset=  {
      'task':'regression',
      'name':'Energy Appliances',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train.values,
      'X_test':X_test.values,
      'y_train':[y_train.values.ravel()],
      'y_test':[y_test.values.ravel()],      
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"http://bit.ly/2QDtQM7",      
      'normalize': 'None',
      }   
    #%%
    return dataset
#-------------------------------------------------------------------------------
def read_data_b2w(filename='./data/data_b2w/cleaned.csv'):
    filename='./data/data_b2w/cleaned.csv'
    df = pd.read_csv(filename, parse_dates=['date'])
    df = df.sort_values(['date','sku_id']) # This sorting works better for splitting train test
    
    df['dow'] = df['date'].dt.dayofweek
    df['dwk'] = df['dow'].apply(lambda x: 1 if x>=5 else 0)
    df['dom'] = df['date'].dt.day
    
    df = df.set_index('date')
    
    n = df['sku_id'].nunique()
    for i in range(7):
        i += 1
        df['y{}'.format(i)] = df.sku_sales.shift(-n*(i))
    
    df = df.dropna() # Drop if any empty
    
    X, y = df.iloc[:,:-7], df.iloc[:,-7:]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n, shuffle=False,random_state=0)
    
    n_samples, n_features = X_train.shape
    variable_names=np.array(X_train.columns)
    target_names=np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    dataset=  {
      'task':'forecast',
      'name':'B2W Sales Forecast',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train.values,
      'X_test':X_test.values,
      'y_train':y_train.T.values,
      'y_test':y_test.T.values,      
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"Falar com Rodrigo Santis",      
      'normalize': 'None',
      }   
    return dataset
#-------------------------------------------------------------------------------
def make_instances_from_dataframe(data, ts, window_size, output_size=1):
    datetime = data.index    
    data = np.asarray(data)
    assert 0 < window_size+output_size < data.shape[0]
    X = np.atleast_3d(np.array([data[start:start + window_size] for start in range(0, data.shape[0] - window_size-output_size)]))
    X = np.asanyarray([ x.T.ravel() for x in X])    
    X = np.atleast_2d(X).astype(float)

    y = np.zeros((X.shape[0],output_size))
    for i in range(y.shape[1]):
        y[:,i] =ts[window_size+i:window_size+i+y.shape[0]].ravel()        
        
    y = np.atleast_2d(y).astype(float)  
    idx = ~np.isnan(np.asarray(np.c_[X, y])).any(axis=1)
    dt=datetime[:datetime.shape[0] - window_size-output_size]
    return X[idx], y[idx]

#-------------------------------------------------------------------------------
def read_data_iraq_monthly(filename='./data/data_iraq_monthly/iraq_monthly_inflow.csv'):

    X = pd.read_csv(filename, sep=';', )
    c='montlhly inflow'
    X.columns = [c]
    #idx = pd.date_range('2010-12-03', periods=len(X), freq='-30.41D')
    X[c]/=X[c].max()

    y_train, y_test = X[len(X)-235:len(X)-48], X[len(X)-235+187: len(X)]
    
    X_train, X_test = [], []
    n=5
    for i in range(n):
        #print(i,[len(X)-235-i-1,len(X)-48-i-1])
        X_train.append(X[len(X)-235-i-1:len(X)-48-i-1].values.ravel())
    
        #print(i,[len(X)-235+187-i-1, len(X)-i-1])
        X_test.append(X[len(X)-235+187-i-1: len(X)-i-1].values.ravel())

    X_train = np.array(X_train).T
    y_train = y_train.values.ravel()
    X_test = np.array(X_test).T
    y_test = y_test.values.ravel()

    #pl.figure(figsize=(16,4)); 
    #pl.plot([a for a in y_train]+[None for a in y_test]); 
    #pl.plot([None for a in y_train]+[a for a in y_test]); 
    #pl.show()
       
    n_samples, n_features = X_train.shape
    variable_names=np.array(['Q_{t-'+str(i+1)+'}' for i in range(X_train.shape[1])])
    target_names=np.array(['Q_{t}'])
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    dataset={
          'task':'forecast',
          'name':'Iraq Inflow',
          'feature_names':variable_names,'target_names':target_names,
          'n_samples':n_samples, 'n_features':n_features,
          'X_train':X_train,
          'X_test':X_test,
          'y_train':np.array([y_train.T]),
          'y_test':np.array([y_test.T]),      
          'targets':target_names,
          'true_labels':None,
          'predicted_labels':None,
          'descriptions':data_description,
          'items':None,
          'reference':"https://www.hindawi.com/journals/complexity/2020/8206245/",
          # https://bit.ly/33DXpC8
          'normalize': 'None',
      }
    return dataset
#-------------------------------------------------------------------------------
#%% 
def read_data_british_columbia_daily_sequence(
            filename='./data/data_british_columbia/Residential_1.tab',
            look_back=14, look_forward=1, kind='ml', roll=False, plot=False,window=1,
        ):
    #%%
    # filename='./data/data_british_columbia/Residential_1.tab'
    # #look_back=14; look_forward=1; kind='ml'; roll=True; plot=True ;window=7;    
    # X = pd.read_csv(filename, sep='\t', )
    # dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in X['date']])
    # dates['hour']=X['hour']
    # X['hour']=X['hour']
    # X.index=pd.to_datetime(dates)
    # X['weekday'] = pd.to_datetime(X['date']).dt.dayofweek  # monday = 0, sunday = 6
    # X['weekend_indi'] = 0          # Initialize the column with default value of 0
    # X.loc[X['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun

    # c='energy_kWh'
    # out_seq=X.groupby('date').agg(sum)[[c]]
    # df=X.groupby('date').agg(np.mean)
    # df['energy_kWh']=df['Target']=out_seq
    
    # df.drop(['hour'], axis=1, inplace=True)
    
    # target_names=[c]
    # dates = df.index
    # ds = df.values
    # n_steps_in, n_steps_out = look_back, look_forward
    # X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    # y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    # dates=df.index[look_forward+look_back-1::]
    filename='./data/data_british_columbia/Residential_1.tab'
    df  = pd.read_csv(filename, delimiter='\t')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date']])
    dates['hour']=df['hour']
    df.index=pd.to_datetime(dates)
    df.sort_index(inplace=True)
    #df['energy_kWh'].plot(); pl.show()        
    
    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek  # monday = 0, sunday = 6
    df['weekend_indi'] = 0          # Initialize the column with default value of 0
    df.loc[df['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun
    
    aux=df.groupby('date').agg(sum)
    rolling_window=3
    aux=aux.rolling(window=rolling_window, win_type=None, min_periods=1).mean()
    
    
    df_solar = pd.read_csv('./data/data_british_columbia/Solar.tab', delimiter='\t')
    dates_solar=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df_solar['date']])
    dates_solar['year']=2000
    df_solar.index=pd.to_datetime(dates_solar)
    df_solar=df_solar.groupby('date').agg(np.sum)
    df_solar = pd.concat([df_solar]*30,axis=0)
    df_solar.index=pd.date_range(start='2000/01/01', periods=len(df_solar))
    mask = (df_solar.index >= aux.index[0]) & (df_solar.index <= aux.index[-1])
    df_solar=df_solar[mask]
    df_solar.drop(['hour', 'dc_output'], axis=1, inplace=True)
    
    df_weather1 = pd.read_csv('./data/data_british_columbia/Weather_YVR.tab', delimiter='\t')
    df_weather1.index=pd.to_datetime(df_weather1['date'])
    weather=pd.get_dummies(df_weather1['weather'],prefix=None)
    weather=weather.groupby(weather.index).agg(sum)
    weather.sort_index(inplace=True)
    
    df_weather1.drop(['hour', 'weather', 'date'], axis=1, inplace=True)
    df_weather1=df_weather1.groupby('date').agg(np.mean)
    #df_weather1=pd.concat([df_weather1,weather],axis=1)
    df_weather1.sort_index(inplace=True)
    
    mask = (df_weather1.index >= aux.index[0]) & (df_weather1.index <= aux.index[-1])
    df_weather1=df_weather1[mask]
    
    aux.drop('hour', axis=1, inplace=True)
    
    df=df.groupby('date').agg(np.mean)
    df.drop('hour', axis=1, inplace=True)
    df=pd.concat([df,df_weather1],axis=1)
    df=pd.concat([df,df_solar],axis=1)
      
    target='energy_kWh'
    out_seq=aux[target].values
    #df.drop([target,], inplace=True, axis=1)
    #df[target]=out_seq
    df['energy_kWh']=df['Target']=out_seq
    #cols=[i for i in reversed(df.columns)]
    #df=df[cols]
    
    df.drop(index=df.index[-1], inplace=True)
    
    dates = df.index
    ds = df.values
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    dates=df.index[look_forward+look_back-1::]

    if roll:
        y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
        y=y_roll
        
        
    # df=X.groupby('date').agg(sum)[[c]]
    # df.columns = [c]
    # out_seq = df[c].values
    # feature_names=df.columns    
    # df['Target']=out_seq
    # target_names=[c]
    # ds = df.values
    # n_steps_in, n_steps_out = look_back, look_forward
    # X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    # y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    
    # if roll:
    #     y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
    #     y=y_roll
    #     #pl.plot([a    for a in  y_roll], 'r-', label='', lw=0.9);    
    #     #pl.plot([a    for a in       y], 'b-', label='', lw=0.1);    
    #     #pl.show()

    train_size = int(X.shape[0]*.7)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    
    if plot:
        pl.figure()
        pl.plot([a    for a in y_train]+[None for a in y_test], 'r-', label='Train', lw=0.9)
        pl.plot([None for a in y_train]+[a    for a in y_test], 'b-', label='Test', lw=0.9)
        pl.legend()
        pl.show()
        
    n_samples, n_features,_ = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
        target_names=['energy_kWh']
       
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'British Columbia '+' day ahead '+kind,
      'feature_names'   : data_description,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "doi.org/10.1016/j.enbuild.2020.109864",      
      'normalize'       : 'MinMax',
      }
    #%%
    return dataset
#-------------------------------------------------------------------------------
def read_data_alameer_sequence(
            filename=None, 
            look_back=1, look_forward=1, kind='ml', roll=False, plot=False,window=7,
            target='coal-australian', train_split=0.7,
        ):
    #%%
    #look_back=1; look_forward=1; kind='ml'; roll=False; plot=True ;window=5; train_split=0.7
    target='coal-australian'
    data_dir='./data/data_alameer/'
    
    dic={'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12, 'Jan':1, 'Feb':2, 'Mar':3}
    
    df=pd.DataFrame()
    fn_list=glob.glob(data_dir+'*.csv')
    for fn in fn_list:
        print(fn)
        #os.system('libreoffice --headless --convert-to csv '+fn)
        aux=pd.read_csv(fn, delimiter=',', header=1)
        nam=fn.split('-360')[0].split('/')[-1]
        df[nam]=aux['Price'].values
        d=[m.split(' ') for m in aux['Month']]
        d=pd.DataFrame(data=d, columns=['month', 'year'])
        d['month']=[ dic[m] for m in d['month']]
        d['day']=1
        df.index=pd.to_datetime(d)
        df.sort_index(inplace=True)
        #df=df.reindex()
        #df[nam].plot(); pl.show()
        
    fn_list=glob.glob(data_dir+'*.tsv')
    for fn in fn_list:
        print(fn)
        #os.system('libreoffice --headless --convert-to csv '+fn)
        aux=pd.read_csv(fn, delimiter='\t', header=0)
        aux=aux[-361:]
        aux.index=pd.to_datetime(aux['Month'])
        aux.sort_index(inplace=True)
        nam=fn.split('.tsv')[0].split('/')[-1]
        df[nam]=aux['Average'].values
    
    out_seq = df[target]
    feature_names=df.columns    
    df['Target']=out_seq 
    target_names=[target]
    dates = df.index
 
    c2r=[
        #'gold', 
        #'copper', 
        #'natural-gas', 'silver', 'crude-oil',
        #'iron-ore', #'aud_usd', 'Target',
        #'coal-australian',
        ]
    for c in c2r:
        df.drop(c, axis=1, inplace=True)
        
    scaler=preprocessing.MinMaxScaler()
    scaled=scaler.fit_transform(df)
    scaled=pd.DataFrame(data=scaled, columns=df.columns, index=df.index)
    
    # scaler = preprocessing.MinMaxScaler()
    # df=pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    ds = df.values
    ds = scaled.values
    
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    dates=df.index[look_forward+look_back-1::]
    
    if roll:
        y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
        y=y_roll
        
    train_size = int(X.shape[0]*train_split)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    
    if plot:
        pl.figure()
        pl.plot([a    for a in y_train]+[None for a in y_test], 'r-', label='Train', lw=0.9)
        pl.plot([None for a in y_train]+[a    for a in y_test], 'b-', label='Test', lw=0.9)
        pl.legend()
        pl.show()
        pl.figure()
        sns.heatmap(df.corr(), annot=True,)
        pl.show()
        
    n_samples, _, n_features = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
        n_features = n_features+look_back
        #X_train=np.c_[X_train, mnth[:train_size]]
        #X_test=np.c_[X_test, mnth[train_size:]]
        n_samples, n_features = X_train.shape
        feature_names=np.array([ str(i)+'_{-'+str(j)+'}' for i in feature_names for j in range(look_back)])
       
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Coal prediction '+' day ahead '+kind,
      'feature_names'   : data_description,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "https://doi.org/10.1016/j.resourpol.2020.101588",      
      'normalize'       : 'MinMax',
      }
    #%%
    return dataset
#-------------------------------------------------------------------------------
def read_data_pollution_sequence(
            filename='./data/data_pollution/pollution.csv',
            look_back=14, look_forward=1, kind='ml', roll=False, plot=False,window=7,
        ):
    #%%
    filename='./data/data_pollution/pollution.csv'
    #look_back=14; look_forward=1; kind='ml'; roll=True; plot=True ;window=7;    
    X = pd.read_csv(filename, sep=',', )
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in X['date']])
    dates['hour']=X['hour']
    X['hour']=X['hour']
    X.index=pd.to_datetime(dates)
    X['weekday'] = pd.to_datetime(X['date']).dt.dayofweek  # monday = 0, sunday = 6
    X['weekend_indi'] = 0          # Initialize the column with default value of 0
    X.loc[X['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun

    c='energy_kWh'
    out_seq=X.groupby('date').agg(sum)[[c]]
    df=X.groupby('date').agg(np.mean)
    df['energy_kWh']=df['Target']=out_seq
    
    df.drop(['hour'], axis=1, inplace=True)
    
    target_names=[c]
    dates = df.index
    ds = df.values
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    dates=df.index[look_forward+look_back-1::]
    
    if roll:
        y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
        y=y_roll
        
        
    # df=X.groupby('date').agg(sum)[[c]]
    # df.columns = [c]
    # out_seq = df[c].values
    # feature_names=df.columns    
    # df['Target']=out_seq
    # target_names=[c]
    # ds = df.values
    # n_steps_in, n_steps_out = look_back, look_forward
    # X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    # y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    
    # if roll:
    #     y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
    #     y=y_roll
    #     #pl.plot([a    for a in  y_roll], 'r-', label='', lw=0.9);    
    #     #pl.plot([a    for a in       y], 'b-', label='', lw=0.1);    
    #     #pl.show()

    train_size = int(X.shape[0]*.7)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    
    if plot:
        pl.figure()
        pl.plot([a    for a in y_train]+[None for a in y_test], 'r-', label='Train', lw=0.9)
        pl.plot([None for a in y_train]+[a    for a in y_test], 'b-', label='Test', lw=0.9)
        pl.legend()
        pl.show()
        
    n_samples, n_features,_ = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
       
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'British Columbia '+' day ahead '+kind,
      'feature_names'   : data_description,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "doi.org/10.1016/j.enbuild.2020.109864",      
      'normalize'       : 'MinMax',
      }
    #%%
    return dataset
#-------------------------------------------------------------------------------
def read_data_iraq_sequence(
            filename='./data/data_iraq_monthly/iraq_monthly_inflow.csv',
            look_back=3, look_forward=1, unit='month', kind='ml'
        ):
    df = pd.read_csv(filename, sep=';', )
    c='montlhly inflow'
    df.columns = [c]
    #idx = pd.date_range('2010-12-03', periods=len(X), freq='-30.41D')
    #df[c]/=df[c].max()   
    out_seq = df[c]
    feature_names=df.columns    
    df['Target']=out_seq 
    target_names=[c]
    ds = df.values
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    
    ts = X.shape[0]-240
    X = X[ts::]
    y = y[ts::]
    
    train_size = int(X.shape[0]*.9)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    
    n_samples, n_features,_ = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
       
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Iraq Inflow '+unit+'s ahead '+kind,
      'feature_names'   : data_description,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "https://bit.ly/33DXpC8",      
      'normalize'       : 'MinMax',
      }
    return dataset     
#%% 
def read_data_ldc_tayfur(filename='./data/data_ldc_vijay/tayfur_2005.csv', case=0):
#%%    
    filename='./data/data_ldc_vijay/tayfur_2005.csv'
    filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='Training')
    df.drop(labels=['Stream'], axis=1, inplace=True)
    col_names = ['$B$', '$H$', '$U$', '$u^*$', '$Q$', '$U/u^*$', '$\\beta$','$\\sigma$', '$K_x$']
    target_names  = ['$K_x$']
    df.columns    = col_names
    
    if   case == 0:
        feature_names = ['$B$', '$H$', '$U$', '$u^*$',   '$Q$', '$U/u^*$', '$\\beta$', '$\\sigma$',]
    elif case == 1:
        feature_names = ['$B$', '$H$', '$U$',                                                      ]
    elif case == 2:
        feature_names = [                                '$Q$',                                    ]
    elif case == 3:
        feature_names = [             '$U$',                                                       ]
    elif case == 4: 
        feature_names = [             '$U$',                               '$\\beta$',             ]
    elif case == 5:
        feature_names = [             '$U$',                               '$\\beta$', '$\\sigma$',]
    elif case == 6:
        feature_names = [                                       '$U/u^*$',                         ]
    elif case == 7:
        feature_names = [                                       '$U/u^*$', '$\\beta$', '$\\sigma$',]
    elif   case == 8:
         feature_names = ['$B$',      '$U$',                                           '$\\sigma$',]
    elif   case == 9:
         feature_names = ['$B$',      '$U$',             '$Q$',                        '$\\sigma$',]
    elif   case == 10:
         feature_names = ['$B$',      '$U$',                               '$\\beta$', '$\\sigma$',]
    elif   case == 11:
         feature_names = ['$B$', '$H$', '$U$', '$u^*$', '$\\sigma$',]
    else:
        sys.exit('Case not found')
        
    X_train = df[feature_names][df.index=='*'].values
    X_test  = df[feature_names][df.index=='**'].values
    y_train = df[df.index=='*'][target_names].values
    y_test  = df[df.index=='**'][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC case '+str(case),
      'feature_names'   : np.array(feature_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
#%%
    return dataset  
#%%
def read_data_ldc_vijay(filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'):
#%%    
    #filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='Training')
    target_names=['Kx (m2 /s)']
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index==1].values
    X_test  = df[feature_names][df.index==0].values
    y_train = df[df.index==1][target_names].values
    y_test  = df[df.index==0][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Longitudinal Dispersion Coefficient',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    return dataset  
#%%
def read_data_ldc_etemad(filename='./data/data_ldc_vijay/etemad_2012.csv'):
#%%    
    #
    filename='./data/data_ldc_vijay/etemad_2012.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='No')
    target_names=['Kx(m2∕s)']
    df.drop(['Stream', 'sigma'], axis=1, inplace=True)
    df['W(m)']/df['H(m)']
    df['U/U*'] =df['U(m∕s)']/df['U*(m∕s)']
    df['Kx/U*H'] =df['Kx(m2∕s)']/df['U*(m∕s)']/df['H(m)']
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index<=119]#.values
    X_test  = df[feature_names][df.index> 119]#.values
    y_train = df[df.index<=119][target_names]#.values
    y_test  = df[df.index> 119][target_names]#.values
    
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC 149',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.T.values,
      'y_test'          : y_test.T.values,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    return dataset
#%%    
def read_data_ldc_noori2017a(filename='./data/data_ldc_vijay/noori2017a.csv'):
#%%    
    #
    filename='./data/data_ldc_vijay/noori2017a.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='Number')
    target_names=['Kx']
    df.drop(['Stream',], axis=1, inplace=True, errors='ignore')
    #df['W(m)']/df['H(m)']
    #df['U/U*'] =df['U(m∕s)']/df['U*(m∕s)']
    #df['Kx/U*H'] =df['Kx(m2∕s)']/df['U*(m∕s)']/df['H(m)']
    feature_names = df.columns.drop(target_names)
    
    n = 70
    X_train = df[feature_names][df.index<=n]#.values
    X_test  = df[feature_names][df.index> n]#.values
    y_train = df[df.index<=n][target_names]#.values
    y_test  = df[df.index> n][target_names]#.values
    
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC Noori 2017a',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.T.values,
      'y_test'          : y_test.T.values,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://tinyurl.com/tz976ts",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
#%%    
    return dataset  
#%%
def read_data_ldc_toprak(filename='./data/data_ldc_vijay/toprak_2004.csv'):
#%%    
    #filename='./data/data_ldc_vijay/toprak_2004.csv'
    df=pd.read_csv(filename, delimiter=';', index_col='Dataset') 
    target_names=[ 'D1(m2/s)']
    df.drop(labels=['No', 'Source', 'Channel',], axis=1, inplace=True)
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index=='C'].values
    X_test  = df[feature_names][df.index=='CC'].values
    y_train = df[df.index=='C'][target_names].values
    y_test  = df[df.index=='CC'][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC Toprak 2008',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "www.doi.org/10.1002/hyp.7012",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset  
#%% 
def read_data_cahora_bassa_sequence(
            filename='./data/data_cahora_bassa/cahora-bassa.csv',
            look_back=21, look_forward=7, unit='day', kind='ml',
            roll=False, window=5, scale=True,
        ):
    #%%
    #look_back=7; look_forward=7; kind='ml'; unit='day'; roll=True; window=7; scale=False
    filename='./data/data_cahora_bassa/cahora-bassa.csv'
    df= pd.read_csv(filename,  delimiter=';')
    df.index = pd.DatetimeIndex(data=df['Data'].values, dayfirst=True)
    df['year']=[a.year for a in df.index]
    df['month']=[a.month for a in df.index]
    df.drop('Data', axis=1, inplace=True) 
    #df['week']=[a.week for a in df.index]
    #
    if unit=='day':
        df['day']=[a.day for a in df.index]
        #df = df.groupby(['day', 'month', 'year']).agg(np.mean)
        #dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':d} for (d,m,y) in df.index.values] )        
        dt=df.index
        df.index=pd.to_datetime(dt, yearfirst=True)
    elif unit=='month':   
        df_std = df.groupby(['month', 'year']).agg(np.std)
        df = df.groupby(['month', 'year']).agg(np.mean)
        dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':15} for (m,y) in df.index.values] )
        df.index=pd.to_datetime(dt, yearfirst=True)
    else:
        sys.exit('Time slot is not defined: day or month')
        
    #>>df['Time']= [a.year+a.dayofyear/366 for a in df.index]   
    df.sort_index(inplace=True)
    #
    idx = df.index < '2013-12-31'
    idx = df.index < '2015-12-31'
    idx = df.index < '2018-12-31'
    df=df[idx]
    
    c='Caudal Afluente (m3/s)'
    df[c]/=1e3
    out_seq=df[c]
    #aux=df.rolling(window=5, min_periods=1, win_type=None).sum()
    #df['Prec. Acum. (mm)']=aux['Precipitacao (mm)']
        
    #df['smooth']=smooth(df[c].values, window_len=10)
    #if unit=='day':
    #    df[c]=smooth(df[c].values, window_len=10)
        
    clstdrp=[]#['Precipitacao (mm)', 'Evaporacao (mm)','Humidade Relativa (%)',]
    if unit=='day':
        cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado (mm3)',  'year', 'month', 'day']
    elif unit=='month':   
        cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado (mm3)', ]
    else:
        sys.exit('Time slot is not defined: day or month')
    
    df.drop(cols_to_drop, axis=1, inplace=True) 
    
    #df.drop(df.columns, axis=1, inplace=True); df[c]=out_seq
    df.columns=['Q', 'R', 'E', 'H', ]

    pl.figure(figsize=(10,12))
    for i,group in enumerate(df.columns):
        pl.subplot(len(df.columns), 1, i+1)
        df[group].plot(marker='o')#pyplot.plot(dataset[group].values)
        pl.title(group, y=0.5, loc='right')
        pl.axvline('2012-06-30', color='k')
    pl.show()

        
    feature_names=df.columns    
    df['Target']=out_seq 
    target_names=[c]
    dates = df.index

    # if unit=='month':
    #     pl.plot(df.index, df[target_names].values)    
    #     pl.fill_between(df.index, 
    #                     (df[target_names].values - df_std[target_names].values).ravel(), 
    #                     (df[target_names].values + df_std[target_names].values).ravel(), 
    #              alpha=0.2, color='k')

    if scale:       
        scaler=preprocessing.MinMaxScaler()
        scaled=scaler.fit_transform(df)
        scaled=pd.DataFrame(data=scaled, columns=df.columns, index=df.index)
    else:
        scaled=df
    
    #ds = df.values
    ds = scaled.values
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    dates=df.index[look_forward+look_back-1::]
    
    if roll:
         y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
         y=y_roll
    
    train_set_date = '2014-12-31' 
    train_set_date = '2012-06-30' 
    train_size, test_size = sum(dates <= train_set_date), sum(dates > train_set_date) 
    X_train, X_test = X[0:train_size], X[train_size:len(dates)]
    y_train, y_test = y[0:train_size], y[train_size:len(dates)]
    #y_std_train, y_std_test = df_std[target].values[0:train_size], df_std[target].values[train_size:len(dates)]
        
    pl.figure(figsize=(16,4)); 
    pl.plot([a for a in y_train]+[None for a in y_test]);
    pl.plot([None for a in y_train]+[a for a in y_test]); 
    pl.show()

    mnth=[a.month for a in dates]
    n_samples, _, n_features = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
        n_features = n_features+look_back
        #X_train=np.c_[X_train, mnth[:train_size]]
        #X_test=np.c_[X_test, mnth[train_size:]]
        n_samples, n_features = X_train.shape
        feature_names=np.array([ str(i)+'_{-'+str(j)+'}' for i in feature_names for j in range(look_back)])
    
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Cahora Bassa '+str(look_back)+' '+unit+'s back '+str(look_forward)+' '+unit+'s ahead',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "Alfeu",      
      'normalize'       : 'MinMax',
      'date_range'      : {'train': dates[0:train_size], 'test': dates[train_size:]},
      }
    #%%
    return dataset
#%%
def read_data_bituminous_4var(fn='./data/data_bituminous/bituminous_4var.csv', 
                                  n_var=4, experiment=None):  
  #%%
  #filename='./data/data_bituminous/arquivo_dados_completos.xls'
  fn='./data/data_bituminous/bituminous_4var.csv'
  n_var=4
  experiment=None
  #aux = pd.read_excel(filename)
  #aux = aux[aux['MISTURA']==3]
  #aux.index = np.round(np.random.uniform(size=len(aux))<0.71)
  
  aux = pd.read_csv(fn, delimiter=';')
  if experiment==None:
      A = aux
  else:
      A = aux[aux['E']== experiment]
      
  if n_var == 4:
    col_inputs	= ['Visc', 't', 'Va', 'T',]
  elif n_var == 6:
    col_inputs	= ['Visc', 't', 'Va', 'VMA', 'VFA',  'T']
  elif n_var == 9:
    col_inputs	= ['Visc', 't', 'Va', 'VMA', 'VFA', 'T', 'AG', 'GAF', 'FAF']
  elif n_var == 10:
    col_inputs	= [ 'MISTURA', 'Visc', 'IST', 't', 'Gmb', 'Gmm', 'Va', 'VMA', 'VFA',  'T',]
  elif n_var == 11:
    col_inputs	= [ 'MISTURA', 'Visc', 'IST', 't', 'Gmb', 'Gmm', 'Va', 'VMA', 'VFA',  'T', 'RTa']
  else:
    print('Please check the number of variables of the problem')
    exit()
      
  #col_output_1 = ['MR1', 'MR2', 'MR3']
  
  #X,y = [],[]
  #for c,df in A.groupby(col_inputs+['Train']):
  #    X.append(c); y.append(df[col_output_1].mean(axis=1).values[0]/1e3)
      
  #X=pd.DataFrame(data=X, columns=col_inputs+['Train'])
  #y=pd.DataFrame(data=y, columns=['MR'])

  train_set = A['Train']==1
  test_set  = A['Train']==0
  X,y = A[col_inputs], A[['MR']]
      
  
  X_train, y_train  = X[train_set][col_inputs],y[train_set]
  X_test , y_test   = X[test_set ][col_inputs],y[test_set] 
  feature_names     = X_train.columns
  target_names      = y_train.columns
  X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
  n_samples, n_features = X_train.shape
  dataset=  {
      'task'            : 'regression',
      'name'            : 'Bituminous Mixes 4 var',
      'feature_names'   : np.array(feature_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "Consultcahoracahoraar Geraldo Marques",      
      'normalize'       : 'MinMax',
      'date_range'      : None
  }
  #%%
  return dataset  
  #%%
#-------------------------------------------------------------------------------
def read_data_tran2019(filename='./data/data_tran2019/tran2019.csv', fold=0):
#%%    
    filename='./data/data_tran2019/tran2019.csv'
    df=pd.read_csv(filename,  delimiter=';', header=1)
    #reference = df['Reference'].unique()[0]
    #df.drop(['Reference'], axis=1, inplace=True)
      
    #target_names=['Power Consumption (kWh)']
    target_names=['Y']
    feature_names = df.columns.drop(target_names)
    
    
    if fold>0:
        X,y = df[feature_names].values, df[target_names].values
        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for k,(train_index, test_index) in enumerate(skf.split(X, y)):
            #print("TRAIN:", train_index, "TEST:", test_index)
            if k==fold:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                #print(k)
    else:        
        X_train = df[feature_names].values
        X_test  = np.array([[],])
        y_train = df[target_names].values
        y_test  = np.array([[],])
    
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Tran 2019'+' fold '+str('n' if fold<0 else fold),
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T if fold>0 else y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://www.sciencedirect.com/science/article/pii/S0360544219322479",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset      
#%%-----------------------------------------------------------------------------
def read_data_cbecs_common(filename='./data/data_cbecs/cbecs_common.csv'):
#%%    
    filename='./data/data_cbecs/cbecs_common.csv'
    df=pd.read_csv(filename,  delimiter=',', header=0)
    target_names=['MFBTU']
   
    df[target_names]=np.log10(df[target_names])
    cols_to_remove=['PUBID']
    for c in cols_to_remove:
        df.drop(c,axis=1, inplace=True)
        
    feature_names = df.columns.drop(target_names)
    X,y = df[feature_names].values, df[target_names].values


    X_train = df[feature_names].values
    X_test  = np.array([[],])
    y_train = df[target_names].values
    y_test  = np.array([[],])
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'CBECS Energy',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://www.sciencedirect.com/science/article/pii/S0306261917313429",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset      
#%%----------------------------------------------------------------------------
def read_data_biswas_2016(filename='./data/data/data_biswas_2016/biswas.csv', plot=False):
#%%    
    filename='./data/data_biswas_2016/biswas.csv'
    df=pd.read_csv(filename,  delimiter=';', header=0)
    target_names=['Electricity (Wh)']
    cols_to_remove=['Electricity (HP)']
    for c in cols_to_remove:
        df.drop(c,axis=1, inplace=True)
    
    feature_names = df.columns.drop(target_names)
    X,y = df[feature_names].values, df[target_names].values


    X_train = df[feature_names].values
    X_test  = np.array([[],])
    y_train = df[target_names].values
    y_test  = np.array([[],])
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Biswas 2016',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://doi.org/10.1016/j.energy.2016.10.066",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset      
#%%----------------------------------------------------------------------------
def read_data_hotspot(filename='./data/data_hotspot/hotspot.csv',day=1, plot=False):
#%%    
    filename='./data/data_hotspot/hotspot.csv'
    df=pd.read_csv(filename,  delimiter=';', header=0)
    
    cols=['Dia', 'Hora da Amostragem', 'Load current (p.u.)', 'Top Oil Temperature (norm.)', 'Hot-spot Temperature (norm.)']
    cols=['Dia', 'Hora da Amostragem', 'LC', 'TOT', 'HST']
    df.columns=cols
    target_names=[cols[4]]
    
    df.index=df['Hora da Amostragem']
    df=df[df['Dia']==day]
    cols_to_remove=['Dia', 'Hora da Amostragem']
    for c in cols_to_remove:
        df.drop(c,axis=1, inplace=True)
    
    if plot:        
         df.plot()
         pl.title('Day '+str(day))
         pl.show()
        
    feature_names = df.columns.drop(target_names)
    X,y = df[feature_names].values, df[target_names].values


    X_train = df[feature_names].values
    X_test  = np.array([[],])
    y_train = df[target_names].values
    y_test  = np.array([[],])
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Hotspot day '+str(day),
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://bit.ly/35XlqWe",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset      
#%%----------------------------------------------------------------------------
def read_data_tahiri(filename='./data/data_tahiri/tahiri.csv', grouped=False):
    #%%
    if grouped:
        filename='./data/data_tahiri/tahiri_grouped.csv'
        df=pd.read_csv(filename,  delimiter=';', header=0)
        train=df['Training']
        cols_to_remove=['Training']
    else:
        filename='./data/data_tahiri/tahiri.csv'
        df=pd.read_csv(filename,  delimiter=';', header=0)
        cols_to_remove=[]
    
    for c in cols_to_remove:
        df.drop(c,axis=1, inplace=True)    
    
    target_names=['Cooling', 'Heating']
    feature_names = df.columns.drop(target_names)
    if grouped:
        X_train = df[train==1][feature_names].values
        X_test  = df[train==0][feature_names].values
        y_train = df[train==1][target_names].values
        y_test  = df[train==0][target_names].values
        n_samples, n_features = X_train.shape
    else:
        X_train = df[feature_names].values
        X_test  = np.array([[],])
        y_train = df[target_names].values
        y_test  = np.array([[],])
        n_samples, n_features = X_train.shape
        
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Tahiri Energy',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T if grouped else y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://doi.org/10.1016/j.csite.2018.03.006",
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset
#------------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [
                 read_data_host_guest(),
               ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
