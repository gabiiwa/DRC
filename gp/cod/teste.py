#%%
from gplearn.genetic import SymbolicClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
import pandas as pd

from read_data import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
#%%


datasets = [
            read_data_cenario('cenario1.csv'),
            # read_data_cenario('cenario2.csv'),
            # read_data_cenario('cenario3.csv'),
            # read_data_cenario('cenario4.csv')
           ]
X = datasets[0]['X_train']
y = datasets[0]['y_train'][0]

#preenchendo valores nulos
X=pd.DataFrame(X).fillna(0)

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                test_size=0.30,
                                random_state=42)
lb = LabelBinarizer()
lb.fit(y_train)
#transformei os dois conjutos de y para duas
# classes, ao invés de 6
y_train =  np.squeeze(lb.transform(y_train))
y_test =  np.squeeze(lb.transform(y_test))
#%%
est = SymbolicClassifier(parsimony_coefficient=.01,
                         random_state=0)
#esse i fazia parte do for. Decidi fazer, considerando 
# uma única coluna, para observar o resultado apenas
i=0
est.fit(X_train,y_train[:,i])

# %%

#o resultado do y_score retorna um shape (12030, 2)
#e o roc_auc_score pede para comparar o y_test[:,i] com um 
#array 1d, por isso fiz y_score[:,1]
y_score = est.predict_proba(X_test)
print("Predict_proba:")
roc_auc_score(y_test[:,i], y_score[:,1],multi_class='ovr')

# %%
