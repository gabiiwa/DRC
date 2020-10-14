##########################################################################################
''' Módulo: drc_treino_teste.py '''                                                      #
# Realiza o teste e o treinamento de um conjunto de dados.                               #
##########################################################################################

import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from fancyimpute import IterativeImputer as MICE

# Extra Trees Classifier: 70% treinamento e 30% teste
def treino_teste(X, y, size_test = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state = 0)
    modelo = ExtraTreesClassifier()
    modelo.fit(X_train, y_train)
    y_pred = cross_val_predict(modelo, X_test, y_test, cv=5)
    return modelo, modelo.score(X_test, y_test), y_test, y_pred

# Prepara um conjunto de dados X para 5 cenários diferentes de inputação de dados
def prepara_X(X):
    X_0 = X.fillna(0) 
    X_media = X.fillna(X.mean())
    X_mediana = X.fillna(X.median()) 
    Xnn = KNNImputer(n_neighbors = int(np.sqrt(X.shape[0]*X.shape[1])) , weights="uniform").fit_transform(X) 
    X_mice = MICE().fit_transform(X) 
    return X_0, X_media, X_mediana, Xnn, X_mice

# Retorna a média da acurácia de uma classificação com um dado número de iterações
def simulacao(X, y, iteracoes):
    soma_score = 0
    for _ in range(iteracoes): soma_score += treino_teste(X, y)[1]
    return soma_score/iteracoes
    