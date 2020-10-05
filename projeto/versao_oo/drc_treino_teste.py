##########################################################################################
''' Módulo: drc_treino_teste.py '''                                                      #
# Realiza o teste e o treinamento de um conjunto de dados.                               #
##########################################################################################

from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees Classifier: 70% treinamento e 30% teste
def treino_teste(X, y, size_test = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state = 0)
    modelo = ExtraTreesClassifier()
    modelo.fit(X_train, y_train)
    y_pred = cross_val_predict(modelo, X_test, y_test, cv=5)
    return modelo, modelo.score(X_test, y_test), y_test, y_pred


    