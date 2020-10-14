########################################################################
''' Treinamento e teste '''                                            #                          
# 'classeTreinoTeste.py'                                               #                       
########################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
# Classe
from classeAnaliseSensibilidade import AnaliseSensibilidade
# Módulos próprios
import drc_funcoes as funcoes
import drc_treino_teste as tt 

class TreinoTeste:

    def __init__(self, banco):
        self.banco = banco
        self.analisador = AnaliseSensibilidade()
        self.__y = 0
        self.__X = 0 
        self.__ytest = 0
        self.__ypred = 0
        self.__modelo = None
        self.__modelo_score = []
        

    # Converte valores literais para numéricos
    def conversao(self):
        self.banco = funcoes.converte_todos(self.banco)
        self.banco = funcoes.converte_todos_estagios(self.banco)
        self.banco = funcoes.converte_renda_sm(self.banco)

    # Realiza o treino e o teste para um ou vários cenários
    def treino_teste(self, var_alvo, col_X, fillna_tipo=0, size_test=0.3, varios=0):
        self.__y = self.banco[var_alvo] # Variável alvo
        if (varios == 0):
            self.__X = self.banco[col_X]
            self.__X = self.__X.fillna(fillna_tipo) 
            # Retorno de X, teste e retorno do modelo e da acurácia
            self.__modelo, score, self.__ytest, self.__ypred = tt.treino_teste(self.__X, self.__y, size_test)
            return score

        for conjuntos in col_X:
            self.__X = self.banco[conjuntos]
            self.__X = self.__X.fillna(fillna_tipo)
            self.__modelo, score, self.__ytest, self.__ypred = tt.treino_teste(self.__X, self.__y, size_test)
            self.__modelo_score.append([self.__X, self.__modelo, score])
    
    # Aplica as métricas de erro MSE e RMSE
    def metricas_erro(self):
        mse = mean_squared_error(self.__ytest, self.__ypred)
        return mse, np.sqrt(mse)

    # Executa o método de Morris para a análise de sensibilidade
    def aplica_analise_sensibilidade(self):
        self.analisador.aplica_morris(self.__X, self.__modelo)
        return self.analisador.converte_e_ordena()
