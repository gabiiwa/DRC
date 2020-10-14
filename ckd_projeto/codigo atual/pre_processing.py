import pandas as pd
import numpy as np

from drc_cenarios import *
from drc_funcoes import processa_banco
from drc_leitura_banco import banco_data
from drc_treino_teste import prepara_X

bd = processa_banco(banco_data)

y = bd['ESTAGIOF_EQ'] 

X1 = bd.filter(items = cenario1)   
X2 = bd.filter(items = cenario2)  
X3 = bd.filter(items = cenario3)  
X4 = bd.filter(items = cenario4) 

todos_X = [X1, X2, X3, X4]
lista_X = []

for X in todos_X:
    X0, Xmedia, Xmediana, Xnn, Xmice = prepara_X(X)
    lista_X.append(X, X0, Xmedia, Xmediana, Xnn, Xmice)


