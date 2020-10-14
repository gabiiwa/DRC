########################################################################
''' Arquivo principal '''                                              #                          
# 'main.py'                                                            #                       
########################################################################
# Classe
from classePipeline import Pipeline
# Testes
import pytest
# Módulo próprio
import drc_cenarios as cen

########################################################################
''' DADOS PARA LER DIFERENTES BANCOS  '''
########################################################################
# Diretório dos dados de leitura
diretorio = 'C:/Users/jpsco/Documents/Professor/Doutorado/PGMC/BD2020/Dados/'

# Banco 7266 pacientes  
file_orig = diretorio+'testeES_orig.csv' 
sep_orig = ','
encoding_orig = "utf8"

# Banco 5689 pacientes
file_5689 = diretorio+'testeES_ler.csv' 
# Banco por Data 
file_data = diretorio+'testeES_data.csv' 
# Separador e codificação dos dados de ambos os bancos acima
sep = ';'
encoding = "ISO-8859-1"

########################################################################
''' DADOS PARA SALVAR O RESULTADO '''
########################################################################
# Salvar banco
file_salvar = diretorio+'testesES_salvar.csv'
index_salvar = 'False'

########################################################################
''' EXECUÇÃO PRINCIPAL '''
########################################################################

# Pipeline de execução de várias etapas com o banco considerado
p = Pipeline(file_orig, sep_orig, encoding_orig, 1)

# Treinamento e teste, métricas de erro e análise de sensibilidade
p.analises_finais('ESTAGIOF_EQ', # y 
                  cen.cenario8)  # X

# Salva o banco em um arquivo CSV 
p.dados.salvar_banco(file_salvar, index_salvar, sep, encoding)

# Conjunto de testes 
pytest.main(['conj_testes.py', '-v'])


