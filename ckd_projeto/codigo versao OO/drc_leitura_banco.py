######################################################################################################
''' Módulo: drc_leitura_banco.py '''                                                                 #
# Carrega três versões diferentes do banco de dados: Original, 5689 pacientes e Organizado Por data. #
######################################################################################################

import pandas as pd

# Banco original: 7266 pacientes
file_name = 'C:/Users/jpsco/Documents/Doutorado/PGMC/BD2020/Dados/Dados2010-2014DesfechoFasesETC.csv' 
banco_original = pd.read_csv(file_name, low_memory = False) 

# Banco original reduzido para pacientes como pelo menos um valor de creatinina: 5689 pacientes
file_name = 'C:/Users/jpsco/Documents/Doutorado/PGMC/BD2020/Dados/resultados/banco_5689.csv' 
banco = pd.read_csv(file_name, sep = ';', encoding = "ISO-8859-1", low_memory = False).drop(['Unnamed: 0'],axis=1)

# Banco com 5689 pacientes reorganizado por cada data de exame realizado por um paciente
file_name = 'C:/Users/jpsco/Documents/Professor/Doutorado/PGMC/BD2020/DRC/projeto/banco de dados/banco_por_data.csv'
banco_data = pd.read_csv(file_name, sep = ';', encoding = "ISO-8859-1", low_memory = False).drop(['Unnamed: 0'], axis=1)





