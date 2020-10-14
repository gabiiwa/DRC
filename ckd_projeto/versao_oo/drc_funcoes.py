##########################################################################################
''' Módulo: drc_funcoes.py  '''                                                          #
# Funções para converter dados literais para numéricos e outras correções.               #
##########################################################################################

import numpy as np
from drc_listas import estagio_eq_todos

# Retorna se oito valores são NaN
def isNaN(n1, n2, n3, n4, n5, n6, n7, n8):
    return n1 != n1 and n2 != n2 and n3 != n3 and n4 != n4 and n5 != n5 and n6 != n6 and n7 != n7 and n8 != n8

# Funções para conversão de valores literais para numéricos        
def converte_raca(valor):
    if valor == 'Branca': return 0
    elif valor == 'Preta': return 1
    elif valor == 'Parda': return 2
    elif valor == 'Amarela': return 3
    elif valor == 'Indigena': return 4
    
def converte_insulina(valor):
    if   valor == 'Não': return 0
    elif valor == 'Sim': return 1
    else: return -1
    
def converte_sexo(valor):
    if valor == 'Masculino': return 0
    elif valor == 'Feminino' : return 1
    else: return -1

def converte_classe_imc(valor):
    if   valor == 'Abaixo de 17 (Muito abaixo do peso)': return 1
    elif valor == 'Entre 17 e 18,49 (Abaixo do peso)': return 2
    elif valor == 'Entre 18,5 e 24,99 (Peso normal)': return 3
    elif valor == 'Entre 25 e 29,99 (Acima do peso)': return 4
    elif valor == 'Entre 30 e 34,99 (Obesidade I)': return 5
    elif valor == 'Entre 35 e 39,99 (Obesidade II - severa)': return 6
    elif valor == 'Acima de 40 (Obesidade III - mórbida)': return 7
    else: return -1
    
def converte_sedentario(valor):
    if   valor == 'Não': return 0
    elif valor == 'Sim': return 1
    else: return -1
    
def converte_tabagismo(valor):
    if   valor == 'Ex': return 0
    elif valor == 'Sim': return 1
    else: return -1
    
def converte_etilismo(valor):
    if   valor == 'Ex': return 0
    elif valor == 'Sim': return 1
    else: return -1
    
def converte_retorno(valor):
    if   valor == 'Não precisa retorno': return 1
    else: return -1
    
def converte_instruc(valor):
    if   valor == 'Não sabe ler/escrever': return 1
    elif valor == 'Alfabetizado': return 2
    elif valor == 'Fundamental Incompleto': return 3
    elif valor == 'Fundamental completo': return 4
    elif valor == 'Médio incompleto': return 5
    elif valor == 'Médio completo': return 6
    elif valor == 'Superior incompleto': return 7
    elif valor == 'Superior completo': return 8
    elif valor == 'Mestrado': return 9
    else: return -1

# Converte todos os dados das funções acima em números e retorna o dataframe modificado
def converte_todos_para_numero(df, total_conversoes=9):
    df['Codsexo']    = df['Codsexo'].replace('Masculino', 0)
    df['Codsexo']    = df['Codsexo'].replace('Feminino', 1)
    df['Raça']       = df['Raça'].map(converte_raca)
    if total_conversoes <= 2: return df # Converte apenas Codsexo e Raça
    df['insulina']   = df['insulina'].map(converte_insulina)
    df['classe_imc'] = df['classe_imc'].map(converte_classe_imc)
    df['sedentario'] = df['sedentario'].map(converte_sedentario)
    df['tabagismo']  = df['tabagismo'].map(converte_tabagismo)
    df['etilismo']   = df['etilismo'].map(converte_etilismo)
    df['retorno']    = df['retorno'].map(converte_retorno)
    df['instruc']    = df['instruc'].map(converte_instruc)
    return df

# Converte todos os estágios para números 
def converte_estagio(valor):
    if   valor == 'Estágio 1 - >= 90 ml':  return 1
    elif valor == 'Estágio 2 - 60-89 ml':  return 2
    elif valor == 'Estágio 3a - 45-59 ml': return 3
    elif valor == 'Estágio 3b - 30-44 ml': return 4
    elif valor == 'Estágio 4 - 15-29 ml':  return 5
    elif valor == 'Estágio 5 - < 15 ml':   return 6
    
# Converte até todos os estágios aplicando a função acima e retorna o dataframe modificado
def converte_todos_estagios(df, lista=estagio_eq_todos):
    for estagio in lista: df[estagio] = df[estagio].map(converte_estagio)  
    return df

# 'RendaSM' possui alguns poucos valores como ' '. Assim, são convertidos para NaN
# A seguir toda a coluna é convertida para float e o dataframe modificado é retornado
def converte_renda_sm(df):
    for linha in range(len(df)):
        if df.loc[linha, 'RendaSM'] == ' ': df.loc[linha, 'RendaSM'] = np.nan
    df['RendaSM'] = df['RendaSM'].astype(float)
    return df

# Processo todas as etapas acima para um mesmo banco de dados
def processa_banco(df):
    df = converte_renda_sm(df)
    df = converte_todos_para_numero(df)
    df = converte_todos_estagios(df)
    return df

    