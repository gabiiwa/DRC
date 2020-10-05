##########################################################################################
''' Módulo: drc_analise_sensibilidade.py '''                                             #
# Realiza a análise de sensibilidade utiizando o método de Morris.                       #
##########################################################################################

import SALib as sal
from SALib.analyze import morris

# Prepara os dados de teste e o modelo correspondente para realizar a análise de sensibilidade
def aplica_morris(X, modelo):
     
    # Seleção das colunas
    lista_cenario =  X.columns.to_list()
    maxmin_cenario = []
    # Obtenção dos pares de valores mínimo e máximo
    for i in lista_cenario: maxmin_cenario.append([X[i].min(), X[i].max()])
    
    # Análise de sensibilidade com o método de Morris
    problem = {'num_vars': len(lista_cenario), 'names': lista_cenario, 'bounds': maxmin_cenario}
    # Generate sample
    X_morris = sal.sample.morris.sample(problem, 1000) 
    # Predict
    Y_morris = modelo.predict(X_morris)
    # Perform analysis
    Si_morris = morris.analyze(problem, X_morris, Y_morris.astype(float), conf_level=0.95, print_to_console=False)
    return Si_morris

# Conversão do resultado, obtido com a aplicação do método, para um dataframe. Ordenação dos valores das colunas 
# decrescentemente em função da coluna 'mu_star'. Por fim, retorna o dataframe modificado
def converte_e_ordena(df):
    df = df.to_df()
    for col in df: df[col] = round(df[col], 2)
    # Ordena o dataframe decrescentemente pela coluna 'mu_star'
    df.sort_values(by=['mu_star'], inplace=True, ascending=False)
    return df 