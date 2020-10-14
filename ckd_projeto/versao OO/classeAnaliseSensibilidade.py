########################################################################
''' Análise de Sensibilidade '''                                       #                          
# 'classeAnaliseSensibilidade.py'                                      #                       
########################################################################

import drc_analise_sensibilidade as an_sens

class AnaliseSensibilidade:

    def __init__(self):
        self.df_analise = 0

    # Aplica o método de Morris para o conjunto de dados X e o modelo correspodente 
    def aplica_morris(self, X, modelo):
        self.df_analise = an_sens.aplica_morris(X, modelo)
        return self.df_analise

    # Conversão do resultado acima para dataframe e ordenação dos valores
    def converte_e_ordena(self):
        return an_sens.converte_e_ordena(self.df_analise)
        

        