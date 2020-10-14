##########################################################################################
''' Módulo: drc_cenarios.py '''                                                          #
# Listas com os dados para diferente cenários de treinamento e teste                     #
##########################################################################################

from drc_listas import consultas, dados_exames_25

cenario1 = ['Codsexo', 'Idade', 'Raça', 'CREATININAI']
cenario2 = dados_exames_25
cenario3 = consultas + ['CREATININAI']
cenario4 = dados_exames_25 + consultas + ['CREATININAI']
