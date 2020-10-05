########################################################################
''' Estatistica '''                                                    #                          
# 'classeEstatistica.py'                                               #                       
########################################################################

import numpy as np
import pandas as pd
# Gráficos
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (10,7)
# Módulo próprio
import drc_listas as listas

class Estatistica:
  
    # 1) Total de pacientes com valores de TFG no quadriênio 2011-2014
    def tfg_por_ano(self, banco, plot=0):
        dados = {'TFG': listas.tfg_orig,
                'Total Pacientes': 
                                  [banco[listas.tfg_orig[0]].count(), 
                                   banco[listas.tfg_orig[1]].count(), 
                                   banco[listas.tfg_orig[2]].count(), 
                                   banco[listas.tfg_orig[3]].count(),
                                   banco[listas.tfg_orig[4]].count(), 
                                   banco[listas.tfg_orig[5]].count(), 
                                   banco[listas.tfg_orig[6]].count(), 
                                   banco[listas.tfg_orig[7]].count()]
                }
        # Dataframe com os valores acima
        df_tfg = pd.DataFrame(dados , columns = ['TFG', 'Total Pacientes'],           
        index=['','','','','','','',''])
        print(df_tfg) 

        # Gráfico
        if plot == 0:
            plt.bar(df_tfg['TFG'], df_tfg['Total Pacientes'], ec="black", alpha=.4, width=.4, color='green')
            plt.title('TOTAL DE VALORES DE TFG POR SEMESTRE NO QUADRIÊNIO 2011-2014')
            plt.xlabel('TOTAL DE VALORES DE TFG')
            plt.ylabel('TOTAL DE PACIENTES')
            plt.tight_layout() 
            plt.show()
        return df_tfg['Total Pacientes'].to_list()

    # 2) Total de pacientes nos estágios inicial e final por categoria
    def distribuicao_estagios(self, banco, plot=0):
        inicial, final = [], []
        for i in range(len(listas.estagio_nomes)):
            inicial.append(len(banco[banco['ESTAGIOI'] == listas.estagio_nomes[i]] == True))
            final.append(len(banco[banco['ESTAGIOF'] == listas.estagio_nomes[i]] == True))
        print("                INICIAL     FINAL")
        for i in range(len(listas.estagio_nomes)):
            print(listas.estagio_nomes[i][:10],'%10d' % inicial[i], '%10d' % final[i])
  
        # Gráfico
        if plot == 0:
            indice = np.arange(len(listas.estagio_nomes))
            bar_larg = 0.4
            transp = 0.6
            plt.bar(indice, inicial, bar_larg, alpha=transp, color='goldenrod', label='Inicial')
            plt.bar(indice + bar_larg, final, bar_larg, alpha=0.6, color='indigo', label='Final')  
            plt.xlabel('ESTÁGIOS') 
            plt.ylabel('TOTAL DE PACIENTES') 
            plt.title('DISTRIBUIÇÃO DOS ESTÁGIOS INICIAL E FINAL DOS PACIENTES') 
            plt.xticks(np.arange(len(listas.estagio_nomes))+bar_larg/2, ('Estágio 1','Estágio 2','Estágio 3A','Estágio 3B','Estágio 4','Estágio 5')) 
            plt.legend() 
            plt.tight_layout() 
            plt.show()
        return inicial, final

    # 3) Total de pacientes por valor de TFG
    def pacientes_por_tfg(self, banco, plot=0):
        total_por_valor = []
        total_nao_nulos = banco[listas.tfg_orig].count(axis=1).to_list()
        for valor in range(1,9): total_por_valor.append(total_nao_nulos.count(valor))
    
        print ("\n  TFG   |  Total Pacientes\n--------------------------")
        for i in range(0,8):  
            print ("  %d     |     %-4d" % (i+1, total_por_valor[i]))

        # Gráfico
        if plot == 0:
            plt.bar([1, 2, 3, 4, 5, 6, 7, 8], total_por_valor, ec = "black", alpha = .8, width = .4, color = 'grey')
            plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], total_por_valor, marker="o", color='red')
            plt.plot([1, 2, 3, 4, 5, 6, 7, 8], total_por_valor, color='blue', linewidth=2)
            plt.title('TOTAL DE VALORES DE TFG POR TOTAL DE PACIENTES')
            plt.xlabel('TOTAL DE VALORES DE TFG')
            plt.ylabel('TOTAL DE PACIENTES')
            plt.tight_layout() 
            plt.show()
        return total_por_valor

    # 4) Total de pacientes com pelo menos um valor de TFG no quadriênio 2011-2014
    def pelo_menos_1_tfg(self, banco):
        return len(banco) - (banco[listas.creatinina].count(axis=1).to_list().count(0)) 

    # 5) Quantidade de pacientes por cada dado, exames e medicamento
    def pacientes_por_dados_exames_medicamentos(self, banco):
        elem_total = []
        for elem in ['Codsexo', 'Raça', 'Idade', 'pesoi', 'pesof'] + listas.exames + listas.medicamentos:
            elem_total.append([elem, banco[elem].count()])
        elem_total.sort(key=lambda x: x[1], reverse = True)

        print('\nQuantidade de pacientes por cada dado, exames e medicamento:')
        print(*elem_total, sep = "\n") 
        return elem_total





        