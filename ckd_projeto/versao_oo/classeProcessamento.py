########################################################################
''' Processamento '''                                                  #                          
# 'classeProcessamento.py'                                             #                       
########################################################################

import pandas as pd
# Módulos próprios
import drc_funcoes as funcoes
import drc_listas as listas

class Processamento:
    
    # 0) Redução do banco para conter somente pacientes com pelo menos um valor de creatinina
    def reduz_banco(self, banco):
        for lin in range(len(banco)):
            if funcoes.isNaN(banco.loc[lin, listas.creatinina[0]], banco.loc[lin, listas.creatinina[1]], 
                             banco.loc[lin, listas.creatinina[2]], banco.loc[lin, listas.creatinina[3]], 
                             banco.loc[lin, listas.creatinina[4]], banco.loc[lin, listas.creatinina[5]], 
                             banco.loc[lin, listas.creatinina[6]], banco.loc[lin, listas.creatinina[7]]): 
                banco.drop(lin, inplace=True)
        banco.reset_index(drop=True, inplace=True)

    # 1) Adiciona ao banco as 8 colunas de TFG calculadas de acordo com a equação MDRD
    def adiciona_tfg(self, banco):
        for num_creat in range(len(listas.creatinina)):
            lista_equacao_tfg = []
            for linha in range(len(banco)):
                # Equação MDRD
                equacao_tfg = 186 * (banco.loc[linha, listas.creatinina[num_creat]])**(-1.154) * (banco.loc[linha, 'Idade'])**(-0.203)
                if banco.loc[linha, 'Codsexo'] == 'Feminino': equacao_tfg*=0.742
                if banco.loc[linha, 'Raça'] == 'Preta': equacao_tfg*=1.212
                lista_equacao_tfg.append(round(equacao_tfg, 2))
            banco.insert(banco.columns.get_loc(listas.tfg_orig[num_creat])+1, listas.tfg_eq[num_creat], pd.DataFrame(lista_equacao_tfg))

    # 2) Adiciona as 8 colunas de estágio calculadas a partir da TFG do banco e a partir da TFG da equação MDRD
    def adiciona_estagio(self, banco):
        for i in range(8):
            estagio_banco, estagio_eq = [], []
            
            for linha in range(len(banco)):
                valor_banco = banco.loc[linha, listas.tfg_orig[i]] # Valor da vez da TFG do banco 
                if   valor_banco >= 90:                      estagio_banco.append(listas.estagio_nomes[0])
                elif valor_banco >= 60 and valor_banco < 90: estagio_banco.append(listas.estagio_nomes[1])
                elif valor_banco >= 45 and valor_banco < 60: estagio_banco.append(listas.estagio_nomes[2])
                elif valor_banco >= 30 and valor_banco < 45: estagio_banco.append(listas.estagio_nomes[3])
                elif valor_banco >= 15 and valor_banco < 30: estagio_banco.append(listas.estagio_nomes[4])
                elif valor_banco >= 0  and valor_banco < 15: estagio_banco.append(listas.estagio_nomes[5])
                else: estagio_banco.append(valor_banco)
                
                valor_eq = banco.loc[linha, listas.tfg_eq[i]] # Valor da vez da TFG da equação 
                if   valor_eq >= 90:                   estagio_eq.append(listas.estagio_nomes[0])
                elif valor_eq >= 60 and valor_eq < 90: estagio_eq.append(listas.estagio_nomes[1])
                elif valor_eq >= 45 and valor_eq < 60: estagio_eq.append(listas.estagio_nomes[2])
                elif valor_eq >= 30 and valor_eq < 45: estagio_eq.append(listas.estagio_nomes[3])
                elif valor_eq >= 15 and valor_eq < 30: estagio_eq.append(listas.estagio_nomes[4])
                elif valor_eq >= 0  and valor_eq < 15: estagio_eq.append(listas.estagio_nomes[5])
                else: estagio_eq.append(valor_eq)
        
            # Inserção da coluna 'ESTAGIO_BANCO'
            banco.insert(banco.columns.get_loc(listas.tfg_orig[i])+1, # Índice da coluna
                        listas.estagio_orig[i], # Nome da coluna
                        pd.DataFrame(estagio_banco)) # Valores da coluna
            # Inserção da coluna 'ESTAGIO_EQ'
            banco.insert(banco.columns.get_loc(listas.estagio_orig[i])+1, # Índice da coluna
                        listas.estagio_eq[i], # Nome da coluna
                        pd.DataFrame(estagio_eq)) # Valores da coluna

    # 3) Adiciona os estágios inicial e final de cada paciente
    def adiciona_estagio_inicial_final(self, banco):
        for linha in range(len(banco)):           
            # Adiciona as colunas de estágios inicial e final a partir da TFG da equação
            for coluna in listas.estagio_eq:
                if not pd.isna(banco.loc[linha, coluna]): 
                    banco.loc[linha, 'ESTAGIOI_EQ'] = banco.loc[linha, coluna]
                    break
            for coluna in listas.estagio_eq[7::-1]:
                if not pd.isna(banco.loc[linha, coluna]): 
                    banco.loc[linha, 'ESTAGIOF_EQ'] = banco.loc[linha, coluna]
                    break
    
    # 4) Adiciona os valores inicial e final de creatinina para cada paciente
    def adiciona_creatinina(self, banco):
        for linha in range(len(banco)): 
            for coluna in listas.creatinina:
                if not pd.isna(banco.loc[linha, coluna]): 
                    banco.loc[linha, 'CREATININAI'] = banco.loc[linha, coluna]
                    break
            for coluna in listas.creatinina[7::-1]:
                if not pd.isna(banco.loc[linha, coluna]): 
                    banco.loc[linha, 'CREATININAF'] = banco.loc[linha, coluna]
                    break

    # 5) Reorganiza o banco de 5689 pacientes pelas datas dos exames realizados por todos os pacientes
    def cria_banco_por_data(self, banco):
        lista_datas, col_sem_data, lista_exames = [], [], []
        # Seleciona em listas distintas, respectivamente: as datas do banco, os exames correspondentes e todas as demais colunas
        for coluna in banco:
            if coluna[:4] == "data": 
                lista_datas.append(coluna)
                lista_exames.append(coluna[4:])
            else: 
                col_sem_data.append(coluna)
                    
        # Remoção e inserção de elementos nas listas devido às particularidades dos exames e datas relativos à pressão
        lista_exames.remove("PressaoI")
        lista_exames.remove("PressaoF")
        lista_exames.insert(2, "PAS_inicial")
        lista_exames.insert(3, "PAD_inicial")
        lista_exames.insert(4, "PAS_final")
        lista_exames.insert(5, "PAD_final") 
        lista_datas.remove('dataPressaoI')
        lista_datas.remove('dataPressaoF')
        lista_datas.insert(2, 'dataPAS_inicial')
        lista_datas.insert(3, 'dataPAD_inicial')
        lista_datas.insert(4, 'dataPAS_final')
        lista_datas.insert(5, 'dataPAD_final')
                
        # Criação do dataframe que será preenchido com todas as datas possíveis na primeira coluna: "DATA"
        col_sem_data.insert(0, 'DATA')
        df_data = pd.DataFrame(columns = col_sem_data)

        # Percorre todas as linhas do banco para formar o dataframe em que
        # cada linha possui a data de um exame realizado por cada paciente
        for lin_banco in range(len(banco)):
            
            # Dicionário para armazenar as datas já inseridas no dataframe
            datas_adicionadas = {}
            
            # Percorrendo todas as datas disponíveis no banco
            for col in lista_datas[2:]: # Excluindo 'datainicial' e 'datafinal'
                nova_linha_df = []

                # Seleção de qual é o exame a ser considerado
                if col[8:] == 'inicial': valor_data = banco.loc[lin_banco, 'dataPressaoI']
                elif col[8:] == 'final': valor_data = banco.loc[lin_banco, 'dataPressaoF']
                else: valor_data = banco.loc[lin_banco, col]

                # Exame a ser considerado. O nome é obtido retirando o sufixo "data"
                exame = col[4:]

                # Se a data já estiver sido adicionada
                if valor_data in datas_adicionadas: 
                    df_data.loc[datas_adicionadas[valor_data], exame] = banco.loc[lin_banco, exame]

                # Data ainda não adicionada
                else:
                    if str(valor_data) != 'nan':
                        # Adiciona primeiramente a data para preencher, a seguir, toda a linha 
                        nova_linha_df.append(valor_data)

                        # Percorre todos as colunas do dataframe (excetuando "DATA" e todas as outras datas)
                        for info in col_sem_data[1:]:

                            # Tudo que não é exame pode ser adicionado
                            if info != exame and info not in lista_exames:
                                nova_linha_df.append(banco.loc[lin_banco, info])

                            # Tudo que é exame e não é o exame da data em questão
                            elif info != exame and info in lista_exames:
                                nova_linha_df.append(0)

                            # Se o exame for um dos quatro tipos de pressão
                            elif exame == 'PAS_inicial':   nova_linha_df.append(banco.loc[lin_banco, 'PAS_inicial'])
                            elif exame == 'PAD_inicial':   nova_linha_df.append(banco.loc[lin_banco, 'PAD_inicial'])
                            elif exame == 'PAS_final':     nova_linha_df.append(banco.loc[lin_banco, 'PAS_final'])
                            elif exame == 'PAD_final':     nova_linha_df.append(banco.loc[lin_banco, 'PAD_final'])

                            # Se o exame for o procurado
                            else: nova_linha_df.append(banco.loc[lin_banco, exame])

                        # Inserção da nova data na lista das datas já adicionadas
                        datas_adicionadas[valor_data] = len(df_data)

                        # Adiciona no dataframe uma linha completa com todos os dados 
                        df_data.loc[len(df_data)] = nova_linha_df



