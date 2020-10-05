########################################################################
''' Funções de teste com Pytest '''                                    #                          
# 'conj_testes.py'                                                     #                       
########################################################################

import pytest 
from classePipeline import Pipeline
from classeDados import BancoCSV
import drc_cenarios as cen
import drc_listas as listas

########################################################################
''' DADOS PARA OS TESTES '''
########################################################################
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
''' FIXTURES PARA OS TESTES '''
########################################################################
@pytest.fixture
def pipeline():
    '''Retorna uma instância da classe Pipeline'''
    return Pipeline(file_5689, sep, encoding)

@pytest.fixture
def banco_orig():
    '''Retorna uma instância da classe BancoCSV com 7266   pacientes'''
    banco = BancoCSV(file_orig, sep_orig, encoding_orig)
    return banco.ler_banco(1)

@pytest.fixture
def banco_5689():
    '''Retorna uma instância da classe BancoCSV com 5689 pacientes'''
    banco = BancoCSV(file_5689, sep, encoding)
    return banco.ler_banco()

@pytest.fixture
def banco_data():
    '''Retorna uma instância da classe BancoCSV do banco organizado por data'''
    banco = BancoCSV(file_data, sep, encoding)
    return banco.ler_banco()

########################################################################
''' 1) TESTES COM AS DIMENSÕES DOS 3 BANCOS '''
########################################################################
#@pytest.mark.skip
def test_bancos_orig(banco_orig, banco_5689, banco_data):
    assert banco_orig.shape == (7266, 255)
    assert banco_5689.shape == (5689, 283)
    assert banco_data.shape == (40100, 196)

########################################################################
''' 2) TESTES COM AS ESTATÍSTICAS '''
########################################################################
#@pytest.mark.skip    
def test_estatistica_banco_5689(pipeline, banco_5689):
    ''' Banco com 5689 pacientes '''
    pipeline.dados = banco_5689
    assert pipeline.est.tfg_por_ano(pipeline.dados, 1) == [1283, 1604, 1694, 1724, 1961, 2093, 2310, 2038]
    assert pipeline.est.distribuicao_estagios(pipeline.dados, 1)[0] == [759, 1838, 1469, 1013, 496, 114]
    assert pipeline.est.distribuicao_estagios(pipeline.dados, 1)[1] == [983, 1959, 1319, 814, 439, 175]
    assert pipeline.est.pacientes_por_tfg(pipeline.dados, 1) == [2214, 1280, 710, 536, 395, 272, 204, 78]
    assert pipeline.est.pacientes_por_dados_exames_medicamentos(pipeline.dados) == listas.dado_exame_medicamento
    assert pipeline.est.pelo_menos_1_tfg(pipeline.dados) == len(banco_5689)

#@pytest.mark.skip   
def test_estatistica_banco_data(banco_data):
    ''' Banco por data '''
    pipeline = Pipeline(file_data, sep, encoding)
    pipeline.dados = banco_data
    assert pipeline.est.tfg_por_ano(pipeline.dados, 1) == [11499, 14757, 16557, 17188, 19319, 19797, 20778, 17955]
    assert pipeline.est.distribuicao_estagios(pipeline.dados, 1)[0]  == [4364, 11593, 10303, 8244, 4629, 967]
    assert pipeline.est.distribuicao_estagios(pipeline.dados, 1)[1] == [6151, 12687, 8964, 6628, 3977, 1693]
    assert pipeline.est.pacientes_por_tfg(pipeline.dados, 1) == [8731, 7963, 5825, 5319, 4496, 3614, 2938, 1214]
    assert pipeline.est.pelo_menos_1_tfg(pipeline.dados) == len(banco_data)

########################################################################
''' 3) TESTES COM O PROCESSAMENTO DO BANCO ORIGINAL '''
########################################################################
#@pytest.mark.skip   
def test_processamento(pipeline, banco_orig):
    # Função 0)
    pipeline.proc.reduz_banco(banco_orig)
    assert banco_orig.shape == (5689, 255) 

    # Função 1)
    pipeline.proc.adiciona_tfg(banco_orig)
    for tfg_eq in listas.tfg_eq: assert tfg_eq in banco_orig.columns
    assert banco_orig.shape == (5689, 263) 

    # Função 2) 
    pipeline.proc.adiciona_estagio(banco_orig)  
    for estagio_eq in listas.estagio_eq: assert estagio_eq in banco_orig.columns
    assert banco_orig.shape == (5689, 279) 

    # Função 3) 
    pipeline.proc.adiciona_estagio_inicial_final(banco_orig)
    assert banco_orig.columns[-1] == 'ESTAGIOF_EQ'
    assert banco_orig.columns[-2] =='ESTAGIOI_EQ'
    assert banco_orig.shape == (5689, 281) 

    # Função 4) 
    pipeline.proc.adiciona_creatinina(banco_orig)
    assert banco_orig.columns[-1] == 'CREATININAF'
    assert banco_orig.columns[-2] =='CREATININAI'
    assert banco_orig.shape == (5689, 283) 

########################################################################
''' 4) TESTES COM O TREINAMENTO E TESTE '''
########################################################################
#@pytest.mark.skip 
''' Banco com 5689 pacientes '''
@pytest.mark.parametrize("cenario, val_inf, val_sup", 
                        [# Cenários de 1 a 11
                        (cen.cenario1,  0.83, 0.91),
                        (cen.cenario2,  0.47, 0.55),
                        (cen.cenario3,  0.85, 0.93),
                        (cen.cenario4,  0.46, 0.54),
                        (cen.cenario5,  0.86, 0.95),
                        (cen.cenario6,  0.51, 0.59),
                        (cen.cenario7,  0.33, 0.41),
                        (cen.cenario8,  0.39, 0.47),
                        (cen.cenario9,  0.38, 0.46),
                        (cen.cenario10, 0.36, 0.46),
                        (cen.cenario11, 0.47, 0.56)
                        ])
def test_treino_teste_banco_5689(pipeline, cenario, val_inf, val_sup):
    score = pipeline.analises_finais('ESTAGIOF_EQ', cenario)[0]
    assert score >= val_inf and score <= val_sup

#@pytest.mark.skip 
''' Banco por data '''
@pytest.mark.parametrize("cenario, vinf_score, vsup_score, vinf_mse, vsup_mse, vinf_rmse, vsup_rmse", 
                        [# Cenários de 1 a 11
                        (cen.cenario1,  0.99, 1.00, 0.01, 0.03, 0.13, 0.17),
                        (cen.cenario2,  0.98, 1.00, 0.24, 0.28, 0.49, 0.53),
                        (cen.cenario3,  0.99, 1.00, 0.01, 0.03, 0.11, 0.15),
                        (cen.cenario4,  0.92, 0.99, 0.28, 0.32, 0.53, 0.57),
                        (cen.cenario5,  0.99, 1.00, 0.00, 0.03, 0.07, 0.15),
                        (cen.cenario6,  0.80, 0.87, 0.37, 0.41, 0.61, 0.65),
                        (cen.cenario7,  0.40, 0.44, 2.08, 2.14, 1.42, 1.48),
                        (cen.cenario8,  0.59, 0.63, 1.53, 1.60, 1.22, 1.28),
                        (cen.cenario9,  0.66, 0.70, 1.40, 1.46, 1.17, 1.23),
                        (cen.cenario10, 0.69, 0.73, 1.20, 1.30, 1.09, 1.15),
                        (cen.cenario11, 0.82, 0.86, 0.51, 0.55, 0.71, 0.75)
                        ])
def test_treino_teste_banco_data(pipeline, banco_data, cenario, vinf_score, vsup_score, vinf_mse, vsup_mse, vinf_rmse, vsup_rmse):
    pipeline.banco = banco_data
    score, mse, rmse = pipeline.analises_finais('ESTAGIOF_EQ', cenario)
    assert score   >= vinf_score   and    score   <= vsup_score
    assert mse     >= vinf_mse     and    mse     <= vsup_mse
    assert rmse    >= vinf_rmse    and    rmse    <= vsup_rmse

########################################################################
''' 5) COMPARAÇÃO ENTRE OS BANCOS 5689 PACIENTES E O POR DATA '''
########################################################################
#@pytest.mark.skip  
def test_compara_bancos(pipeline, banco_5689, banco_data):
    # Verifica se todas as datas
    nao_encontrado = 0
    for i in range(len(banco_5689)):
        banco_id = banco_data[banco_data['Id'] == banco_5689.loc[i, 'Id']]
        banco_id.reset_index(drop=True, inplace=True)
        for j in range(len(banco_id)):
            if banco_id.loc[j, 'DATA'] not in banco_5689.loc[i,:].to_list():
                nao_encontrado = 1

    assert nao_encontrado == 0