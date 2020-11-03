#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from fancyimpute import IterativeImputer as MICE


# In[30]:


file_name = 'C:/Users/jpsco/Documents/Professor/Doutorado/PGMC/BD2020/DRC/projeto/banco de dados/banco_por_data.csv'
banco_data = pd.read_csv(file_name, sep = ';', encoding = "ISO-8859-1", low_memory = False).drop(['Unnamed: 0'], axis=1)


# ## Listas, funções e preparação do banco de dados

# In[31]:


creatinina = ['Creatinina_1_2011', 'Creatinina_2_2011', 'Creatinina_1_2012', 'Creatinina_2_2012', 
              'Creatinina_1_2013', 'Creatinina_2_2013', 'Creatinina_1_2014', 'Creatinina_2_2014']

creatinina_todas = creatinina + ['CREATININAI', 'CREATININAF']

lista_estagio_eq = ['ESTAGIO_EQ_1_2011', 'ESTAGIO_EQ_2_2011', 'ESTAGIO_EQ_1_2012', 'ESTAGIO_EQ_2_2012',
                    'ESTAGIO_EQ_1_2013', 'ESTAGIO_EQ_2_2013', 'ESTAGIO_EQ_1_2014', 'ESTAGIO_EQ_2_2014']

lista_estagio_todos = lista_estagio_eq + ['ESTAGIOI_EQ', 'ESTAGIOF_EQ']

consultas = ['DRC_1_2011',  'DRC_2_2011',  'DRC_1_2012',  'DRC_2_2012',  
             'DRC_1_2013',  'DRC_2_2013',  'DRC_1_2014',  'DRC_2_2014',  'HAS_1_2011',  'HAS_2_2011',  'HAS_1_2012',  
             'HAS_2_2012',  'HAS_1_2013',  'HAS_2_2013',  'HAS_1_2014',  'HAS_2_2014',  'DM_1_2011',  'DM_2_2011',  
             'DM_1_2012',  'DM_2_2012',  'DM_1_2013',  'DM_2_2013',  'DM_1_2014',  'DM_2_2014']

exames = ['PAS_inicial',  'PAD_inicial',  'PAS_final',  'PAD_final', 'TSHI',  'TSHF',  'HemoglobinaI',  'HemoglobinaF',  
          'AcidoUricoI',  'AcidoUricoF',  'CalcioTotalI',  'CalcioTotalF',  'AcidoFolicoI',  'AcidoFolicoF',  
          'SodioUrinarioI',  'SodioUrinarioF',  'VitaminaB12I',  'VitaminaB12F',  'ColesterolHDLI',  'ColesterolHDLF',  
          'ColesterolLDLI',  'ColesterolLDLF',  'ColesterolTotalI',  'ColesterolTotalF',  'GamaGlutamilI',  'GamaGlutamilF',
          'HemoglobinaGlicadaI',  'HemoglobinaGlicadaF',  'TGPI',  'TGPF',  'TrigliceridesI',  'TrigliceridesF', 
          'BilirrubinatotalI',  'BilirrubinatotalF',  'PotassioI',  'PotassioF',  'GlicemiadeJejumI',  'GlicemiadeJejumF',
          'Ureia24hsI',  'Ureia24hsF',  'FerritinaI',  'FerritinaF',  'IndicedeSaturacaodaTransferenciaI',  
          'IndicedeSaturacaodaTransferenciaF',  'FerroSericoI',  'FerroSericoF',  'FosforoI',  'FosforoF',  
          'PTHintactoI',  'PTHintactoF',  'VITAMINADI',  'VITAMINADF',  'AlbuminaI',  'AlbuminaF',  'HBsAGI',  
          'HBsAGF',  'AntiHBsI',  'AntiHBsF',  'AntiHCVI',  'AntiHCVF',  'Rel.AlbuminaCreatininaUAUCI',  
          'Rel.AlbuminaCreatininaUAUCF',  'Proteinuria24hsI',  'Proteinuria24hsF',  'ECOAEI',  'ECOAEF',  'ECOAOI',  
          'ECOAOF',  'ECOSIVI',  'ECOSIVF',  'ECOPPI',  'ECOPPF',  'ECOFEI',  'ECOFEF',  'MicroalbuminuriaI',  
          'MicroalbuminuriaF',  'FosfataseAlcalinaI',  'FosfataseAlcalinaF',  'HematuriaI',  'HematuriaF',  
          'SodioSericoI',  'SodioSericoF',  'CKI',  'CKF',  'UreiaI',  'UreiaF']

# 35 dados e exames mais frequentes (excetuando as creatininas)
exames_35 = ['Codsexo', 'Idade','Raça','PAS_inicial','PAS_final','PAD_inicial','PAD_final','pesoi','pesof','HemoglobinaI',
             'ColesterolTotalI','GlicemiadeJejumI','TrigliceridesI','PotassioI','ColesterolHDLI', 'UreiaI','TSHI',
             'AcidoUricoI', 'HemoglobinaGlicadaI','TGPI','GlicemiadeJejumF','ColesterolTotalF','TrigliceridesF',
             'ColesterolHDLF','HemoglobinaF','SodioSericoI','PotassioF','CKI','CalcioTotalI','VITAMINADI', 
             'HemoglobinaGlicadaF', 'ColesterolLDLI', 'UreiaF', 'Proteinuria24hsI', 'TGPF'] 

# 25 dados e exames mais frequentes (excetuando as creatininas)
exames_25 = exames_35[:25]

# Cenários
c1 = ['Idade',  'Codsexo',  'Raça',  'pesoi',  'pesof',  'Alt',  'instruc',  'RendaSM',  'TamFamilia', 'RendaFamiliarSM',  'sedentario',  'etilismo',  'tabagismo',  'consultasDM',  'consultasDRC',  'consultasHAS', 'IECA1',  'BRAT2',  'BETABLOQ3',  'estatina9',  'AAS8',  'DIUR4',  'BIGUADINA6',  'SULFONIURA7',  'FIBRATO13', 'insulina',  'imc',  'classe_imc',  'tempoAcompanha',  'PAS_inicial',  'PAD_inicial',  'PAS_final', 'PAD_final',  'CreatininaI',  'CreatininaF',  'TSHI',  'TSHF',  'HemoglobinaI',  'HemoglobinaF', 'AcidoUricoI',  'AcidoUricoF',  'CalcioTotalI',  'CalcioTotalF',  'AcidoFolicoI',  'AcidoFolicoF', 'SodioUrinarioI',  'SodioUrinarioF',  'VitaminaB12I',  'VitaminaB12F',  'ColesterolHDLI',  'ColesterolHDLF', 'ColesterolLDLI',  'ColesterolLDLF',  'ColesterolTotalI',  'ColesterolTotalF',  'GamaGlutamilI', 'GamaGlutamilF',  'HemoglobinaGlicadaI',  'HemoglobinaGlicadaF',  'TGPI',  'TGPF',  'TrigliceridesI', 'TrigliceridesF',  'BilirrubinatotalI',  'BilirrubinatotalF',  'PotassioI',  'PotassioF',  'GlicemiadeJejumI', 'GlicemiadeJejumF',  'Ureia24hsI',  'Ureia24hsF',  'FerritinaI',  'FerritinaF', 'IndicedeSaturacaodaTransferenciaI',  'IndicedeSaturacaodaTransferenciaF',  'FerroSericoI',  'FerroSericoF', 'FosforoI',  'FosforoF',  'PTHintactoI',  'PTHintactoF',  'VITAMINADI',  'VITAMINADF',  'AlbuminaI', 'AlbuminaF',  'HBsAGI',  'HBsAGF',  'AntiHBsI',  'AntiHBsF',  'AntiHCVI',  'AntiHCVF', 'Rel.AlbuminaCreatininaUAUCI',  'Rel.AlbuminaCreatininaUAUCF',  'Proteinuria24hsI',  'Proteinuria24hsF', 'ECOAEI',  'ECOAEF',  'ECOAOI',  'ECOAOF',  'ECOSIVI',  'ECOSIVF',  'ECOPPI',  'ECOPPF',  'ECOFEI', 'ECOFEF',  'MicroalbuminuriaI',  'MicroalbuminuriaF',  'FosfataseAlcalinaI',  'FosfataseAlcalinaF', 'HematuriaI',  'HematuriaF',  'SodioSericoI',  'SodioSericoF',  'CKI',  'CKF',  'UreiaI',  'UreiaF', 'DRC_1_2011',  'DRC_2_2011',  'DRC_1_2012',  'DRC_2_2012',  'DRC_1_2013',  'DRC_2_2013',  'DRC_1_2014', 'DRC_2_2014',  'HAS_1_2011',  'HAS_2_2011',  'HAS_1_2012',  'HAS_2_2012',  'HAS_1_2013',  'HAS_2_2013', 'HAS_1_2014',  'HAS_2_2014',  'DM_1_2011',  'DM_2_2011',  'DM_1_2012',  'DM_2_2012',  'DM_1_2013', 'DM_2_2013',  'DM_1_2014',  'DM_2_2014',  'Creatinina_1_2011',  'Creatinina_2_2011',  'Creatinina_1_2012', 'Creatinina_2_2012',  'Creatinina_1_2013',  'Creatinina_2_2013',  'Creatinina_1_2014',  'Creatinina_2_2014', 'TFG_1_2011_EQ',  'ESTAGIO_EQ_1_2011',  'TFG_2_2011_EQ',  'ESTAGIO_EQ_2_2011',  'TFG_1_2012_EQ', 'ESTAGIO_EQ_1_2012',  'TFG_2_2012_EQ',  'ESTAGIO_EQ_2_2012',  'TFG_1_2013_EQ',  'ESTAGIO_EQ_1_2013', 'TFG_2_2013_EQ',  'ESTAGIO_EQ_2_2013',  'TFG_1_2014_EQ',  'ESTAGIO_EQ_1_2014',  'TFG_2_2014_EQ', 'ESTAGIO_EQ_2_2014',  'ESTAGIOI_EQ',  'CREATININAI' ] 
c2 = ['Idade',  'Codsexo',  'Raça',  'pesoi',  'pesof',  'Alt',  'instruc',  'RendaSM',  'TamFamilia', 'RendaFamiliarSM',  'sedentario',  'etilismo',  'tabagismo',  'consultasDM',  'consultasDRC',  'consultasHAS', 'IECA1',  'BRAT2',  'BETABLOQ3',  'estatina9',  'AAS8',  'DIUR4',  'BIGUADINA6',  'SULFONIURA7',  'FIBRATO13', 'insulina',  'imc',  'classe_imc',  'tempoAcompanha',  'PAS_inicial',  'PAD_inicial',  'PAS_final', 'PAD_final',  'CreatininaI',  'CreatininaF',  'TSHI',  'TSHF',  'HemoglobinaI',  'HemoglobinaF', 'AcidoUricoI',  'AcidoUricoF',  'CalcioTotalI',  'CalcioTotalF',  'AcidoFolicoI',  'AcidoFolicoF', 'SodioUrinarioI',  'SodioUrinarioF',  'VitaminaB12I',  'VitaminaB12F',  'ColesterolHDLI',  'ColesterolHDLF', 'ColesterolLDLI',  'ColesterolLDLF',  'ColesterolTotalI',  'ColesterolTotalF',  'GamaGlutamilI', 'GamaGlutamilF',  'HemoglobinaGlicadaI',  'HemoglobinaGlicadaF',  'TGPI',  'TGPF',  'TrigliceridesI', 'TrigliceridesF',  'BilirrubinatotalI',  'BilirrubinatotalF',  'PotassioI',  'PotassioF',  'GlicemiadeJejumI', 'GlicemiadeJejumF',  'Ureia24hsI',  'Ureia24hsF',  'FerritinaI',  'FerritinaF', 'IndicedeSaturacaodaTransferenciaI',  'IndicedeSaturacaodaTransferenciaF',  'FerroSericoI',  'FerroSericoF', 'FosforoI',  'FosforoF',  'PTHintactoI',  'PTHintactoF',  'VITAMINADI',  'VITAMINADF',  'AlbuminaI', 'AlbuminaF',  'HBsAGI',  'HBsAGF',  'AntiHBsI',  'AntiHBsF',  'AntiHCVI',  'AntiHCVF', 'Rel.AlbuminaCreatininaUAUCI',  'Rel.AlbuminaCreatininaUAUCF',  'Proteinuria24hsI',  'Proteinuria24hsF', 'ECOAEI',  'ECOAEF',  'ECOAOI',  'ECOAOF',  'ECOSIVI',  'ECOSIVF',  'ECOPPI',  'ECOPPF',  'ECOFEI', 'ECOFEF',  'MicroalbuminuriaI',  'MicroalbuminuriaF',  'FosfataseAlcalinaI',  'FosfataseAlcalinaF', 'HematuriaI',  'HematuriaF',  'SodioSericoI',  'SodioSericoF',  'CKI',  'CKF',  'UreiaI',  'UreiaF', 'DRC_1_2011',  'DRC_2_2011',  'DRC_1_2012',  'DRC_2_2012',  'DRC_1_2013',  'DRC_2_2013',  'DRC_1_2014', 'DRC_2_2014',  'HAS_1_2011',  'HAS_2_2011',  'HAS_1_2012',  'HAS_2_2012',  'HAS_1_2013',  'HAS_2_2013', 'HAS_1_2014',  'HAS_2_2014',  'DM_1_2011',  'DM_2_2011',  'DM_1_2012',  'DM_2_2012',  'DM_1_2013', 'DM_2_2013',  'DM_1_2014',  'DM_2_2014']

c3 = ['Codsexo', 'Idade', 'Raça']
c4 = ['Codsexo', 'Idade', 'Raça', 'CREATININAI']
c5 = ['Codsexo', 'Idade', 'Raça'] + creatinina

c6 = exames_35
c7 = exames_35 + ['CREATININAI']
c8 = exames_35 + creatinina

c9  = exames_25
c10 = exames_25 + ['CREATININAI']
c11 = exames_25 + creatinina

c12 = consultas
c13 = consultas + ['CREATININAI']

c14 = exames
c15 = exames + ['CREATININAI']
c16 = exames + consultas
c17 = exames + ['Idade', 'Codsexo', 'Raça', 'pesoi', 'pesof'] 
c18 = exames + ['Idade', 'Codsexo', 'Raça', 'pesoi', 'pesof'] + ['CREATININAI']

c19 = exames_25 + consultas
c20 = exames_25 + consultas + ['CREATININAI']


# In[32]:


# Funções para conversão de valores categóricos para numéricos        
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

# Converte todos os dados das funções acima em números
def converte_todos_para_numero(df, total_conversoes=9):
    df['Codsexo'] = df['Codsexo'].replace('Masculino', 0)
    df['Codsexo'] = df['Codsexo'].replace('Feminino', 1)
    df['Raça'] = df['Raça'].map(converte_raca)
    if total_conversoes <= 2: return df # Converte apenas Codsexo e Raça
    df['insulina'] = df['insulina'].map(converte_insulina)
    df['classe_imc'] = df['classe_imc'].map(converte_classe_imc)
    df['sedentario'] = df['sedentario'].map(converte_sedentario)
    df['tabagismo'] = df['tabagismo'].map(converte_tabagismo)
    df['etilismo'] = df['etilismo'].map(converte_etilismo)
    df['retorno'] = df['retorno'].map(converte_retorno)
    df['instruc'] = df['instruc'].map(converte_instruc)
    return df

# Converte todos os estágios para números 
def converte_estagio(valor):
    if   valor == 'Estágio 1 - >= 90 ml':  return 1
    elif valor == 'Estágio 2 - 60-89 ml':  return 2
    elif valor == 'Estágio 3a - 45-59 ml': return 3
    elif valor == 'Estágio 3b - 30-44 ml': return 4
    elif valor == 'Estágio 4 - 15-29 ml':  return 5
    elif valor == 'Estágio 5 - < 15 ml':   return 6
    
# Aplicação da função anterior
def converte_todos_estagios(df, lista=lista_estagio_todos):
    for estagio in lista: df[estagio] = df[estagio].map(converte_estagio)  
    return df

# 'RendaSM' possui alguns poucos valores como ' '. Assim, são convertidos para NaN
for i in range(len(banco_data)):
    if banco_data.loc[i, 'RendaSM'] == ' ': banco_data.loc[i, 'RendaSM'] = np.nan
        
# Conversão da coluna 'RendaSM' para float
banco_data['RendaSM'] = banco_data['RendaSM'].astype(float)


# In[33]:


banco_data = converte_todos_para_numero(banco_data)
banco_data = converte_todos_estagios(banco_data)


# ## Funções para treinamento, teste e classificação

# In[34]:


# Prepara um conjunto de dados X para 5 cenários diferentes de inputação de dados
def prepara_X(X):
    X_0 = X.fillna(0) 
    X_media = X.fillna(X.mean())
    X_mediana = X.fillna(X.median()) 
    Xnn = KNNImputer(n_neighbors=int(np.sqrt(X.shape[0]*X.shape[1])) , weights="uniform").fit_transform(X) 
    X_mice = MICE().fit_transform(X) 
    return X_0, X_media, X_mediana, Xnn, X_mice

# Extra Trees Classifier: 70% treinamento e 30% teste
def treino_teste(X, y, size_test=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state=0)
    modelo = ExtraTreesClassifier()
    modelo.fit(X_train, y_train)
    y_pred = cross_val_predict(modelo, X_test, y_test, cv = 5)
    return modelo, modelo.score(X_test, y_test), y_test, y_pred

# Retorna a média da acurácia de uma classificação com um dado número de iterações
def simulacao(X, y, iteracoes):
    soma_score = 0
    for _ in range(iteracoes): soma_score += treino_teste(X, y)[1]
    return soma_score/iteracoes


# ## Cenários

# ### Variável alvo

# In[35]:


y = banco_data['ESTAGIOF_EQ'] 


# ### Conjuntos X

# In[36]:


X1 = banco_data.filter(items = c4)   # antigo cenário 4
X2 = banco_data.filter(items = c9)   # antigo cenário 9
X3 = banco_data.filter(items = c13)  # antigo cenário 13
X4 = banco_data.filter(items = c20)  # antigo cenário 20
todos_X = [X1, X2, X3, X4]


# ### Simulações

# In[37]:


lista_acuracia = []
num_simulacoes = 100
for x in todos_X:
    X0, Xmedia, Xmediana, Xnn, Xmice = prepara_X(x)
    lista_acuracia.append([
                            simulacao(X0,       y, num_simulacoes), # Preenchimento com 0
                            simulacao(Xmedia,   y, num_simulacoes), # Preenchimento com a média
                            simulacao(Xmediana, y, num_simulacoes), # Preenchimento com a mediana
                            simulacao(Xnn,      y, num_simulacoes), # Preenchimento com os vizinhos mais próximos
                            simulacao(Xmice,    y, num_simulacoes)  # Preenchimento com MICE
                         ])


# ### Impressão dos resultados

# In[38]:


# Dataframe com de cada um dos cenários
df_acuracias = pd.DataFrame(lista_acuracia, 
                            columns = ['Zero', 'Média', 'Mediana', 'Vizinhos', 'MICE'], index = ['1', '2', '3', '4'])
df_acuracias

