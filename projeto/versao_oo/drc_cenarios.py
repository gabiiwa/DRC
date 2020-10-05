##########################################################################################
''' Módulo: drc_cenarios.py '''                                                          #
# Listas com os dados para diferente cenários de treinamento e teste                     #
##########################################################################################

import drc_listas as listas

# Cenário 1 - Maior quantidade possível de dados
cenario1 = ['Idade',  'Codsexo',  'Raça',  'pesoi',  'pesof',  'Alt',  'instruc',  'RendaSM',  'TamFamilia',  'RendaFamiliarSM',  'sedentario',  'etilismo',  'tabagismo',  'consultasDM',  'consultasDRC',  'consultasHAS',  'IECA1',  'BRAT2',  'BETABLOQ3',  'estatina9',  'AAS8',  'DIUR4',  'BIGUADINA6',  'SULFONIURA7',  'FIBRATO13',  'insulina',  'imc',  'classe_imc',  'tempoAcompanha',  'PAS_inicial',  'PAD_inicial',  'PAS_final',  'PAD_final',  'CreatininaI',  'CreatininaF',  'TSHI',  'TSHF',  'HemoglobinaI',  'HemoglobinaF',  'AcidoUricoI',  'AcidoUricoF',  'CalcioTotalI',  'CalcioTotalF',  'AcidoFolicoI',  'AcidoFolicoF',  'SodioUrinarioI',  'SodioUrinarioF',  'VitaminaB12I',  'VitaminaB12F',  'ColesterolHDLI',  'ColesterolHDLF',  'ColesterolLDLI',  'ColesterolLDLF',  'ColesterolTotalI',  'ColesterolTotalF',  'GamaGlutamilI',  'GamaGlutamilF',  'HemoglobinaGlicadaI',  'HemoglobinaGlicadaF',  'TGPI',  'TGPF',  'TrigliceridesI',  'TrigliceridesF',  'BilirrubinatotalI',  'BilirrubinatotalF',  'PotassioI',  'PotassioF',  'GlicemiadeJejumI',  'GlicemiadeJejumF',  'Ureia24hsI',  'Ureia24hsF',  'FerritinaI',  'FerritinaF',  'IndicedeSaturacaodaTransferenciaI',  'IndicedeSaturacaodaTransferenciaF',  'FerroSericoI',  'FerroSericoF',  'FosforoI',  'FosforoF',  'PTHintactoI',  'PTHintactoF',  'VITAMINADI',  'VITAMINADF',  'AlbuminaI',  'AlbuminaF',  'HBsAGI',  'HBsAGF',  'AntiHBsI',  'AntiHBsF',  'AntiHCVI',  'AntiHCVF',  'Rel.AlbuminaCreatininaUAUCI',  'Rel.AlbuminaCreatininaUAUCF',  'Proteinuria24hsI',  'Proteinuria24hsF',  'ECOAEI',  'ECOAEF',  'ECOAOI',  'ECOAOF',  'ECOSIVI',  'ECOSIVF',  'ECOPPI',  'ECOPPF',  'ECOFEI',  'ECOFEF',  'MicroalbuminuriaI',  'MicroalbuminuriaF',  'FosfataseAlcalinaI',  'FosfataseAlcalinaF',  'HematuriaI',  'HematuriaF',  'SodioSericoI',  'SodioSericoF',  'CKI',  'CKF',  'UreiaI',  'UreiaF',  'DRC_1_2011',  'DRC_2_2011',  'DRC_1_2012',  'DRC_2_2012',  'DRC_1_2013',  'DRC_2_2013',  'DRC_1_2014',  'DRC_2_2014',  'HAS_1_2011',  'HAS_2_2011',  'HAS_1_2012',  'HAS_2_2012',  'HAS_1_2013',  'HAS_2_2013',  'HAS_1_2014',  'HAS_2_2014',  'DM_1_2011',  'DM_2_2011',  'DM_1_2012',  'DM_2_2012',  'DM_1_2013',  'DM_2_2013',  'DM_1_2014',  'DM_2_2014',  'Creatinina_1_2011',  'Creatinina_2_2011',  'Creatinina_1_2012',  'Creatinina_2_2012',  'Creatinina_1_2013',  'Creatinina_2_2013',  'Creatinina_1_2014',  'Creatinina_2_2014',  'TFG_1_2011_EQ',  'ESTAGIO_EQ_1_2011',  'TFG_2_2011_EQ',  'ESTAGIO_EQ_2_2011',  'TFG_1_2012_EQ',  'ESTAGIO_EQ_1_2012',  'TFG_2_2012_EQ',  'ESTAGIO_EQ_2_2012',  'TFG_1_2013_EQ',  'ESTAGIO_EQ_1_2013',  'TFG_2_2013_EQ',  'ESTAGIO_EQ_2_2013',  'TFG_1_2014_EQ',  'ESTAGIO_EQ_1_2014',  'TFG_2_2014_EQ',  'ESTAGIO_EQ_2_2014',  'ESTAGIOI_EQ',  'CREATININAI']

# Cenário 2 - Cenário 1 sem todos os valores de creatinina, tfg e estágio
cenario2 = ['Idade',  'Codsexo',  'Raça',  'pesoi',  'pesof',  'Alt',  'instruc',  'RendaSM',  'TamFamilia',  'RendaFamiliarSM',  'sedentario',  'etilismo',  'tabagismo',  'consultasDM',  'consultasDRC',  'consultasHAS',  'IECA1',  'BRAT2',  'BETABLOQ3',  'estatina9',  'AAS8',  'DIUR4',  'BIGUADINA6',  'SULFONIURA7',  'FIBRATO13',  'insulina',  'imc',  'classe_imc',  'tempoAcompanha',  'PAS_inicial',  'PAD_inicial',  'PAS_final',  'PAD_final',  'CreatininaI',  'CreatininaF',  'TSHI',  'TSHF',  'HemoglobinaI',  'HemoglobinaF',  'AcidoUricoI',  'AcidoUricoF',  'CalcioTotalI',  'CalcioTotalF',  'AcidoFolicoI',  'AcidoFolicoF',  'SodioUrinarioI',  'SodioUrinarioF',  'VitaminaB12I',  'VitaminaB12F',  'ColesterolHDLI',  'ColesterolHDLF',  'ColesterolLDLI',  'ColesterolLDLF',  'ColesterolTotalI',  'ColesterolTotalF',  'GamaGlutamilI',  'GamaGlutamilF',  'HemoglobinaGlicadaI',  'HemoglobinaGlicadaF',  'TGPI',  'TGPF',  'TrigliceridesI',  'TrigliceridesF',  'BilirrubinatotalI',  'BilirrubinatotalF',  'PotassioI',  'PotassioF',  'GlicemiadeJejumI',  'GlicemiadeJejumF',  'Ureia24hsI',  'Ureia24hsF',  'FerritinaI',  'FerritinaF',  'IndicedeSaturacaodaTransferenciaI',  'IndicedeSaturacaodaTransferenciaF',  'FerroSericoI',  'FerroSericoF',  'FosforoI',  'FosforoF',  'PTHintactoI',  'PTHintactoF',  'VITAMINADI',  'VITAMINADF',  'AlbuminaI',  'AlbuminaF',  'HBsAGI',  'HBsAGF',  'AntiHBsI',  'AntiHBsF',  'AntiHCVI',  'AntiHCVF',  'Rel.AlbuminaCreatininaUAUCI',  'Rel.AlbuminaCreatininaUAUCF',  'Proteinuria24hsI',  'Proteinuria24hsF',  'ECOAEI',  'ECOAEF',  'ECOAOI',  'ECOAOF',  'ECOSIVI',  'ECOSIVF',  'ECOPPI',  'ECOPPF',  'ECOFEI',  'ECOFEF',  'MicroalbuminuriaI',  'MicroalbuminuriaF',  'FosfataseAlcalinaI',  'FosfataseAlcalinaF',  'HematuriaI',  'HematuriaF',  'SodioSericoI',  'SodioSericoF',  'CKI',  'CKF',  'UreiaI',  'UreiaF',  'DRC_1_2011',  'DRC_2_2011',  'DRC_1_2012',  'DRC_2_2012',  'DRC_1_2013',  'DRC_2_2013',  'DRC_1_2014',  'DRC_2_2014',  'HAS_1_2011',  'HAS_2_2011',  'HAS_1_2012',  'HAS_2_2012',  'HAS_1_2013',  'HAS_2_2013',  'HAS_1_2014',  'HAS_2_2014',  'DM_1_2011',  'DM_2_2011',  'DM_1_2012',  'DM_2_2012',  'DM_1_2013',  'DM_2_2013',  'DM_1_2014',  'DM_2_2014']

# Cenário 3 - 35 exames mais frequentes + 7 Creatininas + 7 TFGs + 7 Estágios + Estágio inicial
cenario3 = listas.dados_exames_35 + listas.creatinina[:7] + listas.tfg_eq[:7] + listas.estagio_eq_todos[:9]

# Cenário 4 - 35 exames mais frequentes + 7 Creatininas 
cenario4 = listas.dados_exames_35 + listas.creatinina[:7]

# Cenário 5 - 20 exames mais frequentes + 7 Creatininas + 7 TFGs + 7 Estágios + Estágio inicial
cenario5 = ['Codsexo',  'Idade',  'Raça',  'PAS_inicial',  'PAS_final',  'PAD_inicial',  'PAD_final',  'pesoi',  'pesof',  'HemoglobinaI',  'ColesterolTotalI',  'GlicemiadeJejumI',  'TrigliceridesI',  'PotassioI',  'ColesterolHDLI',  'UreiaI',  'TSHI',  'AcidoUricoI',  'HemoglobinaGlicadaI',  'TGPI',  'Creatinina_1_2011',  'Creatinina_2_2011',  'Creatinina_1_2012',  'Creatinina_2_2012',  'Creatinina_1_2013',  'Creatinina_2_2013',  'Creatinina_1_2014',  'TFG_1_2011_EQ',  'TFG_2_2011_EQ',  'TFG_1_2012_EQ',  'TFG_2_2012_EQ',  'TFG_1_2013_EQ',  'TFG_2_2013_EQ',  'TFG_1_2014_EQ',  'ESTAGIO_EQ_1_2011',  'ESTAGIO_EQ_2_2011',  'ESTAGIO_EQ_1_2012',  'ESTAGIO_EQ_2_2012',  'ESTAGIO_EQ_1_2013',  'ESTAGIO_EQ_2_2013',  'ESTAGIO_EQ_1_2014',  'ESTAGIO_EQ_2_2014',  'ESTAGIOI_EQ']

# Cenário 6 - Sexo, Idade, Raça e Creatinina inicial
cenario6 = ['Codsexo', 'Idade', 'Raça', 'CREATININAI']

# Cenário 7 - Sexo, Idade e Raça
cenario7 = ['Codsexo', 'Idade', 'Raça']

# Cenário 8 - 35 exames mais frequentes
cenario8 = listas.dados_exames_35

# Cenário 9 - 25 exames mais frequentes
cenario9 = listas.dados_exames_35[:25]

# Cenário 10 - 20 exames mais frequentes
cenario10 = listas.dados_exames_35[:20]

# Cenário 11 - 20 exames mais frequentes + Creatinina inicial
cenario11 = listas.dados_exames_35[:20] + ['CREATININAI']

