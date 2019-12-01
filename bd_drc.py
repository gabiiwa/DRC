#!/usr/bin/python

import pymysql
import pandas as pd
import numpy as np

#CONECTANDO COM O BANCOS
con = pymysql.connect(host="localhost", user="gabic", passwd="1234", db="doenca_renal_cronica")
con.select_db('doenca_renal_cronica')

banco_int= pd.read_csv('/home/gabic9814/Downloads/Engenharia Computacional/bolsa/drc/Dados2010-2014DesfechoFasesETC2.csv')
cfr= pd.read_csv('/home/gabic9814/Downloads/Engenharia Computacional/bolsa/drc/cfr2.csv')


#consertando banco_int
banco_int = banco_int.replace(np.nan,'0.0', regex=True)
banco_int = banco_int.replace(',', '.', regex=True)

#consertando cfr
cfr = cfr.replace(np.nan,'0.0', regex=True)
cfr = cfr.replace(',', '.', regex=True)

#%%
def albuminTaxa(taxa_alb,estagio_tfg,i):
    if estagio_tfg == '1' or estagio_tfg == '2':
        if float(taxa_alb) < 3.0 or (float(taxa_alb)>=3.0 and float(taxa_alb)<30.0):
            return '1'
        elif float(taxa_alb)>=30.0:
            return '2'
    elif estagio_tfg == '3a':
        if float(taxa_alb) < 3.0:
            return '1'
        elif float(taxa_alb)>=3.0 and float(taxa_alb)<30.0:
            return '2'
        elif float(taxa_alb)>=30.0:
            return '3'
    elif estagio_tfg == '3b':
        if float(taxa_alb) < 3.0:
            return '2'
        elif float(taxa_alb)>=3.0 and float(taxa_alb)<30.0:
            return '3'
        elif float(taxa_alb)>=30.0:
            return '4'
    elif estagio_tfg == '4':
        if float(taxa_alb) < 3.0 or (float(taxa_alb)>=3.0 and float(taxa_alb)<30.0):
            return '3'
        elif float(taxa_alb)>=30.0:
            return '4'
    elif estagio_tfg == '5':
        return '4'
    
    else:
        return '0.0'
    
    

#%%
cursor = con.cursor()  
cursor.connection.autocommit(True)
for i in range(len(banco_int)): 

    ##Inserindo exame_fisico
    #muito abaixo do peso
    if str(banco_int.get_values()[i][33]) <= '17.0':
        classe_imc = '1'
    #abaixo do peso    
    elif str(banco_int.get_values()[i][33]) >= '17.0' and str(banco_int.get_values()[i][33]) <= '18.49':
        classe_imc = '2'
    #peso normal    
    elif str(banco_int.get_values()[i][33]) >= '18.5' and str(banco_int.get_values()[i][33]) <= '24.99':
        classe_imc = '3'  
    #acima do peso    
    elif str(banco_int.get_values()[i][33]) >= '25.0' and str(banco_int.get_values()[i][33]) <= '29.99':
        classe_imc = '4'
    #Obesidade I    
    elif str(banco_int.get_values()[i][33]) >= '30.0' and str(banco_int.get_values()[i][33]) <= '34.99':
        classe_imc = '5'
    #Obesidade II
    elif str(banco_int.get_values()[i][33]) >= '35.0' and str(banco_int.get_values()[i][33]) <= '39.99':
        classe_imc = '6'
    #Obesidade III
    elif str(banco_int.get_values()[i][33]) >= '40.0':
        classe_imc = '7'
   
    pesoi = banco_int.get_values()[i][5]
    pesof = banco_int.get_values()[i][6]
    sql3="INSERT INTO exame_fisico(peso_inicial, peso_final, imc, pas_inicial, pad_inicial,pad_final,pas_final, altura, classe_imc, dt_pressao_inicial,dt_pressao_final,sedentarismo,etilismo,tabagismo) VALUES('"+pesoi+"','"+pesof+"','"+banco_int.get_values()[i][33]+"','"+str(banco_int.get_values()[i][37])+"','"+str(banco_int.get_values()[i][38])+"','"+str(banco_int.get_values()[i][41])+"','"+str(banco_int.get_values()[i][40])+"','"+(banco_int.get_values()[i][7])+"','"+classe_imc+"','"+banco_int.get_values()[i][36]+"','"+banco_int.get_values()[i][39]+"','"+banco_int.get_values()[i][14]+"','"+banco_int.get_values()[i][15]+"','"+banco_int.get_values()[i][16]+"');"
    cursor.execute(sql3) 
    cursor.execute('SELECT last_insert_id() INTO @exame_fisico')
    cursor.fetchone()
    
    ##Inserindo cidade
    sql1="INSERT INTO cidade(nome) VALUES ('"+banco_int.get_values()[i][12]+"')"
    cursor.execute(sql1)
    cursor.execute('SELECT last_insert_id() INTO @cidade')
    cursor.fetchone()
    
    ##Inserindo ubs
    sql2="INSERT INTO ubs(nome) VALUES('"+banco_int.get_values()[i][13]+"')"
    cursor.execute(sql2)
    cursor.execute('SELECT last_insert_id() INTO @ubs')
    cursor.fetchone()



    ##Inserindo has
    sql4="INSERT INTO consulta_has(total_1_2011,total_2_2011,total_1_2012,total_2_2012,total_1_2013,total_2_2013,total_1_2014,total_2_2014) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    z=list(banco_int.get_values()[i][221:229])
    cursor.execute(sql4,list(z))
    cursor.execute('SELECT last_insert_id() INTO @has')
    cursor.fetchone()


    ##Inserindo drc
    sql5="INSERT INTO consulta_drc(total_1_2011,total_2_2011,total_1_2012,total_2_2012,total_1_2013,total_2_2013,total_1_2014,total_2_2014) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    r=list(banco_int.get_values()[i][213:221])
    cursor.execute(sql5,list(r))
    cursor.execute('SELECT last_insert_id() INTO @drc')
    cursor.fetchone()    
        


    ##Inserindo dm
    sql6="INSERT INTO consulta_dm(total_1_2011,total_2_2011,total_1_2012,total_2_2012,total_1_2013,total_2_2013,total_1_2014,total_2_2014) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    m=list(banco_int.get_values()[i][229:237])
    cursor.execute(sql6,list(m))
    cursor.execute('SELECT last_insert_id() INTO @dm')
    cursor.fetchone()      
    
    ##Inserindo tfg
    sql7="INSERT INTO calculo_tfg(creatinina_1_2011, creatinina_2_2011,	creatinina_1_2012,creatinina_2_2012,	creatinina_1_2013,creatinina_2_2013,	creatinina_1_2014,creatinina_2_2014,tfg_1_2011,tfg_2_2011,tfg_1_2012,tfg_2_2012,tfg_1_2013,tfg_2_2013,tfg_1_2014,tfg_2_2014) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    t=list(banco_int.get_values()[i][237:253])
    cursor.execute(sql7,list(t))
    cursor.execute('SELECT last_insert_id() INTO @tfg')
    cursor.fetchone() 
    
    
    ##Inserindo trs
    sql8="INSERT INTO preparacao_trs(tgp_inicial	, dt_tgp_inicial, tgp_final, dt_tgp_final,hemog_glicada_inicial, dt_hemog_glicada_inicial, hemog_glicada_final,	dt_hemog_glicada_final) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    trs=list(banco_int.get_values()[i][94:98])
    trs.extend(banco_int.get_values()[i][90:94])
    cursor.execute(sql8,list(trs))
    cursor.execute('SELECT last_insert_id() INTO @trs')
    cursor.fetchone() 
    
     ##Inserindo medicamento
    sql9="INSERT INTO medicamento(ieca1,brat2,betabloq3,estatina9,aas8,diur4,biguadina6,sulfonura7,fibrato13,insulina) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    banco_int['insulina'] = banco_int['insulina'].replace('Sim', '1.00', regex=True)
    banco_int['insulina'] = banco_int['insulina'].replace('Não', '0.00', regex=True)
    med=list(banco_int.get_values()[i][23:33])
    cursor.execute(sql9, list(med))
    cursor.execute('SELECT last_insert_id() INTO @medicamento')
    cursor.fetchone()     
    
     ##Inserindo tratamento
    sql10="INSERT INTO avaliacao_tratamento(sodio_urinario_inicial,dt_sodio_urinario_inicial,sodio_urinario_final,dt_sodio_urinario_final,sodio_serico_inicial,dt_sodio_serico_inicial,sodio_serico_final,dt_sodio_serico_final) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    trat=list(banco_int.get_values()[i][66:70])
    trat.extend(banco_int.get_values()[i][198:202])
    cursor.execute(sql10,list(trat))
    cursor.execute('SELECT last_insert_id() INTO @tratamento')
    cursor.fetchone()   

    ##Inserindo afr
    sql11="INSERT INTO aval_funcao_renal(creatinina_inicial,	dt_creatinina_inicial,creatinina_final,dt_creatinina_final,albumina_inicial,dt_albumina_inicial,albumina_final,dt_albumina_final,rel_alb_creat_uauc_inicial,	dt_rel_alb_creat_uauc_inicial,rel_alb_creat_uauc_final,dt_rel_alb_creat_uauc_final,proteinuria24h_inicial,dt_proteinuria24h_inicial,proteinuria24h_final,dt_proteinuria24h_final,microAlbumi_inicial,dt_microAlbumi_inicial,microAlbumi_final,dt_microAlbumi_final,hematuria_inicial,dt_hematuria_inicial	,hematuria_final,dt_hematuria_final,	tshi_inicial	,dt_tshi_inicial,tshi_final,	dt_tshi_final,potassio_inicial,dt_potassio_inicial,potassio_final,dt_potassio_final,ureia_inicial,dt_ureia_inicial,ureia_final,dt_ureia_final,ureia24h_inicial,dt_ureia24h_inicial,ureia24h_final,dt_ureia24h_final,fosforo_inicial,dt_fosforo_inicial,fosforo_final,dt_fosforo_final,pth_intacto_inicial	,dt_pth_intacto_inicial,	pth_intacto_final,dt_pth_intacto_final,vitaminaD_inicial,dt_vitaminaD_inicial,vitaminaD_final,dt_vitaminaD_final,fosfatase_alc_inicial,dt_fosfatase_alc_inicial,fosfatase_alc_final,dt_fosfatase_alc_final,ck_inicial,dt_ck_inicial,ck_final,dt_ck_final,	ecoae_inicial,dt_ecoae_inicial,ecoae_final,dt_ecoae_final,ecoao_inicial,	dt_ecoao_inicial	,ecoao_final,dt_ecoao_final,	ecosiv_inicial,dt_ecosiv_inicial	,ecosiv_final,dt_ecosiv_final) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    afr=list(banco_int.get_values()[i][42:46])
    afr.extend(banco_int.get_values()[i][142:146])
    afr.extend(banco_int.get_values()[i][158:166])
    afr.extend(banco_int.get_values()[i][186:190])
    afr.extend(banco_int.get_values()[i][194:198])
    afr.extend(banco_int.get_values()[i][46:50]) #tshi_inicial	,dt_tshi_inicial,tshi_final,	dt_tshi_final,
    afr.extend(banco_int.get_values()[i][106:110]) #potassio_inicial,dt_potassio_inicial,potassio_final,dt_potassio_final
    afr.extend(banco_int.get_values()[i][206:210]) #ureia_inicial,dt_ureia_inicial,ureia_final,dt_ureia_final
    afr.extend(banco_int.get_values()[i][114:118]) #ureia24h_inicial,dt_ureia24h_inicial,ureia24h_final,dt_ureia24h_final
    afr.extend(banco_int.get_values()[i][130:142]) #fosforo_inicial,dt_fosforo_inicial,fosforo_final,dt_fosforo_final,pth_intacto_inicial	,dt_pth_intacto_inicial,	pth_intacto_final,dt_pth_intacto_final,vitaminaD_inicial,dt_vitaminaD_inicial,vitaminaD_final,dt_vitaminaD_final
    afr.extend(banco_int.get_values()[i][190:194]) #fosfatase_alc_inicial,dt_fosfatase_alc_inicial,fosfatase_alc_final,dt_fosfatase_alc_final,
    afr.extend(banco_int.get_values()[i][202:206]) #ck_inicial,dt_ck_inicial,ck_final,dt_ck_final,
    afr.extend(banco_int.get_values()[i][166:178]) #ecoae_inicial,dt_ecoae_inicial,ecoae_final,dt_ecoae_final,ecoao_inicial,	dt_ecoao_inicial	,ecoao_final,dt_ecoao_final,	ecosiv_inicial,dt_ecosiv_inicial	,ecosiv_final,dt_ecosiv_final
    cursor.execute(sql11,list(afr))
    cursor.execute('SELECT last_insert_id() INTO @afr')
    cursor.fetchone() 

    ##Inserindo cfr
    sql12="INSERT INTO complicacao_funcao_renal(	hemoglobina_inicial,	dt_hemoglobina_inicial,	hemoglobina_final,	dt_hemoglobina_final	,acido_urico_inicial,	dt_acido_urico_inicial,	acido_urico_final,	dt_acido_urico_final	,ferritina_inicial,dt_ferritina_inicial	,ferritina_final,	dt_ferritina_final,	ferro_serico_inicial	,dt_ferro_serico_inicial,	ferro_serico_final,	dt_ferro_serico_final,	calcio_total_inicial	,dt_calcio_total_inicial,	calcio_total_final,	dt_calcio_total_final,	acido_folico_inicial	,dt_acido_folico_inicial,	acido_folico_final,	dt_acido_folico_final,	bilirrubina_inicial,	dt_bilirrubina_inicial,	bilirrubina_final,	dt_bilirrubina_final,	vitaminaB12_inicial,	dt_vitaminaB12_inicial,	vitaminaB12_final,	dt_vitaminaB12_final	,colest_hdl_inicial,	dt_colest_hdl_inicial,colest_hdl_final	,dt_colest_hdl_final,	colest_ldl_inicial,	dt_colest_ldl_inicial,colest_ldl_final	,dt_colest_ldl_final,	colest_total_inicial,	dt_colest_total_inicial,	colest_total_final,	dt_colest_total_final,	gama_glutamil_inicial,	dt_gama_glutamil_inicial,	gama_glutamil_final,	dt_gama_glutamil_final,triglicerides_inicial,dt_triglicerides_inicial	,triglicerides_final,	dt_triglicerides_final,	glicemia_jejum_inicial,	dt_glicemia_jejum_inicial,glicemia_jejum_final	,dt_glicemia_jejum_final,	ind_sat_transf_inicial,dt_ind_sat_transf_inicial,ind_sat_transf_final	,dt_ind_sat_transf_final,hbsag_inicial,dt_hbsag_inicial	,hbsag_final,dt_hbsag_final,antihbs_inicial,dt_antihbs_inicial,antihbs_final,dt_antihbs_final	,antihcv_inicial,	dt_antihcv_inicial,antihcv_final,dt_antihcv_final) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cfr_list=list(cfr.get_values()[i][:72])   
    cursor.execute(sql12,list(cfr_list))
    cursor.execute('SELECT last_insert_id() INTO @cfr')
    cursor.fetchone()     
    
    #Inserindo dados_consulta
    estagioI = banco_int.get_values()[i][253] # estagio inicial TFG
    estI = estagioI[8:10] 
    estagioF = banco_int.get_values()[i][254] # estagio inicial TFG
    estF = estagioF[8:10]
    ##calculando o estágio das taxas de albuminúria
    #inicial
    taxaI_alb = banco_int.get_values()[i][142]
    estI_alb = albuminTaxa(taxaI_alb,estI,i)
    #final
    taxaF_alb = banco_int.get_values()[i][144]
    estF_alb = albuminTaxa(taxaF_alb,estF,i)
    
    sql16="INSERT INTO dados_consulta(data_inicial,	data_final,	tempoAcomp,	total_consultas_drc,	total_consultas_dm,	total_consultas_has,	desfecho	,desfecho_text,	id_cidade,	id_afr,	id_ubs,	id_has,	id_drc,	id_dm,	id_cfr,	id_exame_fisico,	id_calculo_tfg,	id_preparacao_trs,	id_medicamento,	id_aval_tratamento,estagioI,estagioF,estagioI_alb,estagioF_alb) VALUES('"+str(banco_int.get_values()[i][18])+"','"+str(banco_int.get_values()[i][19])+"','"+str(banco_int.get_values()[i][35])+"','"+str(banco_int.get_values()[i][21])+"','"+str(banco_int.get_values()[i][20])+"','"+str(banco_int.get_values()[i][22])+"','"+str(banco_int.get_values()[i][211])+"','"+str(banco_int.get_values()[i][212])+"',@cidade,@afr,@ubs,@has,@drc,@dm,@cfr,@exame_fisico,@tfg,@trs,@medicamento,@tratamento,'"+str(estI)+"','"+str(estF)+"','"+str(estI_alb)+"','"+str(estF_alb)+"')"   
    cursor.execute(sql16)
    cursor.execute('SELECT last_insert_id() INTO @dados_consulta')
    cursor.fetchone()     
        
    
#
    ##Inserindo raca
    sql13 = "INSERT INTO raca(nome) VALUES('" + banco_int.get_values()[i][4] + "')"
    cursor.execute(sql13)
    cursor.execute('SELECT last_insert_id() INTO @raca')
    cursor.fetchone()
    
    ##Inserindo Instrucao
    sql14 = "INSERT INTO instrucao(nome) VALUES('"+banco_int.get_values()[i][8]+"')"
#   # z=list(banco_int.get_values()[i][8])
    cursor.execute(sql14)
    cursor.execute('SELECT last_insert_id() INTO @instrucao')
    cursor.fetchone()
 
    ##Inserindo Paciente -- ultimo a ser feito
    sql15 = "INSERT INTO paciente(id_original, data_nascimento, sexo, renda_sm, renda_familiar_sm, tamanho_familia,id_dados_consulta, id_raca, id_instrucao) VALUES('"+str(banco_int.get_values()[i][0])+"','"+str(banco_int.get_values()[i][1])+"','"+str(banco_int.get_values()[i][3])+"','"+str(banco_int.get_values()[i][9])+"','"+str(banco_int.get_values()[i][11])+"','"+str(banco_int.get_values()[i][10])+"',@dados_consulta,@raca,@instrucao)"
    cursor.execute(sql15)
    cursor.fetchone()   
    
con.commit()
cursor.close()    

##ENCERRANDO A CONEXAO
con.close()


