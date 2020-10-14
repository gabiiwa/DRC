########################################################################
''' Pipeline de Execução '''                                           #                          
# 'classePipeline.py'                                                  #                       
########################################################################

from classeDados import Dados, BancoCSV, BancoSQL
from classeEstatistica import Estatistica
from classeProcessamento import Processamento
from classeTreinoTeste import TreinoTeste

class Pipeline:
   
    def __init__(self, file_name, sep, encoding, orig=0, tipo=0):
        self.__file_name = file_name
        self.__sep = sep
        self.__encoding = encoding
        self.__orig = orig
        self.__tipo = tipo
        
        # Leitura dos dados
        if (self.__tipo == 0): self.dados = BancoCSV(self.__file_name, self.__sep, self.__encoding)
        else:                  self.dados = BancoSQL(self.__file_name, self.__sep, self.__encoding)
        if (self.__orig == 0): self.banco = self.dados.ler_banco()
        else:                  self.banco = self.dados.ler_banco(1)
           
        # Estatísticas 
        self.est  = Estatistica()
        self.est.tfg_por_ano(self.banco, 1),
        self.est.distribuicao_estagios(self.banco, 1)
        self.est.pacientes_por_tfg(self.banco, 1)
        print('\nTotal de pacientes com pelo menos um valor de TFG:', self.est.pelo_menos_1_tfg(self.banco))
        self.est.pacientes_por_dados_exames_medicamentos(self.banco)
        
        # Processamento
        self.proc = Processamento()
        if (self.__orig != 0): # Se for o banco original
            self.proc.reduz_banco(self.banco)
            self.proc.adiciona_tfg(self.banco)
            self.proc.adiciona_estagio(self.banco)
            self.proc.adiciona_estagio_inicial_final(self.banco)
            self.proc.adiciona_creatinina(self.banco)
        
    # Treinamento e teste, métricas de erro e análise de sensibilidade
    def analises_finais(self, var_alvo, col_X):
        self.trte = TreinoTeste(self.banco)
        
        # Convesão dos dados literais para numéricos
        self.trte.conversao()
        
        # Treinamento e teste
        score = self.trte.treino_teste(var_alvo, col_X)
        print('\nScore: %.4f' % score)
        
        # Métricas de avaliação de erros
        mse, rmse = self.trte.metricas_erro()
        print('\nR2 = %.4f,' % score, 'MSE = %.4f,' % mse, 'RMSE = %.4f' % rmse)
        
        # Aplicação da análise de sensibilidade
        morris = self.trte.aplica_analise_sensibilidade()
        print('\nResultado do método de Morris:\n', morris)
        
        return score, mse, rmse
 
   
