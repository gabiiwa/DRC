########################################################################
''' Banco de dados '''                                                 #                          
# 'classeDados.py'                                                     #                       
########################################################################

import pandas as pd
from abc import ABC, abstractmethod 

class Dados(ABC):

    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def ler_banco(self): pass

    @abstractmethod 
    def shape(self): pass
    
class BancoCSV(Dados): 
    
    def __init__(self, file_name, sep, encoding):
        self.__file_name = file_name
        self.__sep = sep
        self.__encoding = encoding
        self.__banco = pd.DataFrame()
    
    def ler_banco(self, drop_index=0):
        self.__banco = pd.read_csv(self.__file_name, sep=self.__sep, encoding=self.__encoding, low_memory=False)
        if (drop_index == 0): 
            self.__banco = self.__banco.drop(['Unnamed: 0'], axis=1)
        return self.__banco

    def salvar_banco(self, save_file, index_save, sep_save, encoding_save):
        self.__banco.to_csv(save_file, index = index_save, sep = sep_save, encoding = encoding_save)
        print('\nArquivo salvo com sucesso!\n')

    @property
    def shape(self):
        return self.__banco.shape

    @property
    def set_banco(self, banco):
        self.__banco = banco

class BancoSQL(Dados):
    
    def __init__(self, file_name, sep, encoding): pass
    def ler_banco(self, drop_index=0): pass
    def shape(self): pass



    