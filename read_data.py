import pandas as pd
import os
import sys
import re

csv='./CIIS2020/data/Banco35exames.csv'

df=pd.read_csv(csv)
target='ESTAGIOF'

y=df[target]

cols=['Idade', 'Codsexo', 'Raça', 'pesoi','imc','etilismo','HemoglobinaF',
      'PAS_inicial', 'PAD_inicial',]

X = df[cols].dropna()
