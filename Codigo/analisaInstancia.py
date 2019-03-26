from collections import Counter

import pandas as pd

baseDados = pd.read_excel('instancia/medidaresumo.xlsx')
# print(baseDados.describe())
baseDados.describe().to_csv('DescricaoBaseOriginal.csv')
# print(baseDados.AmplitudeNorm)
baseDados.groupby('positivoDeRisco').describe().to_csv('descricaoSeparadoPorRiscoBaseOriginal.csv')
# print(baseDados.corr().to_csv('matrizCorrelacao.csv'))
