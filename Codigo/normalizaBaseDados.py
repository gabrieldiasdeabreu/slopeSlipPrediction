import pandas as pd
import matplotlib.pyplot as plt
def normalizaVariavel(lista):
    return [(i-min(lista))/(max(lista)-min(lista)) for i in lista]

nomeBaseNormalizar = 'medidaresumo.xlsx'
baseDados = pd.read_excel(nomeBaseNormalizar)
baseDadosNormalizada = pd.read_csv('baseDadosRiscoDeslizePronta.csv')
# print(baseDados.keys())
# print(baseDadosNormalizada.keys())
print(min(baseDadosNormalizada['AmplitudeNorm']))
baseDadosNormalizada['AmplitudeNorm'] = normalizaVariavel(baseDados['Amplitude'])
baseDadosNormalizada['AltimetriaNorm'] = normalizaVariavel(baseDados['Altimetria'])
baseDadosNormalizada['concavoConvNorm'] = normalizaVariavel(baseDados['ConcavoConv'])
baseDadosNormalizada['CurvaturaNegativaNorm'] = normalizaVariavel(baseDados['Curvatura-'])
baseDadosNormalizada['CurvaturaPositivaNorm'] = normalizaVariavel(baseDados['Curvatura+'])
baseDadosNormalizada['DeclividadeNorm'] = normalizaVariavel(baseDados['Declividade'])
baseDadosNormalizada.dropna(inplace=True)
baseDadosNormalizada.to_csv('baseDadosDeslizamento.csv')
# baseDadosNormalizada.boxplot()
# print(min(baseDadosNormalizada['AmplitudeNorm']))
# plt.show()
# print(baseDadosNormalizada['altimetria'].min())
# to_csv('analize.csv')

