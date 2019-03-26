import pandas as pd
from sklearn import svm
import Main as ma
import numpy as np
from numpy.linalg._umath_linalg import svd_m
import Analises as an
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers


def rna(x_treino, x_teste, y_treino):
    rna = Sequential()
    rna.add(Dense(units=10, input_dim=29, activation='relu'))
    # rna.add(Dense(units=50, input_dim=29, activation='relu'))
    # rna.add(Dense(units=100, input_dim=29, activation='relu'))
    # rna.add(Dense(units=10, activation='linear'))
    # rna.add(Dense(units=10, input_dim=29, activation='linear'))
    # # rna.add(Dense(units=60, input_dim=7, activation='RELU'))
    # rna.add(Dense(units=320, activation='relu'))
    # rna.add(Dense(units=320, activation='relu'))
    rna.add(Dense(units=1, activation='sigmoid'))
    # print(x_treino)
    rna.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['mean_squared_error'])
    print(rna.fit(x_treino, y_treino, verbose=0, epochs=1000))
    return rna.predict(x_teste, verbose=0)


baseDados = pd.read_csv('instancia/baseDadosDeslizamento.csv')
# modelo = sk.svm
numeroInstanciaTreino=int(len(baseDados)*0.60)
# print(int(len(baseDados)*0.80))
limite = 29
x_treino = baseDados[baseDados.keys()[:limite]][:numeroInstanciaTreino]
y_treino = baseDados[baseDados.keys()[-1]][:numeroInstanciaTreino]
x_teste = baseDados[baseDados.keys()[:limite]][numeroInstanciaTreino:]
y_teste = baseDados[baseDados.keys()[-1]][numeroInstanciaTreino:]
# # print(y_treino)
# np.transpose(x_treino)
# print(baseDados.keys()[:limite])
# predito = ma.regressaoLogistica(x_treino, x_teste, y_treino, c=0.01)
# print(, predito[1].intercept_)
#
# for ind in list(predito[1].coef_[0]):
#     print(ind)
#
# with open("relevanciaFeaturesRegressaoLogistica.csv", 'w') as arq:
#     arq.write('feature, relevanciaRegressaoLogistica'+'\n')
#     for rank, ind in zip(predito[1].coef_[0], baseDados.keys()[:limite]):
#         arq.write(str(ind)+','+ str(rank)+'\n')


predito = rna(x_treino,x_teste,y_treino)
# predito = ma.regressaoLogistica(x_treino,x_teste,y_treino)
predito = np.ravel(predito)
# predito = predito >= 0.5
print(list(predito))
classes = ['Positivo Deslizamento', 'Negativo Deslizamento']
ma.matrizConfusa(y_teste, predito, classes, 'experimentoRegressaoLogistica')
an.AUC(predito, y_teste, 'experimentoRegressaoLogistica')

# print(ma.an.matthews_corrcoef)
