import pandas as pd
from sklearn import svm
import Main as ma
import numpy as np
from numpy.linalg._umath_linalg import svd_m
import Analises as an
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import ranking
from sklearn.metrics import classification

def retornaInstanciaSeparada(inicio, fim, numeroInstanciaTreino):
    x_treino = baseDados[baseDados.keys()[inicio:fim]][:numeroInstanciaTreino]
    y_treino = baseDados[baseDados.keys()[fim]][:numeroInstanciaTreino]
    x_teste = baseDados[baseDados.keys()[inicio:fim]][numeroInstanciaTreino:]
    y_teste = baseDados[baseDados.keys()[fim]][numeroInstanciaTreino:]
    return x_treino, y_treino, x_teste, y_teste

def rna(x_treino, x_teste, y_treino, yTeste, epocas, numNeuroniosCamOculta, optimizer):
    rna = Sequential()
    rna.add(Dense(units=37, input_dim=29, activation='relu'))
    rna.add(Dense(units=numNeuroniosCamOculta, activation='relu'))
    # rna.add(Dense(units=100, input_dim=29, activation='relu'))
    # rna.add(Dense(units=10, activation='linear'))
    # rna.add(Dense(units=10, input_dim=29, activation='linear'))
    # # rna.add(Dense(units=60, input_dim=7, activation='RELU'))
    # rna.add(Dense(units=320, activation='relu'))
    # rna.add(Dense(units=320, activation='relu'))
    rna.add(Dense(units=1, activation='sigmoid'))
    # print(x_treino)
    rna.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[ 'binary_crossentropy'])
    modelo = rna.fit(x_treino, y_treino, verbose=0, epochs=epocas)
    return rna.predict(x_teste, verbose=0), rna.evaluate(xTeste, yTeste), modelo.history['binary_crossentropy']

def plotaMetricaPorEpoca(metrica):
    x = np.arange(1, len(metrica)+1)
    plt.plot(x, metrica)
    plt.savefig('analiseRede/metricaPorEpoca.pdf')
    plt.show()


def plotaMetricaPorEpocaComparada(listaMetricas, listaNomes):
    x = np.arange(1, len(listaMetricas[0])+1)
    for metrica, nome in zip(listaMetricas, listaNomes):
        plt.plot(x, metrica, label=nome)
        # plt.plot(x, metrica2, label=nome2)
    plt.legend()
    plt.savefig('analiseRede/metricaPorEpoca'+str(listaNomes)+'.pdf')
    plt.show()

def classifica(i):
    if i>= 0.5:
         return 1
    else:
         return 0


def mediaErrosMetricas(optimizers, numNeuronios):
    epocas = 700
    media = lambda lista: sum(lista)/len(lista)
    numExecucoes = 5
    listaMetricas = list()
    listaNomes = list()
    listaMatrizes = list()
    listaAp = list()
    listaDescricoes = list()
    for optmizer in optimizers:
        print(optmizer)
        execucoes = list()
        for num in range(numExecucoes):
            predicao, avaliacao, historico = rna(xTreino, xTeste, yTreino, yTeste, epocas, numNeuronios, optmizer)
            execucoes.append(historico)
            listaMatrizes.append(ma.matrizConfusa(yTeste, predicao>= 0.5, ['positivoDeslizamento', 'negativoDeslizamento'], 'rnaAda'+str(num)))
            listaAp.append(ranking.average_precision_score(yTeste, predicao))
            # listaDescricoes.append(classification.classification_report(yTeste, list(map(classifica, predicao)) ,['positivoDeslizamento', 'negativoDeslizamento'] ))
        # np.array(execucoes)
        listaMetricas.append([np.mean(i) for i in zip(execucoes[0], execucoes[1], execucoes[2], execucoes[3], execucoes[4])])
        listaNomes.append(optmizer)
    # print(listaMetricas)
    return listaMatrizes, listaAp
    # plotaMetricaPorEpocaComparada(listaMetricas, listaNomes)

baseDados = pd.read_csv('instancia/baseDadosDeslizamento.csv')
# print(baseDados)
numeroInstanciaTreino=int(len(baseDados)*0.60)
xTreino, yTreino, xTeste, yTeste = retornaInstanciaSeparada(0, 29, numeroInstanciaTreino)
optimizers = ['Adagrad']#, "adam"]#"adam", 'nadam', 'RMSprop',

# predicao, avaliacao, historico = rna(xTreino, xTeste, yTreino, yTeste, 200, 10, 'nadam')
# predicao, avaliacao, historico1 = rna(xTreino, xTeste, yTreino, yTeste, 200, 10, 'adam')
with open('saidasTeste.txt', 'w') as arq:
    with open('saidaResumida.csv', 'w') as arq1:
        arq.write('numNeuronios'+','+ 'meanAp'+','+ 'stdAp'+'\n')
        arq1.write('numNeuronios'+','+ 'meanAp'+','+ 'stdAp'+'\n')
        for i in range(20,51):
            mediaErrosMetrica, listaAp = mediaErrosMetricas(optimizers, i)
            arq.write(str(i) + ',' + str(np.mean(listaAp))
            + ',' + str(np.std(listaAp)) + '\n' )
            arq1.write(str(i) + ',' + str(np.mean(listaAp))
            + ',' + str(np.std(listaAp)) + '\n' )
            # print(mediaErrosMetrica[:])
            # print(mediaErrosMetrica[:][0][0][1] + mediaErrosMetrica[:][0][1][0])

            # mediaErrosMetrica[:][0][1])) + ',' + str(np.std(mediaErrosMetrica[:][0][0]+ mediaErrosMetrica[:][0][1])))
            for i, ap in zip(mediaErrosMetrica, listaAp):
                arq.write(str(i)+'\n')
                arq.write(str(ap)+'\n')
                # arq1.write(str(i)+'\n')

# [
# array([[   0, 4266],
#        [   0, 1951]]),
# array([[4266,    0],
#        [1951,    0]]),
# array([[4266,    0],
#        [1951,    0]]),
# array([[4266,    0],
#        [1951,    0]]),
# array([[4266,    0],
#        [1951,    0]])]



# print(historico)
# plotaMetricaPorEpocaComparada(historico, historico1, 'nadam', 'adam')
# predicao = predicao >= 0.5
# ma.matrizConfusa(yTeste, predicao, ['positivoDeslizamento', 'negativoDeslizamento'], 'experimentoRedeNeural')
# an.AUC(predicao, yTeste, 'experimentoRedeNeural')
