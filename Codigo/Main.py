# from Cython.Includes.libcpp.iterator import insert_iterator
import Analises as an
from LeInstancia import LeInstancia
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import logistic
from sklearn import linear_model
import Grafico3D as graf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def regressaoLogistica(X_train, X_test, y_train, c=1):
    gnb = linear_model.LogisticRegression(C=0.007)#0.004
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return y_pred, gnb


def suportVectorMachine(X_train, X_test, y_train):
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    # print(classifier.coef0)
    return y_pred, classifier


def naiveBayes(X_train, X_test, y_train):
    gnb = GaussianNB()
    # y_pred = gnb.fit(data[:8562], target[:8562]).predict(data[8562:])
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    with open('infosNaiveBayes.txt', 'w') as arquivo:
        arquivo.write(
            'quantidade' + '\n' + 'pele: ' + str(gnb.class_count_[0]) + '\n' + 'Não pele: ' + str(gnb.class_count_[1]))
        arquivo.write('quantidade' + '\n' + 'pele: ' + str(gnb.priors))

    return y_pred, gnb


def matrizConfusa(y_test, y_pred, nomesClasses,  nome):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # an.plot_confusion_matrix(cnf_matrix, nomesClasses, nome, normalize=True,
                             # title='Normalized confusion matrix')
    # plt.savefig('MatrizConfusa_' + nome + '.pdf')
    # plt.close()
    return cnf_matrix


def executaAnalises(X_train, X_test, y_train, y_test):
    nome = 'naiveBayes'
    previsao, classifier = naiveBayes(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    # an.Plota3D(X_test, y_test, classifier, nome)

    nome = 'SVM'
    previsao, classifier = suportVectorMachine(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    an.Plota3D(X_test, y_test, classifier, nome)

    nome = 'RegressaoLogistica'
    previsao, classifier = regressaoLogistica(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    an.Plota3D(X_test, y_test, classifier, nome)
