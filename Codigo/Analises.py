import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import roc_curve, auc



def plot_confusion_matrix(cm, classes, nome, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Matriz de Confusao '+nome)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('verdadeira')
    plt.xlabel('prevista')
    # plt.show()
    # plt.close()

def Plota3D(X, Y, svc, nome):
    z = lambda x, y: (- svc.intercept_[0] - svc.coef_[0][0]*x - svc.coef_[0][1]) / svc.coef_[0][2]
    tmp = np.linspace( 0, 255, 255)
    x, y = np.meshgrid( tmp, tmp)
    # Plot stuff.
    X = np.transpose(X)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    # ax.plot3D(X[0], X[1], X[2], 'ob',  )
    ax.scatter(X[0], X[1], X[2], 'o', c= Y )
    # ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
    ax.plot_surface(x, y, z(x,y))

    plt.xlabel('Blue')
    plt.ylabel('Green')
    # plt.zlabel('Blue')
    ax.set_zlabel('Red')
    plt.title('Gráfico da Classificação '+ nome)
    plt.legend()
    plt.savefig('GraficoClassificacao'+ nome+'.pdf')
    # plt.savefig(Gráfico da Classificação '+ nome)
    plt.close()


def AUC(predictions, actual, nome):
    false_positive_rate, true_positive_rate, thresholds = roc_curve([x == 1 for x in actual] , [x==1 for x in predictions])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('(ROC) Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.xlabel('Taxa de Falso Negativo')
    plt.savefig('ROC'+nome+'.png')
    plt.close()
