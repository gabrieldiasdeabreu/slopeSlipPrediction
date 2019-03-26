from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def plotaSilueta(silueta):
    x = range(2,12)
    plt.plot(x,silueta)
    plt.savefig('plotSilueta.pdf')
    plt.close()

def clusteriza(df, meses):
    with open('gruposSugeridos.csv', 'w') as arq:
        silueta= list()
        clusters = list()
        arq.write('numGrupos')
        # for mes in meses:
        #     arq.write(','+ mes)
        arq.write(','+ 'coefSilueta')
        arq.write('\n')
        for numClusters in range(11,12):
            kmeans = KMeans(n_clusters=numClusters).fit(df)
            y_pred = KMeans(n_clusters=numClusters).fit_predict(df)
            clusters.append(y_pred)
            arq.write(str(numClusters))
            # for i in y_pred:
            #     arq.write(','+str(i))
            coefSil = silhouette_score(df,kmeans.labels_ , metric='euclidean')
            arq.write(','+str(coefSil))
            silueta.append(coefSil)
            arq.write('\n')
        # plotaSilueta(silueta)
        return clusters


    
def pca(df):
    reduced_data = PCA(n_components=2)
    nova = reduced_data.fit_transform(df)
    # nova = nova.transpose()
    return nova

def plota(nova, c, cores):
    for x, y in zip(nova[0], nova[1]):
        plt.scatter(x,y)
        # print(cor)
    plt.legend()
    plt.title('pcaRiscoDeslizamento')
    plt.show()


names=['jan', 'fev', 'mar', 'abr', 'mai','jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
c = ['red','blue', 'green', 'yellow', 'pink', 'darkBlue', 'brown', 'blueviolet', 'black', 'aquamarine', 'antiquewhite', 'bisque']
        # 'np.arange(1, len(names)+1 )
df = pd.read_csv('baseDadosDeslizamento.csv')
# print(df)
pcaFeito = pca(df)
print(pcaFeito)
clusters = None# clusteriza(pcaFeito, names)
plota(pcaFeito.transpose(), clusters, c)
#, names)
# plota(, c)

# print(df)
# print(pca.components_)
# n_digits = 5
# dic = {c:names}
# print(dic)

# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
           # extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           # cmap=plt.cm.Paired,
           # aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
            # marker='x', s=169, linewidths=3,
            # color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          # 'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
