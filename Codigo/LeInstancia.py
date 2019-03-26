import pandas as pirina
import numpy as np
import matplotlib.pyplot as pargo
from sklearn.preprocessing import Imputer
class LeInstancia:
    baseDados = None

    def __init__(self, nomeArquivo):
        self.baseDados = pirina.read_csv(nomeArquivo, sep=',')
        #tirando os tbds
        self.baseDados.replace('?', float('Nan'), inplace=True)
        # lia avaliaçao dos usuarios como uma object(string)
        self.baseDados = self.baseDados.dropna()
        self.baseDados['b'] = pirina.to_numeric(self.baseDados['b'])
        self.baseDados['g'] = pirina.to_numeric(self.baseDados['g'])
        self.baseDados['r'] = pirina.to_numeric(self.baseDados['r'])

        # self.baseDados['avaliacao-usuarios'] = pirina.to_numeric(self.baseDados['avaliacao-usuarios'])
        # print(self.baseDados['avaliacao-usuarios'])


    def normaliza(self, nomeAtributo):
        self.baseDados[nomeAtributo] = [(x - self.baseDados[nomeAtributo].min())
                                                 / (self.baseDados[nomeAtributo].max()
                                                    - self.baseDados[nomeAtributo].min())
                                                    for x in self.baseDados[nomeAtributo]]


    def desnormaliza(self, nomeAtributo):
        self.baseDados[nomeAtributo] = \
            [x * (self.baseDados[nomeAtributo].max() - self.baseDados[nomeAtributo].min())
             + self.baseDados[nomeAtributo].min() for x in self.baseDados[nomeAtributo] ]


    def normalizaTodos(self):
        self.normaliza('g')
        self.normaliza('r')
        self.normaliza('b')
        self.normaliza('pele')
        # self.normaliza('vendas')


    def matrizCorrelacao(self):
        # self.baseDados.replace('tbd',float('Nan'),inplace=True)
        with open('saidaMatrizCorrelacao','w') as arquivo:
            arquivo.write(str('Matriz de correlação: ')+'\n')
            arquivo.write(str(self.baseDados.dropna().corr(method='pearson')))
            #arquivo.write(str(self.baseDados.dropna().groupby(by='genero').corr(method='pearson')))


    def matrizCorrelacaoPorAlgo(self, oQueAgrupar):
        with open('saidaMatrizCorrelacaoGenero','w') as arquivo:
            # print('Matriz de correlação: '+' '+oQueAgrupar)
            #print(self.baseDados.dropna().corr(method='pearson'))
            arquivo.write(str(self.baseDados.dropna().groupby(by=oQueAgrupar).corr(method='pearson')))
        # self.baseDados.dropna().boxplot()


    def boxplot(self):
        # self.normalizaTodos()
        self.baseDados.boxplot()
        pargo.savefig('boxPlot.png')
        pargo.close()


    def descreve(self):
        with open('saidaDescribe','w') as arquivo:
            arquivo.write(str(self.baseDados.describe()))


    def retornaCoeficiente(self, basePorGenero, dividendo, divisor):
        x = basePorGenero[1].dropna()
        # print('oieeeee',x)
        return sum(x[dividendo]) / sum(x[divisor])


    def completaDadosFaltando(self):
        self.completaNumUsuarios()
        self.completaNumCriticos()
        self.completaNotasCriticos()
        self.completaNotasUsuarios()


    def completaNotasCriticos(self):
        L = list()
        for base in self.baseDados[['genero', 'avaliacao-criticos']].groupby(["genero"]):
            l = list()
            # print(np.mean(base[1]['avaliacao-usuarios']))
            # break
            l.append(base[0])
            l.append(np.ceil(np.median(base[1]['avaliacao-criticos'])))
            # print(l)
            L.append(l)
        L = dict(L)
        for i in range(494):
            if np.isnan(self.baseDados.get_value(i, 'avaliacao-criticos')):
                self.baseDados.set_value(i, 'avaliacao-criticos', L[self.baseDados.get_value(i,'genero')])


    def completaNotasUsuarios(self):
        L = list()
        for base in self.baseDados[['genero', 'avaliacao-usuarios']].groupby(["genero"]):
            l = list()
            # print(np.mean(base[1]['avaliacao-usuarios']))
            # break
            l.append(base[0])
            l.append(np.round(np.median(base[1]['avaliacao-usuarios']),decimals=1))
            # print(l)
            L.append(l)
        L = dict(L)
        for i in range(494):
            if np.isnan(self.baseDados.get_value(i, 'avaliacao-usuarios')):
                self.baseDados.set_value(i, 'avaliacao-usuarios', L[self.baseDados.get_value(i,'genero')])


        # dados = dados.dropna(how='all').dropna(subset=['Escolaridade'])
        # print(dados)

    #     oQueCompletar = 'avaliacao-usuarios'
    #     l = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='vendas', filtro='plataforma')
    #     l2 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='numero-usuarios', filtro='plataforma')
    #     l3 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='vendas', filtro='genero')
    #     l4 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='numero-usuarios', filtro='genero')
    #     l5 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='numero-criticos', filtro='plataforma')
    #     l6 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='numero-criticos', filtro='genero')
    #     l7 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='avaliacao-criticos', filtro='plataforma')
    #     l8 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='avaliacao-criticos', filtro='genero')
    #     k = [(x + y + w + z + a + b + c + d) / 8 for x, y, w, z, a ,b, c, d in zip(l, l2, l3, l4, l5, l6, l7, l8)]
    #     self.colocaListaDataFrame('avaliacao-usuarios', k)


    # def completaNotasCriticos(self):
    #     oQueCompletar = 'avaliacao-criticos'
    #     l = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='vendas', filtro='plataforma')
    #     l2 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='numero-usuarios', filtro='plataforma')
    #     l3 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='vendas', filtro='genero')
    #     l4 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='numero-usuarios', filtro='genero')
    #     l5 = self.completaDados(oQueCompletar=oQueCompletar,
    #                            BaseadoEmQue='numero-criticos', filtro='plataforma')
    #     l6 = self.completaDados(oQueCompletar=oQueCompletar,
    #                             BaseadoEmQue='numero-criticos', filtro='genero')
    #     k = [(x + y + w + z + a + b) / 6 for x, y, w, z, a ,b in zip(l, l2, l3, l4, l5, l6)]
    #     self.colocaListaDataFrame('avaliacao-criticos', list(np.round(k)))


    def completaNumCriticos(self):
        l = self.completaDados(oQueCompletar='numero-criticos',
                               BaseadoEmQue='vendas', filtro='plataforma')
        l2 = self.completaDados(oQueCompletar='numero-criticos',
                               BaseadoEmQue='numero-usuarios', filtro='plataforma')
        l3 = self.completaDados(oQueCompletar='numero-criticos',
                               BaseadoEmQue='vendas', filtro='genero')
        l4 = self.completaDados(oQueCompletar='numero-criticos',
                                BaseadoEmQue='numero-usuarios', filtro='genero')
        k = [(x + y + w + z) / 4 for x, y, w, z in zip(l, l2, l3, l4)]
        self.colocaListaDataFrame('numero-criticos', list(np.ceil(k)))


    def completaNumUsuarios(self):
        l = self.completaDados(oQueCompletar = 'numero-usuarios',
                                       BaseadoEmQue = 'vendas',filtro = 'plataforma')
        l2 = self.completaDados(oQueCompletar = 'numero-usuarios',
                                       BaseadoEmQue = 'vendas',filtro = 'genero')
        k = [(x+y)/2 for x,y in zip(l, l2)]
        # self.baseDados['numero-usuarios'] = np.ceil(k)
        self.colocaListaDataFrame('numero-usuarios', list(np.ceil(k)))
        # self.completaDados(oQueCompletar = 'numero-criticos',
        #                                BaseadoEmQue = 'vendas',filtro = 'genero')
        # self.completaDados(oQueCompletar = 'avaliacao-usuarios',
        #                                BaseadoEmQue = 'numero-usuarios',filtro = 'genero')
        # self.completaDados(oQueCompletar = 'avaliacao-criticos',
        #                                BaseadoEmQue = 'numero-criticos',filtro = 'genero')


    def colocaListaDataFrame(self, qualAtributo, lista):
        for i in range(494):
            if np.isnan(self.baseDados.get_value(i, qualAtributo)):
                self.baseDados.set_value(i, qualAtributo, lista.pop(0))



    def retornaCoeficiente(self, basePorGenero, dividendo, divisor):
        x = basePorGenero[1].dropna()
        # print('oieeeee',x)
        return sum(x[dividendo]) / sum(x[divisor])


    def completaDados(self, oQueCompletar, BaseadoEmQue, filtro  ):
        baseSemVazioAgrupada = self.baseDados[[oQueCompletar ,BaseadoEmQue ,filtro ]].groupby(by=filtro)
        coeficientes = list()
        for base in baseSemVazioAgrupada:
            l = list()
            l.append(base[0])
            l.append(self.retornaCoeficiente(base, oQueCompletar , BaseadoEmQue ))
            coeficientes.append(l)
        coeficientes = dict(coeficientes)
        # print(coeficientes)
        resultados = list()
        for i in range(494):
            if np.isnan(self.baseDados.get_value(i, oQueCompletar)):
                j = coeficientes[self.baseDados.get_value(i,filtro )] * self.baseDados.get_value(i,BaseadoEmQue )
                # k = self.retornaCoeficiente(basePorGenero, 'avaliacao-usuarios',
                #                             'numero-usuarios') * self.baseDados.get_value(i, 'numero-usuarios')
                # self.baseDados.set_value(i, oQueCompletar , (j))
                resultados.append(j)
        return resultados


    def completaAvaliacaoCriticos(self):
        baseSemVazioAgrupada = self.baseDados[['avaliacao-criticos', 'numero-criticos', 'genero']].groupby(by='genero')
        for basePorGenero in baseSemVazioAgrupada:
            for i in range(494):
                if np.isnan(self.baseDados.get_value(i, 'avaliacao-criticos')):
                    j = self.retornaCoeficiente(basePorGenero, 'avaliacao-criticos',
                                                'numero-criticos') * self.baseDados.get_value(i, 'numero-criticos')
                    self.baseDados.set_value(i, 'avaliacao-criticos', j)


    def completaNumeroUsuarios(self):
        baseSemVazioAgrupada = self.baseDados[['numero-usuarios','vendas','genero']].groupby(by='genero')
        for basePorGenero in baseSemVazioAgrupada:
            for i in range(494):
                if np.isnan(self.baseDados.get_value(i, 'numero-usuarios')):
                    j = np.ceil(self.retornaCoeficiente(basePorGenero,'numero-usuarios','vendas') * self.baseDados.get_value(i,'vendas'))
                    self.baseDados.set_value(i, 'numero-usuarios', j)
        # print(self.baseDados)


    def completaNumeroCriticos(self):
        baseSemVazioAgrupada = self.baseDados[['numero-criticos','numero-usuarios','genero']].groupby(by='genero')
        # print(baseSemVazioAgrupada.head(500))
        for basePorEditora in baseSemVazioAgrupada:
            # print(basePorEditora)
            for i in range(494):
                if np.isnan(self.baseDados.get_value(i, 'numero-criticos')):
                    j = np.ceil(self.retornaCoeficiente(basePorEditora,'numero-criticos','numero-usuarios')
                                 * self.baseDados.get_value(i,'numero-usuarios'))
                    self.baseDados.set_value(i, 'numero-criticos', j)
        # print(self.baseDados)



    def removeOutlier(self, atributo):
        # dados = pd.read_csv("dados.csv")[["Irmaos"]].dropna()
       # obtendo q1 e q3
        dados = self.baseDados
        q1, q3 = np.percentile(dados, [25, 75])
        # IQR e limites
        iqr = q3 - q1
        min = q1 - iqr * 1.5
        max = q3 + iqr * 1.5
        # imprime os dados removendo os outliers
        # print('outliers:')
        # print(dados[(dados > min) & (dados < max)])
        # print(np.round(len(dados[(dados < min) | (dados > max)])/len(self.baseDados[atributo]),8)*100, '%')
        # for i in range(len(self.baseDados)):
        #     if (self.baseDados[atributo].get_value(i) < min) or (self.baseDados[atributo].get_value(i) > max):
        #         self.baseDados.__delitem__(i)
        self.baseDados=dados[(dados > min) & (dados < max)].dropna()



    def geraCsv(self):
        self.baseDados.to_excel('saidaTrem.xlsx')