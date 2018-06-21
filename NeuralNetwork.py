import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self, in_nodes, out_nodes, lr):
        # Define o layout da rede
        self.in_nodes = in_nodes
        self.hidden = []
        self.weights = []
        self.error = []
        self.out_nodes = out_nodes
        self.lr = lr
        # Define as funcoes suas derivadas
        self.act_func = lambda x: 1 / (1 + np.exp(-x))
        self.error_func = lambda d, y: np.square(d - y)/2
        self.error_derivative = lambda x, y, e: -np.dot(np.square(1+np.exp(-y)), (np.exp(-y).dot(x.T)).T.dot(e).T)
        #self.error_derivative = lambda x, y, e: self.lr * np.dot(e * y * (1 - y),x.T)

    def add(self, n_nodes):
        # Adiciona uma camada oculta
        self.hidden.append(n_nodes)

    def startWeights(self):
        # Define a funcao que rege como os pesos serao iniciados
        normal_dist = lambda in_layer, out_layer: np.random.normal(0, pow(out_layer, -0.5), (out_layer, in_layer))
        # Calcula o peso entre a primeira camada oculta e a camada de entrada
        self.weights.append(normal_dist(self.in_nodes, self.hidden[0]))
        # Calcula o peso para as camadas ocultas
        for i in range(len(self.hidden) - 1):
            self.weights.append(normal_dist(self.hidden[i], self.hidden[i+1]))
        # Calcula o peso entre a ultima camada oculta e a saida
        self.weights.append(normal_dist(self.hidden[-1], self.out_nodes))

    def query(self, in_list):
        # Converte a lista em array
        inputs = np.array([in_list]).T
        # Operacao matricial na primeira camada
        self.hidden[0] = self.act_func(np.dot(self.weights[0], inputs))
        # Operacao matricial nas camadas ocultas
        for i in range(1, len(self.hidden)):
            self.hidden[i] = self.act_func(np.dot(self.weights[i], self.hidden[i-1]))
        # Operacao matricial na ultima camada
        self.out_nodes = self.act_func(np.dot(self.weights[-1], self.hidden[-1]))
        return self.out_nodes

    def train(self, in_list, target_list):
        for epoch in range(100):
            # Iteracao de todas as entradas e saidas
            for actual_in, actual_target in zip(in_list, target_list):
                self.error = []
                # Converte lista para array
                inputs = np.array([actual_in]).T
                targets = np.array([actual_target]).T
                # Operacao matricial na primeira camada
                self.hidden[0] = self.act_func(np.dot(self.weights[0], inputs))
                # Operacao matricial nas camadas ocultas
                for i in range(1, len(self.hidden)):
                    self.hidden[i] = self.act_func(np.dot(self.weights[i], self.hidden[i-1]))
                # Operacao matricial na ultima camada
                self.out_nodes = self.act_func(np.dot(self.weights[-1], self.hidden[-1]))
                # Calcula os erros na ultima camada
                self.error.insert(0, self.error_func(targets, self.out_nodes))
                # Calcula os erros nas camadas ocultas
                #for i in range(len(self.weights) - 1, -1, -1):
                    #self.error.insert(0, np.dot(self.weights[i].T, self.error[0]))
                # Atualiza os pesos na ultima camada
                self.weights[-1] += self.error_derivative(self.hidden[-1], self.out_nodes, self.error[-1])
                # Atualiza os pesos nas camadas ocultas
                #for i in range(len(self.weights) - 2, 0, -1):
                    #self.weights[i] += self.error_derivative(self.hidden[i-1], self.hidden[i], self.error[i+1])
                # Atualiza os pesos na primeira camada
                #self.weights[0] += self.error_derivative(inputs, self.hidden[0], self.error[1])
            plt.plot(epoch, self.error[-1], 'ro')
