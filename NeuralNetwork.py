import numpy as np
np.set_printoptions(suppress=True)


class NeuralNetwork(object):
    def __init__(self, epochs, lr):
        # Define alguns parametros
        self.epochs = epochs
        self.lr = lr
        # Inicia o layout da rede
        self.net = []
        self.weights = []
        self.errors = []
        self.grads = []
        # Funcao de custo
        self.f_cost = lambda e: np.square(sum(e))/2
        # Funcao de erro
        self.f_error = lambda d, y: d - y
        # Funcao de ativacao
        self.f_act = lambda x: 1/(1 + np.exp(-x))
        # Derivada da funcao de ativacao
        self.f_act_derivative = lambda x: np.exp(-x)/np.square(1 + np.exp(-x))
        # Funcao para o somatorio da layer
        self.f_foward = lambda x, w: np.dot(w, x)
        # Funcao para a resultado do neuronio -> f_act(f_foward)
        self.f_out = lambda x, w: self.f_act(self.f_foward(x, w))
        # Funcao do gradiente descendente
        self.f_grad_out = lambda c, x, w: c * self.f_act_derivative(self.f_foward(x, w))
        # Funcao de ajuste dos pesos
        self.f_delta_w_out = lambda x, g: self.lr * np.dot(g, x.T)

    def train(self, inputs, targets):
        # Passa um vetor de linhas para colunas
        to_col = lambda x: np.array([x]).T
        # Itera as epocas
        for epoch in range(self.epochs):
            # Itera todas as entradas
            for actual_in, actual_target in zip(inputs, targets):
                # Limpa as listas de erros e gradientes
                self.error, self.grad = [], []
                # Passa as entradas e targets para um vetor de colunas
                actual_in, actual_target = to_col(actual_in), to_col(actual_target)
                # Atribui o vetor de entradas na primeira camada da rede
                self.net[0] = actual_in
                # FeedFoward
                for i in range(len(self.weights)):
                    self.net[i+1] = self.f_out(self.net[i], self.weights[i])
                # Erro da iteracao
                self.error.insert(0, self.f_error(actual_target, self.net[-1]))
                # Retropropagacao do custo
                self.grad.insert(0, np.multiply(self.error[-1], self.f_act_derivative(self.f_foward(self.net[-2], self.weights[-1]))))
                for i in range(len(self.weights)-2, -1, -1):
                    self.error.insert(0, np.dot(self.weights[i+1].T, self.grad[0]))
                    self.grad.insert(0, np.multiply(self.error[0], self.f_act_derivative(self.f_foward(self.net[i], self.weights[i]))))
                # Ajuste dos pesos
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * np.dot(self.grad[i], self.net[i].T)
            print(epoch)

    def query(self, in_list):
        # Converte a lista em array
        inputs = np.array(in_list).T
        self.net[0] = inputs
        # FeedFoward
        for i in range(len(self.weights)):
            self.net[i+1] = self.f_out(self.net[i], self.weights[i])

        return self.net[-1]

    def startWeights(self):
        # Define a funcao que rege como os pesos serao iniciados
        f_rand = lambda x, y: np.random.rand(x, y) - 0.5
        # Calula os pesos
        for i in range(1, len(self.net)):
            # Tamanho da layer i x tamanho da lamanho da layer i-1
            self.weights.append(f_rand(self.net[i], self.net[i-1]))

    def add(self, num_nodes):
        self.net.append(num_nodes)
=======
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