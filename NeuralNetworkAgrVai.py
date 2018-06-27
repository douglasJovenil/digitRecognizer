import numpy as np


class NeuralNetwork(object):
    def __init__(self, epochs, lr):
        # Define alguns parâmetros
        self.epochs = epochs
        self.lr = lr
        # Inicia o layout da rede
        self.net = []
        self.weights = []
        # Função de custo
        self.f_cost = lambda e: np.square(sum(e))/2
        # Função de erro
        self.f_error = lambda d, y: d - y
        # Função de ativação
        self.f_act = lambda x: 1/(1 + np.exp(-x))
        # Derivada da função de ativação
        self.f_act_derivative = lambda x: np.exp(-x)/np.square(1 + np.exp(-x))
        # Função para o somatório da layer + bias
        self.f_foward = lambda x, w: np.dot(w, x) + 1
        # Função para a resultado do neurônio -> f_act(f_foward) - 1 por causa do bias
        self.f_out = lambda x, w: self.f_act(self.f_foward(x, w))
        # Função do gradiente descendente
        self.f_grad_out = lambda c, x, w: c * self.f_act_derivative(self.f_foward(x, w))
        # Função de ajuste dos pesos
        self.f_delta_w_out = lambda x, g: self.lr * np.dot(g, x.T)

    def train(self, inputs, targets):
        # Inicia o custo
        cost = 0
        # Passa um vetor de linhas para colunas
        to_col = lambda x: np.array([x]).T
        # Itera as epocas
        for epoch in range(self.epochs):
            # Itera todas as entradas
            for actual_in, actual_target in zip(inputs, targets):
                # Passa as entradas e targets para um vetor de colunas
                actual_in, actual_target = to_col(actual_in), to_col(actual_target)
                # Atribui o vetor de entradas na primeira camada da rede
                self.net[0] = actual_in
                # FeedFoward
                for i in range(len(self.weights)):
                    self.net[i+1] = self.f_out(self.net[i], self.weights[i])
                # Custo da iteração -> -1 por causa do bias
                cost = self.f_cost(self.f_error(actual_target, self.net[-1] - 1))
                # Calcula o gradiente na ultima layer
                grad = self.f_grad_out(cost, self.net[-2], self.weights[-1])
                # Calcula o ajuste na ultima layer
                self.weights[-1] += self.f_delta_w_out(self.net[1], grad)
                # AJUSTA NA PENULTIMA LAYER
                p_termo = self.f_act_derivative(self.f_foward(self.net[-3], self.weights[-2]))
                #print(p_termo)
                s_termo = np.dot(self.weights[-1].T, grad)
                grad_oc = p_termo * s_termo
                ajuste_oc = self.lr * self.f_foward(self.net[0].T, grad_oc)
                self.weights[0] += ajuste_oc



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

    def show(self):
        for value in self.net:
            print(value)
