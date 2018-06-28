import numpy as np
np.set_printoptions(suppress=True)


class NeuralNetwork(object):
    def __init__(self, epochs, lr):
        # Define alguns parametros
        self.epochs, self.lr = epochs, lr
        # Inicia o layout da rede
        self.net, self.weights = [], []
        # Funcao de ativacao
        self.f_act = lambda x: 1/(1 + np.exp(-x))
        # Funcao para o somatorio da entrada do neurônio
        self.f_z = lambda x, w: np.dot(w, x)
        # Funcao para a saida do neuronio
        self.f_a = lambda x, w: self.f_act(self.f_z(x, w))

    def train(self, inputs, targets):
        # Derivada da funcao de ativacao
        f_act_derivative = lambda x: np.exp(-x)/np.square(1 + np.exp(-x))
        # Funcao de erro
        f_error = lambda d, y: np.subtract(d, y)
        # Função para calcular o gradiente
        f_grad = lambda e, x, w: np.multiply(e, f_act_derivative(self.f_z(x, w)))
        # Função para atualizar o peso
        f_deltaW = lambda x, g: self.lr * np.dot(g, x.T)
        # Passa um vetor de linhas para colunas
        to_col = lambda x: np.array([x]).T
        # Inicia os pesos
        self._startWeights()
        # Itera as epocas
        for epoch in range(self.epochs):
            # Itera todas as entradas
            for actual_in, actual_target in zip(inputs, targets):
                # Limpa as listas de erros e gradientes
                errors, grads = [], []
                # Passa as entradas e targets para um vetor de colunas
                actual_in, actual_target = to_col(actual_in), to_col(actual_target)
                # Atribui o vetor de entradas na primeira camada da rede
                self.net[0] = actual_in
                # FeedFoward
                for i in range(len(self.weights)):
                    self.net[i+1] = self.f_a(self.net[i], self.weights[i])
                # Erro da iteracao
                errors.insert(0, f_error(actual_target, self.net[-1]))
                # Retropropagacao do custo
                grads.insert(0, f_grad(errors[-1], self.net[-2], self.weights[-1]))
                for i in range(len(self.weights)-2, -1, -1):
                    errors.insert(0, np.dot(self.weights[i+1].T, grads[0]))
                    grads.insert(0, f_grad(errors[0], self.net[i], self.weights[i]))
                # Ajuste dos pesos
                for i in range(len(self.weights)):
                    self.weights[i] += f_deltaW(self.net[i], grads[i])

    def query(self, in_list):
        # Converte a lista em array
        inputs = np.array(in_list).T
        self.net[0] = inputs
        # FeedFoward
        for i in range(len(self.weights)):
            self.net[i+1] = self.f_a(self.net[i], self.weights[i])
        return self.net[-1]

    def add(self, num_nodes):
        self.net.append(num_nodes)
        
    def _startWeights(self):
        # Define a funcao que rege como os pesos serao iniciados
        f_rand = lambda x, y: np.random.rand(x, y) - 0.5
        # Calula os pesos
        for i in range(1, len(self.net)):
            # Tamanho da layer i x tamanho da lamanho da layer i-1
            self.weights.append(f_rand(self.net[i], self.net[i-1]))
