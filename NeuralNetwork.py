import numpy as np
np.set_printoptions(suppress=True)


class NeuralNetwork(object):
    def __init__(self, epochs=100, lr=0.01):
        self.epochs, self.lr = epochs, lr
        self.net, self.weights = [], []
        # Funcao de ativacao, entrada e saida do neuronio
        self.f_act = lambda x: np.divide(1, np.add(1, np.exp(-x)))
        self.f_z = lambda x, w: np.dot(w, x)
        self.f_a = lambda x, w: self.f_act(self.f_z(x, w))

    # Treina a rede
    def train(self, inputs, targets):
        # Derivada de f_act, funcoes de erro, gradiente e atualizar o peso
        f_act_derivative = lambda x: np.divide(np.exp(-x), np.square(np.add(1, np.exp(-x))))
        f_error = lambda d, y: np.subtract(d, y)
        f_grad = lambda e, x, w: np.multiply(e, f_act_derivative(self.f_z(x, w)))
        f_deltaW = lambda x, g: np.multiply(self.lr, np.dot(g, x.T))
        # Passa um vetor de linhas para colunas
        to_col = lambda x: np.array([x]).T
        self._startWeights()

        for epoch in range(self.epochs):
            for actual_in, actual_target in zip(inputs, targets):
                errors, grads = [], []
                actual_in, actual_target = to_col(actual_in), to_col(actual_target)
                self.net[0] = actual_in
                # FeedFoward
                for i in range(len(self.weights)):
                    self.net[i+1] = self.f_a(self.net[i], self.weights[i])
                # Erro na layer de saida
                errors.insert(0, f_error(actual_target, self.net[-1]))
                # Retropropagacao do custo
                grads.insert(0, f_grad(errors[-1], self.net[-2], self.weights[-1]))
                for i in range(len(self.weights)-2, -1, -1):
                    errors.insert(0, np.dot(self.weights[i+1].T, grads[0]))
                    grads.insert(0, f_grad(errors[0], self.net[i], self.weights[i]))
                # Ajuste dos pesos
                for i in range(len(self.weights)):
                    self.weights[i] += f_deltaW(self.net[i], grads[i])
            print(epoch)

    # Passa o parâmetro in_list como entrada e retorna a saída da rede
    def query(self, in_list):
        inputs = np.array(in_list).T
        self.net[0] = inputs
        for i in range(len(self.weights)):
            self.net[i+1] = self.f_a(self.net[i], self.weights[i])
        return self.net[-1]

    # Retorna a precisao da rede
    def acc(self, in_list, out_list):
        corrects = 0
        for i, (in_value, out_value) in enumerate(zip(in_list, out_list)):
            if self.query([in_value]).argmax() == out_value.index(max(out_value)):
                corrects += 1
        return corrects/(i+1)*100

    # Adiciona uma layer com num_nodes de altura
    def add(self, num_nodes):
        self.net.append(num_nodes)

    # Inicia os pesos da rede
    def _startWeights(self):
        # Define a funcao que rege como os pesos serao iniciados
        f_rand = lambda x, y: np.subtract(np.random.rand(x, y), 0.5)
        # Calula os pesos
        for i in range(1, len(self.net)):
            # Tamanho da layer i x tamanho da lamanho da layer i-1
            self.weights.append(f_rand(self.net[i], self.net[i-1]))
