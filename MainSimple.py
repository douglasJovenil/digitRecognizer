import numpy as np
import scipy


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, lr):
        # Define o layout da rede
        self.in_nodes = in_nodes
        self.hid_nodes = hid_nodes
        self.out_nodes = out_nodes
        self.lr = lr
        # Inicia os pesos
        self.wih = np.random.normal(0, pow(self.hid_nodes, -0.5), (self.hid_nodes, self.in_nodes))
        self.who = np.random.normal(0, pow(self.out_nodes, -0.5), (self.out_nodes, self.in_nodes))
        # Define a funcao de ativacao
        self.act_func = lambda x: 1 / (1 + np.exp(-x))
        
        
    def query(self, in_list):
        # Converte a lista em array
        inputs = np.array([in_list]).T
        # Operacao matricial para somar os links nos neuronios escondidos
        hid_inputs = np.dot(self.wih, inputs)
        # Funcao de ativacao nos neuronios escondidos
        hid_outputs = self.act_func(hid_inputs)
        # Operacao matricial para somar os links na saida
        final_inputs = np.dot(self.who, hid_outputs)
        # Funcao de ativacao na saida
        final_outputs = self.act_func(final_inputs)
        return final_outputs
    
    
    def train(self, in_list, target_list):
        # Iteracao das epocas
        for epoch in range(100):
            # Iteracao de todas as entradas e saidas
            for actual_in, actual_target in zip(in_list, target_list):
                # Converte lista para array
                inputs = np.array([actual_in]).T
                targets = np.array([actual_target]).T
                # Iteracao do batch
                for batch in range(10):
                    # Operacao matricial e funcao de ativacao
                    hid_inputs = np.dot(self.wih, inputs)
                    hid_outputs = self.act_func(hid_inputs)
                    final_inputs = np.dot(self.who, hid_outputs)
                    final_outputs = self.act_func(final_inputs)
                # Calcula os erros
                out_errors = targets - final_outputs
                hid_errors = np.dot(self.who.T, out_errors)
                # Atualiza os pesos
                self.who += self.lr * np.dot((out_errors * final_outputs * (1 - final_outputs)), hid_outputs.T)
                self.wih += self.lr * np.dot((hid_errors * hid_outputs * (1 - hid_outputs)), inputs.T)


def main():
    inputs = [[0, 0], [1, 0], [0, 1], [1, 1],
              [1, 1], [0, 0], [1, 1], [1, 0],
              [0, 1], [1, 0], [1, 1], [0, 0],
              [0, 0], [1, 0], [0, 1], [1, 1],
              [1, 1], [0, 0], [1, 1], [1, 0],
              [0, 1], [1, 0], [1, 1], [0, 0]]
    outputs = [[0], [0], [0], [1],
               [1], [0], [1], [0],
               [0], [0], [1], [0],
               [0], [0], [0], [1],
               [1], [0], [1], [0],
               [0], [0], [1], [0]]
    myNet = NeuralNetwork(2, 2, 1, 0.01)
    #myNet.train(inputs, outputs)
    #print(myNet.query([1, 1]))


if __name__ == '__main__':
    main()
