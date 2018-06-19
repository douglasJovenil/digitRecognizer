import numpy as np
import scipy


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, lr):
        # Define o layout da rede
        self.in_nodes = in_nodes
        self.hidden = []
        self.weights = []
        self.out_nodes = out_nodes
        self.lr = lr
        # Define as funcoes suas derivadas
        self.act_func = lambda x: 1 / (1 + np.exp(-x))
        self.act_derivative = lambda x: np.exp(-x)/(np.square(1+np.exp(-x)))
        self.error_func = lambda d, y: np.square(d - y)/2
        self.error_derivative = lambda x: x
        
    
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
        # Operacao matricial para somar os links nos neuronios ocultos
        hid_inputs = np.dot(self.wih, inputs)
        # Funcao de ativacao nos neuronios ocultos
        hid_outputs = self.act_func(hid_inputs)
        # Operacao matricial para somar os links na saida
        final_inputs = np.dot(self.who, hid_outputs)
        # Funcao de ativacao na saida
        final_outputs = self.act_func(final_inputs)
        return final_outputs
    
    
    def train(self, in_list, target_list):
        # Iteracao das epocas
        for epoch in range(1):
            # Iteracao de todas as entradas e saidas
            for actual_in, actual_target in zip(in_list, target_list):
                # Converte lista para array
                inputs = np.array([actual_in]).T
                targets = np.array([actual_target]).T
                # Operacao matricial e aplicacao da funcao de ativacao entre a primeira camada oculta e camada de entrada
                self.hidden[0] = np.dot(self.weights[0], inputs)
                print(self.weights[0])
                print(inputs)
                print(self.hidden[0])
                print('\n')
                # Operacao matricial e aplicacao da funcao de ativacao nas camadas ocultas
                for i in range(len(self.hidden) - 1):
                    self.hidden[i] = self.act_func(np.dot(self.weights[i], self.hidden[i]))
                # Operacao matricial e aplicacao da funcao de ativacao entre a ultima camada oculta e camada de saida
                self.out_nodes = self.act_func(np.dot(self.weights[-1], self.hidden[-1]))
        
            #print(self.out_nodes)
                #    hid_inputs = np.dot(self.wih, inputs)
                #    hid_outputs = self.act_func(hid_inputs)
                #    final_inputs = np.dot(self.who, hid_outputs)
                #    final_outputs = self.act_func(final_inputs)
                # Calcula os erros
                #out_errors = targets - final_outputs
                #hid_errors = np.dot(self.who.T, out_errors)
                # Atualiza os pesos
                #self.who += self.lr * np.dot((out_errors * final_outputs * (1 - final_outputs)), hid_outputs.T)
                #self.wih += self.lr * np.dot((hid_errors * hid_outputs * (1 - hid_outputs)), inputs.T)


def main():
    inputs = [[1, 1]]
    outputs = [[1]]
    myNet = NeuralNetwork(2, 2, 1, 0.01)
    myNet.add(2)
    myNet.startWeights()
    myNet.train(inputs, outputs)
    #print(myNet.query([1, 1]))


if __name__ == '__main__':
    main()
