from NeuralNetwork import NeuralNetwork
from LoadMNIST import generateData
import matplotlib.pyplot as plt


def main():
    in_train, out_train, in_test, out_test = generateData('mnist_src')
    myNet = NeuralNetwork(1, 0.5)
    myNet.load('weights.json')
    #myNet.add(len(in_train[0]))
    myNet.add(len(in_test[0]))
    myNet.add(3)
    myNet.add(4)
    myNet.add(len(out_test[0]))
    #myNet.add(len(out_train[0]))
    #myNet.train(in_train, out_train)
    myNet.train(in_test[0:10], out_test[:10])
    myNet.saveNet('weights')
    #print(myNet.acc(in_test, out_test))

def toMultDimArray(mnist_num):
    result = []
    aux = []

    for i, value in enumerate(mnist_num):
        if(i != 0): aux.append(value)
        if (not(i % 28) and i != 0): 
            result.append(aux)
            aux = []
    return result

def printImg(i):
    imshow(toMultDimArray(in_test[i]), cmap=plt.cm.binary)
    print(out_test[i])
    
def saveWeights():
    with open("weights.csv", 'w') as file:
        file.write(f'layers=len(data);')
        for layer in layers:
            file.write(f'layer;')
            for neuron in layer:
                file.write(f'neuron;')
                for weight in neuron:
                    file.write(f'{weight};')




if __name__ == '__main__':
    main()
