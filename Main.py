from NeuralNetwork import NeuralNetwork
from MyFunctions import generateData


def main():
    in_train, out_train, in_test, out_test = generateData('mnist_src')
    myNet = NeuralNetwork(30, 0.5)
    myNet.add(len(in_train[0]))
    myNet.add(16)
    myNet.add(16)
    myNet.add(len(out_train[0]))
    myNet.train(in_train, out_train)
    print(myNet.acc(in_test, out_test))


if __name__ == '__main__':
    main()
