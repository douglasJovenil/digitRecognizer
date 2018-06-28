from NeuralNetwork import NeuralNetwork
from mnist import MNIST


def main():
    mndata = MNIST('mnist_src')
    train_img, train_lbl = mndata.load_training()
    test_img, test_lbl = mndata.load_testing()
    inputs = [normalize(train_img[i]) for i in range(len(train_img))]
    outputs = [numToArray(train_lbl[i]) for i in range(len(train_lbl))]
    myNet = NeuralNetwork(10, 0.05)
    myNet.add(len(train_img[0]))
    myNet.add(16)
    myNet.add(16)
    myNet.add(10)
    myNet.startWeights()
    myNet.train(inputs, outputs)
    print(myNet.query([test_img[1]]))


def numToArray(num):
    retorno = [0 for x in range(10)]
    retorno[num] = 1
    return retorno


def normalize(x):
    normalize = lambda x: (x - 0)/(255 - 0)
    for i in range(len(x)):
        x[i] = normalize(x[i])
    return x


if __name__ == '__main__':
    main()
=======
from NeuralNetworkAgrVai import NeuralNetwork


def main():
    inputs = [[0, 0], [1, 0], [0, 1], [1, 1],
              [1, 1], [0, 0], [1, 1], [1, 0],
              [0, 1], [1, 0], [1, 1], [0, 0],
              [0, 0], [1, 0], [0, 1], [1, 1],
              [1, 1], [0, 0], [1, 1], [1, 0],
              [0, 1], [1, 0], [1, 1], [0, 0]]
    outputs = [[0, 1], [0, 1], [0, 1], [1, 0],
               [1, 0], [0, 0], [1, 0], [0, 0],
               [0, 0], [0, 0], [1, 0], [0, 0],
               [0, 0], [0, 0], [0, 0], [1, 0],
               [1, 0], [0, 0], [1, 0], [0, 0],
               [0, 0], [0, 0], [1, 0], [0, 0]]
    #inputs = [[1, 0]]
    #outputs = [[1]]
    inputs = [[0,0], [0,1], [1,0], [1,1] ]
    outputs = [ [0], [1],[1],[0] ]

    myNet = NeuralNetwork(13000, 1)
    myNet.add(2)
    myNet.add(4)
    myNet.add(1)
    myNet.startWeights()
    myNet.train(inputs, outputs)
    print(myNet.query([[0, 0]]))
    print(myNet.query([[0, 1]]))
    print(myNet.query([[1, 0]]))
    print(myNet.query([[1, 1]]))


if __name__ == '__main__':
    main()
