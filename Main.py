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