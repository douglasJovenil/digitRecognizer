from NeuralNetwork import NeuralNetwork


def main():
    #inputs = [[0, 0], [1, 0], [0, 1], [1, 1],
    #          [1, 1], [0, 0], [1, 1], [1, 0],
    #          [0, 1], [1, 0], [1, 1], [0, 0],
    #          [0, 0], [1, 0], [0, 1], [1, 1],
    #         [1, 1], [0, 0], [1, 1], [1, 0],
    #         [0, 1], [1, 0], [1, 1], [0, 0]]
    #outputs = [[0], [0], [0], [1],
    #           [1], [0], [1], [0],
    #           [0], [0], [1], [0],
    #           [0], [0], [0], [1],
    #           [1], [0], [1], [0],
    #           [0], [0], [1], [0]]
    inputs = [[1, 1]]
    outputs = [[1]]
    myNet = NeuralNetwork(2, 1, 0.1)
    myNet.add(3)
    myNet.add(2)
    myNet.startWeights()
    myNet.train(inputs, outputs)
    myNet.query([1, 1])


if __name__ == '__main__':
    main()
