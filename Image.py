import matplotlib.pyplot as plt

def _toMultDimArray(mnist_num):
    result = []
    aux = []

    for i, value in enumerate(mnist_num):
        if(i != 0): aux.append(value)
        if (not(i % 28) and i != 0): 
            result.append(aux)
            aux = []
    return result

def arrayToNum(array):
    return array.index(1)

def plotNum(mnist_num, output_value):
    plt.imshow(_toMultDimArray(mnist_num), cmap=plt.cm.binary)
    print(f'The neural network thinks the number is: {output_value}', end='\n\n')
    plt.show()