from NeuralNetwork import NeuralNetwork
from LoadMNIST import generateData
from Image import plotNum, arrayToNum
from os import system

def main():
    in_train, out_train, in_test, out_test = generateData('mnist_src')
    
    #myNet = NeuralNetwork(5, 0.5)
    myNet = NeuralNetwork()    
    myNet.load('weights_97_41.json')
    #myNet.add(len(in_train[0]))
    #myNet.add(300)
    #myNet.add(300)
    #myNet.add(len(out_train[0]))
    #myNet.train(in_train, out_train)
    
    #myNet.save('weights')
    #print(myNet.acc(in_test, out_test))
    
    system('cls')
    choose = str()
    targets = list()
    
    print('You have 10000 images to test the neural network!')
    print('quit: stop the program')
    print('show: prints the numbers that you already choose', end='\n\n')
    
    while(True):
        choose = input('Write a number between 0 and 9999: ')
        if (choose == 'quit'):
            break
        elif (choose == 'show'):
            print('you alredy choose:', end=' ')
            for value in targets: print(value, end=' ,')
            print()
        else:
            plotNum(in_test[int(choose)], myNet.query(in_test[int(choose)]).argmax())
            targets.append(arrayToNum(out_test[int(choose)]))
        
        
if __name__ == '__main__':
    system('cls')
    main()
