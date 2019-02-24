<h1 align="center">Digit Recognizer</h1>
<p align="center"><img src="https://i.imgur.com/NxJZ6OU.png" width=481px height=257></p>

### Resume
Digit Recognizer is a library where the algorithm Multilayer Perceptron is implemented with intent to teach a neural network learn how to recognize handwritten numbers based on MNIST dataset. It's important remember, this is a basic implementation where there's NO OPTIMIZATION at algorithm, in other words, the training is SLOW but the results are satisfactory.<br>
The book used as support material is: **HAYKIN, Simon. Redes Neurais: Princípios e Prática.**

##### What is MNIST?

A definition according to <span><i><a href="http://yann.lecun.com/exdb/mnist">MNIST database of handwritten digits</a></i></span>:<br>
Is a database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image (28 x 28). It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

### Requeriments
![matplotlib](https://img.shields.io/badge/matplotlib-3.0.2-blue.svg?style=popout-square)
![mnist](https://img.shields.io/badge/mnist-0.2.2-blue.svg?style=popout-square)
![python-mnist](https://img.shields.io/badge/python_mnist-0.6-blue.svg?style=popout-square)
![cycler](https://img.shields.io/badge/cycler-%E2%89%A50.10-blue.svg?style=popout-square)
![six](https://img.shields.io/badge/six-%E2%89%A51.5-blue.svg?style=popout-square)
![kiwisolver](https://img.shields.io/badge/kiwisolver-%E2%89%A51.0.1-blue.svg?style=popout-square)
![numpy](https://img.shields.io/badge/numpy-%E2%89%A51.10.0-blue.svg?style=popout-square)
![python-dateutil](https://img.shields.io/badge/python_dateutil-%E2%89%A52.1-blue.svg?style=popout-square)
![pytz](https://img.shields.io/badge/pytz-%E2%89%A52011k-blue.svg?style=popout-square)
![numpy](https://img.shields.io/badge/Numpy-%E2%89%A51.10.0-blue.svg?style=popout-square)
![pyparsing](https://img.shields.io/badge/pyparsing-%E2%89%A52.0.1|%E2%89%A02.1.6|%E2%89%A02.1.2|%E2%89%A02.0.4-blue.svg?style=popout-square)
![setuptools](https://img.shields.io/badge/setuptools-any-blue.svg?style=popout-square)

### Setup
```bash
$ pip install -r requirements.txt
```

### Methods & Files
A brief explanation of each method of this repository.
  - <em>NeuralNetwork.py</em>
    - The constructor <span><i>NeuralNetwork<i></span> receives as argument the number of epochs and learning rate. e.g.
    ```Python
    NeuralNetwork(10, 0.5)
    ```
    - Add a layer to the network with quantity of neurons passed as argument.
    ```Python
    def add(self, num_nodes)
    ```
    - Receives the values of inputs and outputs of training dataset.
    ```Python
    def train(self, inputs, targets)
    ```
    - Receives a value and return the output of neural network.
    ```Python
    def query(self, in_list)
    ```
    - Receives an test input and output and return the precision of network.
    ```Python
    def acc(self, in_list, out_list)
    ```
    - Receives a name and save the neural network at the working directory.
    ```Python
    def save(self, file_name)
    ```
    - Receives a path of saved neural network and load into memory.
    ```Python
    def load(self, path_json)
    ```
  - <em>Image.py</em>
    - Receives an output of mnist as array and return the corresponding value.
    ```Python
    def arrayToNum(array)
    ```
    - Receives an mninst array and the value generated for the network, then plots the image and print the value.
    ```Python
    plotNum(mnist_num, output_value)
    ```
    
  - <em>LoadMNIST.py</em>
    - Receives the path of MNIST files and return the arrays of training and testing normalizeds.
    ```Python
    def generateData(path)
    ```


