# Digit Recognizer

### Setup
```bash
$ pip install -r requirements.txt
$ python Main.py
```

- O construtor da classe [`NeuralNetwork`](https://github.com/douglasJovenil/digitRecognizer/blob/master/NeuralNetwork.py#L6) recebe o número de épocas/repetições e a taxa de aprendizado.<br>
- [`NeuralNetwork.train`](https://github.com/douglasJovenil/digitRecognizer/blob/master/NeuralNetwork.py#L15) - recebe os valores de entradas e saídas do dataset de treino.<br>
- [`NeuralNetwork.query`](https://github.com/douglasJovenil/digitRecognizer/blob/master/NeuralNetwork.py#L46) - recebe um valor e retorna a previsão da rede neural.<br>
- [`NeuralNetwork.acc`](https://github.com/douglasJovenil/digitRecognizer/blob/master/NeuralNetwork.py#L54) - recebe uma entrada e saída de teste e retorna a precisão da rede.<br>
- [`NeuralNetwork.add`](https://github.com/douglasJovenil/digitRecognizer/blob/master/NeuralNetwork.py#L62) - adiciona uma camada na rede com o número do argumento passado para `num_nodes`.<br>
