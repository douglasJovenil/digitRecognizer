# Digit Recognizer

### Setup
```bash
$ pip install -r requirements.txt
$ python Main.py
```

- O construtor da classe `NeuralNetwork` recebe o número de épocas/repetições e a taxa de aprendizado.<br>
- `NeuralNetwork.train` - recebe os valores de entradas e saídas do dataset de treino.<br>
- `NeuralNetwork.query` - recebe um valor e retorna a previsão da rede neural.<br>
- `NeuralNetwork.acc` - recebe uma entrada e saída de teste e retorna a precisão da rede.<br>
- `NeuralNetwork.add` - adiciona uma camada na rede com o número do argumento passado para `num_nodes`.<br>
