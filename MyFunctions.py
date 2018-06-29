from mnist import MNIST


def _numToArray(num):
    retorno = [0 for x in range(10)]
    retorno[num] = 1
    return retorno


def _normalize(x):
    for i in range(len(x)):
        x[i] = (x[i] - 0)/(255 - 0)
    return x


def generateData(path):
    mndata = MNIST(path)
    train_img, train_lbl = mndata.load_training()
    test_img, test_lbl = mndata.load_testing()
    in_train = [_normalize(train_img[i]) for i in range(len(train_img))]
    out_train = [_numToArray(train_lbl[i]) for i in range(len(train_lbl))]
    in_test = [_normalize(test_img[i]) for i in range(len(test_img))]
    out_test = [_numToArray(test_lbl[i]) for i in range(len(test_lbl))]
    return [in_train, out_train, in_test, out_test]
