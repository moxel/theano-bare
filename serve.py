import theano
import theano.tensor as T

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))

logistic = theano.function([x], s)

def predict(x):
    out = logistic(x)
    return {
        'out': out
    }
