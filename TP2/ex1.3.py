import math


def perceptron_simple(x, w, active):
    result = w[0] + x[0]*w[1] + x[1]*w[2]
    # On fait au plus simple

    if (active == 0):
        return np.sign(result)
    elif (active==1):
        return np.tanh(result)
    return;

def sigmoide(v):
    return 1/(1+math.exp(-1*v));

def multiperceptron(x, w1, w2):
    
    return;