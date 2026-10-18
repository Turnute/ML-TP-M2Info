import math
import numpy as np

# fonction d'activation
def sigmoide(v):
    return 1/(1+math.exp(-1*v));

# x : entrée du réseau, vecteur de taille 2
# w1 : poids de la couche cachée, matrice 3x2. w[:,0] : poids du neurone 1, w[:,1] : poids du neurone 2
# w2 : poids de la couche de sortie, vecteur de taille 3

def multiperceptron(x, w1, w2):
    # calcul de la sortie du premier neurone
    v1 = w1[0][0] + w1[1][0]*x[0] + w1[2][0]*x[1]
    y1 = sigmoide(v1)

    # calcul de la sortie du deuxième neurone
    v2 = w1[0][1] + w1[1][1]*x[0] + w1[2][1]*x[1]
    y2 = sigmoide(v2)

    # calcul de la sortie du perceptron
    v3 = w2[0] + w2[1]*y1 + w2[2]*y2
    y3 = sigmoide(v3)

    return y3

# TEST :
x = [1, 1]
w1 = [[-0.5, 0.5], [2,1], [-1, 0.5]]
w2 = [2,-1,1]

print(multiperceptron(x, w1, w2))