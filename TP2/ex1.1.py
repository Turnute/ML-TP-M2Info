import numpy as np
import matplotlib.pyplot as plt

# EXERCICE 1
# Conventions pour toute cette partie.
# - Le coefficient d’apprentissage est fixé à α = 0,1.
# - Le seuil est traité comme un poids associé à une entrée constante égale à +1 : pour une entrée
# - x = (x1, x2), on note ˜x = (1, x1, x2) et le potentiel du neurone vaut v = wT ˜x = w0 +w1x1 +w2x2.
# - Les vecteurs et matrices sont des tableaux NumPy, indexés à partir de 0.

# 1.1 : Mise en place d'un perceptron simple

# Créer une fonction y = perceptron_simple(x, w, active)
# w : poids synaptiques du neurone, vecteur de taille 3 ; w[0] correspond au seuil.
# x : entrée du neurone, vecteur de taille 2.
# active : fonction d’activation utilisée : φ(v) = sign(v) si active==0 ; φ(v) = tanh(v) si active==1.
# résultat y : scalaire, sortie du neurone y = φ(wT ˜x).

def perceptron_simple(x, w, active):
    result = w[0] + x[0]*w[1] + x[1]*w[2]
    # On fait au plus simple

    if (active == 0):
        return np.sign(result)
    elif (active==1):
        return np.tanh(result)
    return;

# TESTS DU OU :

w = [-0.5,1,1]  # poids (tels de donnés dans l'énoncé)
x0=[0,0]        # 0 OU 0 : faux
x1=[1,0]        # 1 OU 0 : vrai
x2=[0,1]        # 0 OU 1 : vrai
x3=[1,1]        # 1 OU 1 : vrai


y0=perceptron_simple(x0, w, 0)
y1=perceptron_simple(x1, w, 0)
y2=perceptron_simple(x2, w, 0)
y3=perceptron_simple(x3, w, 0)

print(y0) # -1
print(y1) # +1
print(y2) # +1
print(y3) # +1

# AFFICHAGE
yd = np.array([y0, y1, y2, y3]) # tableau stockant les sorties trouvées
X = np.array([
    [0, 0, 1, 1],  # coordonnées x1
    [0, 1, 0, 1]   # coordonnées x2
])

t = np.array([-0.5, 1.5]) # abscisses x1 du tracé
z = -(w[0] + w[1]*t) / w[2] # ordonnées x2 correspondantes
plt.plot(t, z, 'k-') # droite séparatrice

plt.plot(X[0, yd == -1], X[1, yd == -1], 'o') # classe -1
plt.plot(X[0, yd == 1], X[1, yd == 1], 'x') # classe +1

plt.xlim(t[0], t[1]); plt.ylim(-0.5, 1.5)
plt.show()