# EXERCICE 1.2 : Apprentissage de Widrow-Hoff

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

alpha = 0.1

# Perceptron défini à l'exercice 1.1
def perceptron_simple(x, w, active):
    result = w[0] + x[0]*w[1] + x[1]*w[2]

    if active == 0:
        return np.sign(result)
    elif active == 1:
        return np.tanh(result)
    return

# Fonction d'affichage
def afficher_epoque(x, yd, w, epoque, epoch, cmpt):
    plt.figure("Apprentissage Widrow-Hoff", figsize=(12, 8))

    # On vide la figure au début d'un nouvel apprentissage
    if epoque == 1:
        plt.clf()

    # Plusieurs graphiques sur une seule figure, 3 par ligne
    lignes = int(np.ceil(epoch / 3))
    plt.subplot(lignes, 3, epoque)

    # Affichage des points
    plt.scatter(x[0, yd == 1], x[1, yd == 1], color="blue")
    plt.scatter(x[0, yd == -1], x[1, yd == -1], color="red")

    # Droite séparatrice
    x1 = np.array([min(x[0])-0.5, max(x[0])+0.5])

    if w[2] != 0:
        x2 = -(w[0] + w[1]*x1) / w[2]
        plt.plot(x1, x2, color="green")
    elif w[1] != 0:
        plt.axvline(-w[0]/w[1], color="green")

    plt.title(f"Époque {epoque} : {cmpt} erreur(s)")
    plt.xlim(min(x[0])-0.5, max(x[0])+0.5)
    plt.ylim(min(x[1])-0.5, max(x[1])+0.5)
    plt.grid()

    # Affichage de la figure à la fin de l'apprentissage
    if cmpt == 0 or epoque == epoch:
        plt.tight_layout()
        plt.show()


# Première version sans batch_size (une époque traite tout le jeu de données)
def apprentissage_widrow_no_batch(x, yd, epoch):
    # On commence par tirer aléatoirement les poids initiaux
    w0 = (random.randrange(-5,5))/10
    w1 = (random.randrange(-5,5))/10
    w2 = (random.randrange(-5,5))/10

    w = np.array([w0, w1, w2])

    # tableau des résultats
    y_result = np.zeros(yd.size)

    # tableau stockant l'erreur quadratique de chaque époque
    erreur = np.zeros(epoch)

    # travail sur les époques
    for i in range(epoch):
        for j in range(yd.size):
            y_result[j] = perceptron_simple(x[:,j], w, 1)

        for j in range(yd.size):
            # ajustement des poids
            xi = np.array([1, x[0, j], x[1, j]])
            w += alpha*(yd[j]-y_result[j])*(1-y_result[j]**2)*xi

            # cumul de l'erreur de cette époque
            erreur[i] += (yd[j]-y_result[j])**2

        # Nombre d'individus mal classés avec les nouveaux poids
        y = np.sign(w[0] + w[1]*x[0] + w[2]*x[1])
        cmpt = np.sum(y != yd)

        # Affichage
        afficher_epoque(x, yd, w, i + 1, epoch, cmpt)

        # Si aucune erreur, on a terminé !
        if cmpt == 0:
            break

    return w, erreur[:i+1]


# Version avec batch_size : tirage aléatoire des indices traités
def apprentissage_widrow(x, yd, epoch, batch_size):
    # On commence par tirer aléatoirement les poids initiaux
    w0 = (random.randrange(-5,5))/10
    w1 = (random.randrange(-5,5))/10
    w2 = (random.randrange(-5,5))/10

    w = np.array([w0, w1, w2])

    print("Poids initiaux :", w)

    # tableau stockant l'erreur quadratique de chaque époque
    erreur = np.zeros(epoch)

    # travail sur les époques
    for i in range(epoch):

        # Tirage aléatoire des indices des individus
        indices = np.random.permutation(yd.size)

        # Parcours des individus par lots
        for debut in range(0, yd.size, batch_size):
            lot = indices[debut:debut + batch_size]

            # tableau des résultats du lot
            y_result = np.zeros(len(lot))

            # remplir les résultats avec les poids actuels
            for j in range(len(lot)):
                indice = lot[j]
                y_result[j] = perceptron_simple(x[:, indice], w, 1)

            # somme des corrections du lot
            delta_w = np.zeros(3)

            for j in range(len(lot)):
                indice = lot[j]

                # calcul de la correction
                xi = np.array([1, x[0, indice], x[1, indice]])
                delta_w += alpha*(yd[indice]-y_result[j])*(1-y_result[j]**2)*xi

                # cumul de l'erreur de cette époque
                erreur[i] += (yd[indice]-y_result[j])**2

            # ajustement des poids après le lot
            w += delta_w

        # Nombre d'individus mal classés avec les nouveaux poids
        y = np.sign(w[0] + w[1]*x[0] + w[2]*x[1])
        cmpt = np.sum(y != yd)

        # Affichage
        afficher_epoque(x, yd, w, i + 1, epoch, cmpt)

        # Si aucune erreur, on a terminé !
        if cmpt == 0:
            break

    return w, erreur[:i+1]


# ==== TESTS ====
def test_ou_logique():
    # Table de vérité du OU logique
    x = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])

    yd = np.array([-1, 1, 1, 1])

    # Pour reproduire le même test
    random.seed(0)
    np.random.seed(0)

    # Apprentissage
    w, erreur = apprentissage_widrow(x, yd, 15, 2)

    print("Poids finaux :", w)
    print("Erreurs par époque :", erreur)

    # Vérification des prédictions
    for j in range(yd.size):
        print(x[:, j], "=>", perceptron_simple(x[:, j], w, 0))
def test_fichier_donnees(X, batch_size):

    # Création des étiquettes correspondantes
    yd = np.concatenate((-np.ones(25), np.ones(25)))

    # Pour reproduire le même test
    random.seed(0)
    np.random.seed(0)

    # Apprentissage
    w, erreur = apprentissage_widrow(X, yd, 15, batch_size)

    print("Poids finaux :", w)
    print("Erreurs par époque :", erreur)

    # Numéros des époques réellement effectuées
    epoques = np.arange(1, len(erreur) + 1)

    plt.figure("Évolution de l'erreur", figsize=(8, 5))
    plt.plot(epoques, erreur, marker="o", color="blue")

    plt.xlabel("Époque")
    plt.ylabel("Somme des erreurs quadratiques")
    plt.title("Évolution de l'erreur pendant l'apprentissage")
    plt.xticks(epoques)
    plt.grid()
    plt.tight_layout()
    plt.show()
def moyenne_epoques(X, yd, batch_size, iterations=10000):
    total_ep = 0

    for _ in range(iterations):
        w, erreur = apprentissage_widrow(X, yd, 15, batch_size)
        total_ep += len(erreur)

    print("Nombre moyen d'époques par apprentissage :", total_ep / iterations)

# Dossier contenant ce fichier Python
dossier_script = Path(__file__).resolve().parent

# Chargement des données dans le sous-dossier ressources
# X = np.loadtxt(dossier_script / ".ressources" / "p2_d1.txt")
X = np.loadtxt(dossier_script / ".ressources" / "p2_d2.txt")
test_fichier_donnees(X, 15)
