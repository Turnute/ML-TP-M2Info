# Définition de l'environnement de jeu :
# Plateau de 4x4 cases, position(ligne, colonne), commence à 0,0 en haut à gauche.
# Valeurs de cases : Départ, Vide, Dragon, Trésor.
# 
# Case vide : le joueur s’y déplace et reçoit R_case.
# Sortie du plateau : le joueur reste sur sa case et obtient R_case. Il peut être intéressant de définir une autre valeur R_case2.
# Dragon : le joueur obtient R_dragon et la partie se termine (fin = True). La partie suivante repart de la case départ.
# Trésor : le joueur obtient R_tresor et la partie se termine.
# Une partie s’arrête aussi au bout de max_coups coups, sans récompense supplémentaire.

# Mouvement : 0 ↑, 1 →, 2 ↓, 3 ←

import numpy as np
from enum import IntEnum

class TypeCase(IntEnum):
    VIDE = 0
    DRAGON = 1
    TRESOR = 2

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

R_case = -1     # Récompense d'une case vide
R_case2 = -5    # Récompense d'une case hors du plateau
R_dragon = -10  # Récompense d'une case dragon
R_tresor = 20   # Récompense d'une case trésor


def creer_plateau(R_case, R_dragon, R_tresor):

    # Initialisation d'un plateau rempli de cases vides
    plateau = np.array([
        [(TypeCase.VIDE, R_case) for _ in range(4)]
        for _ in range(4)
    ])

    # Placement des cases spéciales
    plateau[1, 0] = (TypeCase.DRAGON, R_dragon)
    plateau[1, 2] = (TypeCase.DRAGON, R_dragon)
    plateau[2, 3] = (TypeCase.DRAGON, R_dragon)
    plateau[3, 1] = (TypeCase.DRAGON, R_dragon)
    plateau[3, 3] = (TypeCase.TRESOR, R_tresor)

    return plateau

def application_action(action, position, plateau):

    l, c = position

    match action:
        case Action.UP:
            l -= 1
        case Action.RIGHT:
            c += 1
        case Action.DOWN:
            l += 1
        case Action.LEFT:
            c -= 1
        case _:
            raise ValueError("Action inconnue")

    # Vérification des limites
    if (l < 0 or l >= plateau.shape[0] or c < 0 or c >= plateau.shape[1]):
        return position, R_case2, False

    # On récupère le type et la récompense
    type_case, reward = plateau[l, c]

    # Dragon ou trésor = fin de partie
    fin = type_case in (TypeCase.DRAGON, TypeCase.TRESOR)

    return (l, c), reward, fin


# TESTS

plateau = creer_plateau(R_case, R_dragon, R_tresor)

# 1. Déplacement simple : (0,0) → (0,1)
pos, reward, fin = application_action(Action.RIGHT, (0, 0), plateau)
assert (pos, reward, fin) == ((0, 1), R_case, False)
print("Test déplacement simple : OK")

# 2. Sortie du plateau : (0,0) → haut (impossible)
pos, reward, fin = application_action(Action.UP, (0, 0), plateau)
assert (pos, reward, fin) == ((0, 0), R_case2, False)
print("Test sortie du plateau : OK")

# 3. Dragon : (0,0) → (1,0)
pos, reward, fin = application_action(Action.DOWN, (0, 0), plateau)
assert (pos, reward, fin) == ((1, 0), R_dragon, True)
print("Test dragon : OK")

# 4. Trésor : (3,2) → (3,3)
pos, reward, fin = application_action(Action.RIGHT, (3, 2), plateau)
assert (pos, reward, fin) == ((3, 3), R_tresor, True)
print("Test trésor : OK")