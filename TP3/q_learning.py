import numpy as np
import matplotlib.pyplot as plt
import game

# Table mat_q à 16 lignes et 4 colonnes (état s et action a).
# Contient l'estimation du gain futur actualisé obtenu en jouant l'action a dans l'état s.
def init_mat_q(lignes, colonnes):
    return np.zeros((lignes, colonnes), dtype=float)

def state_to_mq(state) :
    # Convertis un état (position dans le plateau) en sa ligne dans mat_q
    return state[0] * 4 + state[1]

def choose_action(state, epsilon, mat_q) :
    if np.random.uniform(0, 1) < epsilon:      # explorer (probabilité ε)
        return np.random.randint(4)
    q = mat_q[state_to_mq(state), :]      # exploiter (probabilité 1 − ε)
    best = np.flatnonzero(q == q.max())   # les égalités...
    return int(np.random.choice(best))    # ...sont tirées au hasard

def onestep(mat_q, state, epsilon, alpha, gamma, plateau) :
    a = choose_action(state, epsilon, mat_q)
    new_state, r, fin = game.application_action(a, state, plateau)

    s = state_to_mq(state)
    s_ = state_to_mq(new_state)
    cible = r + gamma * (1 - fin) * max(mat_q[s_, :])
    mat_q[s, a] += alpha * (cible - mat_q[s, a])

    return mat_q, new_state, r, fin

def apprentissage(nb_parties, max_coups, epsilon_fct, alpha, gamma, plateau):

    mat_q = init_mat_q(16, 4)
    historique = []

    # On répète l'apprentissage sur plusieurs parties
    for i in range(nb_parties):

        state = (0, 0)  # Départ
        coups = 0
        fin = False
        epsilon = epsilon_fct / (epsilon_fct + i)

        # On joue jusqu'à la fin ou jusqu'au nombre maximum de coups
        while not fin and coups < max_coups:

            mat_q, state, r, fin = onestep(
                mat_q, state, epsilon, alpha, gamma, plateau
            )

            coups += 1

        # La partie est gagnée uniquement si on a atteint le trésor
        gagne = fin and plateau[state[0], state[1], 0] == game.TypeCase.TRESOR

        # On enregistre le résultat de cette partie
        historique.append((gagne, coups))

    return mat_q, historique

def jouer(mat_q, plateau, max_coups=10):

    state = (0, 0)
    coups = 0
    fin = False

    print("Départ :", state)

    while not fin and coups < max_coups:
        a = choose_action(state, 0, mat_q)
        state, r, fin = game.application_action(a, state, plateau)
        coups += 1

        print(f"Coup {coups} : {game.Action(a).name} → {state} (récompense : {r})")

    # Résultat de la partie
    if fin:
        if plateau[state[0], state[1], 0] == game.TypeCase.TRESOR:
            print("Victoire ! Trésor atteint.")
        else:
            print("Défaite ! Dragon rencontré.")
    else:
        print("Fin : nombre maximum de coups atteint.")

def courbe_apprentissage(historique):
    # Récupération des victoires (True = 1, False = 0)
    victoires = [int(gagne) for gagne, coups in historique]

    # Calcul du taux de victoire sur les 50 dernières parties
    taux = [
        np.mean(victoires[i-50:i]) * 100
        for i in range(50, len(victoires) + 1)
    ]

    # Tracé de la courbe
    plt.plot(range(50, len(victoires) + 1), taux)

    plt.xlabel("Nombre de parties")
    plt.ylabel("Taux de victoire (%)")
    plt.title("Courbe d'apprentissage du Q-learning")
    plt.ylim(0, 100)
    plt.grid(True)

    # Première partie où le taux de victoire dépasse 50 %
    if any(t > 50 for t in taux):
        seuil = next(i for i, t in enumerate(taux, start=50) if t > 50)
        plt.axvline(seuil, color="red", linestyle="--", label=f"50 % dépassés : partie {seuil}")
    plt.legend()
    plt.show()
    
# Paramètres
N = 1000        # Nombre de parties
max_coups = 10  # Nombre maximum de coups par partie
alpha = 0.81    # Pas d'apprentissage
gamma = 0.96    # Facteur d'actualisation
epsilon_fct = 100

plateau = game.creer_plateau(0, -1, 1)

# mat_q, historique = apprentissage(N, max_coups, epsilon_fct, alpha, gamma, plateau)
# courbe_apprentissage(historique)
# print(mat_q)

# jouer(mat_q, plateau)

def compare_gamma(gamma1, gamma2):
    plt.figure()

    for gamma in (gamma1, gamma2):
        victoires = np.zeros(N)

        # Moyenne de 5 apprentissages
        for _ in range(100):
            mat_q, hist = apprentissage(
                N, max_coups, epsilon_fct, 0.5, gamma, plateau
            )

            for i in range(len(hist)):
                victoires[i] += int(hist[i][0])

        victoires /= 100

        # Moyenne glissante sur 50 parties
        taux = [
            np.mean(victoires[i-50:i]) * 100
            for i in range(50, N + 1)
        ]

        plt.plot(range(50, N + 1), taux, label=f"Gamma = {gamma}")

    plt.xlabel("Nombre de parties")
    plt.ylabel("Taux de victoire (%)")
    plt.title("Comparaison des valeurs de gamma")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.show()


compare_gamma(0.9, 0.5)