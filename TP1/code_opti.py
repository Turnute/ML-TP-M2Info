import numpy as np

class Jeu_Hanoi:
    def __init__(self):
        self.pic = np.zeros([3,3], dtype= int)
        self.nombre_palet = np.zeros(3, dtype=int)
        pass

# Situation initale :
jeu = Jeu_Hanoi()
jeu.nombre_palet[0] = 3
jeu.pic[0,0] = 3
jeu.pic[0,1] = 2
jeu.pic[0,2] = 1

# Situation finale désirée :
jeu_final = Jeu_Hanoi()
jeu_final.nombre_palet[2] = 3
jeu_final.pic[2,0] = 3
jeu_final.pic[2,1] = 2
jeu_final.pic[2,2] = 1

# Situation intermédiaire :
jeu_intermediaire = Jeu_Hanoi()
jeu_intermediaire.nombre_palet[1] = 2
jeu_intermediaire.pic[1,0] = 3
jeu_intermediaire.pic[1,1] = 2
jeu_intermediaire.pic[1,2] = 0
jeu_intermediaire.nombre_palet[2] = 1
jeu_intermediaire.pic[2,0] = 1

print("Situation initiale :")
print(jeu.pic)

def nombre_situation(jeu):
    return hash(jeu.pic.tobytes())

situation_etudiee = [nombre_situation(jeu)]

def pic_vide(indice_pic, jeu):
    check_nb_palet = jeu.nombre_palet[indice_pic] == 0
    check_valeurs_pic = jeu.pic[indice_pic, 0] == 0
    return check_nb_palet or check_valeurs_pic

def regle_jeu(pic1, pic2, jeu): # On déplace de pic1 vers pic2
    if pic_vide(pic1, jeu):
        return False
    else:
        return jeu.pic[pic1, jeu.nombre_palet[pic1] - 1] < jeu.pic[pic2, jeu.nombre_palet[pic2] - 1] or pic_vide(pic2, jeu)

def effectuer_deplacement(pic1, pic2, jeu): # On déplace de pic1 vers pic2
    # pic1 et pc2 sont des indices de pics (0, 1 ou 2)
    if regle_jeu(pic1, pic2, jeu):
        # obtenir le palet à déplacer
        palet = jeu.pic[pic1, jeu.nombre_palet[pic1] - 1]

        # déplacer le palet
        jeu.pic[pic2, jeu.nombre_palet[pic2]] = palet
        jeu.pic[pic1, jeu.nombre_palet[pic1] - 1] = 0

        # mettre à jour les compteurs de palets
        jeu.nombre_palet[pic1] -= 1
        jeu.nombre_palet[pic2] += 1
    pass

def situation_non_vue(pic1, pic2, jeu, situation_etudiee):
    # simuler le déplacement sur une copie temporaire
    jeu_temp = Jeu_Hanoi()
    jeu_temp.pic = np.copy(jeu.pic)
    jeu_temp.nombre_palet = np.copy(jeu.nombre_palet)

    effectuer_deplacement(pic1, pic2, jeu_temp)

    # vérifier si la situation résultante a déjà été vue
    situation = nombre_situation(jeu_temp)
    return situation not in situation_etudiee

def prolonge_solution(jeu, pic):
    # vérifie si un déplacement de pic vers le dernier pic prolonge la solution
    palet_deplace = jeu.pic[pic, jeu.nombre_palet[pic] - 1]
    palet_destination = jeu.pic[2, jeu.nombre_palet[2] - 1]
    return palet_destination == 0 or palet_deplace == palet_destination - 1

cmpt = 0


while cmpt < 100 and nombre_situation(jeu) != nombre_situation(jeu_final):
    pic1, pic2 = 0, 0
    # selectionne une regle eligible
    if regle_jeu(0, 2, jeu) and situation_non_vue(0, 2, jeu, situation_etudiee) and prolonge_solution(jeu, 0):
        pic1 = 0
        pic2 = 2
    elif regle_jeu(0, 1, jeu) and situation_non_vue(0, 1, jeu, situation_etudiee):
        pic1 = 0
        pic2 = 1
    elif regle_jeu(1, 0, jeu) and situation_non_vue(1, 0, jeu, situation_etudiee):
        pic1 = 1
        pic2 = 0
    elif regle_jeu(1, 2, jeu) and situation_non_vue(1, 2, jeu, situation_etudiee) and prolonge_solution(jeu, 1):
        pic1 = 1
        pic2 = 2
    elif regle_jeu(2, 1, jeu) and situation_non_vue(2, 1, jeu, situation_etudiee):
        pic1 = 2
        pic2 = 1
    else:
        print("Aucun déplacement possible ou toutes les situations ont été vues.")
        break

    # effectue le deplacement
    effectuer_deplacement(pic1, pic2, jeu)

    # met à jour la liste des situations
    situation_etudiee.append(nombre_situation(jeu))

    # incrémente le compteur
    cmpt += 1
    
    # affiche l'etape
    
    print(f"Étape {cmpt}:")
    print(jeu.pic)

print("Nombre d'étapes pour résoudre le jeu : ", cmpt)