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

def nombre_situation(jeu):
    # convertir une situation en nombre (binaire) unqiue (exemple : jeu.pic =[ [ 3 2 1] [0 0 0] [0 0 0 ]] -> nombre en base 2 = 000 000 111)
    pass

def pic_vide(indice_pic, jeu):
    # vérifie si un pic est vide
    pass

def regle_jeu(pic1, pic2, jeu):
    # vérifie si le déplacement d'un palet du pic1 vers le pic2 est autorisé
    pass

def effectuer_deplacement(pic1, pic2, jeu):
    # effectue le déplacement d'un palet du pic1 vers le pic2
    pass

def situation_non_vue(pic1, pic2, jeu, situation_etudiee):
    # empêche les répétitions. Retourne un booléen.
    pass

cmpt = 0
situation_etudiee = [nombre_situation(jeu)]

while jeu != jeu_final:
    # selectionne une regle eligible
    # effectue le deplacement
    # met à jour la liste des situations
    # affiche l'etape
    # incremente le compteur
    cmpt += 1

print("Nombre d'étapes pour résoudre le jeu : ", cmpt)