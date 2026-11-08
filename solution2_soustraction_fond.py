# Solution 2 : soustraction de fond (fond = médiane des 10 images avant et 10 après, même jour).
#
#   python solution2_soustraction_fond.py        -> mesure sur toutes les images (~2 min)
#   python solution2_soustraction_fond.py 508    -> affiche le détail sur l'image 508

import sys
import time
import numpy as np
from commun import charger_dataset, lire_image, afficher, afficher_resultat, fond_median, taches

SEUIL = 30         # assombrissement minimal (niveaux de gris, 0 à 255)
AIRE_MIN = 15      # enlève le bruit ; la plus petite boîte annotée fait 19 px


def solution_soustraction(d, i):
    image = lire_image(d["chemin"][i])
    fond = fond_median(d, i)

    # int pour avoir des négatifs (en uint8, 10 - 20 donnerait 246)
    assombri = fond.astype(int) - image.astype(int)
    assombri = assombri.mean(axis=2)
    assombri[assombri < 0] = 0          # on ne garde que ce qui est devenu plus sombre

    masque = (assombri > SEUIL).astype(np.uint8)
    return assombri, masque, taches(masque, AIRE_MIN)


if __name__ == "__main__":
    d = charger_dataset()

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        carte, masque, detections = solution_soustraction(d, i)
        afficher(d, i, "fond - image", carte, masque, detections)
    else:
        debut = time.time()
        predit = []
        for i in range(len(d)):
            predit.append(len(solution_soustraction(d, i)[2]))
            if i % 200 == 0:
                print(i, "/", len(d))
        print("Solution 2 : soustraction de fond")
        print("temps :", round(1000 * (time.time() - debut) / len(d)), "ms par image")
        afficher_resultat(predit, d["n"])
