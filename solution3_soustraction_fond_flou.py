# Solution 3 (retenue) : soustraction de fond + flou gaussien 11x11.
#
#   python solution3_soustraction_fond_flou.py        -> mesure sur toutes les images (~2 min)
#                                                       + vérité sans ombres, matrice de confusion,
#                                                       résultats dans resultats_comptage.csv
#   python solution3_soustraction_fond_flou.py 508    -> affiche le détail sur l'image 508

import sys
import time
import cv2
import numpy as np
import pandas as pd
from commun import (charger_dataset, lire_image, afficher, afficher_resultat,
                    fond_median, taches, bilan, verite_sans_ombres)

SEUIL = 30
AIRE_MIN = 15
TAILLE_FLOU = 11   # doit être impaire pour cv2.GaussianBlur


def solution_soustraction_flou(d, i):
    image = lire_image(d["chemin"][i])
    fond = fond_median(d, i)

    # le flou étale les fines bandes créées quand la plateforme bouge d'un ou deux pixels
    image = cv2.GaussianBlur(image, (TAILLE_FLOU, TAILLE_FLOU), 0)
    fond = cv2.GaussianBlur(fond, (TAILLE_FLOU, TAILLE_FLOU), 0)

    assombri = fond.astype(int) - image.astype(int)
    assombri = assombri.mean(axis=2)
    assombri[assombri < 0] = 0

    masque = (assombri > SEUIL).astype(np.uint8)
    return assombri, masque, taches(masque, AIRE_MIN)


if __name__ == "__main__":
    d = charger_dataset()

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        carte, masque, detections = solution_soustraction_flou(d, i)
        afficher(d, i, "fond - image (avec flou)", carte, masque, detections)
    else:
        debut = time.time()
        predit = []
        for i in range(len(d)):
            predit.append(len(solution_soustraction_flou(d, i)[2]))
            if i % 200 == 0:
                print(i, "/", len(d))
        d["predit"] = predit
        print("Solution 3 : soustraction de fond + flou", TAILLE_FLOU, "x", TAILLE_FLOU)
        print("temps :", round(1000 * (time.time() - debut) / len(d)), "ms par image")
        afficher_resultat(d["predit"], d["n"])

        d["n_sans_ombres"] = verite_sans_ombres(d)
        mae, biais, pire = bilan(d["predit"], d["n_sans_ombres"])
        print("vérité sans ombres : MAE =", mae, "  biais =", biais)
        print()
        print("Matrice de confusion : vrai nombre (lignes), détecté (colonnes)")
        print(pd.crosstab(d["n"], d["predit"]))

        d[["chemin", "jour", "heure", "n", "n_sans_ombres", "predit"]].to_csv("resultats_comptage.csv", index=False)
        print("Résultats enregistrés dans resultats_comptage.csv")
