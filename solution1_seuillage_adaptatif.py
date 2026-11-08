# Solution 1 : seuillage adaptatif + contours (une seule image).
#
#   python solution1_seuillage_adaptatif.py        -> mesure sur toutes les images
#   python solution1_seuillage_adaptatif.py 100    -> affiche le détail sur l'image 100

import sys
import time
import cv2
from commun import charger_dataset, lire_image, afficher, afficher_resultat

BLOC = 51          # taille du voisinage pour le seuil local
C = 20             # un pixel est blanc s'il est plus sombre que la moyenne locale moins C
AIRE_MIN = 30
AIRE_MAX = 750     # 90 % des boîtes annotées font moins de 737 px
COTE_MAX = 65


def solution_adaptatif(d, i):
    gris = cv2.cvtColor(lire_image(d["chemin"][i]), cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    masque = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, BLOC, C)

    contours, hierarchie = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if AIRE_MIN <= aire <= AIRE_MAX and w <= COTE_MAX and h <= COTE_MAX:
            detections.append([x, y, w, h])
    return gris, masque, detections


if __name__ == "__main__":
    d = charger_dataset()

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        carte, masque, detections = solution_adaptatif(d, i)
        afficher(d, i, "gris flouté", carte, masque, detections)
    else:
        debut = time.time()
        predit = []
        for i in range(len(d)):
            predit.append(len(solution_adaptatif(d, i)[2]))
        print("Solution 1 : seuillage adaptatif")
        print("temps :", round(1000 * (time.time() - debut) / len(d)), "ms par image")
        afficher_resultat(predit, d["n"])
