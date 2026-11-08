# Fonctions communes aux 3 solutions : lecture du dataset, affichage, mesures.
# Le dataset dézippé doit être dans DATA (dossiers train/, valid/, test/).

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "."
TAILLE = 416


# ---------------------------------------------------------------- dataset

def charger_dataset():
    """Une ligne par image, journées d'au moins 50 images, triées par jour et heure."""
    lignes = []
    for dossier in ["train", "valid", "test"]:
        liste_images = sorted(glob.glob(DATA + "/" + dossier + "/images/*.jpg"))
        for chemin in liste_images:
            nom = os.path.basename(chemin)
            chemin_txt = DATA + "/" + dossier + "/labels/" + nom.replace(".jpg", ".txt")

            boites = []
            fichier = open(chemin_txt)
            for ligne in fichier:
                morceaux = ligne.split()
                if len(morceaux) == 5:
                    cx = float(morceaux[1]) * TAILLE
                    cy = float(morceaux[2]) * TAILLE
                    w = float(morceaux[3]) * TAILLE
                    h = float(morceaux[4]) * TAILLE
                    boites.append([cx, cy, w, h])
            fichier.close()

            lignes.append({"chemin": chemin, "dossier": dossier,
                           "jour": nom[0:8], "heure": nom[9:17],
                           "boites": boites, "n": len(boites)})

    tout = pd.DataFrame(lignes)
    if len(tout) == 0:
        raise SystemExit("Aucune image trouvée : vérifier que train/, valid/ et test/ sont dans " + os.path.abspath(DATA))

    compte = tout["jour"].value_counts()
    jours = []
    for jour in compte.index:
        if compte[jour] >= 50:
            jours.append(jour)
    jours.sort()

    d = tout[tout["jour"].isin(jours)]
    d = d.sort_values(["jour", "heure"])
    d = d.reset_index(drop=True)
    print(len(d), "images,", d["n"].sum(), "insectes annotés, journées :", jours)
    return d


def lire_image(chemin):
    image = cv2.imread(chemin)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dessiner_boites(image, boites):
    image = image.copy()
    for cx, cy, w, h in boites:
        cv2.rectangle(image, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (255, 0, 0), 1)
    return image


def afficher(d, i, titre, carte, masque, detections):
    """Image, carte intermédiaire, masque, résultat (rouge = annoté, vert = détecté)."""
    image = lire_image(d["chemin"][i])
    resultat = dessiner_boites(image, d["boites"][i])
    for x, y, w, h in detections:
        cv2.rectangle(resultat, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("image " + str(i))
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.imshow(carte, cmap="gray")
    plt.title(titre)
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.imshow(masque, cmap="gray")
    plt.title("masque")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(resultat)
    plt.title("vrai " + str(d["n"][i]) + ", détecté " + str(len(detections)))
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------- soustraction de fond (solutions 2 et 3)

def fond_median(d, i):
    """Médiane des 10 images avant et 10 images après, le même jour (l'image elle-même exclue)."""
    jour = d["jour"][i]
    meme_jour = list(d[d["jour"] == jour].index)
    position = meme_jour.index(i)
    debut = max(position - 10, 0)
    fin = min(position + 11, len(meme_jour))

    pile = []
    for j in meme_jour[debut:fin]:
        if j != i:
            pile.append(lire_image(d["chemin"][j]))
    fond = np.median(pile, axis=0)
    return fond.astype(np.uint8)


def taches(masque, aire_min):
    """Composantes connexes d'au moins aire_min pixels -> liste de rectangles [x, y, w, h]."""
    nb, etiquettes, stats, centres = cv2.connectedComponentsWithStats(masque)
    detections = []
    for t in range(1, nb):              # la tache 0, c'est le fond noir
        if stats[t, cv2.CC_STAT_AREA] >= aire_min:
            x = stats[t, cv2.CC_STAT_LEFT]
            y = stats[t, cv2.CC_STAT_TOP]
            w = stats[t, cv2.CC_STAT_WIDTH]
            h = stats[t, cv2.CC_STAT_HEIGHT]
            detections.append([x, y, w, h])
    return detections


# ---------------------------------------------------------------- mesures

def bilan(predit, vrai):
    erreurs = np.array(predit) - np.array(vrai)
    mae = np.mean(np.abs(erreurs))
    biais = np.mean(erreurs)
    pire = np.max(np.abs(erreurs))
    return round(mae, 3), round(biais, 3), pire


def taux_ok(predit, vrai):
    """OA : somme de la diagonale de la matrice de confusion / total."""
    C = pd.crosstab(np.array(vrai), np.array(predit))
    classes = sorted(set(C.index) | set(C.columns))
    C = C.reindex(index=classes, columns=classes, fill_value=0).values
    return round(np.trace(C) / C.sum(), 3)


def afficher_resultat(predit, vrai):
    mae, biais, pire = bilan(predit, vrai)
    erreurs = np.array(predit) - np.array(vrai)
    print("MAE =", mae, "  biais =", biais, "  OA =", taux_ok(predit, vrai),
          "  erreur max =", pire, "  images avec >= 5 de trop :", np.sum(erreurs >= 5))


# ---------------------------------------------------------------- ombres (partie 3 du notebook)

def indice_contraste(gris, cx, cy, w, h):
    x1 = max(int(cx - w / 2), 0)
    y1 = max(int(cy - h / 2), 0)
    x2 = min(int(cx + w / 2), TAILLE)
    y2 = min(int(cy + h / 2), TAILLE)
    dedans = gris[y1:y2, x1:x2]

    X1 = max(x1 - 6, 0)
    Y1 = max(y1 - 6, 0)
    X2 = min(x2 + 6, TAILLE)
    Y2 = min(y2 + 6, TAILLE)
    grand = gris[Y1:Y2, X1:X2].astype(float)
    grand[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = np.nan    # il ne reste que l'anneau
    return 1 - np.percentile(dedans, 10) / np.nanmedian(grand)


def verite_sans_ombres(d):
    """Nombre de boîtes annotées par image, sans celles repérées comme ombres."""
    resultat = []
    for i in range(len(d)):
        gris = cv2.imread(d["chemin"][i], cv2.IMREAD_GRAYSCALE)
        boites = d["boites"][i]
        contrastes = [indice_contraste(gris, cx, cy, w, h) for cx, cy, w, h in boites]
        nb_ombres = 0
        for k in range(len(boites)):
            for j in range(len(boites)):
                if j == k:
                    continue
                distance = np.sqrt((boites[j][0] - boites[k][0]) ** 2 + (boites[j][1] - boites[k][1]) ** 2)
                if distance < 2 * max(boites[j][3], boites[k][3]) and contrastes[j] > contrastes[k] + 0.12:
                    nb_ombres = nb_ombres + 1
                    break
        resultat.append(len(boites) - nb_ombres)
    return resultat
