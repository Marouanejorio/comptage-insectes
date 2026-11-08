# Comptage d'insectes par image (Insect Detect)

Test technique Agriodor : compter le nombre d'insectes sur chaque image du dataset [Insect Detect](https://zenodo.org/records/7725941) (version 416×416, annotations YOLO), sur les journées d'au moins 50 images, avec du traitement d'image classique (OpenCV, pas de deep learning).

## Lancer le code

```bash
pip install -r requirements.txt
```

Télécharger `Insect_Detect_detection.v4-insect_detect_416_1class.yolov5pytorch.zip` sur Zenodo et le dézipper à la racine du dépôt (on doit avoir `train/`, `valid/`, `test/` à côté des scripts).

Chaque solution se lance seule :

```bash
python solution1_seuillage_adaptatif.py          # mesure sur les 1101 images (quelques secondes)
python solution2_soustraction_fond.py            # ~2 min
python solution3_soustraction_fond_flou.py       # ~2 min, solution retenue, écrit resultats_comptage.csv

python solution3_soustraction_fond_flou.py 508   # affiche le détail sur une image (ici la 508)
```

Le notebook `comptage_insectes.ipynb` contient toute la démarche (analyse des données, des ombres, les 3 solutions et leurs résultats).

## Structure

```
comptage_insectes.ipynb              démarche complète avec figures
commun.py                            lecture du dataset, fond médian, composantes connexes, mesures
solution1_seuillage_adaptatif.py
solution2_soustraction_fond.py
solution3_soustraction_fond_flou.py  solution retenue
requirements.txt
README.md
```

Les images ne sont pas dans le dépôt (voir `.gitignore`).

## Démarche

**1. Prise en main.** 1335 images sur 23 journées. On garde les 7 journées d'au moins 50 images : 1101 images, 1931 boîtes annotées (1,75 par image, max 6). Le découpage train / valid / test mélange les journées, et il n'y a pas d'apprentissage, donc je travaille sur toutes les images.

**2. Analyse des annotations.** Avant de détecter, j'ai vérifié les boîtes annotées avec un indice de contraste (intérieur de la boîte comparé à un anneau autour). Parmi les boîtes les moins contrastées, environ la moitié sont des **ombres**, pas des insectes : le dataset d'origine avait une classe `shadow` qui a été fusionnée dans `insect`.

**3. Analyse des ombres.** Une boîte est considérée comme une ombre si une boîte proche (moins de 2 hauteurs) est nettement plus contrastée. Résultat : **au moins 149 ombres (7,7 % des boîtes)**. C'est un minimum, les ombres dont l'insecte est hors champ ne sont pas trouvées. La vérité terrain compte donc aussi des ombres.

**4. Trois solutions testées**, chacune mesurée sur les 1101 images :

| Solution | Principe |
|---|---|
| 1. Seuillage adaptatif | seuil local (`adaptiveThreshold`), contours, filtre de taille |
| 2. Soustraction de fond | fond = médiane des 10 images avant et 10 après (même jour), on garde ce qui est devenu plus sombre, seuil puis composantes connexes |
| 3. Soustraction de fond + flou | pareil, avec un flou gaussien 11×11 sur l'image et le fond pour absorber les petits mouvements de la plateforme |

## Mesures

- **MAE** : erreur moyenne en nombre d'insectes par image
- **Biais** : > 0 on compte trop, < 0 pas assez
- **OA** (cours de reconnaissance de formes) : part des images comptées exactement, à partir de la matrice de confusion vrai nombre / nombre détecté

## Résultats

| Solution | MAE | Biais | OA | Images avec ≥ 5 détections de trop |
|---|---|---|---|---|
| 1. Seuillage adaptatif | 9,74 | +9,74 | 0,00 | 882 |
| 2. Soustraction de fond | 2,92 | +2,67 | 0,60 | 182 |
| **3. Soustraction de fond + flou 11×11** | **0,88** | **+0,36** | **0,62** | **56** |

**Solution retenue : 3.** Fond médian des 20 images voisines, flou gaussien 11×11, seuil 30, taches d'au moins 15 pixels. Environ 0,1 s par image.

- La solution 1 trouve les insectes mais aussi tout le décor sombre (bords de fleurs, coupelles, vis, sol) : aucune image juste.
- La solution 2 efface le décor fixe, mais explose quand la plateforme bouge avec le vent (les contours des fleurs ressortent).
- Le flou de la solution 3 divise la MAE par plus de 3.
- En enlevant les ombres de la vérité, le biais de la solution 3 passe de +0,36 à +0,50 : elle compte une partie des ombres.

## Ce qui ne marche pas

- **Plateforme qui bouge** : il reste 56 images avec des contours de fleurs détectés.
- **Insectes ratés** : l'erreur la plus fréquente est « 2 insectes comptés 1 » (119 images). Causes : petit insecte effacé par le flou (45), insecte immobile qui finit dans le fond médian (37), deux insectes ou un insecte et son ombre dans une seule tache (26), autres (11).
- **Ombres** : une partie est comptée comme insectes, et la vérité terrain contient elle-même des ombres.
- **Photo seule** : la solution 3 a besoin des images d'avant et d'après du même jour.
- **Paramètres** (seuil, aire minimale, taille du flou) choisis à la main ou en regardant les résultats, pas validés sur des données séparées.

## Si j'avais plus de temps

- Recaler chaque image sur son fond (`cv2.phaseCorrelate`) pour corriger les mouvements de la plateforme.
- Filtrer les taches selon leur forme