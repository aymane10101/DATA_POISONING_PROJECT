# Projet 6 — Data Poisoning Attacks on ML-based IDS

**Module :** Deep Learning – Cybersécurité (ICCN INE2)  
**Étudiants :** KHALLOUK Aymane & HAMDAOUA Ahmed  
**Encadrant :** Prof. Tarik Fissaa  
**École :** INPT — Institut National des Postes et Télécommunications  
**Année :** 2024–2025

---

## Description

Ce projet étudie les **attaques par empoisonnement des données d'entraînement** appliquées à un système de détection d'intrusions (IDS) basé sur l'apprentissage profond. Le modèle cible est un **AutoencoderMLP** combinant un auto-encodeur et un classifieur binaire (normal / attaque), entraîné sur le dataset **UNSW-NB15**.

Trois attaques de niveaux de capacité croissants sont implémentées et évaluées, suivies d'une défense hybride :

| # | Attaque | Capacité attaquant | Objectif |
|---|---------|-------------------|----------|
| 1 | Clean-Label Feature Poisoning | Modification des features uniquement | Brouiller la frontière de décision |
| 2 | Random Label Flip | Accès aux étiquettes, stratégie aléatoire | Perturber le système globalement |
| 3 | Targeted Label Flip (Atk→Norm) | Accès aux étiquettes, stratégie ciblée | Maximiser le FNR (attaques manquées) |
| — | Défense Hybride | — | Détecter et supprimer les données empoisonnées |

---

## Structure du Projet

```
.
├── deep_learning.ipynb      # Notebook principal (toutes les expériences)
├── README.md                # Ce fichier
├── REPRODUCIBILITY.md       # Commandes et dépendances pour reproduire les résultats
└── deep_learning_report.pdf     # Rapport PDF (soumis séparément en PDF)
```

---

## Dataset

**UNSW-NB15** — disponible à : https://research.unsw.edu.au/projects/unsw-nb15-dataset

Fichiers requis :
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Placer les fichiers dans le répertoire de travail et mettre à jour la variable `DATA_DIR` dans la **Cellule 2** du notebook :

```python
DATA_DIR = "/chemin/vers/les/données"
```

Statistiques du dataset :

| Élément | Valeur |
|---------|--------|
| Échantillons totaux | 257 673 |
| Classe Normale (label=0) | 93 000 (36.1%) |
| Classe Attaque (label=1) | 164 673 (63.9%) |
| Features après encodage | 193 |
| Partition | 70% train / 15% val / 15% test |

---

## Architecture — AutoencoderMLP

```
Input (193)
    │
    ▼
Encoder: Linear(193→512) → BN → ReLU → Dropout(0.25)
         Linear(512→256)  → BN → ReLU → Dropout(0.25)
         Linear(256→128)  → BN → ReLU → Dropout(0.25)
         Linear(128→64)   → ReLU
    │
    ├──► Decoder: Linear(64→128) → ... → Linear(512→193)   [reconstruction]
    │
    └──► Classifier: Linear(64→32) → BN → ReLU → Linear(32→2)  [logits]

Loss = CrossEntropy(logits, y) + 0.3 × MSE(x_hat, x)
```

---

## Contenu du Notebook

| Cellule | Description |
|---------|-------------|
| 0 | Imports et vérification de l'environnement |
| 1 | Fixation des seeds (reproductibilité) |
| 2 | Chargement du dataset UNSW-NB15 |
| 3 | Préprocessing (encodage, normalisation, splits) |
| 4 | Définition de l'architecture AutoencoderMLP |
| 5 | Boucle d'entraînement avec early stopping |
| 6 | Fonction d'évaluation (Accuracy, F1, PR-AUC, FNR, FPR) |
| 7–9 | Entraînement et visualisation du modèle baseline |
| 10 | Attaque 1 — Clean-Label Feature Poisoning 10% |
| 11 | Attaque 2 — Random Label Flip 10% |
| 12 | Attaque 3 — Targeted Label Flip 10% (Atk→Norm) |
| 13–14 | Visualisations comparatives des attaques |
| 15+ | Défense hybride (trajectoires de perte + reconstruction) |

---

## Résultats Principaux

| Condition | Accuracy | F1-Score | PR-AUC | FNR ↓ | FPR |
|-----------|----------|----------|--------|-------|-----|
| Baseline (données propres) | 0.9396 | 0.9524 | 0.9942 | 0.0554 | 0.0692 |
| Attaque 1 — Clean-Label 10% | 0.9395 | 0.9519 | 0.9941 | 0.0628 | 0.0564 |
| Attaque 2 — Flip Aléatoire 10% | 0.9376 | 0.9508 | 0.9930 | 0.0562 | 0.0735 |
| Attaque 3 — Flip Ciblé 10% | 0.9373 | 0.9501 | 0.9932 | 0.0669 | 0.0552 |
| Défense Hybride (vs Att. 3) | 0.9045 | 0.9261 | 0.9841 | 0.0629 | 0.1533 |

> **Métrique clé :** Le FNR (False Negative Rate) est la métrique de sécurité la plus critique — il mesure le taux d'attaques non détectées. L'Attaque 3 est la plus dangereuse car elle maximise le FNR de façon ciblée et furtive.

---

## Utilisation

### Exécution complète (ordre recommandé)

Exécuter toutes les cellules du notebook dans l'ordre, de la cellule 0 à la dernière. Le notebook est auto-contenu et produit toutes les visualisations et métriques.

### Exécution partielle

- **Baseline seulement :** Cellules 0 → 9
- **Attaques seulement :** Cellules 0 → 3, puis 4 → 14 (le baseline doit être exécuté avant)
- **Défense seulement :** Toutes les cellules précédentes + cellules de défense

---

## Temps d'Entraînement (CPU)

| Expérience | Durée approx. |
|------------|---------------|
| Baseline | ~70 min |
| Attaque 1 (Clean-Label) | ~62 min |
| Attaque 2 (Random Flip) | ~55 min |
| Attaque 3 (Targeted Flip) | ~56 min |
| Scout model (défense) | ~15 min |
| Réentraînement défense | ~40 min |

> Les temps sont mesurés sur CPU. Avec un GPU, diviser par un facteur ~5–10.

---

## Politique d'Utilisation de l'IA

Conformément à la politique du module, des outils IA ont été utilisés comme assistance. Tout le code, les choix de conception et les résultats ont été compris, vérifiés et justifiés par les étudiants.
