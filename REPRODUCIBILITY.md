# Fichier de Reproductibilité

**Projet 6 — Data Poisoning Attacks on ML-based IDS**  
**KHALLOUK Aymane & HAMDAOUA Ahmed — INPT ICCN INE2 — 2024–2025**

---

## 1. Environnement d'Exécution

### Système utilisé pour les expériences

| Paramètre | Valeur |
|-----------|--------|
| OS | Windows 10 / Linux Ubuntu 22.04 |
| Python | 3.14.2 |
| Processeur | CPU (pas de GPU requis) |
| RAM recommandée | ≥ 8 Go |

### Versions exactes des bibliothèques

```
torch==2.12.0+cpu
numpy==2.4.2
pandas==3.0.0
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
```

---

## 2. Installation des Dépendances

### Option A — pip (recommandé)

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.2
pip install pandas==3.0.0
pip install scikit-learn==1.5.0
pip install matplotlib==3.9.0
pip install seaborn==0.13.2
pip install jupyter
```

### Option B — environnement virtuel (isolation complète)

```bash
# Créer l'environnement
python -m venv venv_poisoning

# Activer (Linux/macOS)
source venv_poisoning/bin/activate

# Activer (Windows)
venv_poisoning\Scripts\activate

# Installer les dépendances
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.2 pandas==3.0.0 scikit-learn==1.5.0
pip install matplotlib==3.9.0 seaborn==0.13.2 jupyter
```

### Option C — Google Colab

Aucune installation requise. Modifier uniquement la variable `DATA_DIR` dans la Cellule 2 pour pointer vers les fichiers uploadés dans `/content/`.

---

## 3. Préparation des Données

### Téléchargement

1. Accéder à : https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Télécharger les fichiers :
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

### Placement

```
projet6/
├── deep_learning.ipynb
├── UNSW_NB15_training-set.csv    ← placer ici
├── UNSW_NB15_testing-set.csv     ← placer ici
├── README.md
└── REPRODUCIBILITY.md
```

### Configuration dans le notebook

Ouvrir `deep_learning.ipynb` et modifier la **Cellule 2** :

```python
DATA_DIR = "."   # si les CSV sont dans le même dossier que le notebook
# ou
DATA_DIR = "/chemin/absolu/vers/les/données"
```

---

## 4. Exécution

### Lancement du notebook

```bash
# Depuis le répertoire du projet
jupyter notebook deep_learning.ipynb

# ou avec JupyterLab
jupyter lab deep_learning.ipynb
```

### Ordre d'exécution des cellules

Exécuter **toutes les cellules dans l'ordre** (Kernel → Restart & Run All) :

```
Cellule 0  : Imports
Cellule 1  : Seeds fixes (OBLIGATOIRE — doit être exécutée avant tout)
Cellule 2  : Chargement données
Cellule 3  : Préprocessing
Cellule 4  : Architecture AutoencoderMLP
Cellule 5  : Boucle d'entraînement
Cellule 6  : Fonction d'évaluation
Cellule 7  : Entraînement baseline
Cellule 8  : Évaluation baseline
Cellule 9  : Visualisation baseline
Cellule 10 : Attaque 1 — Clean-Label Feature Poisoning 10%
Cellule 11 : Attaque 2 — Random Label Flip 10%
Cellule 12 : Attaque 3 — Targeted Label Flip 10%
Cellule 13 : Visualisation par attaque
Cellule 14 : Comparaison multi-attaques
Cellule 15+: Défense hybride (scout model + filtrage + réentraînement)
```

> ⚠️ Ne pas sauter la Cellule 1. Les seeds doivent être fixés avant tout chargement ou initialisation.

---

## 5. Seeds et Reproductibilité

Tous les générateurs aléatoires sont fixés via la fonction `set_seed(42)` appelée en Cellule 1 et avant chaque entraînement :

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

Les splits train/val/test sont effectués avec `random_state=42` et `stratify=y` pour garantir la même répartition à chaque exécution.

---

## 6. Paramètres des Expériences

### Préprocessing

| Étape | Détail |
|-------|--------|
| Suppression colonne `id` | Identifiant non informatif |
| Encodage one-hot | `proto`, `service`, `state` → `pd.get_dummies(..., drop_first=True)` |
| Normalisation | `StandardScaler` — **fit sur train uniquement**, transform sur val et test |
| Split | 70% train / 15% val / 15% test, stratifié sur le label binaire |

### Entraînement (commun à toutes les expériences)

```python
epochs     = 400
lr         = 3e-4        # Adam optimizer
batch_size = 1024
alpha_rec  = 0.3         # poids de la perte de reconstruction
patience   = 60          # early stopping sur la val F1
```

### Architecture

```python
encoder_dims    = [512, 256, 128]
latent_dim      = 64
classifier_dims = [64, 32]
dropout_rate    = 0.25
```

### Attaque 1 — Clean-Label Feature Poisoning

```python
poison_rate = 0.10     # 10% des échantillons d'attaque
epsilon     = 0.30     # force du déplacement vers le centroïde normal
noise_std   = 0.05     # bruit gaussien ajouté après déplacement
seed        = 42
```

### Attaque 2 — Random Label Flip

```python
flip_rate = 0.10    # 10% de TOUS les labels retournés aléatoirement
seed      = 42
```

### Attaque 3 — Targeted Label Flip

```python
flip_rate = 0.10    # 10% des labels d'ATTAQUE retournés vers Normal
seed      = 42
# Résultat : 6.39% du corpus total empoisonné
```

### Défense Hybride

```python
scout_epochs      = 50     # époques pour le modèle scout
loss_percentile   = 12     # seuil : top 12% décroissances de perte rapides
rec_percentile    = 12     # seuil : top 12% erreurs de reconstruction élevées
combine_method    = 'union'  # flagging si l'un OU l'autre critère est satisfait
```

---

## 7. Résultats Attendus

En exécutant le notebook avec les paramètres ci-dessus, les résultats suivants doivent être reproduits (±0.001 lié aux variations de parallélisme CPU) :

| Condition | Accuracy | F1-Score | PR-AUC | FNR | FPR |
|-----------|----------|----------|--------|-----|-----|
| Baseline | 0.9396 | 0.9524 | 0.9942 | 0.0554 | 0.0692 |
| Clean-Label 10% | 0.9395 | 0.9519 | 0.9941 | 0.0628 | 0.0564 |
| Random Flip 10% | 0.9376 | 0.9508 | 0.9930 | 0.0562 | 0.0735 |
| Targeted Flip 10% | 0.9373 | 0.9501 | 0.9932 | 0.0669 | 0.0552 |
| Défense Hybride | 0.9045 | 0.9261 | 0.9841 | 0.0629 | 0.1533 |

### Matrices de confusion attendues

```
Baseline :
  TN=12 985  FP=965   FN=1 369   TP=23 332

Clean-Label 10% :
  TN=13 163  FP=787   FN=1 552   TP=23 149

Random Flip 10% :
  TN=12 925  FP=1 025  FN=1 387  TP=23 314

Targeted Flip 10% :
  TN=13 180  FP=770   FN=1 653   TP=23 048

Défense Hybride :
  TN=11 767  FP=1 178  FN=1 556  TP=23 150
```

---

## 8. Résolution de Problèmes

### Erreur : fichier CSV introuvable

```
FileNotFoundError: [Errno 2] No such file or directory: '.../UNSW_NB15_training-set.csv'
```
→ Vérifier le chemin dans `DATA_DIR` (Cellule 2). Utiliser un chemin absolu si nécessaire.

### Résultats légèrement différents

Les petites variations (< 0.001) sont normales sur CPU en raison du non-déterminisme des opérations matricielles parallèles. Vérifier que la Cellule 1 (`set_seed(42)`) a bien été exécutée en premier.

### Entraînement trop lent

Réduire `epochs=400` à `epochs=100` pour un test rapide. Les résultats seront sous-optimaux mais reproductibles en structure.

### Mémoire insuffisante

Réduire `batch_size` de 1024 à 512 ou 256. Les résultats peuvent varier légèrement mais restent comparables.
