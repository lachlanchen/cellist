[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Votre plateforme efficace de détection et de profilage cellulaires 3D</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-blue" alt="Statut" />
  <img src="https://img.shields.io/badge/backend-Tornado-00A3E0" alt="Backend" />
  <img src="https://img.shields.io/badge/ML-PyTorch%20%2B%20Pyro%20%2B%20Cellpose-orange" alt="ML" />
  <img src="https://img.shields.io/badge/database-MySQL-4479A1" alt="DB" />
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey" alt="Plateforme" />
  <img src="https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3" alt="UI" />
  <img src="https://img.shields.io/badge/port-8887-success" alt="Port" />
</p>

<p align="center">
  <a href="#-vue-densemble"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Vue d'ensemble" /></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-utilisation"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Utilisation" /></a>
  <a href="#-dépannage"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Dépannage" /></a>
  <a href="#-contribution"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contribution" /></a>
</p>

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist : votre plateforme efficace de détection et de profilage cellulaires 3D

Bienvenue dans Lazeal Cellist, une plateforme complète et efficace de détection, segmentation et profilage cellulaires pour les images de microscopie 3D.

Notre plateforme est conçue pour détecter les cellules via l'apprentissage non supervisé, des techniques de seuillage et des algorithmes de pointe comme Cellpose. Lazeal Cellist fournit aussi une interface intuitive et interactive qui permet d'affiner les résultats de détection. Ces résultats affinés sont ensuite réinjectés dans le réseau d'apprentissage semi-supervisé, ce qui améliore en continu les performances du modèle.

Lazeal Cellist se distingue par un modèle 3D efficace qui demande peu d'effort d'entraînement et d'affinage, ce qui en fait une plateforme pratique pour les scientifiques, chercheurs et passionnés.

> ℹ️ **Note de périmètre**
> La vision du projet et l'UI incluent des concepts 3D (`/3d`, `templates/cellist_3d.html`), tandis que le flux principal d'entraînement dans le code est surtout basé sur du slicing 2D + affinage.

---

## Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités clés](#-fonctionnalités-clés)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Exemples](#-exemples)
- [Inspiré par la recherche](#-inspiré-par-la-recherche)
- [Notes de développement](#-notes-de-développement)
- [Dépannage](#-dépannage)
- [Feuille de route](#-feuille-de-route)
- [Contribution](#-contribution)
- [Remerciements](#-remerciements)
- [Support](#-support)
- [Licence](#-licence)

## 🔍 Vue d'ensemble

Lazeal Cellist est une plateforme web Python/Tornado pour les workflows d'images de microscopie avec :

- Téléversement via navigateur, création de modèle et édition d'annotations.
- Initialisation assistée par algorithme (mode noyaux Cellpose).
- Affinage itératif humain-dans-la-boucle via actions WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Persistance adossée à la base de données pour les modèles, coupes d'images et annotations.

> ℹ️ Note sur le comportement actuel : même si la vision du projet et l'UI incluent des concepts 3D (`/3d`, `templates/cellist_3d.html`), le flux principal d'entraînement dans le code repose surtout sur du slicing 2D + affinage du modèle.

### Vue rapide

| Zone | Implémentation actuelle |
|---|---|
| Serveur | Tornado (`app.py`) |
| Port | `8887` |
| Base de données | MySQL (`cellist.sql`) |
| Stack ML principale | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Initialisation d'inférence | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Statut du packaging | Prototype de recherche (pas de `pyproject.toml`/`setup.py`) |
| Statut tests/CI | Pas de suite de tests dédiée ni de config CI à la racine du dépôt |

### Langues de la documentation

Ce dépôt inclut déjà des README multilingues dans `i18n/` :

| Langue | Fichier |
|---|---|
| Arabe | `README.ar.md` |
| Allemand | `README.de.md` |
| Espagnol | `README.es.md` |
| Japonais | `README.ja.md` |
| Coréen | `README.ko.md` |
| Russe | `README.ru.md` |
| Vietnamien | `README.vi.md` |
| Chinois (simplifié) | `README.zh-Hans.md` |
| Chinois (traditionnel) | `README.zh-Hant.md` |

## ✨ Fonctionnalités clés

- **Détection cellulaire 3D non supervisée** : identifiez des cellules dans des images de microscopie 3D avec des techniques avancées de machine learning.
- **Interface interactive d'affinage des résultats** : affinez les résultats de détection avec une interface intuitive et simple à utiliser.
- **Réseau d'apprentissage semi-supervisé efficace** : améliorez la performance du modèle dans le temps grâce aux résultats affinés.
- **Segmentation et profilage cellulaires** : allez au-delà de la détection avec des capacités avancées de segmentation et de profilage.

Fonctionnalités d'implémentation supplémentaires actuellement présentes :

- Serveur Tornado REST + WebSocket (`app.py`) sur le port `8887`.
- Découpage automatique d'image (`256x256` par défaut) pour l'ingestion du modèle.
- Schéma MySQL inclus sous forme de dump : [`cellist.sql`](cellist.sql).
- La stack frontend inclut Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Tâches modèle asynchrones via un pool de threads (`max_workers=64`).

## 🗂️ Structure du projet

```text
cellist/
├── app.py                               # Main Tornado server + REST/WebSocket handlers
├── cellist/                             # Core ML/model code
│   ├── model_init.py                    # Main 2D model class and train flow
│   ├── model_pretrain.py                # Pretrain variant
│   ├── model_2d_components.py           # Encoder/Decoder/SPAIR components
│   ├── model_2d_utilities.py            # DB-backed model metadata + transforms
│   ├── image_preprocessing.py           # Slice/stitch helpers
│   └── utils/constants.py               # Runtime paths + MySQL config
├── templates/
│   ├── cellist.html                     # Primary 2D UI
│   └── cellist_3d.html                  # 3D UI variant/prototype
├── statics/                             # Frontend assets and npm dependencies
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Translated README files
├── notebooks/                           # Exploratory notebooks
├── polygon_sample/                      # Polygon annotation experiments
├── figs/                                # Branding assets
├── cellist.sql                          # MySQL schema/data dump
├── cellist.yaml                         # Conda environment specification
├── create_data_folder.py                # Legacy data-folder creation helper
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Extended architecture/TODO notes
└── README.md
```

## ✅ Prérequis

| Exigence | Notes |
|---|---|
| OS | Linux recommandé (les commandes ci-dessous supposent un shell Linux). |
| Python/Conda | Conda disponible pour créer l'environnement depuis [`cellist.yaml`](cellist.yaml). |
| Base de données | Serveur MySQL en cours d'exécution sur `localhost` avec la base `cellist`. |
| GPU | Environnement NVIDIA/CUDA fortement recommandé/attendu par les chemins actuels du code. |
| Node.js + npm | Requis pour installer les dépendances frontend `statics/node_modules`. |
| Accès en écriture disque | Nécessaire pour les données d'exécution sous `<repo>/data`. |

## 🛠️ Installation

### 1. Cloner et entrer dans le dépôt

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Créer l'environnement Python

Utilisez le nom de fichier du dépôt `cellist.yaml` :

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Note de compatibilité conservée depuis les anciennes docs : la documentation précédente utilisait `celist.yaml` (sans un `l`), mais le fichier dans ce dépôt est `cellist.yaml`.

Commande legacy (conservée) :

```bash
conda env create -f celist.yaml
```

### 3. Installer les dépendances frontend

```bash
cd statics
npm install
cd ..
```

### 4. Préparer les répertoires de données d'exécution

L'application attend une arborescence `data/` (et `.gitignore` exclut déjà `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Note : [`create_data_folder.py`](create_data_folder.py) existe, mais il crée actuellement des dossiers dans le répertoire de travail courant (pas sous `data/`). Gardez cela en tête si vous l'utilisez.

### 5. Préparer l'authentification MySQL (si nécessaire)

Si l'authentification root est basée sur un socket et bloque l'accès de l'app, les anciennes docs du projet suggèrent de passer à une auth par mot de passe :

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Créer la base et restaurer le schéma/données

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Exemple de documentation legacy (conservé) :

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configurer les identifiants MySQL pour l'exécution

Le code actuel lit les identifiants depuis [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` et `mysqlurl`).

Les valeurs par défaut du code incluent actuellement :

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Pour la sécurité locale, mettez à jour ces valeurs avant exécution dans votre environnement.

### 8. Vérifications optionnelles de l'environnement

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Utilisation

### Démarrer le serveur web

```bash
python app.py
```

Commande de démarrage legacy issue d'anciennes docs (conservée) :

```bash
python app.py -m cellist
```

Routes serveur observées par défaut dans le code :

- UI principale : `http://localhost:8887/`
- Page 3D : `http://localhost:8887/3d`

### Workflow typique

1. Ouvrez l'UI et connectez-vous.
2. Téléversez des images de microscopie depuis le panneau Create Model.
3. Choisissez l'algorithme de base (`Cellpose`) et créez le modèle.
4. Laissez le backend découper les images et initialiser les détections.
5. Chargez les images découpées, puis vérifiez/ajustez les annotations rectangulaires.
6. Lancez les cycles `initialize`, `pretrain` et `train`.
7. Utilisez `Pretrain Stop` / `Stop` (`train-stop`) / `reset` selon le besoin.
8. Persistez les mises à jour manuelles via les actions `Update Model`/annotation.

### Identifiants de connexion UI intégrés (comportement actuel des templates)

Le frontend vérifie actuellement ces identifiants statiques côté client :

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

C'est un comportement de prototype, pas une authentification de production.

### Surface API/Socket actuellement utilisée par l'UI

Endpoints HTTP :

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

Endpoint WebSocket :

- `ws://localhost:8887/websocket/<ws_uuid>`

Messages d'action `data_type` reconnus dans le handler WebSocket :

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuration

### Backend et endpoints

Configurés dans [`app.py`](app.py) :

- Port : `8887`
- Routes :
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Comportement modèle/données

- La taille du pool de threads est `max_workers=64`.
- Les tuiles d'images sont en `256x256` par défaut.
- L'initialisation Cellpose utilise `model_type='nuclei'` et `gpu=True`.
- L'entraînement et le pré-entraînement s'exécutent de manière asynchrone via des actions déclenchées en WebSocket.
- La racine de données est résolue depuis le répertoire de travail courant en `<repo>/data`.

### Constantes base/runtime

Depuis [`cellist/utils/constants.py`](cellist/utils/constants.py) :

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` inclut des clés host/user/password
- `mysqlurl` cible la base `cellist`

### Aperçu des dépendances frontend

Depuis [`statics/package.json`](statics/package.json) :

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Points clés de l'environnement Conda

Depuis [`cellist.yaml`](cellist.yaml) :

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Exemples

### Exemple : structure d'un message WebSocket create

```json
{
  "data_type": "create",
  "arguments": {
    "username": "lachlan",
    "model_name": "experiment_001",
    "based_on_algorithm": "Cellpose",
    "local_images_uuid": ["<image_uuid_1>", "<image_uuid_2>"],
    "slice_height": 256,
    "slice_width": 256
  }
}
```

### Exemple : mise à jour manuelle d'annotation via WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Exemple : requête de chargement de modèle

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Exemple : démarrage local minimal de bout en bout

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 Inspiré par la recherche

Lazeal Cellist est inspiré par des travaux de pointe en deep learning, notamment :

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Ces travaux fournissent des enseignements précieux qui ont guidé le développement des algorithmes et méthodologies de la plateforme.

(Remarque : pour une citation exacte, référez-vous directement aux articles originaux.)

## 🧭 Notes de développement

- Les classes cœur du modèle sont sous `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La logique interactive principale de l'UI est intégrée directement dans `templates/cellist.html`.
- Le schéma SQL et les données de type seed sont dans `cellist.sql`.
- Les notebooks de `notebooks/` et `polygon_sample/` servent de références exploratoires.
- Des notes étendues plateforme/modèle se trouvent dans [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Il n'existe actuellement pas de suite de tests automatisée dédiée ni de configuration CI à la racine du dépôt.

### Hypothèses et contraintes actuelles

- Ce dépôt semble viser d'abord un usage local orienté recherche.
- Certains chemins de code supposent la disponibilité d'un GPU (`cuda:0`).
- L'authentification et la gestion des secrets sont de niveau prototype.
- Les interfaces 3D existent, mais le workflow dominant d'entraînement reste orienté tuiles 2D.

## 🧯 Dépannage

| Symptôme | Vérifications suggérées |
|---|---|
| `ModuleNotFoundError` ou problèmes d'import | Vérifiez que `conda activate cellist` a été appliqué avant `python app.py`. |
| L'UI s'affiche sans styles/scripts | Exécutez `npm install` dans `statics/` et vérifiez l'existence de `statics/node_modules`. |
| Accès MySQL refusé | Vérifiez identifiant/mot de passe dans `cellist/utils/constants.py` et le mode plugin/auth MySQL. |
| L'app démarre mais les actions modèle échouent | Vérifiez la disponibilité CUDA/GPU ; les chemins actuels supposent CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Le téléversement réussit mais aucune tuile/aucun modèle n'apparaît | Assurez-vous que les sous-répertoires `data/` existent et sont accessibles en écriture. |
| Erreurs de requêtes REST/WebSocket | Vérifiez que le serveur tourne sur `http://localhost:8887` et que les clés de payload correspondent aux noms actuels des templates. |
| `FileNotFoundError` sous `data/` | Démarrez l'app depuis la racine du dépôt pour une résolution cohérente des chemins relatifs. |

### Diagnostics rapides

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Feuille de route

Les éléments suivants sont conservés et organisés à partir de la documentation existante du projet et des notes TODO :

- Polygon sample : utiliser une annotation polygonale au lieu d'une annotation rectangulaire.
- Optimiser le modèle pour le comportement `float32` avec de très petites/grandes valeurs.
- Réduire la taille du modèle quand c'est possible.
- Améliorer la robustesse avec des approches inspirées notamment des Transformers/stable diffusion.
- Ajouter des options de modèle de base (Threshold, Cellpose) et de modèle cible (AIR, Transformer, SD).
- Optimisation de l'interface (y compris la sélection multiple).
- Optimisation du backend (y compris une meilleure gestion mémoire/cache).
- Packaging simple d'utilisation avec configuration DB minimale (ex. option SQLite).

## 🤝 Contribution

### Contribuer à Lazeal Cellist

Lazeal Cellist est un projet open source, et nous accueillons les contributions de tout le monde, quel que soit le niveau d'expérience. Nous invitons les contributions qui :

- Améliorent l'efficacité et les performances algorithmiques
- Améliorent l'interface utilisateur et l'expérience utilisateur
- Étendent la documentation et les exemples
- Corrigent des bugs et améliorent la stabilité du système

Avant de commencer à contribuer, discutez d'abord du changement souhaité via une issue. Cela aide à coordonner les efforts et à éviter les travaux dupliqués ou conflictuels.

Pour plus d'informations pour démarrer, lisez les directives de contribution.

Documentation de contribution supplémentaire du dépôt :

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Remerciements

- Le concept et l'implémentation de Lazeal Cellist s'appuient fortement sur la lignée de recherche AIR/SPAIR listée ci-dessus.
- Le dépôt inclut volontairement de la documentation/commandes historiques (legacy) pour la continuité avec les usages précédents du projet.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 Licence

Ce projet est sous licence MIT. Pour plus d'informations, consultez le fichier [LICENSE](LICENSE) dans ce dépôt.

Note sur l'état du dépôt : aucun fichier `LICENSE` à la racine n'est actuellement présent dans ce checkout. La ligne ci-dessus est conservée depuis le README précédent comme intention canonique du projet ; ajoutez un fichier `LICENSE` local dans une modification ultérieure si besoin.
