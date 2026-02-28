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
  <a href="#-depannage"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Dépannage" /></a>
  <a href="#-contribution"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contribution" /></a>
</p>

## 🎬 Aperçu

![Capture d’écran 2D](screenshot2d.png)

## Lazeal Cellist : votre plateforme efficace de détection et de profilage cellulaires 3D

Bienvenue dans Lazeal Cellist, une plateforme complète et efficace de détection, segmentation et profilage cellulaires pour les images de microscopie 3D.

La plateforme a été conçue pour détecter les cellules via l’apprentissage non supervisé, des techniques de seuillage et des algorithmes de pointe comme Cellpose. Lazeal Cellist offre également une interface interactive et intuitive qui permet aux utilisateurs d’affiner les résultats de détection. Ces résultats affinés sont ensuite réinjectés dans le réseau d’apprentissage semi-supervisé, améliorant progressivement les performances du modèle.

Lazeal Cellist se distingue par un modèle 3D efficace qui demande peu d’efforts pour l’entraînement et l’affinage, ce qui en fait une plateforme pratique pour les scientifiques, chercheurs et amateurs.

> ℹ️ **Note de périmètre**
> La vision du projet et l’interface utilisateur incluent des concepts 3D (`/3d`, `templates/cellist_3d.html`), tandis que le flux d’entraînement principal implémenté aujourd’hui repose majoritairement sur du découpage 2D + affinage.

---

## Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités clés](#-fonctionnalit%C3%A9s-cl%C3%A9s)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-pr%C3%A9requis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Exemples](#-exemples)
- [Inspiré par la recherche](#-inspir%C3%A9-par-la-recherche)
- [Notes de développement](#-notes-de-d%C3%A9veloppement)
- [Dépannage](#-d%C3%A9pannage)
- [Feuille de route](#-feuille-de-route)
- [Contribution](#-contribution)
- [Remerciements](#-remerciements)
- [Support](#-support)
- [Licence](#-licence)

## 🔍 Vue d'ensemble

Lazeal Cellist est une plateforme web Python/Tornado pour les flux de travail d’imagerie microscopique avec :

- Chargement depuis le navigateur, création de modèle et édition d’annotations.
- Initialisation assistée par algorithmes (mode noyaux de Cellpose).
- Raffinement itératif en boucle homme-machine via des actions WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Persistance basée sur la base de données pour les modèles, tranches d’image et annotations.

> ℹ️ Note sur le comportement actuel : bien que la vision du projet et l’UI incluent des concepts 3D (`/3d`, `templates/cellist_3d.html`), le flux principal d’entraînement dans le code est pour l’instant principalement basé sur du découpage 2D + affinage du modèle.

### En un coup d’œil

| Zone | Mise en œuvre actuelle |
|---|---|
| Serveur | Tornado (`app.py`) |
| Port | `8887` |
| Base de données | MySQL (`cellist.sql`) |
| Stack ML principale | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Initialisation d’inférence | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Statut du packaging | Prototype de recherche (pas de `pyproject.toml`/`setup.py`) |
| Statut des tests/CI | Pas de suite de tests dédiée ni de configuration CI dans la racine du dépôt |

### Langues de la documentation

Ce dépôt inclut déjà des fichiers README multilingues dans `i18n/` :

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

- **Détection cellulaire 3D non supervisée** : identifier des cellules dans des images de microscopie 3D grâce à des techniques de machine learning avancées.
- **Interface interactive d’affinage des résultats** : améliorer les résultats de détection via une interface intuitive et conviviale.
- **Réseau d’apprentissage semi-supervisé efficace** : améliorer progressivement les performances du modèle grâce aux résultats affinés.
- **Segmentation et profilage cellulaires** : aller au-delà de la détection avec des capacités de segmentation et de profilage avancées.

Fonctionnalités d’implémentation supplémentaires présentes actuellement :

- Serveur REST + WebSocket Tornado (`app.py`) sur le port `8887`.
- Tuilage automatique des images (`256x256` par défaut) pour l’ingestion du modèle.
- Schéma MySQL inclus sous forme de dump : [`cellist.sql`](cellist.sql).
- Le frontend inclut Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Tâches asynchrones via un pool de threads (`max_workers=64`).

## 🗂️ Structure du projet

```text
cellist/
├── app.py                               # Serveur Tornado principal + handlers REST/WebSocket
├── cellist/                             # Code principal ML/modèle
│   ├── model_init.py                    # Classe principale du modèle 2D et flux d'entraînement
│   ├── model_pretrain.py                # Variante de pré-entraînement
│   ├── model_2d_components.py           # Composants Encodeur/Décodeur/SPAIR
│   ├── model_2d_utilities.py            # Métadonnées modèle adossées à la BDD + transformations
│   ├── image_preprocessing.py           # Outils de découpe/rassemblage
│   └── utils/constants.py               # Chemins runtime + config MySQL
├── templates/
│   ├── cellist.html                     # UI 2D principale
│   └── cellist_3d.html                  # Variante/prototype d'UI 3D
├── statics/                             # Ressources frontend et dépendances npm
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Fichiers README traduits
├── notebooks/                           # Notebooks exploratoires
├── polygon_sample/                      # Expériences d’annotation polygonale
├── figs/                                # Actifs de marque
├── cellist.sql                          # Schéma et données MySQL
├── cellist.yaml                         # Spécification d’environnement Conda
├── create_data_folder.py                # Utilitaire ancien de création de dossier de données
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Notes d’architecture et TODO étendues
└── README.md
```

## ✅ Prérequis

| Exigence | Remarques |
|---|---|
| OS | Linux recommandé (les commandes ci-dessous supposent un comportement de shell Linux). |
| Python/Conda | Conda disponible pour créer l’environnement depuis [`cellist.yaml`](cellist.yaml). |
| Base de données | Serveur MySQL tournant sur `localhost` avec la base `cellist`. |
| GPU | Environnement NVIDIA/CUDA fortement recommandé/prévu par les chemins de code actuels. |
| Node.js + npm | Nécessaire pour installer les dépendances frontend dans `statics/node_modules`. |
| Accès en écriture disque | Nécessaire pour les données runtime sous `<repo>/data`. |

## 🛠️ Installation

### 1. Cloner et ouvrir le dépôt

```bash

git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Créer l’environnement Python

Utilisez le fichier de dépôt `cellist.yaml` :

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Note de compatibilité conservée dans l’ancien texte : la documentation précédente utilisait `celist.yaml` (sans le `l`), mais le fichier présent dans ce dépôt est bien `cellist.yaml`.

Commande historique (conservée) :

```bash
conda env create -f celist.yaml
```

### 3. Installer les dépendances frontend

```bash
cd statics
npm install
cd ..
```

### 4. Préparer les répertoires de données runtime

L’application attend une arborescence `data/` (et `.gitignore` exclut déjà `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Note : [`create_data_folder.py`](create_data_folder.py) existe, mais crée actuellement les répertoires dans le répertoire courant (et non sous `data/`). Gardez cela à l’esprit si vous l’utilisez.

### 5. Préparer l’authentification MySQL (si nécessaire)

Si l’authentification `root` est basée sur socket et bloque l’accès de l’application, l’ancienne documentation du projet suggère de passer à une authentification par mot de passe :

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Créer la base et restaurer le schéma/les données

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Exemple de documentation historique (conservé) :

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configurer les identifiants MySQL pour l’exécution

Le code lit actuellement les identifiants depuis [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` et `mysqlurl`).

Valeurs par défaut actuelles dans le code :

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Pour la sécurité locale, mettez à jour ces valeurs avant d’exécuter l’application.

### 8. Vérifications optionnelles de l’environnement

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

Commande de démarrage legacy conservée (historique) :

```bash
python app.py -m cellist
```

Routes serveur observées dans le code :

- UI principale : `http://localhost:8887/`
- page 3D : `http://localhost:8887/3d`

### Workflow typique

1. Ouvrir l’interface et se connecter.
2. Importer des images de microscopie depuis le panneau **Créer un modèle**.
3. Choisir l’algorithme de base (`Cellpose`) et créer le modèle.
4. Laisser le backend découper les images et initialiser les détections.
5. Charger les images rognées, revoir/ajuster les annotations rectangulaires.
6. Exécuter les cycles `initialize`, `pretrain` et `train`.
7. Utiliser `Pretrain Stop` / `Stop` (`train-stop`) / `reset` si besoin.
8. Sauvegarder les mises à jour manuelles via `Update Model`/actions d’annotation.

### Identifiants de connexion UI intégrés (comportement template actuel)

Le frontend vérifie actuellement ces identifiants statiques côté client :

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Il s’agit d’un comportement de prototype et non d’une authentification de production.

### Endpoints API/WebSocket utilisés actuellement par l’UI

Endpoints HTTP :

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket :

- `ws://localhost:8887/websocket/<ws_uuid>`

Messages `data_type` reconnus par le gestionnaire WebSocket :

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

Configuré dans [`app.py`](app.py) :

- Port : `8887`
- Routes :
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Comportement modèle/données

- La taille du pool de threads est `max_workers=64`.
- Les tuiles d’image sont en `256x256` par défaut.
- L’initialisation Cellpose utilise `model_type='nuclei'` et `gpu=True`.
- L’entraînement et le pré-entraînement s’exécutent de manière asynchrone via des actions déclenchées par WebSocket.
- La racine des données est résolue depuis le répertoire courant comme `<repo>/data`.

### Constantes runtime/base de données

Depuis [`cellist/utils/constants.py`](cellist/utils/constants.py) :

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` inclut les clés host/user/password
- `mysqlurl` cible la base de données nommée `cellist`

### Instantané des dépendances frontend

Depuis [`statics/package.json`](statics/package.json) :

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Points clés de l’environnement Conda

Depuis [`cellist.yaml`](cellist.yaml) :

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Exemples

### Exemple : forme d’un message WebSocket `create`

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

### Exemple : mise à jour manuelle d’annotation via WebSocket

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

Lazeal Cellist s’inspire de travaux récents en deep learning, notamment :

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Ces publications apportent des idées précieuses qui ont guidé le développement des algorithmes et des méthodes de notre plateforme.

(Note : pour des citations exactes, référez-vous directement aux articles originaux.)

## 🧭 Notes de développement

- Les classes de modèle principales se trouvent dans `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La logique d’UI interactive principale est directement intégrée dans `templates/cellist.html`.
- Le schéma SQL et les données de type seed sont dans `cellist.sql`.
- Les notebooks dans `notebooks/` et `polygon_sample/` fournissent des références exploratoires.
- Les notes de plateforme/modèle étendues sont dans [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Il n’existe actuellement pas de suite de tests automatisée dédiée ni de configuration CI dans la racine du dépôt.

### Hypothèses et contraintes actuelles

- Ce dépôt semble cibler d’abord une utilisation locale orientée recherche.
- Certains chemins de code supposent la disponibilité d’un GPU (`cuda:0`).
- L’authentification et la gestion des secrets sont au niveau prototype.
- Les interfaces 3D existent, mais le flux d’entraînement dominant reste orienté découpage 2D.

## 🧯 Dépannage

| Symptôme | Vérifications suggérées |
|---|---|
| `ModuleNotFoundError` ou erreurs d’import | Vérifiez que `conda activate cellist` a bien été exécuté avant `python app.py`. |
| L’UI s’affiche sans style/scripts | Lancez `npm install` dans `statics/` et confirmez que `statics/node_modules` existe. |
| Accès refusé MySQL | Vérifiez le nom d’utilisateur/mot de passe dans `cellist/utils/constants.py` et le mode de plugin/auth MySQL. |
| L’app démarre mais les actions modèle échouent | Vérifiez la disponibilité CUDA/GPU ; les chemins actuels supposent CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| L’upload fonctionne mais aucune tuile/modèle n’apparaît | Assurez-vous que les sous-répertoires `data/` existent et sont inscriptibles. |
| Erreurs de requêtes REST/WebSocket | Confirmez que le serveur tourne sur `http://localhost:8887` et que les clés du payload correspondent aux noms actuels des templates. |
| `FileNotFoundError` sous `data/` | Démarrez l’application depuis la racine du dépôt afin que les chemins relatifs se résolvent de manière cohérente. |

### Diagnostics rapides

```bash
# Vérifier l’environnement Python et les imports clés
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Vérifier que le port du serveur est ouvert après démarrage
ss -ltnp | rg 8887

# Vérifier la connectivité MySQL
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Feuille de route

Les éléments suivants sont conservés et organisés à partir de la documentation/TODO existantes du projet :

- Échantillon polygonal : utiliser des polygones au lieu d’annotations rectangulaires.
- Optimiser le modèle pour un comportement `float32` avec des valeurs très petites/grandes.
- Réduire la taille du modèle quand c’est possible.
- Améliorer la robustesse avec des approches inspirées du Transformer/stable-diffusion.
- Ajouter des options de modèle de base (Threshold, Cellpose) et d’objectif (AIR, Transformer, SD).
- Optimisation de l’interface (dont la sélection multiple).
- Optimisation backend (dont une meilleure gestion mémoire/cache).
- Packaging facile avec configuration DB minimale (par ex. option SQLite).

## 🤝 Contribution

### Contribuer à Lazeal Cellist

Lazeal Cellist est un projet open source, et les contributions sont les bienvenues quel que soit le niveau d’expérience. Nous accueillons les apports qui :

- améliorent l’efficacité et la performance des algorithmes,
- améliorent l’interface utilisateur et l’expérience utilisateur,
- enrichissent la documentation et les exemples,
- corrigent des bugs et renforcent la stabilité du système.

Avant de commencer une contribution, veuillez d’abord discuter du changement prévu via une issue. Cela permet de coordonner les efforts et d’éviter les doublons ou conflits.

Pour plus d’informations sur la prise en main, consultez les consignes de contribution.

Documents complémentaires du dépôt :

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Remerciements

- Le concept et l’implémentation de Lazeal Cellist s’inspirent fortement de la lignée de recherche AIR/SPAIR mentionnée ci-dessus.
- Le dépôt inclut une documentation et des commandes historiques/legacy conservées intentionnellement pour maintenir la continuité avec l’usage antérieur du projet.

## 📄 Licence

Ce projet est sous licence MIT. Pour plus d’informations, consultez le fichier [LICENSE](LICENSE) du dépôt.

Note de statut du dépôt : aucun fichier `LICENSE` racine n’est actuellement présent dans ce checkout. La ligne ci-dessus est conservée depuis le README d’origine comme intention canonique du projet ; ajoutez un fichier `LICENSE` local dans un changement ultérieur si nécessaire.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
