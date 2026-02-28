[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="Bannière LazyingArt" />
</p>

# Lazeal Cellist

![Status](https://img.shields.io/badge/status-research%20prototype-blue)
![Backend](https://img.shields.io/badge/backend-Tornado-00A3E0)
![ML](https://img.shields.io/badge/ML-PyTorch%20%2B%20Pyro%20%2B%20Cellpose-orange)
![DB](https://img.shields.io/badge/database-MySQL-4479A1)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![UI](https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3)

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: votre plateforme efficace de détection et profilage cellulaire 3D

Bienvenue sur Lazeal Cellist, une plateforme complète et efficace de détection, segmentation et profilage cellulaire pour les images de microscopie 3D.

Notre plateforme est conçue pour détecter des cellules à l'aide d'apprentissage non supervisé, de techniques de seuillage et d'algorithmes de pointe comme Cellpose. Lazeal Cellist propose aussi une interface interactive et intuitive permettant d'affiner les résultats de détection. Ces résultats affinés sont ensuite réinjectés dans le réseau d'apprentissage semi-supervisé, ce qui améliore en continu les performances du modèle.

Lazeal Cellist se distingue en offrant un modèle 3D efficace, nécessitant peu d'efforts d'entraînement et d'ajustement, ce qui en fait une plateforme pratique pour scientifiques, chercheurs et passionnés.

---

## 🔍 Vue d'ensemble

Lazeal Cellist est une plateforme web Python/Tornado pour les workflows d'images de microscopie, avec:

- Téléversement via navigateur, création de modèles et édition d'annotations.
- Initialisation assistée par algorithme (mode noyaux Cellpose).
- Raffinement itératif humain-dans-la-boucle via actions WebSocket (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Persistance adossée à une base de données pour les modèles, tranches d'images et annotations.

Note sur le comportement actuel: bien que la vision du projet et l'interface incluent des concepts 3D (`/3d`, `templates/cellist_3d.html`), le flux d'entraînement principal actuel dans le code est surtout basé sur des tranches 2D + raffinement de modèle.

### Aperçu rapide

| Domaine | Implémentation actuelle |
|---|---|
| Serveur | Tornado (`app.py`) |
| Port | `8887` |
| Base de données | MySQL (`cellist.sql`) |
| Pile ML principale | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Initialisation inférence | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Fonctionnalités clés

- **Détection cellulaire 3D non supervisée**: Identifiez des cellules dans des images de microscopie 3D avec des techniques avancées de machine learning.
- **Interface interactive d'affinage des résultats**: Affinez les résultats de détection avec une interface intuitive et conviviale.
- **Réseau d'apprentissage semi-supervisé efficace**: Améliorez les performances du modèle au fil du temps grâce aux résultats affinés.
- **Segmentation et profilage cellulaire**: Allez au-delà de la détection avec des capacités avancées de segmentation et de profilage.

Fonctionnalités d'implémentation supplémentaires actuellement présentes:

- Serveur Tornado REST + WebSocket (`app.py`) sur le port `8887`.
- Découpage automatique des images (`256x256` par défaut) pour l'ingestion modèle.
- Schéma MySQL inclus sous forme de dump: [`cellist.sql`](cellist.sql).
- La pile frontend inclut Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Structure du projet

```text
cellist/
├── app.py
├── cellist/                     # Code ML/modèle (PyTorch + Pyro)
├── templates/                   # Interface HTML (pages 2D + 3D)
├── statics/                     # Ressources frontend + dépendances npm
├── notebooks/                   # Expériences et notebooks exploratoires
├── polygon_sample/              # Exploration d'annotations polygonales
├── cellist.sql                  # Dump schéma/données MySQL
├── cellist.yaml                 # Environnement Conda
├── create_data_folder.py        # Outil historique de bootstrap de dossiers
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md
└── i18n/                        # Présent, actuellement vide
```

## ✅ Prérequis

| Exigence | Notes |
|---|---|
| OS | Linux recommandé (les commandes ci-dessous supposent un shell Linux). |
| Python/Conda | Conda disponible pour créer l'environnement depuis [`cellist.yaml`](cellist.yaml). |
| Base de données | Serveur MySQL en cours d'exécution sur `localhost` avec la base `cellist`. |
| GPU | Environnement NVIDIA/CUDA fortement recommandé/attendu par les chemins de code actuels. |
| Node.js + npm | Requis pour installer les dépendances frontend dans `statics/node_modules`. |

## 🛠️ Installation

### 1. Cloner et entrer dans le dépôt

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Créer l'environnement Python

Utilisez le fichier du dépôt nommé `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Note de compatibilité conservée depuis l'ancienne documentation: une documentation précédente utilisait `celist.yaml` (sans un `l`), mais le fichier de ce dépôt est `cellist.yaml`.

### 3. Installer les dépendances frontend

```bash
cd statics
npm install
cd ..
```

### 4. Préparer l'authentification MySQL (si nécessaire)

Si l'authentification root est basée sur socket et bloque l'accès de l'application, l'ancienne documentation du projet suggère de passer à une authentification par mot de passe:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Créer la base et restaurer le schéma/données

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Exemple historique de la documentation (conservé):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. Configurer les identifiants MySQL à l'exécution

Le code actuel lit les identifiants dans [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` et `mysqlurl`).

Les valeurs par défaut actuellement présentes dans le code incluent:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Pour la sécurité locale, mettez ces valeurs à jour avant l'exécution dans votre environnement.

### 7. Préparer les répertoires de données à l'exécution

L'application attend une arborescence `data/` (et `.gitignore` exclut déjà `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Note: [`create_data_folder.py`](create_data_folder.py) existe, mais crée actuellement les répertoires dans le dossier courant (pas sous `data/`). Gardez cela en tête si vous l'utilisez.

## 🚀 Utilisation

### Démarrer le serveur web

```bash
python app.py
```

Commande de démarrage historique issue de l'ancienne documentation (conservée):

```bash
python app.py -m cellist
```

Routes serveur par défaut observées dans le code:

- Interface principale: `http://localhost:8887/`
- Page 3D: `http://localhost:8887/3d`

### Workflow typique

1. Ouvrez l'interface et connectez-vous.
2. Téléversez des images de microscopie depuis le panneau Create Model.
3. Choisissez l'algorithme de base (`Cellpose`) et créez le modèle.
4. Laissez le backend découper les images et initialiser les détections.
5. Chargez les images recadrées, puis vérifiez/ajustez les annotations rectangulaires.
6. Exécutez les cycles `initialize`, `pretrain` et `train`.
7. Persistez les mises à jour manuelles via les actions `Update Model`/annotation.

### Identifiants de connexion UI intégrés (comportement actuel des templates)

Le frontend vérifie actuellement ces identifiants statiques côté client:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Il s'agit d'un comportement de prototype et non d'une authentification de production.

## ⚙️ Configuration

### Backend et endpoints

Configurés dans [`app.py`](app.py):

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Comportement modèle/données

- La taille du pool de threads est `max_workers=64`.
- Les tuiles d'image sont en `256x256` par défaut.
- L'initialisation Cellpose utilise `model_type='nuclei'` et `gpu=True`.
- L'entraînement et le pré-entraînement s'exécutent de manière asynchrone via des actions déclenchées en WebSocket.

## 🧪 Exemples

### Exemple: forme d'un message WebSocket de création

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

### Exemple: mise à jour WebSocket d'annotation manuelle

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Exemple: requête de chargement de modèle

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

## 📚 Inspiré par la recherche

Lazeal Cellist s'inspire de recherches de pointe en deep learning, notamment:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Ces travaux apportent des enseignements précieux qui ont guidé le développement des algorithmes et méthodologies de la plateforme.

(Note: pour une citation précise, référez-vous directement aux articles originaux.)

## 🧭 Notes de développement

- Les classes de modèle principales se trouvent dans `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La logique principale de l'interface interactive est intégrée directement dans `templates/cellist.html`.
- Le schéma SQL et des données de type seed se trouvent dans `cellist.sql`.
- Les notebooks dans `notebooks/` et `polygon_sample/` fournissent des références exploratoires.
- Il n'existe actuellement pas de suite de tests automatisés dédiée ni de configuration CI à la racine du dépôt.

## 🧯 Dépannage

| Symptôme | Vérifications suggérées |
|---|---|
| `ModuleNotFoundError` ou problèmes d'import | Vérifiez que `conda activate cellist` a bien été appliqué avant `python app.py`. |
| L'UI s'affiche sans styles/scripts | Exécutez `npm install` dans `statics/` et vérifiez que `statics/node_modules` existe. |
| Accès MySQL refusé | Vérifiez le nom d'utilisateur/mot de passe dans `cellist/utils/constants.py` ainsi que le mode plugin/auth MySQL. |
| L'application démarre mais les actions modèle échouent | Vérifiez la disponibilité CUDA/GPU; les chemins actuels supposent CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Le téléversement réussit mais aucune tuile/modèle n'apparaît | Assurez-vous que les sous-répertoires `data/` existent et sont inscriptibles. |

## 🗺️ Feuille de route

Les éléments suivants sont conservés et organisés depuis la documentation et les notes TODO existantes du projet:

- Exemple polygonal: utiliser des annotations polygonales au lieu de rectangles.
- Optimiser le modèle pour le comportement `float32` avec des valeurs très petites/très grandes.
- Réduire la taille du modèle lorsque possible.
- Améliorer la robustesse avec des approches de type Transformer/composants inspirés de stable-diffusion.
- Ajouter des options de modèle de base (Threshold, Cellpose) et de modèle cible (AIR, Transformer, SD).
- Optimisation de l'interface (y compris la sélection multiple).
- Optimisation backend (y compris une meilleure gestion mémoire/cache).
- Packaging simple à utiliser avec une configuration DB minimale (ex.: option SQLite).

## 🤝 Contribution

### Contribuer à Lazeal Cellist

Lazeal Cellist est un projet open source, et nous accueillons les contributions de toutes et tous, quel que soit le niveau d'expérience. Nous encourageons les contributions qui:

- Améliorent l'efficacité et les performances algorithmiques
- Améliorent l'interface utilisateur et l'expérience utilisateur
- Étendent la documentation et les exemples
- Corrigent les bugs et renforcent la stabilité du système

Avant de commencer à contribuer, veuillez d'abord discuter du changement souhaité via une issue. Cela aide à coordonner les efforts et à éviter le travail dupliqué ou conflictuel.

Pour plus d'informations sur la marche à suivre, veuillez lire les directives de contribution.

Documents additionnels de contribution du dépôt:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 Licence

Ce projet est distribué sous licence MIT. Pour plus d'informations, veuillez consulter le fichier [LICENSE](https://chat.openai.com/LICENSE) de ce dépôt.

Note sur l'état du dépôt: aucun fichier `LICENSE` à la racine n'est actuellement présent dans ce checkout. La ligne ci-dessus est conservée depuis le README précédent comme intention canonique du projet; ajoutez un fichier `LICENSE` local dans un changement ultérieur si souhaité.
