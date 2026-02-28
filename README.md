[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# Lazeal Cellist

![Status](https://img.shields.io/badge/status-research%20prototype-blue)
![Backend](https://img.shields.io/badge/backend-Tornado-00A3E0)
![ML](https://img.shields.io/badge/ML-PyTorch%20%2B%20Pyro%20%2B%20Cellpose-orange)
![DB](https://img.shields.io/badge/database-MySQL-4479A1)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![UI](https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3)

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Your Efficient 3D Cell Detection and Profiling Platform

Welcome to Lazeal Cellist, a comprehensive and efficient cell detection, segmentation, and profiling platform for 3D microscopy images.

Our platform is designed to detect cells using unsupervised learning, thresholding techniques, and state-of-the-art algorithms such as Cellpose. Lazeal Cellist also provides an intuitive, interactive interface that allows users to refine detection results. These refined results are then fed back into the semi-supervised learning network, continuously improving model performance.

Lazeal Cellist stands out by offering an efficient 3D model that requires minimal effort to train and refine, making it a practical platform for scientists, researchers, and hobbyists.

---

## 🔍 Overview

Lazeal Cellist is a Python/Tornado web platform for microscopy-image workflows with:

- Browser-based upload, model creation, and annotation editing.
- Algorithm-assisted initialization (Cellpose nuclei mode).
- Iterative human-in-the-loop refinement via WebSocket actions (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Database-backed persistence for models, image slices, and annotations.

Note on current behavior: although the project vision and UI include 3D concepts (`/3d`, `templates/cellist_3d.html`), the current main training flow in code is primarily 2D slicing + model refinement.

### Quick At-a-Glance

| Area | Current implementation |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Database | MySQL (`cellist.sql`) |
| Core ML stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inference init | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Key Features

- **Unsupervised 3D Cell Detection**: Identify cells in 3D microscopy images using advanced machine learning techniques.
- **Interactive Result Refinement Interface**: Refine detection results with an intuitive, user-friendly interface.
- **Efficient Semi-Supervised Learning Network**: Enhance model performance over time with refined results.
- **Cell Segmentation and Profiling**: Go beyond detection with advanced segmentation and profiling capabilities.

Additional implementation features currently present:

- Tornado REST + WebSocket server (`app.py`) on port `8887`.
- Automatic image tiling (`256x256` by default) for model ingestion.
- MySQL schema included as dump: [`cellist.sql`](cellist.sql).
- Frontend stack includes Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Project Structure

```text
cellist/
├── app.py
├── cellist/                     # ML/model code (PyTorch + Pyro)
├── templates/                   # HTML UI (2D + 3D pages)
├── statics/                     # Frontend assets + npm deps
├── notebooks/                   # Experiments and exploratory notebooks
├── polygon_sample/              # Polygon annotation exploration
├── cellist.sql                  # MySQL schema/data dump
├── cellist.yaml                 # Conda environment
├── create_data_folder.py        # Legacy folder bootstrap helper
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md
└── i18n/                        # Present, currently empty
```

## ✅ Prerequisites

| Requirement | Notes |
|---|---|
| OS | Linux recommended (commands below assume Linux shell behavior). |
| Python/Conda | Conda available to create environment from [`cellist.yaml`](cellist.yaml). |
| Database | MySQL server running on `localhost` with database `cellist`. |
| GPU | NVIDIA/CUDA environment strongly recommended/expected by current code paths. |
| Node.js + npm | Required to install `statics/node_modules` frontend dependencies. |

## 🛠️ Installation

### 1. Clone and enter repository

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Create Python environment

Use the repository file name `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Compatibility note preserved from older docs: previous documentation used `celist.yaml` (missing an `l`), but the file in this repository is `cellist.yaml`.

### 3. Install frontend dependencies

```bash
cd statics
npm install
cd ..
```

### 4. Prepare MySQL authentication (if needed)

If root authentication is socket-based and blocks app access, older project docs suggest switching to password auth:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Create database and restore schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Legacy documentation example (preserved):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. Configure MySQL credentials for runtime

Current code reads credentials from [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` and `mysqlurl`).

Default code values currently include:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

For local security, update these values before running in your environment.

### 7. Prepare runtime data directories

The app expects a `data/` tree (and `.gitignore` already excludes `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Note: [`create_data_folder.py`](create_data_folder.py) exists, but it currently creates directories in the current working directory (not under `data/`). Keep this in mind if you use it.

## 🚀 Usage

### Start the web server

```bash
python app.py
```

Legacy startup command from prior docs (preserved):

```bash
python app.py -m cellist
```

Server route defaults observed in code:

- Main UI: `http://localhost:8887/`
- 3D page: `http://localhost:8887/3d`

### Typical workflow

1. Open the UI and log in.
2. Upload microscopy images from the Create Model panel.
3. Choose base algorithm (`Cellpose`) and create model.
4. Let backend slice images and initialize detections.
5. Load cropped images, review/adjust rectangle annotations.
6. Run `initialize`, `pretrain`, and `train` cycles.
7. Persist manual updates via `Update Model`/annotation actions.

### Built-in UI login credentials (current template behavior)

The frontend currently checks these static credentials client-side:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

This is prototype behavior and not production authentication.

## ⚙️ Configuration

### Backend and endpoints

Configured in [`app.py`](app.py):

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Model/data behavior

- Thread pool size is `max_workers=64`.
- Image tiles are `256x256` by default.
- Cellpose initialization uses `model_type='nuclei'` and `gpu=True`.
- Training and pretraining run asynchronously through WebSocket-triggered actions.

## 🧪 Examples

### Example: WebSocket create message shape

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

### Example: WebSocket manual annotation update

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Example: Model load request

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

## 📚 Inspired by Research

Lazeal Cellist is inspired by cutting-edge research in deep learning, including:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

These works provide valuable insights that have guided the development of our platform's algorithms and methodologies.

(Note: For accurate citation, please refer directly to the original papers.)

## 🧭 Development Notes

- Core model classes are under `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Main interactive UI logic is embedded directly in `templates/cellist.html`.
- SQL schema and seed-style data are in `cellist.sql`.
- Notebooks in `notebooks/` and `polygon_sample/` provide exploratory references.
- There is currently no dedicated automated test suite or CI configuration in the repository root.

## 🧯 Troubleshooting

| Symptom | Suggested checks |
|---|---|
| `ModuleNotFoundError` or import issues | Confirm `conda activate cellist` was applied before running `python app.py`. |
| UI renders without styling/scripts | Run `npm install` inside `statics/` and confirm `statics/node_modules` exists. |
| MySQL access denied | Verify username/password in `cellist/utils/constants.py` and MySQL plugin/auth mode. |
| App starts but model actions fail | Check CUDA/GPU availability; current paths assume CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload succeeds but no tiles/models appear | Ensure `data/` subdirectories exist and are writable. |

## 🗺️ Roadmap

The following items are preserved and organized from existing project documentation/TODO notes:

- Polygon sample: use polygon instead of rectangle annotation.
- Optimize model for `float32` behavior with very small/large values.
- Shrink model size where possible.
- Improve robustness with approaches such as Transformer/stable-diffusion-inspired components.
- Add base-model options (Threshold, Cellpose) and target-model options (AIR, Transformer, SD).
- Interface optimization (including multiple selection).
- Backend optimization (including improved memory/cache handling).
- Easy-to-use packaging with minimal DB configuration (e.g., SQLite option).

## 🤝 Contributing

### Contribute to Lazeal Cellist

Lazeal Cellist is an open-source project, and we welcome contributions from everyone, regardless of experience level. We invite contributions that:

- Enhance algorithmic efficiency and performance
- Improve the user interface and user experience
- Expand documentation and examples
- Fix bugs and enhance system stability

Before you start contributing, please first discuss the change you wish to make via an issue. This helps coordinate efforts and avoid duplicate or conflicting work.

For more information on how to get started, please read the contributing guidelines.

Additional repository contribution docs:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 License

This project is licensed under the MIT License. For more information, please refer to the [LICENSE](https://chat.openai.com/LICENSE) file in this repository.

Repository status note: no root `LICENSE` file is currently present in this checkout. The line above is preserved from the prior README as canonical project intent; add a local `LICENSE` file in a follow-up change if desired.
