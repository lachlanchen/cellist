[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


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

## Lazeal Cellist：高效率的 3D 細胞偵測與分析平台

歡迎使用 Lazeal Cellist。這是一個完整且高效率的 3D 顯微影像細胞偵測、分割與分析平台。

本平台透過非監督式學習、閾值法，以及 Cellpose 等先進演算法進行細胞偵測。Lazeal Cellist 也提供直覺且可互動的介面，讓使用者可精修偵測結果。這些精修後的結果會回饋至半監督式學習網路，持續提升模型效能。

Lazeal Cellist 的特色在於提供一個高效率且易於訓練與精修的 3D 模型，讓科學家、研究人員與愛好者都能在實務上有效使用。

---

## 🔍 概覽

Lazeal Cellist 是一個以 Python/Tornado 建構的顯微影像工作流程 Web 平台，提供：

- 以瀏覽器完成上傳、模型建立與標註編輯。
- 演算法輔助初始化（Cellpose nuclei 模式）。
- 透過 WebSocket 動作（`initialize`, `pretrain`, `train`, `update`, `reset`）進行反覆的人機協作式精修。
- 以資料庫持久化儲存模型、影像切片與標註。

目前行為說明：雖然專案願景與 UI 包含 3D 概念（`/3d`, `templates/cellist_3d.html`），但目前程式碼中的主要訓練流程仍以 2D 切片 + 模型精修為主。

### 快速總覽

| 領域 | 目前實作 |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Database | MySQL (`cellist.sql`) |
| Core ML stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inference init | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ 主要功能

- **非監督式 3D 細胞偵測**：使用先進機器學習技術，在 3D 顯微影像中識別細胞。
- **互動式結果精修介面**：透過直覺且友善的介面精修偵測結果。
- **高效率半監督式學習網路**：利用精修結果持續提升模型效能。
- **細胞分割與分析**：不只偵測，亦提供進階分割與分析能力。

目前程式內已具備的其他實作功能：

- 在 `8887` 埠提供 Tornado REST + WebSocket 伺服器（`app.py`）。
- 自動影像切磚（預設 `256x256`）供模型使用。
- 以 dump 形式提供 MySQL schema：[`cellist.sql`](cellist.sql)。
- 前端技術棧包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。

## 🗂️ 專案結構

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

## ✅ 先決條件

| Requirement | Notes |
|---|---|
| OS | 建議 Linux（以下命令假設 Linux shell 行為）。 |
| Python/Conda | 需可使用 Conda，並能依 [`cellist.yaml`](cellist.yaml) 建立環境。 |
| Database | MySQL 伺服器於 `localhost` 執行，且有 `cellist` 資料庫。 |
| GPU | 目前程式路徑強烈建議/預期使用 NVIDIA/CUDA 環境。 |
| Node.js + npm | 需安裝前端相依套件 `statics/node_modules`。 |

## 🛠️ 安裝

### 1. Clone 並進入儲存庫

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 建立 Python 環境

請使用儲存庫中的檔名 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

保留舊文件相容性說明：先前文件曾使用 `celist.yaml`（少一個 `l`），但本儲存庫中的檔案是 `cellist.yaml`。

### 3. 安裝前端相依套件

```bash
cd statics
npm install
cd ..
```

### 4. 準備 MySQL 驗證（如有需要）

若 root 驗證為 socket-based 且導致 app 無法存取，舊版專案文件建議改為密碼驗證：

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. 建立資料庫並還原 schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

保留舊版文件範例：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. 設定執行期 MySQL 憑證

目前程式會從 [`cellist/utils/constants.py`](cellist/utils/constants.py) 讀取憑證（`mysqlconfig` 與 `mysqlurl`）。

目前程式中的預設值包含：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

為了本機安全，請在你的環境中執行前更新這些值。

### 7. 準備執行期資料目錄

此 app 預期存在 `data/` 目錄樹（且 `.gitignore` 已排除 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

備註：[`create_data_folder.py`](create_data_folder.py) 存在，但目前會在目前工作目錄建立資料夾（不是建在 `data/` 之下）。若要使用，請留意此行為。

## 🚀 使用方式

### 啟動 Web 伺服器

```bash
python app.py
```

保留舊版文件中的啟動命令：

```bash
python app.py -m cellist
```

程式碼中觀察到的伺服器預設路由：

- Main UI: `http://localhost:8887/`
- 3D page: `http://localhost:8887/3d`

### 典型工作流程

1. 開啟 UI 並登入。
2. 從 Create Model 面板上傳顯微影像。
3. 選擇基礎演算法（`Cellpose`）並建立模型。
4. 讓後端切片影像並初始化偵測。
5. 載入裁切後影像，檢視/調整矩形標註。
6. 執行 `initialize`、`pretrain`、`train` 週期。
7. 透過 `Update Model`/標註動作儲存手動更新。

### UI 內建登入帳密（目前 template 行為）

前端目前在 client-side 檢查以下靜態帳密：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

這是原型行為，並非生產環境等級的驗證機制。

## ⚙️ 設定

### 後端與端點

設定於 [`app.py`](app.py)：

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型/資料行為

- Thread pool 大小為 `max_workers=64`。
- 影像切磚預設為 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 與 `gpu=True`。
- Training 與 pretraining 透過 WebSocket 觸發動作非同步執行。

## 🧪 範例

### 範例：WebSocket create 訊息格式

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

### 範例：WebSocket 手動標註更新

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### 範例：模型載入請求

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

## 📚 研究啟發

Lazeal Cellist 受到以下深度學習前沿研究啟發：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

這些研究為本平台演算法與方法論的發展提供了重要洞見。

（註：如需精確引用資訊，請直接參閱原始論文。）

## 🧭 開發備註

- 核心模型類別位於 `cellist/`（`ModelD2Init`, `ModelD2Pretrain`）。
- 主要互動式 UI 邏輯直接寫在 `templates/cellist.html`。
- SQL schema 與 seed 風格資料位於 `cellist.sql`。
- `notebooks/` 與 `polygon_sample/` 中的 notebook 提供探索性參考。
- 目前在儲存庫根目錄尚無專用自動化測試套件或 CI 設定。

## 🧯 疑難排解

| Symptom | Suggested checks |
|---|---|
| `ModuleNotFoundError` 或 import 問題 | 執行 `python app.py` 前，確認已先執行 `conda activate cellist`。 |
| UI 有渲染但無樣式/腳本 | 在 `statics/` 內執行 `npm install`，並確認 `statics/node_modules` 存在。 |
| MySQL access denied | 檢查 `cellist/utils/constants.py` 中的使用者名稱/密碼，以及 MySQL plugin/auth 模式。 |
| App 可啟動但 model 動作失敗 | 檢查 CUDA/GPU 可用性；目前路徑假設使用 CUDA（`torch.device('cuda:0')`, Cellpose `gpu=True`）。 |
| Upload 成功但沒有 tiles/models | 確認 `data/` 子目錄存在且具可寫入權限。 |

## 🗺️ 路線圖

以下項目為從既有專案文件/TODO 備註中保留並整理而來：

- Polygon sample：使用 polygon 取代 rectangle 標註。
- 針對極小/極大值情境，優化模型在 `float32` 下的行為。
- 在可行情況下縮小模型尺寸。
- 以 Transformer/stable-diffusion 啟發元件等方法提升穩健性。
- 新增 base-model 選項（Threshold、Cellpose）與 target-model 選項（AIR、Transformer、SD）。
- 介面優化（包含多選）。
- 後端優化（包含更好的記憶體/cache 處理）。
- 提供容易使用且最少 DB 設定的封裝方式（例如 SQLite 選項）。

## 🤝 貢獻

### 為 Lazeal Cellist 做出貢獻

Lazeal Cellist 是開源專案，歡迎所有人參與，不限經驗程度。我們歡迎能夠：

- 提升演算法效率與效能
- 改善使用者介面與使用體驗
- 擴充文件與範例
- 修復錯誤並強化系統穩定性

在開始貢獻之前，請先透過 issue 討論你想進行的變更。這有助於協調工作，避免重複或衝突。

關於如何開始的更多資訊，請閱讀貢獻指南。

儲存庫內其他貢獻文件：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 授權

本專案採用 MIT License。更多資訊請參考本儲存庫中的 [LICENSE](https://chat.openai.com/LICENSE) 檔案。

儲存庫狀態備註：目前此 checkout 的根目錄尚無 `LICENSE` 檔案。上列內容為沿用既有 README 的專案原始意圖；若需要，可於後續變更中新增本機 `LICENSE` 檔案。
