[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>您的高效3D細胞檢測與分析平台</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-blue" alt="Status" />
  <img src="https://img.shields.io/badge/backend-Tornado-00A3E0" alt="Backend" />
  <img src="https://img.shields.io/badge/ML-PyTorch%20%2B%20Pyro%20%2B%20Cellpose-orange" alt="ML" />
  <img src="https://img.shields.io/badge/database-MySQL-4479A1" alt="DB" />
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey" alt="Platform" />
  <img src="https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3" alt="UI" />
  <img src="https://img.shields.io/badge/port-8887-success" alt="Port" />
</p>

<p align="center">
  <a href="#-概覽"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-安裝"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-使用"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-故障排查"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-貢獻"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist：您的高效3D細胞檢測與分析平台

歡迎使用 Lazeal Cellist，這是一個完整且高效的 3D 顯微鏡影像細胞檢測、分割與分析平台。

本平台透過無監督學習、閾值法以及 Cellpose 等先進演算法進行細胞檢測。Lazeal Cellist 也提供直覺且可互動的介面，讓使用者可精修檢測結果。這些精修後的結果會回饋至半監督式學習網路，持續提升模型效能。

Lazeal Cellist 的特色在於提供一個高效率且易於訓練與精修的 3D 模型，讓科學家、研究人員與愛好者都能在實務上有效使用。

> ℹ️ **範圍說明**
> 專案願景與 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），但目前程式碼中的主要訓練流程仍以 2D 切片 + 精修為主。

---

## 目錄

- [概覽](#-概覽)
- [核心特性](#-核心特性)
- [專案結構](#-專案結構)
- [先決條件](#-先決條件)
- [安裝](#-安裝)
- [使用](#-使用)
- [配置](#-配置)
- [範例](#-範例)
- [研究啟發](#-研究啟發)
- [開發說明](#-開發說明)
- [故障排查](#-故障排查)
- [路線圖](#-路線圖)
- [貢獻](#-貢獻)
- [致謝](#-致謝)
- [Support](#-support)
- [許可證](#-許可證)

## 🔍 概覽

Lazeal Cellist 是一個基於 Python/Tornado 的顯微影像工作流程 Web 平台，提供：

- 在瀏覽器中完成上傳、模型建立與標註編輯。
- 演算法輔助初始化（Cellpose nuclei 模式）。
- 透過 WebSocket 動作（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）進行人機協作式反覆精修。
- 資料庫持久化儲存模型、影像切片與標註。

> ℹ️ 當前行為說明：雖然專案願景與 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），但目前程式碼中的主要訓練流程仍以 2D 切片 + 模型精修為主。

### 快速一覽

| 模組 | 目前實作 |
|---|---|
| 伺服器 | Tornado（`app.py`） |
| 連接埠 | `8887` |
| 資料庫 | MySQL（`cellist.sql`） |
| 核心 ML 技術棧 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推理初始化 | Cellpose（`model_type='nuclei'`，`gpu=True`） |
| 打包狀態 | 研究原型（無 `pyproject.toml`/`setup.py`） |
| 測試/CI 狀態 | 倉庫根目錄未配置專用自動化測試套件或 CI |

### 文件語言

本倉庫已在 `i18n/` 下包含多語系 README：

| 語言 | 檔案 |
|---|---|
| 阿拉伯文 | `README.ar.md` |
| 德文 | `README.de.md` |
| 西班牙文 | `README.es.md` |
| 法文 | `README.fr.md` |
| 日文 | `README.ja.md` |
| 韓文 | `README.ko.md` |
| 俄文 | `README.ru.md` |
| 越南文 | `README.vi.md` |

## ✨ 核心特性

- **無監督式 3D 細胞檢測**：使用先進機器學習技術，在 3D 顯微影像中識別細胞。
- **互動式結果精修介面**：透過直覺且友好的介面反覆精修檢測結果。
- **高效半監督式學習網路**：利用精修結果持續提升模型效能。
- **細胞分割與分析**：除檢測外，亦提供進階分割與分析能力。

目前程式內已具備的其他實作特性：

- 在 `8887` 埠提供 Tornado REST + WebSocket 伺服器（`app.py`）。
- 自動影像切片（預設 `256x256`）供模型使用。
- 以 dump 形式提供 MySQL schema：[`cellist.sql`](cellist.sql)。
- 前端技術棧包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。
- 透過執行緒池非同步執行模型任務（`max_workers=64`）。

## 🗂️ 專案結構

```text
cellist/
├── app.py                               # 主 Tornado 服務 + REST/WebSocket 處理器
├── cellist/                             # 核心 ML/模型代碼
│   ├── model_init.py                    # 2D 主模型類與訓練流程
│   ├── model_pretrain.py                # 預訓練變體
│   ├── model_2d_components.py           # Encoder/Decoder/SPAIR 元件
│   ├── model_2d_utilities.py            # 資料庫驅動的模型中介資料與變換
│   ├── image_preprocessing.py           # 切片/拼接工具
│   └── utils/constants.py               # 運行路徑 + MySQL 設定
├── templates/
│   ├── cellist.html                     # 主要 2D 介面
│   └── cellist_3d.html                  # 3D 介面變體（原型）
├── statics/                             # 前端資源與 npm 依賴
│   ├── package.json
│   └── node_modules/
├── i18n/                                # 翻譯版 README
├── notebooks/                           # 探索型筆記本
├── polygon_sample/                      # 多邊形標註實驗
├── figs/                                # 品牌素材
├── cellist.sql                          # MySQL schema 與資料導出
├── cellist.yaml                         # Conda 環境規格
├── create_data_folder.py                # 舊式資料目錄建立腳本
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # 擴展架構/TODO 說明
└── README.md
```

## ✅ 先決條件

| 需求 | 說明 |
|---|---|
| 作業系統 | 推薦 Linux（以下命令按 Linux shell 行為執行）。 |
| Python/Conda | 需要 Conda，可基於 [`cellist.yaml`](cellist.yaml) 建立環境。 |
| 資料庫 | 本地 `localhost` 上需執行 MySQL，並存在 `cellist` 資料庫。 |
| GPU | 當前程式路徑強烈建議/預期在 NVIDIA/CUDA 環境下運行。 |
| Node.js + npm | 需要用於安裝 `statics/node_modules` 前端依賴。 |
| 磁碟寫入權限 | 執行時需要 `<repo>/data` 下的資料寫入權限。 |

## 🛠️ 安裝

### 1. 複製並進入倉庫

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 建立 Python 環境

使用倉庫檔名 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

兼容性說明保留於舊文件：先前文件曾使用 `celist.yaml`（少一個 `l`），但本倉庫檔案為 `cellist.yaml`。

保留的舊命令：

```bash
conda env create -f celist.yaml
```

### 3. 安裝前端依賴

```bash
cd statics
npm install
cd ..
```

### 4. 準備執行時資料目錄

應用需要 `data/` 目錄樹（`.gitignore` 已經排除了 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

備註：[`create_data_folder.py`](create_data_folder.py) 存在，但目前會在目前工作目錄建立目錄（而非建在 `data/` 之下）。若要使用請留意此行為。

### 5. 準備 MySQL 認證（如需）

若 root 認證為 socket-based 且阻擋應用存取，舊版文件建議改為密碼認證：

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. 建立資料庫並還原 schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

保留的舊示例（用於參考）：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 設定執行期 MySQL 憑證

目前程式從 [`cellist/utils/constants.py`](cellist/utils/constants.py) 讀取憑證（`mysqlconfig` 與 `mysqlurl`）。

目前程式的預設值包含：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

為了本機安全，請在你的環境中執行前更新這些值。

### 8. 可選：環境健康檢查

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 使用

### 啟動 Web 伺服器

```bash
python app.py
```

保留的舊啟動命令（來自歷史文檔）：

```bash
python app.py -m cellist
```

程式中可見的預設路由：

- 主介面：`http://localhost:8887/`
- 3D 頁面：`http://localhost:8887/3d`

### 典型工作流

1. 開啟 UI 並登入。
2. 在 Create Model 面板上傳顯微鏡影像。
3. 選擇基礎演算法（`Cellpose`）並建立模型。
4. 等待後端切片並初始化檢測。
5. 載入裁切後影像，檢視並調整矩形標註。
6. 執行 `initialize`、`pretrain` 與 `train` 週期。
7. 依需求使用 `Pretrain Stop`/`Stop`（`train-stop`）/`reset`。
8. 透過 `Update Model`/標註動作持久化手動更新。

### 內建 UI 登入憑證（目前模板行為）

前端目前在客戶端檢查以下靜態憑證：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

這是原型行為，並非生產環境等級的認證機制。

### API/WebSocket 介面（UI 當前使用）

HTTP 介面：

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 介面：

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket 處理器中已識別的 `data_type` 動作：

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ 配置

### 後端與端點

配置位於 [`app.py`](app.py)：

- 端口：`8887`
- 路由：
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型/資料行為

- 執行緒池大小為 `max_workers=64`。
- 預設影像切片大小為 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 與 `gpu=True`。
- 訓練與預訓練透過 WebSocket 觸發動作非同步執行。
- 資料根路徑從目前工作目錄解析為 `<repo>/data`。

### 資料庫/執行期常數

來自 [`cellist/utils/constants.py`](cellist/utils/constants.py)：

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` 包含 host/user/password 欄位
- `mysqlurl` 指向資料庫 `cellist`

### 前端依賴快照

來自 [`statics/package.json`](statics/package.json)：

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda 環境要點

來自 [`cellist.yaml`](cellist.yaml)：

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2`（pip）
- Pyro（`pyro-ppl==1.8.1`）
- PyMySQL + SQLAlchemy

## 🧪 範例

### 範例：WebSocket create 訊息結構

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

### 範例：最小端對端本機啟動

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 研究啟發

Lazeal Cellist 受到前沿深度學習研究的啟發，包括：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

這些工作為本平台的演算法與方法論提供了重要設計方向。

（註：如需精確引用，請直接參考原始論文。）

## 🧭 開發說明

- 核心模型類位於 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要互動式 UI 邏輯直接嵌入 `templates/cellist.html`。
- SQL schema 與種子風格資料位於 `cellist.sql`。
- `notebooks/` 與 `polygon_sample/` 提供探索性參考。
- 擴展的平台/模型說明位於 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)。
- 倉庫根目錄目前沒有專用自動化測試套件或 CI 設定。

### 當前假設與限制

- 倉庫似乎優先面向本地研究場景使用。
- 部分程式路徑預設可用 GPU（`cuda:0`）。
- 認證與密鑰管理仍為原型級。
- 3D 介面已有實作，但主訓練流程仍是以 2D 切片為核心。

## 🧯 故障排查

| 症狀 | 建議檢查 |
|---|---|
| `ModuleNotFoundError` 或匯入問題 | 確認執行 `python app.py` 前已執行 `conda activate cellist`。 |
| UI 有渲染但無樣式/腳本 | 在 `statics/` 下執行 `npm install`，確認 `statics/node_modules` 已存在。 |
| MySQL access denied | 檢查 `cellist/utils/constants.py` 中的使用者名稱、密碼與 MySQL plugin/auth 模式。 |
| 應用已啟動但模型動作失敗 | 檢查 CUDA/GPU 可用性；目前路徑預設使用 CUDA（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| 上傳成功但無 tiles/models | 確認 `data/` 子目錄存在且可寫。 |
| REST/WebSocket 請求報錯 | 確認服務執行於 `http://localhost:8887`，且請求 payload 鍵名與目前模板一致。 |
| `data/` 下報 `FileNotFoundError` | 從倉庫根目錄啟動應用，以保證相對路徑解析一致。 |

### 快速診斷

```bash
# 驗證 Python 環境和關鍵匯入
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 確認服務啟動後端口是否監聽
ss -ltnp | rg 8887

# 檢查 MySQL 連通性
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ 路線圖

以下條目來源於既有專案文件/TODO，並依現況整理：

- Polygon sample：使用 polygon 替代 rectangle 標註。
- 優化 `float32` 在極小/極大值場景下的行為。
- 盡可能縮小模型體積。
- 採用 Transformer / stable-diffusion 啟發組件等方法提升穩健性。
- 增加更多基模型選項（Threshold、Cellpose）與目標模型選項（AIR、Transformer、SD）。
- 介面優化（含批次多選）。
- 後端優化（包含更好的記憶體／快取處理）。
- 提供可即用的打包方案，降低資料庫配置門檻（例如 SQLite 選項）。

## 🤝 貢獻

### 為 Lazeal Cellist 貢獻

Lazeal Cellist 是一個開源專案，歡迎所有人參與，無論經驗程度。以下是我們鼓勵的貢獻類型：

- 提升演算法效率與效能
- 改進介面與使用者體驗
- 豐富文件與範例
- 修正缺陷並提高系統穩定性

開始前請先透過 issue 討論你想做的改動，以便協調工作，避免重複或衝突。

更多上手資訊請閱讀貢獻指南。

倉庫補充貢獻文件：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 致謝

- Lazeal Cellist 的概念與實作大量借鏡上述 AIR/SPAIR 研究脈絡。
- 倉庫保留了歷史/遺留文件與命令，以維持與先前專案使用方式的連續性。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 許可證

本專案採用 MIT 授權。更多資訊請參考本倉庫的 [LICENSE](LICENSE) 檔案。

倉庫狀態說明：目前檢出的根目錄中尚未包含 `LICENSE` 檔案。上方說明保留自先前 README 作為專案規範意圖；如需要可在後續更新中補充本地 `LICENSE` 檔案。
