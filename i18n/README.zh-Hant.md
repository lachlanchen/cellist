[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>您的高效 3D 細胞偵測與分析平台</strong>
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
  <a href="#-故障排除"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-參與"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 預覽

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist：您的高效 3D 細胞偵測與分析平台

歡迎使用 Lazeal Cellist，這是一個面向 3D 顯微鏡影像的完整且高效的細胞偵測、分割與分析平台。

本平台採用無監督學習、閾值方法與如 Cellpose 等先進演算法來辨識細胞。Lazeal Cellist 也提供直覺的互動式介面，讓使用者可以精修偵測結果；這些精修結果會回饋至半監督學習網路，持續改善模型效能。

Lazeal Cellist 的突出之處在於其高效能 3D 模型，只需投入極少的訓練與精修工作量，成為研究人員、研究者與愛好者的實用平台。

> ℹ️ **範圍註記**
> 專案願景與 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），但目前程式碼中的主要訓練流程仍為 2D 切片 + 精修。

---

## 目錄

- [概覽](#-概覽)
- [關鍵功能](#-關鍵功能)
- [專案結構](#-專案結構)
- [先決條件](#-先決條件)
- [安裝](#-安裝)
- [使用](#-使用)
- [設定](#-設定)
- [範例](#-範例)
- [研究啟發](#-研究啟發)
- [開發說明](#-開發說明)
- [故障排除](#-故障排除)
- [路線圖](#-路線圖)
- [參與](#-參與)
- [致謝](#-致謝)
- [Support](#-support)
- [授權條款](#-授權條款)

## 🔍 概覽

Lazeal Cellist 是一個針對顯微鏡影像流程的 Python/Tornado 網站平台，具備：

- 瀏覽器上傳、模型建立與標註編輯。
- 演算法輔助初始化（Cellpose nuclei 模式）。
- 透過 WebSocket 動作（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）進行人機協作的反覆精修。
- 資料庫持久化儲存模型、影像切片與標註。

> ℹ️ 目前行為說明：雖然專案願景與 UI 含有 3D 概念（`/3d`、`templates/cellist_3d.html`），但目前程式碼中的主要訓練流程仍以 2D 切片 + 模型精修為主。

### 快速總覽

| 區域 | 目前實作 |
|---|---|
| 伺服器 | Tornado（`app.py`） |
| 連接埠 | `8887` |
| 資料庫 | MySQL（`cellist.sql`） |
| 核心 ML 技術堆疊 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推論初始化 | Cellpose（`model_type='nuclei'`，`gpu=True`） |
| 打包狀態 | 研究原型（未提供 `pyproject.toml`/`setup.py`） |
| 測試/CI 狀態 | 倉庫根目錄未配置專用自動化測試套件或 CI |

### 文件語言

此倉庫已在 `i18n/` 下包含多語系 README：

| 語言 | 檔案 |
|---|---|
| 阿拉伯語 | `README.ar.md` |
| 德文 | `README.de.md` |
| 西班牙文 | `README.es.md` |
| 法文 | `README.fr.md` |
| 日文 | `README.ja.md` |
| 韓文 | `README.ko.md` |
| 俄文 | `README.ru.md` |
| 越南文 | `README.vi.md` |

## ✨ 關鍵功能

- **無監督 3D 細胞偵測**：使用進階機器學習技術，在 3D 顯微鏡影像中識別細胞。
- **互動式結果精修介面**：透過直覺且友善的介面精修偵測結果。
- **高效半監督學習網路**：利用精修結果持續提升模型效能。
- **細胞分割與分析**：不只做偵測，亦支援更深入的分割與特徵分析。

目前實作中尚有以下補充功能：

- 基於 Tornado 的 REST + WebSocket 伺服器（`app.py`），連接埠為 `8887`。
- 自動影像切片（預設 `256x256`）供模型輸入。
- MySQL schema 以匯出檔形式提供：[`cellist.sql`](cellist.sql)。
- 前端技術堆疊包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。
- 透過執行緒池非同步執行模型任務（`max_workers=64`）。

## 🗂️ 專案結構

```text
cellist/
├── app.py                               # 主要 Tornado 服務 + REST/WebSocket 處理器
├── cellist/                             # 核心 ML/模型程式碼
│   ├── model_init.py                    # 2D 主模型類別與訓練流程
│   ├── model_pretrain.py                # 預訓練變體
│   ├── model_2d_components.py           # 編碼器/解碼器/SPAIR 元件
│   ├── model_2d_utilities.py            # 基於資料庫的模型中繼資料 + transforms
│   ├── image_preprocessing.py           # 切片/拼接工具
│   └── utils/constants.py               # 執行時路徑 + MySQL 設定
├── templates/
│   ├── cellist.html                     # 主要 2D 介面
│   └── cellist_3d.html                  # 3D 介面變體/原型
├── statics/                             # 前端資源與 npm 依賴
│   ├── package.json
│   └── node_modules/
├── i18n/                                # 翻譯版 README
├── notebooks/                           # 探索型筆記本
├── polygon_sample/                      # 多邊形標註實驗
├── figs/                                # 品牌素材
├── cellist.sql                          # MySQL schema 與資料匯出
├── cellist.yaml                         # Conda 環境規範
├── create_data_folder.py                # 舊版資料夾建立腳本
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # 擴充架構/TODO 說明
└── README.md
```

## ✅ 先決條件

| 需求 | 說明 |
|---|---|
| 作業系統 | 建議使用 Linux（以下指令依 Linux shell 行為執行）。 |
| Python/Conda | 請確認可用 Conda，用以依據 [`cellist.yaml`](cellist.yaml) 建立環境。 |
| 資料庫 | 需要於 `localhost` 上執行 MySQL 並建立 `cellist` 資料庫。 |
| GPU | 現有程式碼路徑強烈建議/預期搭配 NVIDIA/CUDA 環境。 |
| Node.js + npm | 需安裝以完成 `statics/node_modules` 前端依賴。 |
| 磁碟寫入權限 | 運行時需在 `<repo>/data` 下有寫入權限。 |

## 🛠️ 安裝

### 1. 複製並進入倉庫

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 建立 Python 環境

使用倉庫內檔名 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

版本相容性備註沿用舊文件：舊文件曾使用 `celist.yaml`（少了 `l`），但本倉庫檔案為 `cellist.yaml`。

保留的歷史指令：

```bash
conda env create -f celist.yaml
```

### 3. 安裝前端依賴

```bash
cd statics
npm install
cd ..
```

### 4. 準備執行期資料目錄

應用預期存在 `data/` 目錄樹（`.gitignore` 已排除 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

備註：[`create_data_folder.py`](create_data_folder.py) 已存在，但目前會在目前工作目錄下建立目錄（非在 `data/` 下），使用時請留意。

### 5. 準備 MySQL 認證（如有需要）

如果 root 認證為 socket 且阻擋應用程式存取，舊版文件建議改為密碼認證：

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

保留的歷史文件示例（僅供參考）：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 設定執行時 MySQL 認證

目前程式碼從 [`cellist/utils/constants.py`](cellist/utils/constants.py) 讀取認證（`mysqlconfig` 與 `mysqlurl`）。

目前預設值為：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

為了本機安全，請先更新這些值再在你的環境中執行。

### 8. 選用：環境自我檢查

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

保留的歷史啟動指令（沿用）：

```bash
python app.py -m cellist
```

程式中預設路由觀察如下：

- 主介面：`http://localhost:8887/`
- 3D 頁面：`http://localhost:8887/3d`

### 常見流程

1. 開啟介面並登入。
2. 從「建立模型」面板上傳顯微影像。
3. 選擇基礎演算法（`Cellpose`）並建立模型。
4. 讓後端切片並初始化偵測結果。
5. 載入裁切影像，檢閱/調整矩形標註。
6. 執行 `initialize`、`pretrain` 與 `train` 週期。
7. 視需要使用 `Pretrain Stop` / `Stop`（`train-stop`）/ `reset`。
8. 透過 `Update Model`/標註操作持續寫回手動更新。

### 內建 UI 登入憑證（目前模板行為）

前端目前在客戶端檢查以下靜態憑證：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

這是原型行為，並非正式上線認證。

### UI 目前使用的 API/Socket 介面

HTTP 端點：

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 端點：

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket handler 已辨識的 `data_type` 動作訊息：

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ 設定

### 後端與端點

設定於 [`app.py`](app.py)：

- 連接埠：`8887`
- 路由：
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型與資料行為

- 執行緒池大小為 `max_workers=64`。
- 預設影像切片大小為 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 與 `gpu=True`。
- 訓練與預訓練透過 WebSocket 觸發的動作非同步執行。
- 資料根目錄以目前工作目錄解析為 `<repo>/data`。

### 資料庫/執行時常數

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

### Conda 環境重點

來自 [`cellist.yaml`](cellist.yaml)：

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2`（pip）
- Pyro（`pyro-ppl==1.8.1`）
- PyMySQL + SQLAlchemy

## 🧪 範例

### 範例：WebSocket 建立訊息結構

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

### 範例：WebSocket 手工標註更新

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

### 範例：最小化本機啟動流程

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

Lazeal Cellist 借鑑了前沿深度學習研究，包括：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

這些成果為平台的演算法和方法設計提供了重要參考。

（註：若需精確引用，請直接參閱原始論文。）

## 🧭 開發說明

- 核心模型類別位於 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要互動式 UI 邏輯直接嵌在 `templates/cellist.html`。
- SQL schema 與種子式資料位於 `cellist.sql`。
- `notebooks/` 與 `polygon_sample/` 中的筆記本可作為探索參考。
- 擴充的平台/模型說明請見 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)。
- 倉庫根目錄目前沒有專用自動化測試套件或 CI 設定。

### 假設與目前限制

- 這個倉庫似乎主要服務於本機研究用途。
- 某些程式路徑假設 GPU（`cuda:0`）可用。
- 認證與祕密管理屬原型層級。
- 3D 介面雖已存在，但主訓練流程仍以 2D 切片為主。

## 🧯 故障排除

| 症狀 | 建議排查 |
|---|---|
| `ModuleNotFoundError` 或匯入錯誤 | 確認執行 `python app.py` 前已啟用 `conda activate cellist`。 |
| UI 顯示無樣式/腳本 | 在 `statics/` 內執行 `npm install`，並確認 `statics/node_modules` 存在。 |
| MySQL 存取被拒絕 | 檢查 `cellist/utils/constants.py` 中的使用者名稱/密碼與 MySQL 插件/認證模式。 |
| 應用啟動但模型動作失敗 | 檢查 CUDA/GPU 可用性；目前路徑預設假設使用 CUDA（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| 上傳成功但未出現切片/模型 | 確保 `data/` 子目錄存在且可寫。 |
| REST/WebSocket 請求錯誤 | 確認服務已在 `http://localhost:8887` 運行，且請求負載欄位與目前模板名稱一致。 |
| `data/` 下出現 `FileNotFoundError` | 請從倉庫根目錄啟動應用，以保持相對路徑一致。 |

### 快速診斷

```bash
# 驗證 Python 環境與關鍵套件匯入
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 確認啟動後連接埠是否開放
ss -ltnp | rg 8887

# 檢查 MySQL 連線
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ 路線圖

以下項目保留並整理自既有專案文件/TODO 備註：

- 多邊形範例：改用 polygon 取代矩形標註。
- 為非常小/非常大值優化 `float32` 行為。
- 盡可能縮小模型尺寸。
- 借助 Transformer / stable-diffusion 等思路提升魯棒性。
- 增加基礎模型選項（Threshold、Cellpose）與目標模型選項（AIR、Transformer、SD）。
- 介面優化（含多重選取）。
- 後端優化（含改善記憶體/快取處理）。
- 提供更易於使用的打包方案，降低資料庫設定門檻（例如提供 SQLite 選項）。

## 🤝 參與

### 參與 Lazeal Cellist

Lazeal Cellist 是一個開源專案，歡迎所有程度的貢獻者參與。我們邀請以下方向的貢獻：

- 提升演算法效率與效能
- 改善使用者介面與使用者體驗
- 擴充文件與範例
- 修復缺陷並加強系統穩定性

在開始貢獻前，請先在 issue 中討論你想做的變更，這可協助協調工作、避免重複或衝突。

若要了解如何開始，請參閱貢獻指南。

補充的倉庫貢獻文件：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 致謝

- Lazeal Cellist 的概念與實作大量借鑑了前述 AIR/SPAIR 研究體系。
- 倉庫保留歷史文件與命令，以維持與先前專案使用方式的連續性。

## 📄 授權條款

本專案採用 MIT 授權條款。詳細資料請參閱本倉庫中的 [LICENSE](LICENSE) 檔案。

倉庫狀態說明：目前 checkout 中尚未包含根目錄的 `LICENSE` 檔案。以上文字保留自先前 README，代表專案原始授權意圖；如有需要，請在後續變更中加入本地 `LICENSE` 檔。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
