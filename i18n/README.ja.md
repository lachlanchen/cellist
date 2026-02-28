[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>効率的な 3D 細胞検出・プロファイリングプラットフォーム</strong>
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
  <a href="#-overview"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-usage"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-troubleshooting"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-contributing"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: 効率的な 3D 細胞検出・プロファイリングプラットフォーム

Lazeal Cellist へようこそ。3D 顕微鏡画像向けの細胞検出、セグメンテーション、プロファイリングを包括的かつ効率的に行えるプラットフォームです。

本プラットフォームは、教師なし学習、しきい値手法、Cellpose などの最先端アルゴリズムを用いた細胞検出を想定して設計されています。さらに、検出結果を直感的なインターフェースで対話的に修正できます。修正結果は半教師あり学習ネットワークにフィードバックされ、モデル性能が継続的に向上します。

Lazeal Cellist は、学習と改善にかかる労力を最小限に抑えた効率的な 3D モデルを提供する点が特長で、科学者、研究者、ホビーユーザーにとって実用的な選択肢です。

> ℹ️ **スコープに関する注記**
> プロジェクトのビジョンと UI には 3D 概念（`/3d`, `templates/cellist_3d.html`）が含まれますが、現行コードの主要な学習フローは主に 2D スライシング + リファインです。

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Inspired by Research](#-inspired-by-research)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)
- [Support](#-support)
- [License](#-license)

## 🔍 Overview

Lazeal Cellist は、顕微鏡画像ワークフロー向けの Python/Tornado Web プラットフォームで、次の機能を提供します。

- ブラウザ上でのアップロード、モデル作成、アノテーション編集。
- アルゴリズム支援の初期化（Cellpose nuclei モード）。
- WebSocket アクション（`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`）を通じた、human-in-the-loop の反復的リファイン。
- モデル、画像スライス、アノテーションのデータベース永続化。

> ℹ️ 現在の挙動に関する注記: プロジェクトのビジョンと UI には 3D 概念（`/3d`, `templates/cellist_3d.html`）が含まれますが、現行コードの主な学習フローは主に 2D スライシング + モデル改善です。

### Quick At-a-Glance

| Area | Current implementation |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Database | MySQL (`cellist.sql`) |
| Core ML stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inference init | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Packaging status | Research prototype (no `pyproject.toml`/`setup.py`) |
| Tests/CI status | No dedicated automated test suite or CI config in repository root |

### Documentation Languages

このリポジトリには、`i18n/` 配下に多言語 README が含まれています。

| Language | File |
|---|---|
| Arabic | `README.ar.md` |
| German | `README.de.md` |
| Spanish | `README.es.md` |
| French | `README.fr.md` |
| Japanese | `README.ja.md` |
| Korean | `README.ko.md` |
| Russian | `README.ru.md` |
| Vietnamese | `README.vi.md` |
| Chinese (Simplified) | `README.zh-Hans.md` |
| Chinese (Traditional) | `README.zh-Hant.md` |

## ✨ Key Features

- **教師なし 3D 細胞検出**: 高度な機械学習技術で 3D 顕微鏡画像中の細胞を検出。
- **対話型の結果リファイン UI**: 直感的で使いやすいインターフェースで検出結果を調整。
- **効率的な半教師あり学習ネットワーク**: リファイン結果を活用してモデル性能を継続的に改善。
- **細胞セグメンテーションとプロファイリング**: 検出だけでなく、より高度な解析まで対応。

現在実装されている追加機能:

- Tornado REST + WebSocket サーバー（`app.py`）がポート `8887` で稼働。
- モデル入力向けの画像自動タイル分割（デフォルト `256x256`）。
- MySQL スキーマをダンプとして同梱: [`cellist.sql`](cellist.sql)。
- フロントエンドは Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload を使用。
- スレッドプール（`max_workers=64`）による非同期モデルタスク実行。

## 🗂️ Project Structure

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

## ✅ Prerequisites

| Requirement | Notes |
|---|---|
| OS | Linux 推奨（以下のコマンドは Linux シェル挙動を前提）。 |
| Python/Conda | [`cellist.yaml`](cellist.yaml) から環境作成できる Conda が必要。 |
| Database | `localhost` で `cellist` データベースを利用できる MySQL サーバー。 |
| GPU | 現行コードパスでは NVIDIA/CUDA 環境を強く推奨・想定。 |
| Node.js + npm | `statics/node_modules` フロントエンド依存の導入に必要。 |
| Disk write access | `<repo>/data` 配下へのランタイム書き込みに必要。 |

## 🛠️ Installation

### 1. リポジトリをクローンして移動

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 環境を作成

リポジトリ内のファイル名 `cellist.yaml` を使用します。

```bash
conda env create -f cellist.yaml
conda activate cellist
```

古いドキュメントとの互換メモ: 以前の文書では `celist.yaml`（`l` が 1 つ不足）を使っていましたが、このリポジトリ内の実ファイルは `cellist.yaml` です。

Legacy command (preserved):

```bash
conda env create -f celist.yaml
```

### 3. フロントエンド依存をインストール

```bash
cd statics
npm install
cd ..
```

### 4. ランタイムデータディレクトリを準備

アプリは `data/` ツリーを前提としています（`.gitignore` で `data` はすでに除外）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

注: [`create_data_folder.py`](create_data_folder.py) は存在しますが、現状はカレントワーキングディレクトリ直下にディレクトリを作成し（`data/` 配下ではない）、その点に注意が必要です。

### 5. MySQL 認証を準備（必要な場合）

root 認証がソケットベースでアプリ接続を阻害する場合、過去のプロジェクト文書ではパスワード認証への切り替えを案内しています。

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. データベース作成とスキーマ/データ復元

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Legacy documentation example (preserved):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 実行時 MySQL 資格情報を設定

現行コードは [`cellist/utils/constants.py`](cellist/utils/constants.py) の `mysqlconfig` と `mysqlurl` から資格情報を読み取ります。

現在のコード既定値には次が含まれます。

- host: `localhost`
- user: `root`
- password: `lazeal0626`

ローカル環境のセキュリティのため、実行前にこれらの値を更新してください。

### 8. 任意の環境ヘルスチェック

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Usage

### Web サーバーを起動

```bash
python app.py
```

旧ドキュメントの起動コマンド（保持）:

```bash
python app.py -m cellist
```

コードから確認できるサーバールート既定値:

- Main UI: `http://localhost:8887/`
- 3D page: `http://localhost:8887/3d`

### 典型的なワークフロー

1. UI を開いてログイン。
2. Create Model パネルから顕微鏡画像をアップロード。
3. ベースアルゴリズム（`Cellpose`）を選択してモデルを作成。
4. バックエンドで画像をスライスし、検出を初期化。
5. 切り出し画像を読み込み、矩形アノテーションを確認・調整。
6. `initialize`, `pretrain`, `train` のサイクルを実行。
7. 必要に応じて `Pretrain Stop` / `Stop`（`train-stop`）/ `reset` を使用。
8. `Update Model`/アノテーション操作で手動更新を永続化。

### 内蔵 UI ログイン資格情報（現行テンプレート挙動）

フロントエンドは現在、次の静的資格情報をクライアント側でチェックします。

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

これはプロトタイプ挙動であり、本番向け認証ではありません。

### UI が現在利用している API/Socket サーフェス

HTTP エンドポイント:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket エンドポイント:

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket ハンドラで認識される `data_type` アクションメッセージ:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuration

### バックエンドとエンドポイント

[`app.py`](app.py) で設定:

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### モデル/データの挙動

- スレッドプールサイズは `max_workers=64`。
- 画像タイルはデフォルトで `256x256`。
- Cellpose 初期化は `model_type='nuclei'` と `gpu=True` を使用。
- 学習と事前学習は WebSocket 起動アクション経由で非同期実行。
- データルートはカレントワーキングディレクトリ基準で `<repo>/data` に解決。

### データベース/ランタイム定数

[`cellist/utils/constants.py`](cellist/utils/constants.py) より:

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` には host/user/password キーが含まれる
- `mysqlurl` はデータベース名 `cellist` を対象

### フロントエンド依存スナップショット

[`statics/package.json`](statics/package.json) より:

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda 環境の主要ポイント

[`cellist.yaml`](cellist.yaml) より:

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Examples

### 例: WebSocket create メッセージ形式

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

### 例: WebSocket 手動アノテーション更新

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### 例: モデル読み込みリクエスト

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### 例: ローカル最小エンドツーエンド起動

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 Inspired by Research

Lazeal Cellist は、次の先端ディープラーニング研究から着想を得ています。

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

これらの研究は、本プラットフォームのアルゴリズムおよび方法論の開発指針になっています。

（注: 正確な引用情報は原論文を直接参照してください。）

## 🧭 Development Notes

- コアモデルクラスは `cellist/` 配下（`ModelD2Init`, `ModelD2Pretrain`）。
- メインの対話 UI ロジックは `templates/cellist.html` に直接埋め込まれています。
- SQL スキーマおよびシード相当データは `cellist.sql` にあります。
- `notebooks/` と `polygon_sample/` は探索的リファレンスを提供します。
- 拡張プラットフォーム/モデルノートは [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md) にあります。
- 現時点で、リポジトリルートに専用自動テストスイートや CI 設定はありません。

### Assumptions and current constraints

- このリポジトリは、まずローカルの研究用途を主対象としているように見えます。
- 一部のコードパスは GPU（`cuda:0`）利用を前提とします。
- 認証およびシークレット管理はプロトタイプ水準です。
- 3D インターフェースは存在しますが、学習ワークフローの中心は依然として 2D タイル指向です。

## 🧯 Troubleshooting

| Symptom | Suggested checks |
|---|---|
| `ModuleNotFoundError` or import issues | `python app.py` の前に `conda activate cellist` を実行したか確認。 |
| UI renders without styling/scripts | `statics/` で `npm install` を実行し、`statics/node_modules` が存在するか確認。 |
| MySQL access denied | `cellist/utils/constants.py` のユーザー名/パスワードと MySQL の plugin/auth モードを確認。 |
| App starts but model actions fail | CUDA/GPU 利用可否を確認（現行パスは CUDA 前提: `torch.device('cuda:0')`, Cellpose `gpu=True`）。 |
| Upload succeeds but no tiles/models appear | `data/` サブディレクトリが存在し、書き込み可能か確認。 |
| REST/WebSocket request errors | サーバーが `http://localhost:8887` で稼働中か、ペイロードキー名が現行テンプレートと一致するか確認。 |
| `FileNotFoundError` under `data/` | 相対パス解決を安定させるため、リポジトリルートからアプリを起動。 |

### Quick diagnostics

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Roadmap

以下の項目は、既存プロジェクト文書/TODO ノートから保持して整理したものです。

- Polygon sample: 矩形ではなくポリゴン注釈を使用。
- 非常に小さい/大きい値に対する `float32` 挙動を最適化。
- 可能な範囲でモデルサイズを縮小。
- Transformer や stable-diffusion 発想コンポーネントなどによる堅牢性向上。
- ベースモデル選択肢（Threshold, Cellpose）とターゲットモデル選択肢（AIR, Transformer, SD）の追加。
- インターフェース最適化（複数選択を含む）。
- バックエンド最適化（メモリ/キャッシュ処理改善を含む）。
- 最小限の DB 設定で使える簡易パッケージ化（例: SQLite オプション）。

## 🤝 Contributing

### Lazeal Cellist への貢献

Lazeal Cellist はオープンソースプロジェクトであり、経験レベルを問わずコントリビューションを歓迎します。特に次の貢献を歓迎しています。

- アルゴリズム効率と性能の向上
- UI/UX の改善
- ドキュメントと例の拡充
- バグ修正とシステム安定性の強化

貢献を始める前に、まず issue で予定変更を相談してください。重複作業や競合を避けるためです。

開始方法の詳細は、コントリビューションガイドラインを参照してください。

追加のリポジトリ貢献ドキュメント:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Lazeal Cellist のコンセプトと実装は、上記の AIR/SPAIR 系研究の影響を強く受けています。
- このリポジトリには、過去のプロジェクト利用との連続性を保つため、歴史的/レガシー文書とコマンドが意図的に保持されています。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

このプロジェクトは MIT License の下でライセンスされています。詳細は、このリポジトリ内の [LICENSE](LICENSE) ファイルを参照してください。

リポジトリ状態に関する注記: 現在のチェックアウトにはルート `LICENSE` ファイルが存在しません。上記文言は、以前の README における正規のプロジェクト意図として保持しています。必要であれば、後続変更でローカル `LICENSE` ファイルを追加してください。
