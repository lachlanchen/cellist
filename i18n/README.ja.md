[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>効率的な 3D 細胞検出とプロファイリングのためのプラットフォーム</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-blue?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/backend-Tornado-00A3E0?style=for-the-badge" alt="Backend" />
  <img src="https://img.shields.io/badge/ML-PyTorch%20%2B%20Pyro%20%2B%20Cellpose-orange?style=for-the-badge" alt="ML" />
  <img src="https://img.shields.io/badge/database-MySQL-4479A1?style=for-the-badge" alt="DB" />
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey?style=for-the-badge" alt="Platform" />
  <img src="https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3?style=for-the-badge" alt="UI" />
  <img src="https://img.shields.io/badge/port-8887-success?style=for-the-badge" alt="Port" />
</p>

<p align="center">
  <a href="#-overview"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-usage"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-troubleshooting"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-contributing"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 Preview

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: 効率的な 3D 細胞検出とプロファイリングのためのプラットフォーム

Lazeal Cellist へようこそ。3D 顕微鏡画像向けの細胞検出、セグメンテーション、プロファイリングを包括的かつ効率的に行えるプラットフォームです。

本プラットフォームは、教師なし学習、しきい値手法、Cellpose などの最先端アルゴリズムを使って細胞検出を行うよう設計されています。さらに、直感的で対話的なインターフェースにより検出結果を微調整できます。これらの修正結果は半教師あり学習ネットワークへフィードバックされ、モデル性能が継続的に向上します。

Lazeal Cellist の強みは、学習と改善に必要な労力を最小限に抑えた効率的な 3D モデルを提供できる点で、科学者、研究者、趣味ユーザーにとって実用的な選択肢です。

> ℹ️ **スコープに関する注記**
> プロジェクトのビジョンと UI には 3D のコンセプト（`/3d`、`templates/cellist_3d.html`）が含まれますが、現行の主要な学習フローは主に 2D スライス処理 + リファインメントになっています。

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

Lazeal Cellist は、顕微鏡画像ワークフロー向けの Python/Tornado ベース Web プラットフォームで、以下を提供します。

- ブラウザ上でのアップロード、モデル作成、アノテーション編集。
- アルゴリズム支援による初期化（Cellpose nuclei モード）。
- WebSocket アクション（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）を通じた human-in-the-loop の反復的リファイン。
- モデル・画像スライス・アノテーションをデータベースで永続化。

> ℹ️ 現在の挙動に関する注記: プロジェクトのビジョンと UI には 3D コンセプト（`/3d`、`templates/cellist_3d.html`）が含まれますが、現在の主要な学習フローは主として 2D スライス + モデル改善です。

### Quick At-a-Glance

| エリア | 現在の実装 |
|---|---|
| サーバー | Tornado (`app.py`) |
| ポート | `8887` |
| データベース | MySQL (`cellist.sql`) |
| コア ML スタック | PyTorch + Pyro + Cellpose |
| フロントエンド | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| 推論初期化 | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| パッケージ状態 | リサーチプロトタイプ（`pyproject.toml` / `setup.py` は未使用） |
| テスト/CI 状態 | リポジトリルートに専用の自動テストスイート/CI 設定はありません |

### Documentation Languages

このリポジトリには `i18n/` 配下に多言語 README が含まれます。

| 言語 | ファイル |
|---|---|
| Arabic | `README.ar.md` |
| German | `README.de.md` |
| Spanish | `README.es.md` |
| French | `README.fr.md` |
| Japanese | `README.ja.md` |
| Korean | `README.ko.md` |
| Russian | `README.ru.md` |
| Vietnamese | `README.vi.md` |
| Chinese (簡体字) | `README.zh-Hans.md` |
| Chinese (繁体字) | `README.zh-Hant.md` |

## ✨ Key Features

- **教師なし 3D 細胞検出**: 高度な機械学習技術で 3D 顕微鏡画像中の細胞を検出します。
- **対話型の結果リファインインターフェース**: 直感的で使いやすい UI で検出結果を調整します。
- **効率的な半教師あり学習ネットワーク**: リファイン結果を用いてモデル性能を継続的に改善します。
- **細胞セグメンテーションとプロファイリング**: 検出に加えて、より高度な解析のためのセグメンテーションとプロファイリングを行えます。

現在実装されている追加機能:

- Tornado REST + WebSocket サーバー（`app.py`）がポート `8887` で起動。
- モデル投入向けの自動画像タイル分割（デフォルト `256x256`）。
- MySQL スキーマがダンプとして同梱されています: [`cellist.sql`](cellist.sql)。
- フロントエンドには Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload を使用。
- スレッドプール（`max_workers=64`）による非同期モデル処理。

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

| 要件 | 備考 |
|---|---|
| OS | Linux を推奨（以下のコマンドは Linux シェル動作を前提）。 |
| Python/Conda | [`cellist.yaml`](cellist.yaml) から環境を作成できる Conda が必要です。 |
| Database | `localhost` 上でデータベース `cellist` を利用できる MySQL サーバー。 |
| GPU | 現行のコードパスでは NVIDIA/CUDA 環境が強く推奨・前提。
| Node.js + npm | `statics/node_modules` のフロントエンド依存解決に必要。 |
| Disk write access | ランタイムデータの `<repo>/data` への書き込み権限。 |

## 🛠️ Installation

### 1. リポジトリをクローンして移動

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 環境を作成

このリポジトリでは `cellist.yaml` を使用します。

```bash
conda env create -f cellist.yaml
conda activate cellist
```

互換性ノート: 旧ドキュメントでは `celist.yaml`（`l` が 1 文字不足）を使っていましたが、このリポジトリ内の実体は `cellist.yaml` です。

旧コマンド（互換性維持）:

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

アプリは `data/` ツリーを想定しています（`.gitignore` で `data` は既に除外）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

注: [`create_data_folder.py`](create_data_folder.py) は存在しますが、現状では現在の作業ディレクトリ直下にディレクトリを作成し、`data/` 配下には作成しません。利用時は注意してください。

### 5. MySQL 認証を準備（必要に応じて）

root 認証がソケットベースでアプリ接続を妨げる場合、過去ドキュメントではパスワード認証への切り替えを案内しています。

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

旧ドキュメント例（互換維持）:

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 実行時の MySQL 資格情報を設定

現行コードは [`cellist/utils/constants.py`](cellist/utils/constants.py) の `mysqlconfig` と `mysqlurl` から資格情報を読み取ります。

現在の既定値には以下が含まれます。

- host: `localhost`
- user: `root`
- password: `lazeal0626`

ローカル環境での安全性向上のため、実行前に環境に合わせてこれらを更新してください。

### 8. 任意の環境サニティチェック

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Usage

### Web サーバー起動

```bash
python app.py
```

旧起動コマンド（保持）:

```bash
python app.py -m cellist
```

コード上で確認できるサーバールート:

- Main UI: `http://localhost:8887/`
- 3D page: `http://localhost:8887/3d`

### Typical workflow

1. UI を開いてログインします。
2. Create Model パネルから顕微鏡画像をアップロードします。
3. ベースアルゴリズム（`Cellpose`）を選択してモデルを作成します。
4. バックエンドで画像をスライスし、検出を初期化します。
5. 切り出した画像を読み込んで、矩形アノテーションを確認・調整します。
6. `initialize`、`pretrain`、`train` のサイクルを実行します。
7. 必要に応じて `Pretrain Stop` / `Stop`（`train-stop`） / `reset` を使用します。
8. `Update Model` / アノテーション操作で手動更新を保存します。

### Built-in UI login credentials (current template behavior)

フロントエンドは現在、以下の静的認証情報をクライアント側でチェックします。

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

これはプロトタイプとしての挙動であり、本番向けの認証ではありません。

### API/Socket surface currently used by the UI

HTTP エンドポイント:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket エンドポイント:

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket ハンドラで認識される `data_type` アクション:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuration

### Backend and endpoints

[`app.py`](app.py) で設定されています:

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Model/data behavior

- スレッドプールサイズは `max_workers=64`。
- 画像タイルはデフォルトで `256x256`。
- Cellpose 初期化は `model_type='nuclei'` と `gpu=True` を使用。
- 学習と事前学習は WebSocket トリガーで非同期実行。
- データのルートはカレントワーキングディレクトリ基準で `<repo>/data`。

### Database/runtime constants

[`cellist/utils/constants.py`](cellist/utils/constants.py) から:

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` に host/user/password キーが含まれる
- `mysqlurl` はデータベース名 `cellist` を指します

### Frontend dependency snapshot

[`statics/package.json`](statics/package.json) から:

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda environment highlights

[`cellist.yaml`](cellist.yaml) から:

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2`（pip）
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

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

### Example: minimal end-to-end local start

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

Lazeal Cellist は、以下の先端ディープラーニング研究にインスパイアされています。

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

これらの研究は、プラットフォームのアルゴリズムと方法論の開発における重要な指針です。

（注記: 正確な引用情報は、原論文を直接参照してください。）

## 🧭 Development Notes

- コアモデルクラスは `cellist/` 配下（`ModelD2Init`、`ModelD2Pretrain`）にあります。
- メインのインタラクティブ UI ロジックは `templates/cellist.html` に直接実装されています。
- SQL スキーマとシード相当データは `cellist.sql` に含まれます。
- `notebooks/` と `polygon_sample/` は探索的なリファレンスを提供します。
- 拡張プラットフォーム/モデルノートは [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md) にあります。
- 現時点で、リポジトリルートには専用の自動テストスイートまたは CI 設定はありません。

### Assumptions and current constraints

- このリポジトリは、まずローカルの研究用途向けであるように見えます。
- 一部のコードパスは GPU（`cuda:0`）の利用を前提としています。
- 認証・シークレット管理はプロトタイプレベルです。
- 3D インターフェースは存在しますが、現在の学習ワークフローの中心は依然として 2D のタイル指向です。

## 🧯 Troubleshooting

| 症状 | 確認すべき項目 |
|---|---|
| `ModuleNotFoundError` または import エラー | `python app.py` 実行前に `conda activate cellist` が適用されているか確認します。 |
| UI がスタイルやスクリプトなしで表示される | `statics/` 内で `npm install` を実行し、`statics/node_modules` が存在することを確認します。 |
| MySQL access denied | `cellist/utils/constants.py` のユーザー名/パスワードと MySQL の plugin/auth モードを確認します。 |
| アプリは起動するがモデル処理が失敗する | CUDA/GPU 利用可否を確認します。現行パスは CUDA 前提です（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| アップロードは成功するがタイル/モデルが現れない | `data/` サブディレクトリが存在し、書き込み可能であることを確認します。 |
| REST/WebSocket リクエストでエラー | サーバーが `http://localhost:8887` で起動しているか、要求ペイロードのキー名が現行テンプレートと一致しているか確認します。 |
| `FileNotFoundError` under `data/` | 相対パスが安定するようリポジトリルートからアプリを起動します。 |

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

以下の項目は既存のプロジェクト文書/TODO ノートから引き継いで整理しています。

- Polygon sample: 矩形ではなくポリゴン注釈を使う。
- 非常に小さい/大きい値に対する `float32` 挙動を最適化する。
- 可能な限りモデルサイズを縮小する。
- Transformer や stable-diffusion 着想のコンポーネントなどを使ってロバスト性を改善する。
- ベースモデル選択肢（Threshold、Cellpose）とターゲットモデル選択肢（AIR、Transformer、SD）を追加する。
- インターフェース最適化（複数選択を含む）。
- バックエンド最適化（メモリ/キャッシュ処理の改善を含む）。
- 最小限の DB 設定で使いやすいパッケージ化（例: SQLite オプション）。

## 🤝 Contributing

### Contribute to Lazeal Cellist

Lazeal Cellist はオープンソースプロジェクトであり、経験の有無を問わずコントリビューションを歓迎します。

以下のような貢献を歓迎します。

- アルゴリズム効率と性能の改善
- UI/UX の改善
- ドキュメントとサンプルの拡充
- バグ修正とシステム安定性の向上

貢献を始める前に、まず issue で変更内容を相談してください。これにより、重複作業や競合を防げます。

開始手順の詳細は、コントリビューションガイドラインを参照してください。

追加のリポジトリ貢献ドキュメント:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Lazeal Cellist のコンセプトと実装は、上で触れた AIR/SPAIR 系の研究成果から強く影響を受けています。
- このリポジトリには過去のプロジェクト運用との連続性を保つため、歴史的なレガシー文書とコマンドが意図的に保持されています。

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

このプロジェクトは MIT License の下でライセンスされています。詳細はこのリポジトリ内の [LICENSE](LICENSE) ファイルを参照してください。

リポジトリ状態に関する注記: 現在のチェックアウトにはルート `LICENSE` ファイルが存在しません。上記文言は以前の README での正規のプロジェクト意図として保持されています。必要に応じて次回変更でローカルの `LICENSE` ファイルを追加してください。
