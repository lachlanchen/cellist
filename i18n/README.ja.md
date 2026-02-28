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

## Lazeal Cellist: 効率的な 3D 細胞検出・プロファイリング基盤

Lazeal Cellist へようこそ。3D 顕微鏡画像に対する細胞検出・セグメンテーション・プロファイリングを包括的かつ効率的に行うためのプラットフォームです。

本プラットフォームは、教師なし学習、しきい値手法、Cellpose などの最先端アルゴリズムを用いて細胞を検出するよう設計されています。さらに、ユーザーが検出結果を直感的かつ対話的に修正できる UI を提供します。修正結果は半教師あり学習ネットワークへフィードバックされ、モデル性能を継続的に改善します。

Lazeal Cellist は、学習と改善に必要な労力を最小限に抑えた効率的な 3D モデルを提供する点で際立っており、科学者、研究者、ホビーユーザーにとって実用的な基盤です。

---

## 🔍 概要

Lazeal Cellist は、顕微鏡画像ワークフロー向けの Python/Tornado Web プラットフォームで、以下を備えています。

- ブラウザベースのアップロード、モデル作成、アノテーション編集。
- アルゴリズム支援による初期化（Cellpose nuclei モード）。
- WebSocket アクション（`initialize`, `pretrain`, `train`, `update`, `reset`）による反復的な Human-in-the-loop 改善。
- モデル、画像スライス、アノテーションの DB 永続化。

現在の挙動に関する注記: プロジェクト構想と UI には 3D の概念（`/3d`, `templates/cellist_3d.html`）が含まれますが、現行コードの主な学習フローは主に 2D スライシング + モデル改善です。

### クイック一覧

| 項目 | 現在の実装 |
|---|---|
| サーバー | Tornado (`app.py`) |
| ポート | `8887` |
| データベース | MySQL (`cellist.sql`) |
| コア ML スタック | PyTorch + Pyro + Cellpose |
| フロントエンド | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| 推論初期化 | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ 主な機能

- **教師なし 3D 細胞検出**: 高度な機械学習手法で 3D 顕微鏡画像から細胞を検出。
- **対話的な結果修正インターフェース**: 直感的で使いやすい UI で検出結果を修正。
- **効率的な半教師あり学習ネットワーク**: 修正結果を活用して継続的に性能向上。
- **細胞セグメンテーションとプロファイリング**: 検出にとどまらず、より高度な解析を実施。

現在実装済みの追加機能:

- ポート `8887` で動作する Tornado REST + WebSocket サーバー（`app.py`）。
- モデル取り込み向け自動画像タイル分割（デフォルト `256x256`）。
- MySQL スキーマをダンプとして同梱: [`cellist.sql`](cellist.sql)。
- フロントエンドは Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload を利用。

## 🗂️ プロジェクト構成

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

## ✅ 前提条件

| 要件 | 補足 |
|---|---|
| OS | Linux 推奨（以下コマンドは Linux シェル挙動を前提）。 |
| Python/Conda | [`cellist.yaml`](cellist.yaml) から環境作成できる Conda が必要。 |
| Database | `localhost` 上で `cellist` DB を利用可能な MySQL サーバー。 |
| GPU | 現行コードパスでは NVIDIA/CUDA 環境を強く推奨/実質想定。 |
| Node.js + npm | `statics/node_modules` の依存導入に必要。 |

## 🛠️ インストール

### 1. リポジトリをクローンして移動

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 環境を作成

リポジトリ内ファイル名 `cellist.yaml` を使用します:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

旧ドキュメントとの互換メモ: 以前の資料では `celist.yaml`（`l` が 1 つ不足）が使われていますが、このリポジトリの実ファイルは `cellist.yaml` です。

### 3. フロントエンド依存をインストール

```bash
cd statics
npm install
cd ..
```

### 4. MySQL 認証を準備（必要な場合）

root 認証がソケット方式でアプリ接続を妨げる場合、旧プロジェクト資料ではパスワード認証への変更が案内されています:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. DB 作成とスキーマ/データの復元

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

旧ドキュメント例（保持）:

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. 実行時の MySQL 資格情報を設定

現行コードは [`cellist/utils/constants.py`](cellist/utils/constants.py) の認証設定（`mysqlconfig` と `mysqlurl`）を参照します。

現在のデフォルト値には以下が含まれます:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

ローカル環境の安全性のため、実行前にこれらを更新してください。

### 7. 実行用データディレクトリを準備

アプリは `data/` ツリーを想定しており（`.gitignore` でも `data` は除外済み）、以下を作成します。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

注: [`create_data_folder.py`](create_data_folder.py) は存在しますが、現状は `data/` 配下ではなくカレントディレクトリに作成します。利用時は注意してください。

## 🚀 使い方

### Web サーバー起動

```bash
python app.py
```

旧ドキュメントの起動コマンド（保持）:

```bash
python app.py -m cellist
```

コードで確認できるデフォルトルート:

- メイン UI: `http://localhost:8887/`
- 3D ページ: `http://localhost:8887/3d`

### 典型的なワークフロー

1. UI を開いてログイン。
2. Create Model パネルから顕微鏡画像をアップロード。
3. ベースアルゴリズム（`Cellpose`）を選んでモデル作成。
4. バックエンドで画像をスライスし、初期検出を実行。
5. クロップ画像を読み込み、矩形アノテーションを確認・修正。
6. `initialize`, `pretrain`, `train` を反復実行。
7. `Update Model` やアノテーション操作で手動更新を保存。

### 組み込み UI ログイン情報（現行テンプレート挙動）

フロントエンドでは現在、以下の固定認証情報をクライアント側でチェックしています:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

これはプロトタイプ挙動であり、本番向け認証ではありません。

## ⚙️ 設定

### バックエンドとエンドポイント

[`app.py`](app.py) で設定:

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### モデル/データ挙動

- スレッドプールサイズは `max_workers=64`。
- 画像タイルはデフォルト `256x256`。
- Cellpose 初期化は `model_type='nuclei'` と `gpu=True` を使用。
- 学習・事前学習は WebSocket トリガーのアクションで非同期実行。

## 🧪 例

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

## 📚 参考にした研究

Lazeal Cellist は、以下の最先端 Deep Learning 研究から着想を得ています。

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

これらの研究は、本プラットフォームのアルゴリズムと方法論の開発に有益な示唆を与えています。

（注: 正確な引用情報は元論文を直接参照してください。）

## 🧭 開発メモ

- コアモデルクラスは `cellist/` 配下（`ModelD2Init`, `ModelD2Pretrain`）。
- メインの対話 UI ロジックは `templates/cellist.html` に直接埋め込まれています。
- SQL スキーマとシード相当データは `cellist.sql` にあります。
- `notebooks/` と `polygon_sample/` には探索的な参考実装があります。
- 現在、リポジトリルートには専用の自動テストスイートや CI 設定はありません。

## 🧯 トラブルシューティング

| 症状 | 確認ポイント |
|---|---|
| `ModuleNotFoundError` や import 問題 | `python app.py` 実行前に `conda activate cellist` を適用したか確認。 |
| UI は表示されるがスタイル/スクリプトが効かない | `statics/` で `npm install` を実行し、`statics/node_modules` が存在するか確認。 |
| MySQL access denied | `cellist/utils/constants.py` のユーザー名/パスワード、および MySQL の plugin/auth mode を確認。 |
| アプリは起動するがモデル操作が失敗 | CUDA/GPU 可用性を確認（現行パスは CUDA 前提: `torch.device('cuda:0')`, Cellpose `gpu=True`）。 |
| アップロード成功後もタイル/モデルが出ない | `data/` のサブディレクトリが存在し書き込み可能か確認。 |

## 🗺️ ロードマップ

以下の項目は、既存ドキュメントおよび TODO メモから保持・整理した内容です。

- Polygon sample: 矩形ではなくポリゴンアノテーションを使用。
- 極小値/極大値を含む `float32` 挙動向けのモデル最適化。
- 可能な範囲でモデルサイズを縮小。
- Transformer / stable-diffusion 着想コンポーネント等による堅牢性向上。
- ベースモデル（Threshold, Cellpose）とターゲットモデル（AIR, Transformer, SD）の選択肢追加。
- インターフェース最適化（複数選択を含む）。
- バックエンド最適化（メモリ/キャッシュ改善を含む）。
- DB 設定を最小化した使いやすいパッケージ化（例: SQLite オプション）。

## 🤝 コントリビューション

### Lazeal Cellist への貢献

Lazeal Cellist はオープンソースで、経験レベルを問わずあらゆる貢献を歓迎します。特に次のような貢献を歓迎します。

- アルゴリズム効率と性能の改善
- UI/UX の改善
- ドキュメントやサンプルの拡充
- バグ修正とシステム安定性向上

貢献を開始する前に、まず Issue で変更内容を相談してください。重複作業や競合を避け、連携しやすくなります。

開始方法の詳細は、コントリビューションガイドラインを参照してください。

追加の貢献関連ドキュメント:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 ライセンス

本プロジェクトは MIT License で提供されています。詳細は、このリポジトリ内の [LICENSE](https://chat.openai.com/LICENSE) ファイルを参照してください。

リポジトリ状態に関する注記: このチェックアウトには現時点でルート `LICENSE` ファイルが存在しません。上記文言は prior README の正準方針として保持されています。必要であれば、後続変更でローカル `LICENSE` ファイルを追加してください。
