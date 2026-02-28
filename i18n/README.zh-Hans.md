[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>高效的 3D 细胞检测与表型分析平台</strong>
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

## Lazeal Cellist：高效 3D 细胞检测与表型分析平台

欢迎使用 Lazeal Cellist，这是一款面向 3D 显微镜图像的全面且高效细胞检测、分割与表型分析平台。

本平台通过无监督学习、阈值法以及 Cellpose 等先进算法实现细胞检测。同时提供一个直观、交互式界面，帮助用户精修检测结果。精修后的结果会回流到半监督学习网络中，持续提升模型性能。

Lazeal Cellist 的核心优势在于其高效的 3D 模型，仅需较少的训练和精修投入，适合科研人员、研究者以及业余用户。

> ℹ️ **范围说明**
> 项目愿景与 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），但当前代码中的主要训练流程主要是 2D 切片 + 精修。

---

## 目录

- [概览](#-overview)
- [核心特性](#-key-features)
- [项目结构](#-project-structure)
- [先决条件](#-prerequisites)
- [安装](#-installation)
- [使用](#-usage)
- [配置](#-configuration)
- [示例](#-examples)
- [研究启发](#-inspired-by-research)
- [开发说明](#-development-notes)
- [故障排查](#-troubleshooting)
- [路线图](#-roadmap)
- [贡献](#-contributing)
- [致谢](#-acknowledgements)
- [Support](#-support)
- [许可](#-license)

## 🔍 概览

Lazeal Cellist 是一个面向显微镜影像流程的 Python/Tornado Web 平台，具备以下能力：

- 浏览器上传、模型创建与标注编辑。
- 算法辅助初始化（Cellpose nuclei 模式）。
- 通过 WebSocket 动作（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）进行人机协同的迭代式精修。
- 数据库持久化保存模型、图像切片与标注。

> ℹ️ 当前行为说明：虽然项目愿景与 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），当前主训练流程仍以 2D 切片 + 模型精修为主。

### 快速总览

| 模块 | 当前实现 |
|---|---|
| 服务器 | Tornado（`app.py`） |
| 端口 | `8887` |
| 数据库 | MySQL（`cellist.sql`） |
| 核心 ML 技术栈 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推理初始化 | Cellpose（`model_type='nuclei'`、`gpu=True`） |
| 打包状态 | 研究原型（未提供 `pyproject.toml`/`setup.py`） |
| 测试/CI 状态 | 仓库根目录未配置专用自动化测试套件或 CI |

### 文档语言

本仓库已在 `i18n/` 下包含多语言 README：

| 语言 | 文件 |
|---|---|
| 阿拉伯语 | `README.ar.md` |
| 德语 | `README.de.md` |
| 西班牙语 | `README.es.md` |
| 法语 | `README.fr.md` |
| 日语 | `README.ja.md` |
| 韩语 | `README.ko.md` |
| 俄语 | `README.ru.md` |
| 越南语 | `README.vi.md` |

## ✨ 核心特性

- **无监督 3D 细胞检测**：使用先进机器学习技术识别 3D 显微镜图像中的细胞。
- **交互式结果精修界面**：通过直观友好的界面调优检测结果。
- **高效半监督学习网络**：利用精修结果持续提升模型性能。
- **细胞分割与表型分析**：不仅支持检测，还支持更深入的分割与分析。

当前实现中还包括以下补充功能：

- Tornado REST + WebSocket 服务器（`app.py`）运行于 `8887` 端口。
- 自动影像分块（默认 `256x256`）用于模型接收。
- MySQL Schema 以转储形式提供：[`cellist.sql`](cellist.sql)。
- 前端技术栈包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。
- 通过线程池异步执行模型任务（`max_workers=64`）。

## 🗂️ 项目结构

```text
cellist/
├── app.py                               # 主 Tornado 服务器 + REST/WebSocket 处理器
├── cellist/                             # 核心 ML/模型代码
│   ├── model_init.py                    # 主 2D 模型类与训练流程
│   ├── model_pretrain.py                # 预训练变体
│   ├── model_2d_components.py           # Encoder/Decoder/SPAIR 组件
│   ├── model_2d_utilities.py            # 数据库支撑的模型元数据 + transforms
│   ├── image_preprocessing.py           # 切片/拼接工具
│   └── utils/constants.py               # 运行时路径 + MySQL 配置
├── templates/
│   ├── cellist.html                     # 主 2D UI
│   └── cellist_3d.html                  # 3D UI 变体/原型
├── statics/                             # 前端资源与 npm 依赖
│   ├── package.json
│   └── node_modules/
├── i18n/                                # 翻译版 README 文件
├── notebooks/                           # 探索型 notebook
├── polygon_sample/                      # 多边形标注实验
├── figs/                                # 品牌素材
├── cellist.sql                          # MySQL schema/data 转储
├── cellist.yaml                         # Conda 环境说明
├── create_data_folder.py                # 旧版数据目录创建脚本
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # 扩展架构/TODO 说明
└── README.md
```

## ✅ 先决条件

| 要求 | 说明 |
|---|---|
| 操作系统 | 推荐 Linux（以下命令基于 Linux shell 行为）。 |
| Python/Conda | 需要 Conda，可通过 [`cellist.yaml`](cellist.yaml) 创建环境。 |
| 数据库 | 需要本地运行 MySQL，数据库名为 `cellist`。 |
| GPU | 现有代码路径强烈推荐并默认预期 NVIDIA/CUDA 环境。 |
| Node.js + npm | 需要安装前端依赖到 `statics/node_modules`。 |
| 磁盘写入权限 | 运行时需要对 `<repo>/data` 有写权限。 |

## 🛠️ 安装

### 1. 克隆仓库并进入目录

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 创建 Python 环境

使用仓库文件名为 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

兼容性说明（沿用旧文档）：旧说明文档曾使用 `celist.yaml`（少了一个 `l`），但本仓库实际文件为 `cellist.yaml`。

保留的历史命令：

```bash
conda env create -f celist.yaml
```

### 3. 安装前端依赖

```bash
cd statics
npm install
cd ..
```

### 4. 准备运行时数据目录

应用期待存在 `data/` 目录树（`.gitignore` 已排除 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

说明：[`create_data_folder.py`](create_data_folder.py) 已存在，但当前会在当前工作目录下创建目录（非 `data/` 下），使用时请注意。

### 5. 准备 MySQL 认证（如需）

若 root 认证为 socket 方式且阻止应用访问，旧文档建议切换为密码认证：

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. 创建数据库并恢复 schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

保留的历史示例（仅作参考）：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 配置运行时 MySQL 凭据

当前代码从 [`cellist/utils/constants.py`](cellist/utils/constants.py) 读取凭据（`mysqlconfig` 与 `mysqlurl`）。

默认值目前包含：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

为安全起见，请在你的本地环境中更新这些值后再运行。

### 8. 可选环境检查

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 使用

### 启动 Web 服务器

```bash
python app.py
```

保留的历史启动命令（兼容旧文档）：

```bash
python app.py -m cellist
```

代码中的默认路由：

- 主界面：`http://localhost:8887/`
- 3D 页面：`http://localhost:8887/3d`

### 典型流程

1. 打开界面并登录。
2. 在“Create Model”面板上传显微镜图像。
3. 选择基础算法（`Cellpose`）并创建模型。
4. 让后端切片并初始化检测结果。
5. 加载裁剪图像，查看并调整矩形标注。
6. 运行 `initialize`、`pretrain`、`train` 循环。
7. 按需使用 `Pretrain Stop` / `Stop`（`train-stop`）/ `reset`。
8. 通过 `Update Model`/标注操作保存手工更新。

### 内置 UI 登录凭据（当前模板行为）

前端当前在客户端会检查以下静态凭据：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

这是原型级行为，不是生产级身份认证。

### API/WebSocket 接口（当前 UI 使用）

HTTP 端点：

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 端点：

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket 处理器中已识别的 `data_type` 动作：

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ 配置

### 后端和端点

配置位于 [`app.py`](app.py)：

- 端口: `8887`
- 路由:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型与数据行为

- 线程池大小为 `max_workers=64`。
- 默认图像分块大小为 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 与 `gpu=True`。
- 训练和预训练通过 WebSocket 触发动作异步执行。
- 数据根目录通过当前工作目录解析为 `<repo>/data`。

### 数据库/运行时常量

来自 [`cellist/utils/constants.py`](cellist/utils/constants.py)：

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` 包含 host/user/password 字段
- `mysqlurl` 指向数据库 `cellist`

### 前端依赖清单

来自 [`statics/package.json`](statics/package.json)：

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda 环境要点

来自 [`cellist.yaml`](cellist.yaml)：

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2`（pip）
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 示例

### 示例：WebSocket create 消息结构

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

### 示例：WebSocket 手工标注更新

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### 示例：模型加载请求

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### 示例：最小本地启动流程

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 研究启发

Lazeal Cellist 借鉴了前沿的深度学习研究，包括：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

这些工作为我们平台的算法与方法提供了有价值的启发。

（备注：如需准确引用，请直接参考原始论文。）

## 🧭 开发说明

- 核心模型类位于 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要交互式 UI 逻辑直接写在 `templates/cellist.html` 中。
- SQL schema 与种子式数据位于 `cellist.sql`。
- `notebooks/` 与 `polygon_sample/` 中有探索性参考材料。
- 更完整的平台/模型说明见 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)。
- 仓库根目录目前无专用自动测试套件或 CI 配置。

### 假设与当前约束

- 本仓库看起来优先面向本地研究用途。
- 部分代码路径默认假设 GPU（`cuda:0`）可用。
- 认证与密钥管理仍处于原型级。
- 3D 界面已存在，但当前主训练流程仍以 2D 切片为主。

## 🧯 故障排查

| 症状 | 建议排查 |
|---|---|
| `ModuleNotFoundError` 或 import 问题 | 确认运行 `python app.py` 前已执行 `conda activate cellist`。 |
| UI 无样式/脚本 | 在 `statics/` 中运行 `npm install`，并确认 `statics/node_modules` 存在。 |
| MySQL access denied | 检查 [`cellist/utils/constants.py`](cellist/utils/constants.py) 中的用户名/密码及 MySQL 插件/认证模式。 |
| 应用启动后模型动作失败 | 检查 CUDA/GPU 可用性；当前路径默认假设 CUDA（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| 上传成功但未出现切片/模型 | 确保 `data/` 子目录存在且可写。 |
| REST/WebSocket 请求报错 | 确认服务运行在 `http://localhost:8887`，并且请求 payload 字段与当前模板一致。 |
| `data/` 下出现 `FileNotFoundError` | 从仓库根目录启动应用，以确保相对路径一致。 |

### 快速诊断

```bash
# 验证 Python 环境和关键导入
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 确认启动后端口是否打开
ss -ltnp | rg 8887

# 检查 MySQL 连接性
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ 路线图

以下条目保留并整理自现有项目文档/TODO 记录：

- 多边形示例：改用 polygon 标注替代矩形。
- 优化 `float32` 在非常小或非常大数值下的行为。
- 尽可能缩小模型规模。
- 使用 Transformer/受 stable-diffusion 启发的组件提升鲁棒性。
- 增加基础模型选项（Threshold、Cellpose）与目标模型选项（AIR、Transformer、SD）。
- 界面优化（含多选）。
- 后端优化（含更好的内存/缓存处理）。
- 提供更易用的打包方案，降低数据库配置门槛（例如 SQLite 选项）。

## 🤝 贡献

### 参与 Lazeal Cellist

Lazeal Cellist 是开源项目，欢迎所有人参与，不分经验水平。我们欢迎以下贡献：

- 提升算法效率和性能
- 改进用户界面与体验
- 扩展文档与示例
- 修复缺陷并提升系统稳定性

在开始贡献前，请先在 issue 中讨论你要做的修改，以便协调并避免重复或冲突。

更多上手信息请查看贡献指南。

附加仓库贡献文档：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 致谢

- Lazeal Cellist 的概念与实现大量借鉴了上述 AIR/SPAIR 研究路线。
- 仓库保留了历史性文档和命令，以维持与先前项目用法的连续性。

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

本项目采用 MIT License 进行授权。详细信息请参见仓库中的 [LICENSE](LICENSE) 文件。

仓库状态说明：当前检出版本中根目录未包含 `LICENSE` 文件。以上说明保留自既有 README 的项目规范；如果你愿意，可在后续提交中补充本地 `LICENSE` 文件。
