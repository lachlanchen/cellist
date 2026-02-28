[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>您的高效3D细胞检测与分析平台</strong>
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
  <a href="#-概览"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-安装"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-使用"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-故障排查"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-贡献"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist：您的高效3D细胞检测与分析平台

欢迎使用 Lazeal Cellist，这是一款面向3D显微镜图像的全面、轻量且高效的细胞检测、分割与分析平台。

本平台通过无监督学习、阈值方法以及 Cellpose 等前沿算法实现细胞检测。Lazeal Cellist 还提供直观的交互式界面，帮助用户细化检测结果。这些精修后的结果会回流到半监督学习网络，实现模型持续提升。

Lazeal Cellist 的优势在于提供高效的3D模型，训练和精修所需工作量较低，适合科研人员、研究人员和业余爱好者。

> ℹ️ **范围说明**
> 项目愿景与UI包含3D概念（`/3d`、`templates/cellist_3d.html`），但当前代码中的主要训练流程仍以2D切片 + 精修为主。

---

## 目录

- [概览](#-概览)
- [核心特性](#-核心特性)
- [项目结构](#-项目结构)
- [先决条件](#-先决条件)
- [安装](#-安装)
- [使用](#-使用)
- [配置](#-配置)
- [示例](#-示例)
- [研究启发](#-研究启发)
- [开发说明](#-开发说明)
- [故障排查](#-故障排查)
- [路线图](#-路线图)
- [贡献](#-贡献)
- [致谢](#-致谢)
- [Support](#-support)
- [许可证](#-许可证)

## 🔍 概览

Lazeal Cellist 是一个基于 Python/Tornado 的显微图像工作流 Web 平台，支持：

- 浏览器中上传、创建模型与编辑标注。
- 算法辅助初始化（Cellpose nuclei 模式）。
- 通过 WebSocket 动作（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）进行人机协同迭代精修。
- 数据库持久化存储模型、图像切片和标注。

> ℹ️ 当前行为说明：尽管项目愿景与UI包含3D概念（`/3d`、`templates/cellist_3d.html`），当前代码中的主训练流程主要仍是2D切片 + 模型精修。

### 快速一览

| 模块 | 当前实现 |
|---|---|
| 服务器 | Tornado（`app.py`） |
| 端口 | `8887` |
| 数据库 | MySQL（`cellist.sql`） |
| 核心 ML 技术栈 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推理初始化 | Cellpose（`model_type='nuclei'`，`gpu=True`） |
| 打包状态 | 研究原型（无 `pyproject.toml`/`setup.py`） |
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

- **无监督3D细胞检测**：使用先进机器学习技术，在3D显微镜图像中识别细胞。
- **交互式结果精修界面**：通过直观友好的界面反复细化检测结果。
- **高效半监督学习网络**：利用精修结果持续提升模型表现。
- **细胞分割与分析**：除检测外，还可进行更深入的分割和分析。

当前代码中已具备的附加实现特性：

- 在 `8887` 端口运行的 Tornado REST + WebSocket 服务器（`app.py`）。
- 自动图像切片（默认 `256x256`）用于模型摄入。
- 以 dump 形式提供 MySQL schema：[`cellist.sql`](cellist.sql)。
- 前端栈包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。
- 通过线程池异步执行模型任务（`max_workers=64`）。

## 🗂️ 项目结构

```text
cellist/
├── app.py                               # 主 Tornado 服务 + REST/WebSocket 处理器
├── cellist/                             # 核心 ML/模型代码
│   ├── model_init.py                    # 2D 主模型类与训练流程
│   ├── model_pretrain.py                # 预训练变体
│   ├── model_2d_components.py           # 编码器/解码器/SPAIR 组件
│   ├── model_2d_utilities.py            # 数据库驱动的模型元数据与变换
│   ├── image_preprocessing.py           # 切片/拼接工具
│   └── utils/constants.py               # 运行路径 + MySQL 配置
├── templates/
│   ├── cellist.html                     # 主要2D界面
│   └── cellist_3d.html                  # 3D界面变体（原型）
├── statics/                             # 前端资源与 npm 依赖
│   ├── package.json
│   └── node_modules/
├── i18n/                                # 翻译版 README
├── notebooks/                           # 探索型笔记本
├── polygon_sample/                      # 多边形标注实验
├── figs/                                # 品牌素材
├── cellist.sql                          # MySQL schema 与数据导出
├── cellist.yaml                         # Conda 环境定义
├── create_data_folder.py                # 旧式数据目录创建脚本
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # 扩展架构/TODO 说明
└── README.md
```

## ✅ 先决条件

| 需求 | 说明 |
|---|---|
| 操作系统 | 推荐 Linux（以下命令按 Linux shell 行为执行）。 |
| Python/Conda | 需要 Conda，可基于 [`cellist.yaml`](cellist.yaml) 创建环境。 |
| 数据库 | 本地 `localhost` 上需运行 MySQL，并存在 `cellist` 数据库。 |
| GPU | 当前代码路径强烈建议/预期在 NVIDIA/CUDA 环境下运行。 |
| Node.js + npm | 需要用于安装 `statics/node_modules` 前端依赖。 |
| 磁盘写权限 | 运行时需要 `<repo>/data` 下的数据写入权限。 |

## 🛠️ 安装

### 1. 克隆并进入仓库

```bash

git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 创建 Python 环境

使用仓库文件名 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

兼容性说明保留于旧文档：先前文档曾使用 `celist.yaml`（缺少一个 `l`），但本仓库文件为 `cellist.yaml`。

保留的旧命令：

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

应用需要 `data/` 目录树（`.gitignore` 已经排除了 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

说明：[`create_data_folder.py`](create_data_folder.py) 存在，但目前会在当前工作目录创建目录，而非 `data/` 下。

### 5. 准备 MySQL 认证（如需要）

若 root 认证为 socket 模式且拦截应用访问，旧版文档建议改为密码认证：

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

保留旧文档示例（用于参考）：

```bash
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 配置运行时 MySQL 凭据

当前代码从 [`cellist/utils/constants.py`](cellist/utils/constants.py) 读取配置（`mysqlconfig` 与 `mysqlurl`）。

当前代码默认值为：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

为了本地安全性，请在你的环境运行前更新这些值。

### 8. 可选：环境健康检查

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

保留的旧启动命令（来自历史文档）：

```bash
python app.py -m cellist
```

代码中可见的默认路由：

- 主界面：`http://localhost:8887/`
- 3D 页面：`http://localhost:8887/3d`

### 典型工作流

1. 打开 UI 并登录。
2. 在 Create Model 面板上传显微镜图像。
3. 选择基础算法（`Cellpose`）并创建模型。
4. 等待后端切片并初始化检测。
5. 加载裁剪后的图像，审阅并调整矩形标注。
6. 运行 `initialize`、`pretrain` 与 `train` 周期。
7. 按需使用 `Pretrain Stop` / `Stop`（`train-stop`）/`reset`。
8. 通过 `Update Model`/标注动作持久化手动更新。

### 内置 UI 登录凭据（当前模板行为）

前端当前在客户端侧检查以下静态凭据：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

这属于原型级行为，不是生产级认证方案。

### API/WebSocket 接口（UI 当前使用）

HTTP 接口：

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 接口：

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

### 后端与端点

配置位于 [`app.py`](app.py)：

- 端口：`8887`
- 路由：
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型/数据行为

- 线程池大小为 `max_workers=64`。
- 默认图像切片大小为 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 与 `gpu=True`。
- 训练和预训练通过 WebSocket 触发动作异步执行。
- 数据根路径从当前工作目录解析为 `<repo>/data`。

### 数据库/运行时常量

来自 [`cellist/utils/constants.py`](cellist/utils/constants.py)：

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` 包含 host/user/password 字段
- `mysqlurl` 指向数据库 `cellist`

### 前端依赖快照

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
- Pyro（`pyro-ppl==1.8.1`）
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

### 示例：WebSocket 手动标注更新

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

### 示例：最小端到端本地启动

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

Lazeal Cellist 受到前沿深度学习研究的启发，包括：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

这些工作为我们平台的算法与方法论提供了重要的设计思路。

（说明：如需准确引用，请直接参考原始论文。）

## 🧭 开发说明

- 核心模型类位于 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要交互式 UI 逻辑直接嵌入 `templates/cellist.html`。
- SQL schema 与种子风格数据位于 `cellist.sql`。
- `notebooks/` 与 `polygon_sample/` 提供探索性参考。
- 扩展的平台/模型说明位于 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)。
- 仓库根目录目前没有专用自动化测试套件或 CI 配置。

### 当前假设与约束

- 仓库似乎优先面向本地研究场景使用。
- 某些代码路径默认可用 GPU（`cuda:0`）。
- 鉴权与密钥管理仍为原型级。
- 3D 界面已有实现，但主训练流程仍是以2D切片为中心。

## 🧯 故障排查

| 症状 | 建议检查 |
|---|---|
| `ModuleNotFoundError` 或导入问题 | 确认运行 `python app.py` 前已执行 `conda activate cellist`。 |
| UI 有渲染但无样式/脚本 | 在 `statics/` 下执行 `npm install`，确认 `statics/node_modules` 已存在。 |
| MySQL access denied | 检查 `cellist/utils/constants.py` 中的用户名/密码和 MySQL plugin/auth 模式。 |
| 应用已启动但模型动作失败 | 检查 CUDA/GPU 可用性；当前路径默认使用 CUDA（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| 上传成功但无 tiles/models | 确认 `data/` 子目录存在且可写。 |
| REST/WebSocket 请求报错 | 确认服务运行于 `http://localhost:8887`，且请求 payload 键名与当前模板一致。 |
| `data/` 下报 `FileNotFoundError` | 从仓库根目录启动应用，以保证相对路径解析一致。 |

### 快速诊断

```bash
# 验证 Python 环境和关键导入
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 确认服务启动后端口是否监听
ss -ltnp | rg 8887

# 检查 MySQL 连通性
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ 路线图

以下条目来源于现有项目文档/TODO，并按现状整理：

- Polygon sample：使用 polygon 替代 rectangle 标注。
- 在极小/极大值场景下优化 `float32` 行为。
- 尽可能减小模型体积。
- 使用 Transformer / stable-diffusion 启发组件等方法提升鲁棒性。
- 增加基模型选项（Threshold、Cellpose）与目标模型选项（AIR、Transformer、SD）。
- 界面优化（含多选）。
- 后端优化（包括更好的内存/缓存处理）。
- 提供开箱即用的打包方案，减少数据库配置门槛（例如 SQLite 选项）。

## 🤝 贡献

### 为 Lazeal Cellist 贡献

Lazeal Cellist 是一个开源项目，欢迎所有人参与，无论经验水平。我们鼓励以下贡献：

- 提升算法效率与性能
- 改进界面与用户体验
- 丰富文档与示例
- 修复缺陷并提高系统稳定性

开始前请先通过 issue 讨论你想做的改动，以便协调工作，避免重复或冲突。

更多上手信息请阅读贡献指南。

仓库补充贡献文档：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 致谢

- Lazeal Cellist 的概念与实现大量借鉴了上文提到的 AIR/SPAIR 研究脉络。
- 仓库保留了历史/遗留文档与命令，以便与先前项目使用方式保持连续性。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 许可证

本项目采用 MIT 许可证。更多信息请参考本仓库的 [LICENSE](LICENSE) 文件。

仓库状态说明：当前检出的根目录中暂无 `LICENSE` 文件。上方说明保留自先前 README 作为项目规范意图；如需要可在后续更新中补充本地 `LICENSE` 文件。
