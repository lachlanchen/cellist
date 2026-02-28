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

## 🎬 预览

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist：您的高效3D细胞检测与分析平台

欢迎使用 Lazeal Cellist，这是一款面向 3D 显微镜图像的、完整且高效的细胞检测、分割与分析平台。

本平台采用无监督学习、阈值方法和如 Cellpose 等先进算法来识别细胞。Lazeal Cellist 还提供直观的交互界面，让用户可以细化检测结果。细化后的结果会反馈给半监督学习网络，持续提升模型表现。

Lazeal Cellist 的优势在于其高效的 3D 模型，只需极少的训练与细化工作量，适合科研人员、研究者和爱好者使用。

> ℹ️ **范围说明**
> 项目愿景与界面中包含 3D 概念（`/3d`、`templates/cellist_3d.html`），但当前代码的主要训练流程仍是 2D 切片 + 细化。

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
- [许可证](#-license)

## 🔍 概览

Lazeal Cellist 是一个面向显微镜影像流程的 Python/Tornado Web 平台，具有以下能力：

- 浏览器端上传、模型创建与标注编辑。
- 算法辅助初始化（Cellpose nuclei 模式）。
- 通过 WebSocket 动作（`create`、`initialize`、`pretrain`、`pretrain-stop`、`train`、`train-stop`、`update`、`reset`）进行人机协同的迭代式细化。
- 数据库持久化存储模型、图像切片和标注。

> ℹ️ 当前行为说明：尽管项目愿景与 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），当前代码中的主要训练流程仍以 2D 切片 + 模型细化为主。

### 快速一览

| 模块 | 当前实现 |
|---|---|
| 服务器 | Tornado（`app.py`） |
| 端口 | `8887` |
| 数据库 | MySQL（`cellist.sql`） |
| 核心 ML 技术栈 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推理初始化 | Cellpose（`model_type='nuclei'`，`gpu=True`） |
| 打包状态 | 研究原型（未包含 `pyproject.toml`/`setup.py`） |
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

- **无监督 3D 细胞检测**：使用先进的机器学习技术，在 3D 显微镜图像中识别细胞。
- **交互式结果细化界面**：通过直观友好的界面细化检测结果。
- **高效半监督学习网络**：利用细化结果持续提升模型表现。
- **细胞分割与分析**：不仅支持检测，还支持更深度的分割与特征分析。

当前实现中还具备的附加特性：

- 基于 Tornado 的 REST + WebSocket 服务器（`app.py`），端口为 `8887`。
- 自动影像切片（默认 `256x256`）用于模型输入。
- MySQL schema 以导出文件形式提供：[`cellist.sql`](cellist.sql)。
- 前端技术栈包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。
- 通过线程池异步执行模型任务（`max_workers=64`）。

## 🗂️ 项目结构

```text
cellist/
├── app.py                               # 主 Tornado 服务 + REST/WebSocket 处理器
├── cellist/                             # 核心 ML/模型代码
│   ├── model_init.py                    # 2D 主模型类与训练流程
│   ├── model_pretrain.py                # 预训练变体
│   ├── model_2d_components.py           # 编码器/解码器/SPAIR 组件
│   ├── model_2d_utilities.py            # 基于数据库的模型元数据 + transforms
│   ├── image_preprocessing.py           # 切片/拼接工具
│   └── utils/constants.py               # 运行时路径 + MySQL 配置
├── templates/
│   ├── cellist.html                     # 主要 2D 界面
│   └── cellist_3d.html                  # 3D 界面变体/原型
├── statics/                             # 前端资源和 npm 依赖
│   ├── package.json
│   └── node_modules/
├── i18n/                                # 翻译版 README
├── notebooks/                           # 探索型笔记本
├── polygon_sample/                      # 多边形标注实验
├── figs/                                # 品牌图形素材
├── cellist.sql                          # MySQL schema 与数据导出
├── cellist.yaml                         # Conda 环境规范
├── create_data_folder.py                # 旧版数据文件夹创建脚本
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # 扩展架构/TODO 说明
└── README.md
```

## ✅ 先决条件

| 需求 | 说明 |
|---|---|
| 操作系统 | 建议使用 Linux（下列命令按 Linux shell 行为执行）。 |
| Python/Conda | 可用 Conda，用于依据 [`cellist.yaml`](cellist.yaml) 创建环境。 |
| 数据库 | 本地运行 MySQL，`localhost` 上需存在 `cellist` 数据库。 |
| GPU | 现有代码路径强烈推荐/默认预期使用 NVIDIA/CUDA。 |
| Node.js + npm | 需要用于安装 `statics/node_modules` 前端依赖。 |
| 磁盘写权限 | 运行时需在 `<repo>/data` 下有写入权限。 |

## 🛠️ 安装

### 1. 克隆并进入仓库

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. 创建 Python 环境

使用仓库中的文件名 `cellist.yaml`：

```bash
conda env create -f cellist.yaml
conda activate cellist
```

兼容性说明保留自旧文档：旧文档曾使用 `celist.yaml`（少了一个 `l`），但本仓库中的文件为 `cellist.yaml`。

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

应用期望存在 `data/` 目录树（`.gitignore` 已将 `data` 排除）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

说明：[`create_data_folder.py`](create_data_folder.py) 已存在，但当前会在当前工作目录下创建目录（不是在 `data/` 下），如使用请注意。

### 5. 准备 MySQL 认证（如需要）

如果 root 认证是基于 socket 且拦截了应用访问，旧版文档建议切换为密码认证：

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

保留的历史文档示例（仅作参考）：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 配置运行时 MySQL 凭据

当前代码从 [`cellist/utils/constants.py`](cellist/utils/constants.py) 读取凭据（`mysqlconfig` 与 `mysqlurl`）。

当前默认值如下：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

出于本地安全考虑，请在你的环境中运行前更新这些值。

### 8. 可选：环境自检

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

保留的历史启动命令（来自旧文档）：

```bash
python app.py -m cellist
```

代码中的默认路由：

- 主界面：`http://localhost:8887/`
- 3D 页面：`http://localhost:8887/3d`

### 常规流程

1. 打开界面并登录。
2. 在“创建模型”面板上传显微镜图像。
3. 选择基础算法（`Cellpose`）并创建模型。
4. 让后端切片并初始化检测结果。
5. 加载裁剪后的图像，审阅/调整矩形标注。
6. 执行 `initialize`、`pretrain` 与 `train` 周期。
7. 按需使用 `Pretrain Stop` / `Stop`（`train-stop`）/ `reset`。
8. 通过 `Update Model`/标注操作持久化手工更新。

### 内置 UI 登录凭据（当前模板行为）

前端当前在客户端进行以下静态凭据校验：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

这是原型行为，不属于生产环境认证方案。

### UI 当前使用的 API/Socket 接口

HTTP 端点：

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 端点：

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket handler 中已识别的 `data_type` 动作消息：

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

配置于 [`app.py`](app.py)：

- 端口：`8887`
- 路由：
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型与数据行为

- 线程池大小为 `max_workers=64`。
- 默认图像切片大小为 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 与 `gpu=True`。
- 训练与预训练通过 WebSocket 触发的动作异步执行。
- 数据根目录使用当前工作目录解析为 `<repo>/data`。

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
- Pyro（`pyro-ppl==1.8.1`）
- PyMySQL + SQLAlchemy

## 🧪 示例

### 示例：WebSocket 创建消息结构

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

这些成果为平台的算法和方法设计提供了重要参考。

（注：如需准确引用，请直接参考原始论文。）

## 🧭 开发说明

- 核心模型类位于 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要交互式 UI 逻辑直接嵌在 `templates/cellist.html` 中。
- SQL schema 与类似种子数据位于 `cellist.sql`。
- `notebooks/` 与 `polygon_sample/` 中的笔记本可作为探索参考。
- 扩展的平台/模型说明见 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)。
- 仓库根目录目前没有独立的自动化测试套件或 CI 配置。

### 假设与当前约束

- 本仓库似乎优先服务于本地研究用途。
- 某些代码路径假设 GPU（`cuda:0`）可用。
- 认证与密钥管理处于原型级别。
- 3D 界面已存在，但主训练流程仍以 2D 切片为主。

## 🧯 故障排查

| 症状 | 建议排查 |
|---|---|
| `ModuleNotFoundError` 或导入报错 | 确认运行 `python app.py` 前已执行 `conda activate cellist`。 |
| UI 无样式或脚本未加载 | 在 `statics/` 内执行 `npm install`，并确认存在 `statics/node_modules`。 |
| MySQL 被拒绝访问 | 检查 `cellist/utils/constants.py` 中的用户名/密码与 MySQL 插件/认证模式。 |
| 应用启动但模型动作失败 | 检查 CUDA/GPU 可用性；当前路径默认假设使用 CUDA（`torch.device('cuda:0')`，Cellpose `gpu=True`）。 |
| 上传成功但无切片/模型出现 | 确保 `data/` 各子目录存在且可写。 |
| REST/WebSocket 请求报错 | 确认服务运行在 `http://localhost:8887`，且请求载荷字段与当前模板名称一致。 |
| `data/` 下出现 `FileNotFoundError` | 请从仓库根目录启动应用，以保证相对路径一致解析。 |

### 快速诊断

```bash
# 验证 Python 环境与关键导入
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 验证启动后端口是否开放
ss -ltnp | rg 8887

# 检查 MySQL 连接
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ 路线图

以下条目保留自现有项目文档/TODO 说明：

- 多边形样例：改用 polygon 代替矩形标注。
- 针对非常小/非常大数值优化 `float32` 行为。
- 尽可能缩小模型规模。
- 借鉴 Transformer / stable-diffusion 等思路提升鲁棒性。
- 增加基础模型选项（Threshold、Cellpose）与目标模型选项（AIR、Transformer、SD）。
- 界面优化（包括多选）。
- 后端优化（包括更好的内存/缓存处理）。
- 提供更易用的打包方案，降低数据库配置门槛（如支持 SQLite 选项）。

## 🤝 贡献

### 参与 Lazeal Cellist

Lazeal Cellist 是开源项目，欢迎各水平的贡献者加入。我们欢迎以下方向的贡献：

- 提升算法效率和性能
- 改进用户界面与交互体验
- 扩展文档与示例
- 修复缺陷并增强系统稳定性

在开始提交更改前，请先在 issue 中讨论你的改动思路。这样有助于协调工作、避免重复或冲突。

如需了解如何开始，请阅读贡献指南。

补充的仓库贡献文档：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 致谢

- Lazeal Cellist 的概念与实现大量借鉴了上述 AIR/SPAIR 研究体系。
- 仓库保留了历史和旧版文档与命令，以确保与之前的项目使用方式兼容。



## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
