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

## Lazeal Cellist：高效的 3D 细胞检测与分析平台

欢迎使用 Lazeal Cellist，这是一个面向 3D 显微图像的综合高效细胞检测、分割与分析平台。

本平台通过无监督学习、阈值技术以及 Cellpose 等先进算法进行细胞检测。Lazeal Cellist 还提供直观的交互式界面，便于用户细化检测结果。经过细化的结果会回流至半监督学习网络，从而持续提升模型性能。

Lazeal Cellist 的突出优势是提供一个高效的 3D 模型，训练与细化成本低，适用于科学家、研究人员和爱好者。

---

## 🔍 概览

Lazeal Cellist 是一个基于 Python/Tornado 的显微图像工作流 Web 平台，提供：

- 基于浏览器的上传、模型创建和标注编辑。
- 算法辅助初始化（Cellpose nuclei 模式）。
- 通过 WebSocket 动作（`initialize`、`pretrain`、`train`、`update`、`reset`）实现迭代式 human-in-the-loop 精修。
- 基于数据库的模型、图像切片和标注持久化。

当前行为说明：尽管项目愿景和 UI 包含 3D 概念（`/3d`、`templates/cellist_3d.html`），代码中的主训练流程目前主要是 2D 切片 + 模型精修。

### 快速总览

| 区域 | 当前实现 |
|---|---|
| 服务器 | Tornado（`app.py`） |
| 端口 | `8887` |
| 数据库 | MySQL（`cellist.sql`） |
| 核心 ML 栈 | PyTorch + Pyro + Cellpose |
| 前端 | Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload |
| 推理初始化 | Cellpose（`model_type='nuclei'`、`gpu=True`） |

## ✨ 关键特性

- **无监督 3D 细胞检测**：使用先进机器学习技术识别 3D 显微图像中的细胞。
- **交互式结果精修界面**：通过直观易用的界面细化检测结果。
- **高效半监督学习网络**：利用精修结果持续提升模型性能。
- **细胞分割与分析**：不仅能检测，还支持更深入的分割与分析能力。

当前已具备的额外实现特性：

- 运行在 `8887` 端口的 Tornado REST + WebSocket 服务器（`app.py`）。
- 用于模型输入的自动图像切片（默认 `256x256`）。
- 以 dump 形式提供的 MySQL schema：[`cellist.sql`](cellist.sql)。
- 前端技术栈包含 Bootstrap、jQuery、jQuery UI、Three.js、blueimp-file-upload。

## 🗂️ 项目结构

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

## ✅ 前置要求

| 要求 | 说明 |
|---|---|
| OS | 推荐 Linux（以下命令默认 Linux shell 行为）。 |
| Python/Conda | 可使用 [`cellist.yaml`](cellist.yaml) 创建环境。 |
| 数据库 | 在 `localhost` 运行 MySQL，并存在 `cellist` 数据库。 |
| GPU | 当前代码路径强烈推荐/预期使用 NVIDIA/CUDA 环境。 |
| Node.js + npm | 安装 `statics/node_modules` 前端依赖所需。 |

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

保留旧文档兼容说明：此前文档使用的是 `celist.yaml`（少一个 `l`），但本仓库中的文件是 `cellist.yaml`。

### 3. 安装前端依赖

```bash
cd statics
npm install
cd ..
```

### 4. 准备 MySQL 认证（如有需要）

如果 root 认证使用 socket 且阻止应用访问，旧项目文档建议切换为密码认证：

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. 创建数据库并恢复 schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

保留旧文档示例：

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. 配置运行时 MySQL 凭据

当前代码从 [`cellist/utils/constants.py`](cellist/utils/constants.py) 读取凭据（`mysqlconfig` 和 `mysqlurl`）。

当前代码默认值包括：

- host: `localhost`
- user: `root`
- password: `lazeal0626`

为本地安全起见，请在你的环境中运行前更新这些值。

### 7. 准备运行时数据目录

应用期望存在 `data/` 目录树（且 `.gitignore` 已排除 `data`）。

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

说明：[`create_data_folder.py`](create_data_folder.py) 存在，但它目前会在当前工作目录下创建目录（而不是 `data/` 下）。使用时请注意。

## 🚀 使用

### 启动 Web 服务器

```bash
python app.py
```

保留旧文档中的启动命令：

```bash
python app.py -m cellist
```

代码中可见的默认服务路由：

- 主界面：`http://localhost:8887/`
- 3D 页面：`http://localhost:8887/3d`

### 典型工作流

1. 打开 UI 并登录。
2. 在 Create Model 面板上传显微图像。
3. 选择基础算法（`Cellpose`）并创建模型。
4. 等待后端切片图像并初始化检测结果。
5. 加载 cropped 图像，检查/调整矩形标注。
6. 运行 `initialize`、`pretrain` 和 `train` 循环。
7. 通过 `Update Model`/标注动作持久化手动更新。

### 内置 UI 登录凭据（当前模板行为）

前端目前在客户端侧检查以下静态凭据：

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

这是原型行为，不是生产级认证方案。

## ⚙️ 配置

### 后端与端点

在 [`app.py`](app.py) 中配置：

- 端口：`8887`
- 路由：
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 模型/数据行为

- 线程池大小为 `max_workers=64`。
- 图像 tile 默认大小为 `256x256`。
- Cellpose 初始化使用 `model_type='nuclei'` 和 `gpu=True`。
- 训练与预训练通过 WebSocket 触发的动作异步运行。

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

## 📚 研究启发

Lazeal Cellist 受到前沿深度学习研究启发，包括：

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

这些工作为本平台算法和方法学的开发提供了重要思路。

（注：准确引用请直接参考原始论文。）

## 🧭 开发说明

- 核心模型类位于 `cellist/`（`ModelD2Init`、`ModelD2Pretrain`）。
- 主要交互式 UI 逻辑直接嵌入在 `templates/cellist.html`。
- SQL schema 与 seed 风格数据位于 `cellist.sql`。
- `notebooks/` 与 `polygon_sample/` 中包含探索性参考内容。
- 仓库根目录目前没有专门的自动化测试套件或 CI 配置。

## 🧯 故障排查

| 症状 | 建议检查 |
|---|---|
| `ModuleNotFoundError` 或导入问题 | 运行 `python app.py` 前确认已执行 `conda activate cellist`。 |
| UI 渲染但无样式/脚本 | 在 `statics/` 内运行 `npm install`，并确认 `statics/node_modules` 存在。 |
| MySQL access denied | 检查 `cellist/utils/constants.py` 中用户名/密码，以及 MySQL plugin/auth 模式。 |
| 应用已启动但模型动作失败 | 检查 CUDA/GPU 可用性；当前路径假设 CUDA（`torch.device('cuda:0')`、Cellpose `gpu=True`）。 |
| 上传成功但未出现 tiles/models | 确认 `data/` 子目录存在且可写。 |

## 🗺️ 路线图

以下条目从现有项目文档/TODO 说明中保留并整理：

- Polygon sample：使用 polygon 替代矩形标注。
- 针对极小/极大值场景优化模型 `float32` 行为。
- 在可行情况下缩小模型体积。
- 采用如 Transformer/受 stable-diffusion 启发组件等方案提高鲁棒性。
- 增加基础模型选项（Threshold、Cellpose）与目标模型选项（AIR、Transformer、SD）。
- 界面优化（包括多选能力）。
- 后端优化（包括改进内存/缓存处理）。
- 提供易用打包方案并减少 DB 配置门槛（如 SQLite 选项）。

## 🤝 贡献

### 为 Lazeal Cellist 做贡献

Lazeal Cellist 是开源项目，欢迎所有经验水平的贡献者参与。我们欢迎以下方面的贡献：

- 提升算法效率与性能
- 改进用户界面与用户体验
- 扩展文档与示例
- 修复缺陷并增强系统稳定性

在开始贡献前，请先通过 issue 讨论你希望进行的变更。这有助于协调工作并避免重复或冲突。

更多入门信息请阅读贡献指南。

仓库中的额外贡献文档：

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 许可证

本项目采用 MIT 许可证。更多信息请参阅本仓库中的 [LICENSE](https://chat.openai.com/LICENSE) 文件。

仓库状态说明：当前检出的根目录中暂无 `LICENSE` 文件。上面的说明按先前 README 作为项目规范意图予以保留；如有需要，可在后续变更中补充本地 `LICENSE` 文件。
