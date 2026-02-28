[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>효율적인 3D 세포 검출 및 프로파일링 플랫폼</strong>
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

## Lazeal Cellist: 효율적인 3D 세포 검출 및 프로파일링 플랫폼

Lazeal Cellist에 오신 것을 환영합니다. 이 프로젝트는 3D 현미경 이미지를 위한 세포 검출, 분할, 프로파일링을 포괄적이고 효율적으로 수행하도록 설계된 플랫폼입니다.

이 플랫폼은 비지도 학습, 임계값 기법, 그리고 Cellpose 같은 최신 알고리즘을 활용해 세포를 검출합니다. 또한 직관적이고 상호작용적인 인터페이스를 제공하여 사용자가 검출 결과를 직접 보정할 수 있습니다. 보정된 결과는 다시 준지도 학습 네트워크로 피드백되어 모델 성능을 지속적으로 개선합니다.

Lazeal Cellist는 적은 학습/보정 노력으로도 운용 가능한 효율적인 3D 모델을 제공한다는 점에서 차별화되며, 과학자, 연구자, 취미 사용자 모두에게 실용적인 플랫폼입니다.

> ℹ️ **범위 참고**
> 프로젝트 비전과 UI에는 3D 개념(`/3d`, `templates/cellist_3d.html`)이 포함되어 있지만, 현재 코드의 주요 학습 플로우는 주로 2D 슬라이싱 + 보정 중심입니다.

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

Lazeal Cellist는 현미경 이미지 워크플로를 위한 Python/Tornado 웹 플랫폼으로, 다음 기능을 제공합니다.

- 브라우저 기반 업로드, 모델 생성, 주석 편집
- 알고리즘 보조 초기화(Cellpose nuclei 모드)
- WebSocket 액션(`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`)을 통한 반복형 human-in-the-loop 보정
- 모델, 이미지 슬라이스, 주석 데이터를 위한 데이터베이스 기반 영속화

> ℹ️ 현재 동작 참고: 프로젝트 비전과 UI에는 3D 개념(`/3d`, `templates/cellist_3d.html`)이 포함되어 있지만, 현재 코드의 메인 학습 플로우는 주로 2D 슬라이싱 + 모델 보정입니다.

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
| Tests/CI status | 저장소 루트에 전용 자동 테스트 스위트/CI 설정 없음 |

### Documentation Languages

이 저장소에는 `i18n/` 아래에 다국어 README가 포함되어 있습니다.

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

- **비지도 3D 세포 검출**: 고급 머신러닝 기법으로 3D 현미경 이미지에서 세포를 식별합니다.
- **상호작용형 결과 보정 인터페이스**: 직관적이고 사용자 친화적인 UI로 검출 결과를 보정합니다.
- **효율적인 준지도 학습 네트워크**: 보정 결과를 반영해 시간이 지날수록 모델 성능을 향상합니다.
- **세포 분할 및 프로파일링**: 검출을 넘어 고급 분할/프로파일링 기능을 제공합니다.

현재 구현에 포함된 추가 기능:

- 포트 `8887`에서 동작하는 Tornado REST + WebSocket 서버(`app.py`).
- 모델 입력을 위한 자동 이미지 타일링(기본 `256x256`).
- MySQL 스키마 덤프 포함: [`cellist.sql`](cellist.sql).
- 프런트엔드 스택: Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- 스레드 풀(`max_workers=64`) 기반 비동기 모델 작업.

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
| OS | Linux 권장(아래 명령은 Linux shell 동작 기준). |
| Python/Conda | [`cellist.yaml`](cellist.yaml)로 환경을 생성하려면 Conda 필요. |
| Database | `localhost`에서 `cellist` 데이터베이스를 사용하는 MySQL 서버 실행 필요. |
| GPU | 현재 코드 경로 특성상 NVIDIA/CUDA 환경을 강력히 권장/사실상 요구. |
| Node.js + npm | `statics/node_modules` 프런트엔드 의존성 설치에 필요. |
| Disk write access | `<repo>/data` 하위 런타임 데이터 쓰기 권한 필요. |

## 🛠️ Installation

### 1. 저장소 클론 및 이동

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 환경 생성

저장소의 파일명은 `cellist.yaml`입니다.

```bash
conda env create -f cellist.yaml
conda activate cellist
```

구 문서와의 호환성 참고: 이전 문서에서는 `celist.yaml`(`l` 하나 누락)을 사용했지만, 이 저장소의 실제 파일명은 `cellist.yaml`입니다.

레거시 명령(보존):

```bash
conda env create -f celist.yaml
```

### 3. 프런트엔드 의존성 설치

```bash
cd statics
npm install
cd ..
```

### 4. 런타임 데이터 디렉터리 준비

앱은 `data/` 트리를 필요로 하며(`.gitignore`에서 `data`는 이미 제외됨), 아래처럼 생성할 수 있습니다.

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

참고: [`create_data_folder.py`](create_data_folder.py)는 존재하지만, 현재는 `data/` 하위가 아니라 현재 작업 디렉터리에 디렉터리를 만듭니다. 사용 시 이 점을 유의하세요.

### 5. MySQL 인증 준비(필요한 경우)

루트 인증이 소켓 기반이라 앱 접근이 막히는 경우, 과거 프로젝트 문서에서는 비밀번호 인증 전환을 권장합니다.

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. 데이터베이스 생성 및 스키마/데이터 복원

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

레거시 문서 예시(보존):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 런타임용 MySQL 자격 증명 설정

현재 코드는 [`cellist/utils/constants.py`](cellist/utils/constants.py)의 `mysqlconfig`와 `mysqlurl`에서 자격 증명을 읽습니다.

현재 코드 기본값에는 다음이 포함됩니다.

- host: `localhost`
- user: `root`
- password: `lazeal0626`

로컬 보안을 위해 실행 전에 환경에 맞게 값을 수정하세요.

### 8. 선택적 환경 점검

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Usage

### 웹 서버 시작

```bash
python app.py
```

이전 문서의 레거시 시작 명령(보존):

```bash
python app.py -m cellist
```

코드에서 확인되는 기본 서버 경로:

- Main UI: `http://localhost:8887/`
- 3D page: `http://localhost:8887/3d`

### 일반 워크플로

1. UI를 열고 로그인합니다.
2. Create Model 패널에서 현미경 이미지를 업로드합니다.
3. 기본 알고리즘(`Cellpose`)을 선택해 모델을 생성합니다.
4. 백엔드가 이미지를 슬라이스하고 초기 검출을 수행하도록 합니다.
5. 크롭 이미지를 불러와 사각형 주석을 검토/수정합니다.
6. `initialize`, `pretrain`, `train` 사이클을 실행합니다.
7. 필요 시 `Pretrain Stop` / `Stop` (`train-stop`) / `reset`을 사용합니다.
8. `Update Model`/주석 액션으로 수동 업데이트를 반영합니다.

### 내장 UI 로그인 자격 증명(현재 템플릿 동작)

프런트엔드는 현재 아래 정적 자격 증명을 클라이언트 사이드에서 검사합니다.

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

이는 프로토타입 동작이며 운영 환경용 인증 방식이 아닙니다.

### 현재 UI에서 사용하는 API/소켓 표면

HTTP 엔드포인트:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 엔드포인트:

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket 핸들러가 인식하는 `data_type` 액션 메시지:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuration

### 백엔드 및 엔드포인트

[`app.py`](app.py)에 설정되어 있습니다.

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 모델/데이터 동작

- 스레드 풀 크기: `max_workers=64`
- 기본 이미지 타일 크기: `256x256`
- Cellpose 초기화: `model_type='nuclei'`, `gpu=True`
- 학습/사전학습은 WebSocket 트리거 액션으로 비동기 실행
- 데이터 루트는 현재 작업 디렉터리 기준 `<repo>/data`로 해석

### 데이터베이스/런타임 상수

[`cellist/utils/constants.py`](cellist/utils/constants.py) 기준:

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig`에 host/user/password 키 포함
- `mysqlurl`은 데이터베이스명 `cellist` 대상

### 프런트엔드 의존성 스냅샷

[`statics/package.json`](statics/package.json) 기준:

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda 환경 핵심 항목

[`cellist.yaml`](cellist.yaml) 기준:

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Examples

### 예시: WebSocket create 메시지 형태

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

### 예시: WebSocket 수동 주석 업데이트

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### 예시: 모델 로드 요청

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### 예시: 최소 end-to-end 로컬 시작

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

Lazeal Cellist는 다음과 같은 딥러닝 최신 연구에서 영감을 받았습니다.

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

이 연구들은 플랫폼의 알고리즘과 방법론 개발에 중요한 인사이트를 제공했습니다.

(참고: 정확한 인용은 원 논문을 직접 확인해 주세요.)

## 🧭 Development Notes

- 핵심 모델 클래스는 `cellist/` 아래(`ModelD2Init`, `ModelD2Pretrain`)에 있습니다.
- 메인 상호작용 UI 로직은 `templates/cellist.html`에 직접 포함되어 있습니다.
- SQL 스키마 및 시드 성격 데이터는 `cellist.sql`에 있습니다.
- `notebooks/`와 `polygon_sample/`에는 탐색적 참고 자료가 포함되어 있습니다.
- 확장 플랫폼/모델 메모는 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)에 있습니다.
- 현재 저장소 루트에는 전용 자동 테스트 스위트나 CI 설정이 없습니다.

### 가정 및 현재 제약

- 이 저장소는 우선 로컬 연구용 사용을 목표로 하는 것으로 보입니다.
- 일부 코드 경로는 GPU(`cuda:0`) 가용성을 전제로 합니다.
- 인증 및 시크릿 관리는 프로토타입 수준입니다.
- 3D 인터페이스는 존재하지만, 주요 학습 워크플로는 여전히 2D 타일 중심입니다.

## 🧯 Troubleshooting

| Symptom | Suggested checks |
|---|---|
| `ModuleNotFoundError` 또는 import 문제 | `python app.py` 실행 전에 `conda activate cellist` 적용 여부 확인 |
| UI는 뜨지만 스타일/스크립트가 깨짐 | `statics/`에서 `npm install` 실행 후 `statics/node_modules` 존재 확인 |
| MySQL access denied | `cellist/utils/constants.py`의 사용자명/비밀번호와 MySQL plugin/auth 모드 확인 |
| 앱은 시작되지만 모델 액션 실패 | CUDA/GPU 가용성 확인(현재 경로는 CUDA(`torch.device('cuda:0')`, Cellpose `gpu=True`) 가정) |
| 업로드는 되지만 타일/모델이 생성되지 않음 | `data/` 하위 디렉터리 존재 및 쓰기 권한 확인 |
| REST/WebSocket 요청 오류 | 서버가 `http://localhost:8887`에서 실행 중인지, 요청 payload 키가 템플릿 이름과 일치하는지 확인 |
| `data/` 하위 `FileNotFoundError` | 상대 경로 일관성을 위해 저장소 루트에서 앱 시작 |

### 빠른 진단

```bash
# Python 환경 및 핵심 import 확인
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 서버 시작 후 포트 오픈 확인
ss -ltnp | rg 8887

# MySQL 연결 확인
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Roadmap

아래 항목은 기존 프로젝트 문서/TODO 메모를 정리해 보존한 것입니다.

- Polygon 샘플: 사각형 대신 폴리곤 주석 사용.
- 매우 작은/큰 값에 대한 `float32` 동작 기준 모델 최적화.
- 가능한 범위에서 모델 크기 축소.
- Transformer/stable-diffusion 계열 아이디어를 포함한 강건성 개선.
- 베이스 모델 옵션(Threshold, Cellpose) 및 타깃 모델 옵션(AIR, Transformer, SD) 추가.
- 인터페이스 최적화(다중 선택 포함).
- 백엔드 최적화(메모리/캐시 처리 개선 포함).
- 최소한의 DB 설정으로 쉽게 사용할 수 있는 패키징(예: SQLite 옵션).

## 🤝 Contributing

### Lazeal Cellist에 기여하기

Lazeal Cellist는 오픈소스 프로젝트이며, 경험 수준과 관계없이 모든 기여를 환영합니다. 특히 다음과 같은 기여를 기다립니다.

- 알고리즘 효율성과 성능 향상
- 사용자 인터페이스 및 사용자 경험 개선
- 문서와 예제 확장
- 버그 수정 및 시스템 안정성 강화

기여를 시작하기 전에, 변경하고자 하는 내용을 이슈로 먼저 논의해 주세요. 이는 작업 조율과 중복/충돌 방지에 도움이 됩니다.

시작 방법은 기여 가이드를 참고해 주세요.

저장소 내 추가 기여 문서:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Lazeal Cellist의 개념과 구현은 위에 나열한 AIR/SPAIR 계열 연구에 크게 기반합니다.
- 저장소에는 과거 사용 맥락의 연속성을 위해 의도적으로 보존된 레거시 문서/명령이 포함되어 있습니다.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 저장소의 [LICENSE](LICENSE) 파일을 참고하세요.

저장소 상태 참고: 현재 체크아웃에는 루트 `LICENSE` 파일이 없습니다. 위 문구는 기존 README의 프로젝트 의도를 보존한 것이며, 원하면 후속 변경에서 로컬 `LICENSE` 파일을 추가하세요.
