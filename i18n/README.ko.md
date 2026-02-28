[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>효율적인 3D 세포 검출 및 프로파일링 플랫폼</strong>
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

## Lazeal Cellist: 효율적인 3D 세포 검출 및 프로파일링 플랫폼

3D 현미경 이미지를 위한 포괄적이고 효율적인 세포 검출, 분할, 프로파일링 플랫폼인 Lazeal Cellist에 오신 것을 환영합니다.

이 플랫폼은 비지도 학습, 임계값 기법, Cellpose 같은 최신 알고리즘을 활용해 세포를 검출하도록 설계되었습니다. 또한 Lazeal Cellist는 사용자가 검출 결과를 정제할 수 있는 직관적인 대화형 인터페이스를 제공합니다. 이렇게 정제된 결과는 반지도학습(semi-supervised) 네트워크로 다시 전달되어 모델 성능을 지속적으로 향상시킵니다.

Lazeal Cellist의 차별점은 적은 노력으로 학습과 정제를 반복할 수 있는 효율적인 3D 모델을 제공한다는 점으로, 연구자, 과학자, 취미 사용자 모두에게 실용적인 플랫폼을 제공합니다.

> ℹ️ **범위 관련 메모**
> 프로젝트 비전과 UI에는 3D 컨셉(`/3d`, `templates/cellist_3d.html`)이 반영되어 있지만, 현재 코드의 주요 학습 흐름은 2D 슬라이싱 + 정제 중심입니다.

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

## <a id="-overview"></a>🔍 개요

Lazeal Cellist는 현미경 이미지 워크플로를 위한 Python/Tornado 웹 플랫폼입니다. 주요 기능은 다음과 같습니다.

- 브라우저 기반 업로드, 모델 생성, 주석 편집
- 알고리즘 기반 초기화(Cellpose nuclei 모드)
- WebSocket 액션(`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`)을 통한 반복적 human-in-the-loop 정제
- 모델, 이미지 슬라이스, 주석의 데이터베이스 기반 영속화

> ℹ️ 현재 동작 관련 참고: 프로젝트 비전과 UI에는 3D 컨셉(`/3d`, `templates/cellist_3d.html`)이 반영되어 있지만, 현재 코드의 주 학습 흐름은 주로 2D 슬라이스 처리와 모델 정제로 진행됩니다.

### Quick At-a-Glance

| 영역 | 현재 구현 |
|---|---|
| 서버 | Tornado (`app.py`) |
| 포트 | `8887` |
| 데이터베이스 | MySQL (`cellist.sql`) |
| 핵심 ML 스택 | PyTorch + Pyro + Cellpose |
| 프런트엔드 | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| 추론 초기화 | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| 패키징 상태 | 연구용 프로토타입 (`pyproject.toml`/`setup.py` 미포함) |
| 테스트/CI 상태 | 저장소 루트에 전용 자동 테스트 스위트 또는 CI 구성 없음 |

### Documentation Languages

이 저장소에는 `i18n/` 디렉터리 아래에 다국어 README가 포함되어 있습니다.

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

## <a id="-key-features"></a>✨ 주요 기능

- **비지도 3D 세포 검출**: 고급 머신러닝 기법으로 3D 현미경 이미지에서 세포를 검출합니다.
- **대화형 결과 정제 인터페이스**: 직관적이고 사용하기 쉬운 인터페이스로 검출 결과를 정제할 수 있습니다.
- **효율적인 반지도학습 네트워크**: 정제된 결과를 활용해 시간이 지날수록 모델 성능이 향상됩니다.
- **세포 분할 및 프로파일링**: 검출을 넘어 고급 세분화 및 프로파일링 기능을 제공합니다.

현재 코드에서 구현된 추가 기능:

- 포트 `8887`에서 동작하는 Tornado REST + WebSocket 서버(`app.py`).
- 모델 입력을 위한 자동 이미지 타일링(기본값 `256x256`).
- MySQL 스키마가 덤프 형식으로 포함됨: [`cellist.sql`](cellist.sql).
- 프런트엔드 스택에 Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload 포함.
- 스레드 풀(`max_workers=64`)을 통한 비동기 모델 작업.

## <a id="-project-structure"></a>🗂️ 프로젝트 구조

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

## <a id="-prerequisites"></a>✅ 사전 요구사항

| 요구사항 | 설명 |
|---|---|
| OS | Linux 권장 (아래 명령은 Linux 셸 동작을 기준으로 작성됨). |
| Python/Conda | [`cellist.yaml`](cellist.yaml) 파일로 Conda 환경을 생성할 수 있어야 함. |
| Database | `localhost`에서 `cellist` 데이터베이스를 사용하는 MySQL 서버 실행 필요. |
| GPU | 현재 코드 경로상 NVIDIA/CUDA 환경이 강하게 권장되며, 사실상 요구됩니다. |
| Node.js + npm | `statics/node_modules`의 프런트엔드 의존성을 설치하는 데 필요. |
| 디스크 쓰기 권한 | 런타임 데이터가 `<repo>/data`에 저장되므로 쓰기 권한 필요. |

## <a id="-installation"></a>🛠️ 설치

### 1. 저장소 복제 및 이동

```bash

git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 환경 생성

이 저장소는 `cellist.yaml` 파일명을 사용합니다.

```bash
conda env create -f cellist.yaml
conda activate cellist
```

레거시 호환 참고: 예전 문서에서는 파일명 `celist.yaml`(`l`이 누락된 이름)을 사용했지만, 이 저장소의 실제 파일은 `cellist.yaml`입니다.

레거시 명령(유지용):

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

앱은 `data/` 트리를 가정합니다(`.gitignore`에서 `data`는 이미 제외되어 있음).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

참고: [`create_data_folder.py`](create_data_folder.py)가 존재하지만 현재는 현재 작업 디렉터리 기준으로 디렉터리를 생성하며 `data/` 하위가 아닙니다. 사용 시 이 점에 유의하세요.

### 5. MySQL 인증 준비(필요 시)

루트 인증이 소켓 기반이고 앱 접근을 차단할 때는, 이전 문서에서 비밀번호 인증으로 전환하는 방법을 제시합니다.

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

레거시 문서 예시(유지용):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. 런타임용 MySQL 인증 정보 설정

현재 코드는 [`cellist/utils/constants.py`](cellist/utils/constants.py)의 `mysqlconfig` 및 `mysqlurl`에서 인증 정보를 읽습니다.

현재 기본값에는 다음이 포함됩니다.

- host: `localhost`
- user: `root`
- password: `lazeal0626`

로컬 환경 보안을 위해 실행 전 환경에 맞게 값들을 업데이트하세요.

### 8. 선택적 환경 점검

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## <a id="-usage"></a>🚀 사용법

### 웹 서버 시작

```bash
python app.py
```

이전 문서의 레거시 실행 명령(유지용):

```bash
python app.py -m cellist
```

코드에서 관찰되는 기본 라우트:

- 메인 UI: `http://localhost:8887/`
- 3D 페이지: `http://localhost:8887/3d`

### 일반 워크플로

1. UI를 열고 로그인합니다.
2. Create Model 패널에서 현미경 이미지를 업로드합니다.
3. 기본 알고리즘(`Cellpose`)을 선택하고 모델을 생성합니다.
4. 백엔드가 이미지를 슬라이싱해 검출을 초기화합니다.
5. 크롭 이미지를 로드하고, 사각형 주석을 확인/조정합니다.
6. `initialize`, `pretrain`, `train` 순환을 실행합니다.
7. 필요할 때 `Pretrain Stop` / `Stop`(`train-stop`) / `reset`을 사용합니다.
8. `Update Model` 및 주석 액션으로 수동 업데이트를 반영합니다.

### 내장 UI 로그인 계정(현재 템플릿 동작)

현재 프런트엔드는 정적 자격 정보를 클라이언트 측에서 검사합니다.

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

이는 프로토타입 동작이며 운영용 인증 체계가 아닙니다.

### UI에서 현재 사용 중인 API/소켓 인터페이스

HTTP 엔드포인트:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket 엔드포인트:

- `ws://localhost:8887/websocket/<ws_uuid>`

WebSocket 핸들러가 인식하는 `data_type` 액션:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## <a id="-configuration"></a>⚙️ 설정

### 백엔드 및 엔드포인트

[`app.py`](app.py)에서 설정됩니다.

- 포트: `8887`
- 라우트:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 모델/데이터 동작

- 스레드 풀 크기는 `max_workers=64`입니다.
- 이미지 타일 기본 크기는 `256x256`입니다.
- Cellpose 초기화는 `model_type='nuclei'` 및 `gpu=True`를 사용합니다.
- 학습/사전학습은 WebSocket 트리거 액션을 통해 비동기적으로 실행됩니다.
- 데이터 루트는 현재 작업 디렉터리 기준으로 `<repo>/data`를 사용합니다.

### 데이터베이스/런타임 상수

[`cellist/utils/constants.py`](cellist/utils/constants.py) 기준:

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig`에 host/user/password 키 포함
- `mysqlurl`은 데이터베이스 이름 `cellist`를 가리킵니다.

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

## <a id="-examples"></a>🧪 예시

### 예시: WebSocket `create` 메시지 형식

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

### 예시: 최소 엔드투엔드 로컬 실행

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## <a id="-inspired-by-research"></a>📚 연구 기반

Lazeal Cellist는 다음의 최첨단 딥러닝 연구로부터 영감을 받았습니다.

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

이들 연구는 플랫폼 알고리즘과 방법론 설계에 중요한 통찰을 제공했습니다.

(참고: 정확한 인용은 원문 논문을 직접 확인해 주세요.)

## <a id="-development-notes"></a>🧭 개발 노트

- 핵심 모델 클래스는 `cellist/` 하위(`ModelD2Init`, `ModelD2Pretrain`)에 있습니다.
- 주요 대화형 UI 로직은 `templates/cellist.html`에 직접 구현되어 있습니다.
- SQL 스키마와 시드 스타일 데이터는 `cellist.sql`에 포함되어 있습니다.
- `notebooks/` 및 `polygon_sample/`에는 탐색용 참고 자료가 있습니다.
- 확장 아키텍처/모델 노트는 [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md)에 있습니다.
- 현재 저장소 루트에는 전용 자동 테스트 스위트 또는 CI 구성 파일이 없습니다.

### Assumptions and current constraints

- 본 저장소는 우선 로컬 연구용 사용을 겨냥한 것으로 보입니다.
- 일부 코드 경로는 GPU(`cuda:0`) 사용 가능성을 전제로 합니다.
- 인증 및 비밀 정보 관리는 프로토타입 수준입니다.
- 3D 인터페이스가 존재하지만, 현재 학습 워크플로는 여전히 2D 타일 중심입니다.

## <a id="-troubleshooting"></a>🧯 문제 해결

| 증상 | 점검 항목 |
|---|---|
| `ModuleNotFoundError` 또는 import 문제 | `python app.py` 실행 전에 `conda activate cellist`가 적용되었는지 확인하세요. |
| UI가 스타일/스크립트 없이 표시됨 | `statics/`에서 `npm install`을 실행하고 `statics/node_modules` 존재 여부를 확인하세요. |
| MySQL access denied | `cellist/utils/constants.py`의 사용자/비밀번호와 MySQL plugin/auth 모드를 확인하세요. |
| 앱은 시작되지만 모델 액션 실패 | CUDA/GPU 사용 가능 여부를 확인하세요. 현재 경로는 CUDA(`torch.device('cuda:0')`, Cellpose `gpu=True`)를 가정합니다. |
| 업로드는 성공했지만 타일/모델이 나타나지 않음 | `data/` 하위 디렉터리가 존재하고 쓰기 가능한지 확인하세요. |
| REST/WebSocket 요청 오류 | 서버가 `http://localhost:8887`에서 실행 중인지, 요청 payload 키가 현재 템플릿 이름과 일치하는지 확인하세요. |
| `data/` 하위에서 `FileNotFoundError` | 상대 경로를 일관되게 처리하려면 저장소 루트에서 앱을 시작하세요. |

### 빠른 진단

```bash
# Python 환경과 핵심 import 점검
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# 시작 후 서버 포트가 열렸는지 확인
ss -ltnp | rg 8887

# MySQL 연결 확인
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## <a id="-roadmap"></a>🛣️ 로드맵

다음 항목은 기존 프로젝트 문서/TODO 노트에서 보존된 항목입니다.

- Polygon 샘플: 사각형 주석 대신 폴리곤 주석을 사용.
- 매우 작거나 큰 값에서의 `float32` 동작 최적화.
- 가능한 범위에서 모델 크기 축소.
- Transformer/Stable Diffusion 기반 구성요소와 같은 방식으로 강건성 개선.
- 기본 모델 옵션(Threshold, Cellpose)과 대상 모델 옵션(AIR, Transformer, SD) 추가.
- 인터페이스 최적화(다중 선택 포함).
- 백엔드 최적화(메모리/캐시 처리 개선 포함).
- DB 설정을 최소화한 사용성 높은 패키징(예: SQLite 옵션).

## <a id="-contributing"></a>🤝 기여

### Lazeal Cellist에 기여하기

Lazeal Cellist는 오픈 소스 프로젝트이며, 모든 수준의 기여를 환영합니다. 다음과 같은 기여를 권장합니다.

- 알고리즘 효율 및 성능 향상
- 사용자 인터페이스/사용자 경험 개선
- 문서 및 예제 확장
- 버그 수정 및 시스템 안정성 강화

기여를 시작하기 전에 먼저 이슈로 변경하고자 하는 내용을 논의해 주세요. 이는 작업 중복과 충돌을 줄이는 데 도움됩니다.

시작 방법은 기여 가이드를 참고하세요.

추가 저장소 기여 문서:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## <a id="-acknowledgements"></a>🙏 감사의 말

- Lazeal Cellist의 개념과 구현은 앞서 언급한 AIR/SPAIR 계열 연구로부터 큰 영향을 받았습니다.
- 저장소에는 과거 프로젝트 사용 맥락의 연속성을 위해 의도적으로 레거시 문서와 명령이 보존되어 있습니다.

## <a id="-support"></a>❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## <a id="-license"></a>📄 License

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 이 저장소의 [LICENSE](LICENSE) 파일을 참고하세요.

현재 체크아웃에는 루트 `LICENSE` 파일이 없습니다. 위 문구는 기존 README의 표준 프로젝트 의도를 유지하기 위해 보존된 것으로, 원하시면 다음 변경에서 로컬 `LICENSE` 파일을 추가하세요.
