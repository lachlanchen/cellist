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

## Lazeal Cellist: 효율적인 3D 세포 검출 및 프로파일링 플랫폼

Lazeal Cellist에 오신 것을 환영합니다. 이 플랫폼은 3D 현미경 이미지를 위한 포괄적이고 효율적인 세포 검출, 분할, 프로파일링 환경을 제공합니다.

우리 플랫폼은 비지도 학습, 임계값(Threshold) 기법, Cellpose 같은 최신 알고리즘을 활용해 세포를 검출하도록 설계되었습니다. 또한 사용자가 검출 결과를 정교하게 보정할 수 있는 직관적이고 상호작용적인 인터페이스를 제공합니다. 이렇게 보정된 결과는 반지도 학습 네트워크에 다시 반영되어 모델 성능을 지속적으로 향상시킵니다.

Lazeal Cellist는 적은 노력으로 학습 및 보정이 가능한 효율적인 3D 모델을 제공한다는 점에서 강점이 있으며, 과학자, 연구자, 취미 사용자 모두에게 실용적인 플랫폼입니다.

---

## 🔍 개요

Lazeal Cellist는 현미경 이미지 워크플로를 위한 Python/Tornado 웹 플랫폼으로, 다음을 제공합니다.

- 브라우저 기반 업로드, 모델 생성, 어노테이션 편집
- 알고리즘 기반 초기화(Cellpose nuclei 모드)
- WebSocket 액션(`initialize`, `pretrain`, `train`, `update`, `reset`)을 통한 반복적 human-in-the-loop 보정
- 모델, 이미지 슬라이스, 어노테이션을 위한 데이터베이스 기반 영속성

현재 동작 관련 참고: 프로젝트 비전과 UI에는 3D 개념(``/3d``, `templates/cellist_3d.html`)이 포함되어 있지만, 현재 코드의 메인 학습 흐름은 주로 2D 슬라이싱 + 모델 보정 중심입니다.

### 빠른 요약

| 영역 | 현재 구현 |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Database | MySQL (`cellist.sql`) |
| Core ML stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inference init | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ 주요 기능

- **비지도 3D 세포 검출**: 고급 머신러닝 기법으로 3D 현미경 이미지에서 세포를 식별합니다.
- **상호작용형 결과 보정 인터페이스**: 직관적이고 사용하기 쉬운 인터페이스로 검출 결과를 보정합니다.
- **효율적인 반지도 학습 네트워크**: 보정 결과를 반영해 시간이 지날수록 모델 성능을 향상합니다.
- **세포 분할 및 프로파일링**: 단순 검출을 넘어 고급 분할과 프로파일링 기능을 제공합니다.

현재 구현에 포함된 추가 기능:

- `8887` 포트에서 동작하는 Tornado REST + WebSocket 서버(`app.py`)
- 모델 입력을 위한 자동 이미지 타일링(기본 `256x256`)
- MySQL 스키마가 덤프로 포함됨: [`cellist.sql`](cellist.sql)
- 프런트엔드 스택: Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload

## 🗂️ 프로젝트 구조

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

## ✅ 사전 요구사항

| 요구사항 | 비고 |
|---|---|
| OS | Linux 권장(아래 명령은 Linux 셸 동작을 기준으로 작성됨). |
| Python/Conda | [`cellist.yaml`](cellist.yaml)에서 환경을 생성할 수 있도록 Conda가 준비되어 있어야 함. |
| Database | `localhost`에서 `cellist` 데이터베이스를 사용하는 MySQL 서버가 실행 중이어야 함. |
| GPU | 현재 코드 경로는 NVIDIA/CUDA 환경을 강하게 권장/사실상 요구함. |
| Node.js + npm | `statics/node_modules` 프런트엔드 의존성 설치에 필요함. |

## 🛠️ 설치

### 1. 저장소 클론 및 진입

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python 환경 생성

저장소 파일명 `cellist.yaml`을 사용하세요.

```bash
conda env create -f cellist.yaml
conda activate cellist
```

이전 문서와의 호환성 참고: 과거 문서에서는 `celist.yaml`(l 하나 누락)을 사용했지만, 이 저장소의 실제 파일은 `cellist.yaml`입니다.

### 3. 프런트엔드 의존성 설치

```bash
cd statics
npm install
cd ..
```

### 4. MySQL 인증 준비(필요 시)

루트 인증이 소켓 기반이라 앱 접근이 막히는 경우, 과거 프로젝트 문서에서는 비밀번호 인증으로 전환을 권장합니다.

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. 데이터베이스 생성 및 스키마/데이터 복원

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

기존 문서 예시(보존):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. 런타임 MySQL 자격 증명 설정

현재 코드는 [`cellist/utils/constants.py`](cellist/utils/constants.py)의 `mysqlconfig` 및 `mysqlurl`에서 자격 증명을 읽습니다.

현재 코드 기본값은 다음을 포함합니다.

- host: `localhost`
- user: `root`
- password: `lazeal0626`

로컬 보안을 위해, 실제 환경에서 실행하기 전에 이 값을 수정하세요.

### 7. 런타임 데이터 디렉터리 준비

앱은 `data/` 트리를 필요로 하며(그리고 `.gitignore`는 이미 `data`를 제외함), 아래와 같이 생성할 수 있습니다.

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

참고: [`create_data_folder.py`](create_data_folder.py)가 존재하지만, 현재는 `data/` 하위가 아니라 현재 작업 디렉터리에 폴더를 생성합니다. 사용 시 이 점을 유의하세요.

## 🚀 사용법

### 웹 서버 시작

```bash
python app.py
```

이전 문서의 시작 명령(보존):

```bash
python app.py -m cellist
```

코드에서 확인되는 서버 기본 경로:

- 메인 UI: `http://localhost:8887/`
- 3D 페이지: `http://localhost:8887/3d`

### 일반적인 워크플로

1. UI를 열고 로그인합니다.
2. Create Model 패널에서 현미경 이미지를 업로드합니다.
3. 기본 알고리즘(`Cellpose`)을 선택하고 모델을 생성합니다.
4. 백엔드가 이미지를 슬라이싱하고 초기 검출을 수행하도록 둡니다.
5. 크롭 이미지를 불러와 사각형 어노테이션을 검토/조정합니다.
6. `initialize`, `pretrain`, `train` 사이클을 실행합니다.
7. `Update Model`/어노테이션 액션으로 수동 업데이트를 저장합니다.

### UI 내장 로그인 자격 증명(현재 템플릿 동작)

현재 프런트엔드는 아래의 정적 자격 증명을 클라이언트 측에서 검사합니다.

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

이는 프로토타입 동작이며 운영 환경용 인증 방식이 아닙니다.

## ⚙️ 설정

### 백엔드 및 엔드포인트

[`app.py`](app.py)에서 설정됩니다.

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### 모델/데이터 동작

- 스레드 풀 크기는 `max_workers=64`입니다.
- 이미지 타일 기본값은 `256x256`입니다.
- Cellpose 초기화는 `model_type='nuclei'`, `gpu=True`를 사용합니다.
- 학습 및 사전학습은 WebSocket 트리거 액션을 통해 비동기로 실행됩니다.

## 🧪 예시

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

### 예시: WebSocket 수동 어노테이션 업데이트

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

## 📚 참고한 연구

Lazeal Cellist는 다음과 같은 딥러닝 최첨단 연구에서 영감을 받았습니다.

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

이 연구들은 플랫폼의 알고리즘 및 방법론 개발을 이끄는 중요한 인사이트를 제공합니다.

(참고: 정확한 인용 정보는 원문 논문을 직접 확인해 주세요.)

## 🧭 개발 노트

- 핵심 모델 클래스는 `cellist/`(`ModelD2Init`, `ModelD2Pretrain`) 아래에 있습니다.
- 메인 상호작용 UI 로직은 `templates/cellist.html`에 직접 포함되어 있습니다.
- SQL 스키마와 시드 형태 데이터는 `cellist.sql`에 있습니다.
- `notebooks/`와 `polygon_sample/`의 노트북은 탐색적 참고 자료를 제공합니다.
- 현재 저장소 루트에는 전용 자동화 테스트 스위트나 CI 설정이 없습니다.

## 🧯 문제 해결

| 증상 | 점검 사항 |
|---|---|
| `ModuleNotFoundError` 또는 import 문제 | `python app.py` 실행 전에 `conda activate cellist`가 적용되었는지 확인하세요. |
| UI는 보이지만 스타일/스크립트가 적용되지 않음 | `statics/`에서 `npm install`을 실행하고 `statics/node_modules` 존재 여부를 확인하세요. |
| MySQL access denied | `cellist/utils/constants.py`의 사용자명/비밀번호와 MySQL plugin/auth 모드를 확인하세요. |
| 앱은 시작되지만 모델 액션 실패 | CUDA/GPU 사용 가능 여부를 점검하세요. 현재 경로는 CUDA(`torch.device('cuda:0')`, Cellpose `gpu=True`)를 가정합니다. |
| 업로드는 성공하지만 타일/모델이 생성되지 않음 | `data/` 하위 디렉터리가 존재하고 쓰기 가능한지 확인하세요. |

## 🗺️ 로드맵

다음 항목은 기존 프로젝트 문서/TODO 노트에서 보존해 정리한 내용입니다.

- 폴리곤 샘플: 사각형 대신 폴리곤 어노테이션 사용.
- 매우 작은/큰 값에서 `float32` 동작을 고려해 모델 최적화.
- 가능한 범위에서 모델 크기 축소.
- Transformer/stable-diffusion 계열 컴포넌트 등 접근으로 견고성 향상.
- 베이스 모델 옵션(Threshold, Cellpose) 및 타깃 모델 옵션(AIR, Transformer, SD) 추가.
- 인터페이스 최적화(다중 선택 포함).
- 백엔드 최적화(메모리/캐시 처리 개선 포함).
- 최소한의 DB 설정으로 쉽게 패키징(예: SQLite 옵션).

## 🤝 기여

### Lazeal Cellist에 기여하기

Lazeal Cellist는 오픈소스 프로젝트이며, 경험 수준과 관계없이 모든 분의 기여를 환영합니다. 특히 다음과 같은 기여를 기대합니다.

- 알고리즘 효율성과 성능 향상
- UI/UX 개선
- 문서와 예시 확장
- 버그 수정 및 시스템 안정성 향상

기여를 시작하기 전에, 먼저 이슈를 통해 원하는 변경 사항을 논의해 주세요. 이렇게 하면 협업 조율이 쉬워지고 중복 또는 충돌 작업을 줄일 수 있습니다.

시작 방법은 기여 가이드를 참고해 주세요.

저장소 내 추가 기여 문서:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 이 저장소의 [LICENSE](https://chat.openai.com/LICENSE) 파일을 참고하세요.

저장소 상태 참고: 현재 체크아웃에는 루트 `LICENSE` 파일이 없습니다. 위 문구는 기존 README의 정식 프로젝트 의도를 보존한 것이며, 원한다면 후속 변경에서 로컬 `LICENSE` 파일을 추가하세요.
