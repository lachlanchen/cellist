[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Ваша эффективная платформа для обнаружения и профилирования клеток в 3D</strong>
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

## 🎬 Preview

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Ваша эффективная платформа для обнаружения и профилирования клеток в 3D

Добро пожаловать в Lazeal Cellist — полноценную и эффективную платформу для детектирования, сегментации и профилирования клеток на 3D-микроскопических изображениях.

Платформа предназначена для обнаружения клеток с помощью обучения без учителя, пороговой фильтрации и современных алгоритмов, таких как Cellpose. Lazeal Cellist также предоставляет интуитивный интерактивный интерфейс, позволяющий уточнять результаты детекции. Эти уточнённые результаты затем возвращаются в полуавтоматическую модель обучения, постепенно повышая её качество.

Lazeal Cellist выделяется тем, что предлагает эффективную 3D-модель, требующую минимальных усилий для обучения и донастройки, поэтому платформа практична для учёных, исследователей и энтузиастов.

> ℹ️ **Примечание по объёму проекта**
> Видение проекта и UI включают 3D-концепции (`/3d`, `templates/cellist_3d.html`), тогда как текущий основной сценарий обучения в коде в основном опирается на 2D-срезы с последующей доработкой.

---

## Table of Contents

- [Обзор](#-overview)
- [Ключевые возможности](#-key-features)
- [Структура проекта](#-project-structure)
- [Предварительные требования](#-prerequisites)
- [Установка](#-installation)
- [Использование](#-usage)
- [Настройка](#-configuration)
- [Примеры](#-examples)
- [Вдохновлено исследованиями](#-inspired-by-research)
- [Заметки по разработке](#-development-notes)
- [Устранение неполадок](#-troubleshooting)
- [План развития](#-roadmap)
- [Вклад](#-contributing)
- [Благодарности](#-acknowledgements)
- [Поддержка](#-support)
- [Лицензия](#-license)

## 🔍 Overview

Lazeal Cellist — это веб-платформа на Python/Tornado для рабочих сценариев с микроскопическими изображениями, которая поддерживает:

- Загрузку через браузер, создание модели и редактирование аннотаций.
- Инициализацию с помощью алгоритма (режим ядер Cellpose).
- Итеративную доработку в человекоориентированном цикле через действия WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Хранение в БД для моделей, срезов изображений и аннотаций.

> ℹ️ Примечание по текущему поведению: хотя видение проекта и UI включают 3D-компоненты (`/3d`, `templates/cellist_3d.html`), текущий основной процесс обучения в коде преимущественно использует 2D-разбиение + донастройку модели.

### Quick At-a-Glance

| Область | Текущая реализация |
|---|---|
| Сервер | Tornado (`app.py`) |
| Порт | `8887` |
| База данных | MySQL (`cellist.sql`) |
| Ядро ML-стека | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Инициализация инференса | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Статус упаковки | Исследовательский прототип (без `pyproject.toml`/`setup.py`) |
| Состояние tests/CI | Нет выделенного набора автоматических тестов или конфигурации CI в корне репозитория |

### Языки документации

В этом репозитории уже есть мультиязычные версии README в `i18n/`:

| Язык | Файл |
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

- **Unsupervised 3D Cell Detection**: Обнаружение клеток на 3D-микроскопических изображениях с помощью продвинутых методов машинного обучения.
- **Интерактивный интерфейс доработки результатов**: Уточнение результатов детекции через удобный и понятный интерфейс.
- **Эффективная полу‑контролируемая сеть обучения**: Постепенное улучшение качества модели на основе доработанных результатов.
- **Сегментация и профилирование клеток**: Не ограничивается только детекцией, включая расширенные возможности сегментации и профиля.

Дополнительные реализованные возможности:

- Tornado REST + WebSocket сервер (`app.py`) на порту `8887`.
- Автоматическое разбиение изображений на тайлы (`256x256` по умолчанию) для подачи в модель.
- Схема MySQL поставляется как дамп: [`cellist.sql`](cellist.sql).
- Frontend-стек включает Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Асинхронное выполнение задач модели через пул потоков (`max_workers=64`).

## 🗂️ Project Structure

```text
cellist/
├── app.py                               # Основной Tornado server + REST/WebSocket handlers
├── cellist/                             # Основной код ML/моделей
│   ├── model_init.py                    # Основной 2D-класс модели и training flow
│   ├── model_pretrain.py                # Вариант предобучения
│   ├── model_2d_components.py           # Компоненты Encoder/Decoder/SPAIR
│   ├── model_2d_utilities.py            # Метаданные модели в БД + transforms
│   ├── image_preprocessing.py           # Вспомогательные функции нарезки/склейки
│   └── utils/constants.py               # Пути во время выполнения + конфигурация MySQL
├── templates/
│   ├── cellist.html                     # Основной 2D UI
│   └── cellist_3d.html                  # Вариант/прототип 3D UI
├── statics/                             # Frontend-ресурсы и зависимости npm
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Переведённые README
├── notebooks/                           # Исследовательские ноутбуки
├── polygon_sample/                      # Эксперименты с полигональной разметкой
├── figs/                                # Брендовые материалы
├── cellist.sql                          # Схема/дамп MySQL
├── cellist.yaml                         # Описание среды Conda
├── create_data_folder.py                # Устаревший помощник для создания папок данных
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Расширенные заметки по архитектуре/TODO
└── README.md
```

## ✅ Prerequisites

| Требование | Примечания |
|---|---|
| OS | Linux рекомендуется (ниже команды предполагают поведение Linux shell). |
| Python/Conda | Conda должна быть доступна для создания окружения по [`cellist.yaml`](cellist.yaml). |
| База данных | Запущенный MySQL на `localhost` с БД `cellist`. |
| GPU | Для текущих путей выполнения рекомендуется/ожидается NVIDIA/CUDA. |
| Node.js + npm | Требуется для установки зависимостей в `statics/node_modules`. |
| Права на запись на диск | Нужны для runtime-данных в `<repo>/data`. |

## 🛠️ Installation

### 1. Clone and enter repository

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Create Python environment

Используйте файл `cellist.yaml` в репозитории:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Примечание совместимости сохранено из более ранних инструкций: ранее использовалось `celist.yaml` (без `l`), но в этом репозитории файл называется `cellist.yaml`.

Устаревшая команда (сохраняется):

```bash
conda env create -f celist.yaml
```

### 3. Install frontend dependencies

```bash
cd statics
npm install
cd ..
```

### 4. Prepare runtime data directories

Приложению нужна структура `data/` (и `.gitignore` уже исключает `data`):

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Примечание: [`create_data_folder.py`](create_data_folder.py) существует, но сейчас создаёт каталоги в текущей рабочей директории (а не внутри `data/`). Учтите это, если используете этот скрипт.

### 5. Prepare MySQL authentication (if needed)

Если root-аутентификация идёт через сокет и блокирует доступ приложения, в старой документации предлагалось перейти на password-аутентификацию:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Create database and restore schema/data

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Пример устаревшей документации (сохраняется):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configure MySQL credentials for runtime

Текущий код считывает учётные данные из [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` и `mysqlurl`).

Текущие значения по умолчанию:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Для локальной безопасности обновите их для вашей среды.

### 8. Optional environment sanity checks

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Usage

### Start the web server

```bash
python app.py
```

Устаревшая команда запуска из прошлой документации (сохраняется):

```bash
python app.py -m cellist
```

Наблюдаемые в коде маршруты сервера по умолчанию:

- Основной UI: `http://localhost:8887/`
- 3D-страница: `http://localhost:8887/3d`

### Typical workflow

1. Откройте UI и войдите в систему.
2. Загрузите микроскопические изображения из панели Create Model.
3. Выберите базовый алгоритм (`Cellpose`) и создайте модель.
4. Позвольте backend нарезать изображения и инициализировать детекции.
5. Загрузите обрезанные изображения, проверьте/отредактируйте прямоугольные аннотации.
6. Запускайте циклы `initialize`, `pretrain` и `train`.
7. Используйте `Pretrain Stop` / `Stop` (`train-stop`) / `reset` по мере необходимости.
8. Сохраняйте ручные доработки через действия `Update Model`/аннотаций.

### Built-in UI login credentials (current template behavior)

Frontend сейчас проверяет следующие статические учётные данные на стороне клиента:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Это поведение прототипа и не является production-аутентификацией.

### API/Socket surface currently used by the UI

HTTP endpoints:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket endpoint:

- `ws://localhost:8887/websocket/<ws_uuid>`

Распознанные `data_type`-сообщения в обработчике WebSocket:

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

Настроено в [`app.py`](app.py):

- Port: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Model/data behavior

- Размер пула потоков `max_workers=64`.
- По умолчанию изображения режутся на тайлы `256x256`.
- Инициализация Cellpose использует `model_type='nuclei'` и `gpu=True`.
- Обучение и предобучение выполняются асинхронно через действия, запускаемые WebSocket.
- Корень данных берётся из текущей рабочей директории как `<repo>/data`.

### Database/runtime constants

Из [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` содержит ключи host/user/password
- `mysqlurl` целится в базу `cellist`

### Frontend dependency snapshot

Из [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda environment highlights

Из [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
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

Lazeal Cellist опирается на передовые исследования в глубоком обучении, включая:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Эти работы дают полезные идеи, которые повлияли на построение алгоритмов и методологий платформы.

(Примечание: для корректных цитат обращайтесь к оригинальным статьям.)

## 🧭 Development Notes

- Основные классы моделей находятся в `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Основная логика интерактивного UI встроена прямо в `templates/cellist.html`.
- SQL-схема и seed-подобные данные находятся в `cellist.sql`.
- Ноутбуки в `notebooks/` и `polygon_sample/` дают исследовательские справочные материалы.
- Расширенные notes по платформе и моделям находятся в [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- В корне репозитория пока нет отдельного набора автоматических тестов или конфигурации CI.

### Assumptions and current constraints

- Репозиторий ориентирован прежде всего на локальное исследовательское использование.
- Некоторые ветки кода предполагают доступность GPU (`cuda:0`).
- Аутентификация и управление секретами находятся на уровне прототипа.
- 3D-интерфейсы присутствуют, но доминирующий рабочий процесс обучения остаётся ориентированным на 2D-срезы.

## 🧯 Troubleshooting

| Симптом | Рекомендуемые проверки |
|---|---|
| `ModuleNotFoundError` или проблемы с импортами | Проверьте, что перед запуском `python app.py` выполнен `conda activate cellist`. |
| UI рендерится без стилей/скриптов | Запустите `npm install` в `statics/` и убедитесь, что существует `statics/node_modules`. |
| Доступ к MySQL отклонён | Проверьте имя пользователя/пароль в `cellist/utils/constants.py` и режим плагина/аутентификации MySQL. |
| Приложение запускается, но действия модели падают | Проверьте доступность CUDA/GPU; текущие пути предполагают CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Загрузка проходит, но тайлы/модели не появляются | Проверьте, что подкаталоги `data/` существуют и доступны для записи. |
| Ошибки REST/WebSocket-запросов | Убедитесь, что сервер запущен на `http://localhost:8887`, а ключи payload соответствуют текущим именам шаблонов. |
| `FileNotFoundError` в `data/` | Запускайте приложение из корня репозитория, чтобы относительные пути разрешались стабильно. |

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

Ниже перечислены пункты, сохранённые и упорядоченные из текущей проектной документации/TODO:

- Polygon sample: использовать полигональную разметку вместо прямоугольной.
- Оптимизировать поведение модели для `float32` с очень малыми/очень большими значениями.
- Снизить размер модели там, где возможно.
- Повысить устойчивость с подходами, вдохновлёнными Transformer/stable-diffusion.
- Добавить базовые модели (Threshold, Cellpose) и целевые модели (AIR, Transformer, SD).
- Оптимизация интерфейса (включая множественный выбор).
- Оптимизация backend (включая улучшенную работу с памятью и кэшем).
- Удобная упаковка с минимальной конфигурацией БД (например, вариант SQLite).

## 🤝 Contributing

### Contribute to Lazeal Cellist

Lazeal Cellist — проект с открытым исходным кодом, и мы приветствуем вклад на любом уровне. Мы принимаем вклад, который:

- Повышает эффективность и производительность алгоритмов
- Улучшает интерфейс и пользовательский опыт
- Расширяет документацию и примеры
- Исправляет ошибки и повышает стабильность системы

Перед началом изменений обсудите ваши планы в issue. Это помогает координировать усилия и избегать дублирования или конфликтов.

Подробнее о том, как начать, см. правила для участников.

Дополнительные документы репозитория:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Концепция и реализация Lazeal Cellist во многом опираются на направление исследований AIR/SPAIR, упомянутое выше.
- Репозиторий содержит историческую/устаревшую документацию и команды, сохранённые для непрерывности с прошлым использованием проекта.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

This project is licensed under the MIT License. For more information, please refer to the [LICENSE](LICENSE) file in this repository.

Repository status note: in this checkout there is no root `LICENSE` file at the moment. The line above is preserved from the previous README as canonical project intent; add a local `LICENSE` file in a follow-up change if needed.
