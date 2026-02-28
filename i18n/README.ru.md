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

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Ваша эффективная платформа для детекции и профилирования клеток в 3D

Добро пожаловать в Lazeal Cellist — удобную и эффективную платформу для детекции, сегментации и профилирования клеток на 3D-микроскопических изображениях.

Наша платформа предназначена для определения клеток с помощью обучения без учителя, методов пороговой обработки и современных алгоритмов, включая Cellpose. Lazeal Cellist также предоставляет интуитивно понятный интерактивный интерфейс, который позволяет пользователю уточнять результаты детекции. Эти уточнённые результаты затем возвращаются в полу-supervised обучение и непрерывно улучшают качество модели.

Lazeal Cellist выделяется за счёт эффективной 3D-модели, требующей минимальных усилий для обучения и донастройки, что делает её практичным инструментом для учёных, исследователей и энтузиастов.

> ℹ️ **Примечание по охвату**
> Видение проекта и UI включают 3D-концепции (`/3d`, `templates/cellist_3d.html`), в то время как текущий основной тренировочный поток в коде в основном строится на 2D-срезах и донастройке.

---

## Содержание

- [Обзор](#-overview)
- [Ключевые возможности](#-key-features)
- [Структура проекта](#-project-structure)
- [Требования](#-prerequisites)
- [Установка](#-installation)
- [Использование](#-usage)
- [Конфигурация](#-configuration)
- [Примеры](#-examples)
- [Вдохновлено исследованиями](#-inspired-by-research)
- [Заметки по разработке](#-development-notes)
- [Устранение неполадок](#-troubleshooting)
- [План развития](#-roadmap)
- [Участие](#-contributing)
- [Благодарности](#-acknowledgements)
- [Поддержка](#-support)
- [Лицензия](#-license)

## 🔍 Overview

Lazeal Cellist — это веб-платформа на Python/Tornado для рабочих процессов с микроскопическими изображениями, которая поддерживает:

- Загрузку из браузера, создание модели и редактирование аннотаций.
- Инициализацию на основе алгоритма (режим ядер Cellpose).
- Итеративную корректировку человеком через действия WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Персистентное хранение в БД для моделей, срезов изображений и аннотаций.

> ℹ️ Примечание по текущему поведению: хотя концепция проекта и UI включают 3D-элементы (`/3d`, `templates/cellist_3d.html`), текущий основной поток обучения в коде в основном выполняется как 2D-слайсинг + донастройка модели.

### Quick At-a-Glance

| Область | Текущая реализация |
|---|---|
| Сервер | Tornado (`app.py`) |
| Порт | `8887` |
| База данных | MySQL (`cellist.sql`) |
| Базовый ML-стек | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Инициализация инференса | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Состояние упаковки | Исследовательский прототип (без `pyproject.toml`/`setup.py`) |
| Состояние тестов/CI | Нет выделенного набора автоматических тестов или конфигурации CI в корне репозитория |

### Документационные языки

В этом репозитории уже есть многоязычные версии README в `i18n/`:

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

- **Unsupervised 3D Cell Detection**: Определение клеток на 3D-микроскопических изображениях с использованием современных методов машинного обучения.
- **Интерактивный интерфейс уточнения результатов**: Уточняйте результаты детекции с помощью удобного и понятного интерфейса.
- **Эффективная полу-supervised обучающая сеть**: Постепенно повышает производительность модели за счёт уточнённых результатов.
- **Сегментация и профилирование клеток**: Выход за пределы простой детекции благодаря продвинутым возможностям сегментации и профиля.

Дополнительные возможности текущей реализации:

- Tornado REST + WebSocket сервер (`app.py`) на порту `8887`.
- Автоматическое разбиение изображений на тайлы (`256x256` по умолчанию) для подачи в модель.
- Схема MySQL включена в виде дампа: [`cellist.sql`](cellist.sql).
- В frontend входят Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Асинхронная обработка задач модели через пул потоков (`max_workers=64`).

## 🗂️ Project Structure

```text
cellist/
├── app.py                               # Основной сервер Tornado + обработчики REST/WebSocket
├── cellist/                             # Базовый ML/модульный код
│   ├── model_init.py                    # Основной 2D-класс модели и поток обучения
│   ├── model_pretrain.py                # Вариант предварительного обучения
│   ├── model_2d_components.py           # Компоненты encoder/decoder/SPAIR
│   ├── model_2d_utilities.py            # Метаданные модели и преобразования поверх БД
│   ├── image_preprocessing.py           # Вспомогательные функции нарезки/склейки
│   └── utils/constants.py               # Пути выполнения + конфигурация MySQL
├── templates/
│   ├── cellist.html                     # Основной 2D UI
│   └── cellist_3d.html                  # Вариант/прототип 3D-интерфейса
├── statics/                             # Frontend-ресурсы и npm-зависимости
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Переводы README
├── notebooks/                           # Исследовательские тетради
├── polygon_sample/                      # Эксперименты с полигональной разметкой
├── figs/                                # Брендовые изображения
├── cellist.sql                          # Схема/дамп данных MySQL
├── cellist.yaml                         # Спецификация окружения Conda
├── create_data_folder.py                # Устаревший помощник для создания папок данных
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Расширенные заметки по архитектуре/TODO
└── README.md
```

## ✅ Prerequisites

| Требование | Примечания |
|---|---|
| ОС | Рекомендуется Linux (команды ниже предполагают поведение командной строки Linux). |
| Python/Conda | Conda должна быть доступна для создания окружения из [`cellist.yaml`](cellist.yaml). |
| База данных | Запущенный сервер MySQL на `localhost` с базой `cellist`. |
| GPU | Для текущих путей в коде настоятельно рекомендуется/ожидается окружение NVIDIA/CUDA. |
| Node.js + npm | Нужно для установки зависимостей frontend в `statics/node_modules`. |
| Доступ на запись в диск | Нужен для данных выполнения в `<repo>/data`. |

## 🛠️ Installation

### 1. Clone and enter repository

```bash

git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Create Python environment

Используйте файл в репозитории `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Примечание по совместимости из старой документации: ранее использовалось `celist.yaml` (без `l`), но файл в этом репозитории — `cellist.yaml`.

Старый вариант команды (сохранён):

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

Приложение ожидает структуру `data/` (и `.gitignore` уже исключает `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Примечание: [`create_data_folder.py`](create_data_folder.py) существует, но в текущей версии создаёт каталоги в текущей рабочей директории (не под `data/`). Учитывайте это, если используете его.

### 5. Prepare MySQL authentication (if needed)

Если аутентификация root работает через сокет и блокирует доступ приложения, старая документация проекта рекомендует перейти на парольную аутентификацию:

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

Пример из устаревшей документации (сохранён):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configure MySQL credentials for runtime

Текущий код читает учётные данные из [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` и `mysqlurl`).

Значения по умолчанию в текущем коде:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Для локальной безопасности обновите их перед запуском в вашей среде.

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

Устаревшая команда запуска из предыдущих инструкций (сохранена):

```bash
python app.py -m cellist
```

Наблюдаемые по коду маршруты сервера по умолчанию:

- Основной UI: `http://localhost:8887/`
- 3D-страница: `http://localhost:8887/3d`

### Typical workflow

1. Откройте UI и выполните вход.
2. Загрузите микроскопические изображения из панели Create Model.
3. Выберите базовый алгоритм (`Cellpose`) и создайте модель.
4. Дайте бэкенду разбить изображения и инициализировать детекции.
5. Загрузите обрезанные изображения, проверьте и поправьте прямоугольные аннотации.
6. Запускайте циклы `initialize`, `pretrain` и `train`.
7. Используйте `Pretrain Stop` / `Stop` (`train-stop`) / `reset` по необходимости.
8. Сохраняйте ручные доработки через действия `Update Model`/аннотаций.

### Built-in UI login credentials (current template behavior)

Фронтенд сейчас проверяет следующие статические учётные данные на стороне клиента:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Это прототипное поведение и не является production-аутентификацией.

### API/Socket surface currently used by the UI

HTTP endpoints:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket endpoint:

- `ws://localhost:8887/websocket/<ws_uuid>`

Распознанные сообщения `data_type` в обработчике WebSocket:

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

- Порт: `8887`
- Маршруты:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Model/data behavior

- Размер пула потоков равен `max_workers=64`.
- Тайлы изображений по умолчанию `256x256`.
- Инициализация Cellpose использует `model_type='nuclei'` и `gpu=True`.
- Обучение и предобучение выполняются асинхронно через действия, инициируемые WebSocket.
- Корень данных берётся из текущей рабочей директории как `<repo>/data`.

### Database/runtime constants

Из [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` включает ключи host/user/password
- `mysqlurl` указывает на базу `cellist`

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

Lazeal Cellist вдохновлен новейшими исследованиями в области deep learning, включая:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Эти работы дают ценные идеи, которые повлияли на разработку алгоритмов и методологий нашей платформы.

(Примечание: для точного цитирования обратитесь к оригинальным статьям.)

## 🧭 Development Notes

- Базовые классы модели находятся в `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Основная логика интерактивного UI встроена непосредственно в `templates/cellist.html`.
- SQL-схема и seed-подобные данные содержатся в `cellist.sql`.
- Ноутбуки в `notebooks/` и `polygon_sample/` дают исследовательские справки.
- Расширенные заметки по платформе/модели находятся в [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- В текущем репозитории пока нет выделенного набора автоматических тестов или конфигурации CI в корне.

### Assumptions and current constraints

- Репозиторий ориентирован в первую очередь на локальное исследовательское использование.
- Некоторые ветки кода предполагают доступность GPU (`cuda:0`).
- Аутентификация и управление секретами находятся на уровне прототипа.
- 3D-интерфейсы присутствуют, но доминирующий тренировочный workflow остаётся 2D-ориентированным по тайлам.

## 🧯 Troubleshooting

| Симптом | Рекомендуемые проверки |
|---|---|
| `ModuleNotFoundError` или ошибки импорта | Убедитесь, что перед запуском `python app.py` выполнено `conda activate cellist`. |
| UI рендерится без стилей/скриптов | Запустите `npm install` в `statics/` и убедитесь, что существует `statics/node_modules`. |
| Отказ MySQL в доступе | Проверьте имя пользователя/пароль в `cellist/utils/constants.py` и режим plugin/auth в MySQL. |
| Приложение запускается, но действия модели падают | Проверьте доступность CUDA/GPU; текущие пути предполагают CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Загрузка проходит, но тайлы/модели не появляются | Убедитесь, что подкаталоги `data/` существуют и доступны для записи. |
| Ошибки REST/WebSocket-запросов | Убедитесь, что сервер запущен на `http://localhost:8887` и ключи запроса совпадают с текущими именами шаблонов. |
| `FileNotFoundError` в `data/` | Запускайте приложение из корня репозитория, чтобы относительные пути были корректны. |

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

Следующие пункты сохранены и структурированы из существующей проектной документации/заметок TODO:

- Polygon sample: использовать полигон вместо прямоугольной разметки.
- Оптимизация модели для поведения `float32` с очень маленькими/очень большими значениями.
- Уменьшить размер модели, где это возможно.
- Повысить устойчивость с подходами, такими как компоненты Transformer/stable-diffusion.
- Добавить базовые модели (Threshold, Cellpose) и целевые модели (AIR, Transformer, SD).
- Оптимизация интерфейса (включая множественный выбор).
- Оптимизация backend (включая лучшее управление памятью/кэшем).
- Простая упаковка с минимальной конфигурацией БД (например, вариант SQLite).

## 🤝 Contributing

### Contribute to Lazeal Cellist

Lazeal Cellist — проект с открытым исходным кодом, и мы приветствуем вклад любого уровня. Мы приглашаем вас вносить вклад, который:

- Повышает эффективность и производительность алгоритмов
- Улучшает пользовательский интерфейс и опыт работы
- Расширяет документацию и примеры
- Исправляет ошибки и повышает устойчивость системы

Перед началом внесения изменений, пожалуйста, обсудите их в issue. Это помогает скоординировать работу и избежать дублирования или конфликтов.

Дополнительные инструкции по началу работы см. в документах по внесению вклада.

Дополнительные документы репозитория:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Концепция и реализация Lazeal Cellist во многом опираются на линию исследований AIR/SPAIR, упомянутую выше.
- Репозиторий содержит исторические и устаревшие документы и команды, сохранённые для совместимости с прошлой практикой использования проекта.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

This project is licensed under the MIT License. For more information, please refer to the [LICENSE](LICENSE) file in this repository.
