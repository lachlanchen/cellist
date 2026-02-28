[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Ваша эффективная платформа для обнаружения и профилирования клеток в 3D</strong>
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

## 🎬 Предварительный просмотр

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Ваша эффективная платформа для обнаружения и профилирования клеток в 3D

Добро пожаловать в Lazeal Cellist — всестороннюю и производительную платформу для детекции, сегментации и профилирования клеток на 3D‑микроскопических изображениях.

Платформа спроектирована для поиска клеток с использованием обучения без разметки, пороговых техник и современных алгоритмов, включая Cellpose. Lazeal Cellist также предоставляет интуитивный интерактивный интерфейс, позволяющий дорабатывать результаты детекции вручную. Эти уточнённые результаты возвращаются в полу‑наблюдаемую сеть обучения и постепенно улучшают качество модели.

Lazeal Cellist выделяется тем, что предлагает эффективную 3D‑модель, требующую минимальных усилий для обучения и донастройки, поэтому платформа подходит для учёных, исследователей и энтузиастов.

> ℹ️ **Примечание по области применения**
> Видение проекта и интерфейс включают 3D‑концепты (`/3d`, `templates/cellist_3d.html`), тогда как текущий основной процесс обучения в коде преимущественно опирается на 2D‑срезы + уточнение.

---

## Содержание

- [Обзор](#-overview)
- [Ключевые возможности](#-key-features)
- [Структура проекта](#-project-structure)
- [Требования](#-prerequisites)
- [Установка](#-installation)
- [Использование](#-usage)
- [Настройка](#-configuration)
- [Примеры](#-examples)
- [Исследовательские источники](#-inspired-by-research)
- [Заметки по разработке](#-development-notes)
- [Устранение неполадок](#-troubleshooting)
- [План развития](#-roadmap)
- [Вклад в проект](#-contributing)
- [Благодарности](#-acknowledgements)
- [Поддержка](#-support)
- [Лицензия](#-license)

<a id="-overview"></a>
## 🔍 Обзор

Lazeal Cellist — это веб‑платформа на Python/Tornado для рабочих процессов с микроскопическими изображениями, которая предоставляет:

- Загрузку через браузер, создание модели и редактирование аннотаций.
- Инициализацию с помощью алгоритма (режим ядер Cellpose).
- Итеративное уточнение в цикле человек-в-петле через действия WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Постоянное хранение в базе данных моделей, срезов изображений и аннотаций.

> ℹ️ Текущее поведение: хотя видение проекта и интерфейс включают 3D-компоненты (`/3d`, `templates/cellist_3d.html`), текущий основной поток обучения в коде всё ещё в основном 2D‑слайсинг + уточнение модели.

### Быстрый обзор

| Область | Текущая реализация |
|---|---|
| Сервер | Tornado (`app.py`) |
| Порт | `8887` |
| База данных | MySQL (`cellist.sql`) |
| Основной ML-стек | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Инициализация инференса | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Статус упаковки | Исследовательский прототип (без `pyproject.toml`/`setup.py`) |
| Статус тестов/CI | В репозитории корня нет отдельного набора автоматических тестов и конфигурации CI |

### Языки документации

В этом репозитории уже есть мультиязычные README-файлы в `i18n/`:

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

<a id="-key-features"></a>
## ✨ Ключевые возможности

- **Unsupervised 3D Cell Detection**: Выявление клеток на 3D‑микроскопических снимках с помощью современных методов машинного обучения.
- **Интерактивный интерфейс доработки результатов**: Уточняйте результаты обнаружения в удобном и понятном интерфейсе.
- **Эффективная полу-наблюдаемая обучающая сеть**: Постепенно повышайте качество модели на основе уточнённых результатов.
- **Сегментация и профилирование клеток**: Возможности выходят за рамки детекции за счёт расширенной сегментации и профилирования.

Дополнительные реализованные возможности:

- REST + WebSocket сервер Tornado (`app.py`) на порту `8887`.
- Автоматическая разрезка изображений на тайлы (`256x256` по умолчанию) для подачи в модель.
- Схема MySQL включена как дамп: [`cellist.sql`](cellist.sql).
- Frontend-стек: Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Асинхронные задачи модели через пул потоков (`max_workers=64`).

<a id="-project-structure"></a>
## 🗂️ Структура проекта

```text
cellist/
├── app.py                               # Основной сервер Tornado + обработчики REST/WebSocket
├── cellist/                             # Основной код ML/моделей
│   ├── model_init.py                    # Основной 2D класс модели и поток обучения
│   ├── model_pretrain.py                # Вариант предварительного обучения
│   ├── model_2d_components.py           # Компоненты Encoder/Decoder/SPAIR
│   ├── model_2d_utilities.py            # Метаданные модели в БД + преобразования
│   ├── image_preprocessing.py           # Утилиты разрезки/склейки
│   └── utils/constants.py               # Пути runtime + конфигурация MySQL
├── templates/
│   ├── cellist.html                     # Основной 2D-интерфейс
│   └── cellist_3d.html                  # Вариант/прототип 3D-интерфейса
├── statics/                             # Frontend-ресурсы и зависимости npm
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Переведённые файлы README
├── notebooks/                           # Исследовательские ноутбуки
├── polygon_sample/                      # Эксперименты с полигональной разметкой
├── figs/                                # Брендовые материалы
├── cellist.sql                          # Схема и дамп MySQL
├── cellist.yaml                         # Спецификация окружения Conda
├── create_data_folder.py                # Устаревший помощник создания папок данных
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Расширенные заметки по архитектуре/TODO
└── README.md
```

<a id="-prerequisites"></a>
## ✅ Предварительные требования

| Требование | Примечания |
|---|---|
| ОС | Рекомендуется Linux (ниже перечисленные команды предполагают поведение оболочки Linux). |
| Python/Conda | Conda должна быть установлена для создания окружения через [`cellist.yaml`](cellist.yaml). |
| База данных | Сервер MySQL на `localhost` с базой данных `cellist`. |
| GPU | Текущие пути выполнения ожидают среду NVIDIA/CUDA. |
| Node.js + npm | Нужны для установки зависимостей `statics/node_modules`. |
| Доступ на запись к диску | Нужен для runtime-данных в `<repo>/data`. |

<a id="-installation"></a>
## 🛠️ Установка

### 1. Клонируйте репозиторий и зайдите в него

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Создайте Python-окружение

Используйте файл репозитория `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Сохранили совместимость с историческими инструкциями: раньше документация использовала `celist.yaml` (без `l`), но в текущем репозитории файл называется `cellist.yaml`.

Альтернативная устаревшая команда (сохранена):

```bash
conda env create -f celist.yaml
```

### 3. Установите frontend-зависимости

```bash
cd statics
npm install
cd ..
```

### 4. Подготовьте каталоги runtime-данных

Приложению нужен каталог `data/` (и `.gitignore` уже исключает `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Примечание: [`create_data_folder.py`](create_data_folder.py) существует, но сейчас создаёт директории в текущей рабочей папке (не внутри `data/`). Учтите это, если используете скрипт.

### 5. Подготовьте аутентификацию MySQL (при необходимости)

Если корневая аутентификация через сокет блокирует доступ приложения, в старой документации предлагается перейти на парольную схему:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Создайте базу и восстановите схему/данные

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Устаревший пример из документации (сохранён):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Настройте учетные данные MySQL для runtime

Текущий код читает параметры из [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` и `mysqlurl`).

Текущие значения по умолчанию:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Для локальной безопасности обновите эти значения под своё окружение.

### 8. Опциональные проверки окружения

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

<a id="-usage"></a>
## 🚀 Использование

### Запуск веб-сервера

```bash
python app.py
```

Устаревшая команда запуска из ранних заметок (сохранена):

```bash
python app.py -m cellist
```

Текущие маршруты сервера, которые наблюдаются в коде:

- Основной интерфейс: `http://localhost:8887/`
- 3D-страница: `http://localhost:8887/3d`

### Типовой рабочий процесс

1. Откройте интерфейс и войдите в систему.
2. Загрузите микроскопические изображения из панели создания модели.
3. Выберите базовый алгоритм (`Cellpose`) и создайте модель.
4. Дайте backend разрезать изображения и инициализировать детекции.
5. Загрузите обрезанные изображения, проверьте и скорректируйте прямоугольные аннотации.
6. Запускайте циклы `initialize`, `pretrain` и `train`.
7. При необходимости используйте `Pretrain Stop` / `Stop` (`train-stop`) / `reset`.
8. Сохраняйте ручные доработки через действия `Update Model`/операции аннотаций.

### Встроенные учетные данные UI (поведение текущего шаблона)

Фронтенд сейчас проверяет следующие статические учетные данные на стороне клиента:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Это поведение прототипа и не относится к production-аутентификации.

### API/WebSocket поверхность, которую использует UI

HTTP-маршруты:

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

<a id="-configuration"></a>
## ⚙️ Конфигурация

### Backend и endpoints

Настройка задаётся в [`app.py`](app.py):

- Порт: `8887`
- Маршруты:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Поведение модели/данных

- Размер пула потоков — `max_workers=64`.
- По умолчанию изображения режутся на тайлы `256x256`.
- Инициализация Cellpose использует `model_type='nuclei'` и `gpu=True`.
- Обучение и предварительное обучение выполняются асинхронно через действия, инициируемые WebSocket.
- Корень данных определяется относительно текущей рабочей директории как `<repo>/data`.

### Константы базы и runtime

Из [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` включает ключи host/user/password
- `mysqlurl` указывает на базу `cellist`

### Снимок зависимостей frontend

Из [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Конфигурация Conda

Из [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

<a id="-examples"></a>
## 🧪 Примеры

### Пример: форма WebSocket-сообщения `create`

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

### Пример: ручное обновление аннотации через WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Пример: запрос загрузки модели

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Пример: минимальный локальный end-to-end запуск

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

<a id="-inspired-by-research"></a>
## 📚 Вдохновлено исследованиями

Lazeal Cellist опирается на новейшие исследования в глубокому обучении, включая:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Эти работы дали ценную информацию, которая повлияла на алгоритмы и методологию разработки платформы.

(Примечание: для корректных ссылок лучше обращаться к оригинальным статьям.)

<a id="-development-notes"></a>
## 🧭 Заметки по разработке

- Основные классы моделей находятся в `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Основная логика интерактивного UI встроена прямо в `templates/cellist.html`.
- SQL-схема и seed-подобные данные находятся в `cellist.sql`.
- Ноутбуки в `notebooks/` и `polygon_sample/` дают исследовательские ориентиры.
- Расширенные заметки по платформе и моделям находятся в [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- В корне репозитория пока нет отдельного набора автоматических тестов или конфигурации CI.

### Предпосылки и текущие ограничения

- Репозиторий ориентирован прежде всего на локальное исследовательское использование.
- Некоторые пути в коде предполагают доступность GPU (`cuda:0`).
- Аутентификация и управление секретами находятся на уровне прототипа.
- 3D-интерфейсы существуют, но основной рабочий процесс обучения остаётся ориентированным на 2D-срезы.

<a id="-troubleshooting"></a>
## 🧯 Устранение неполадок

| Симптом | Рекомендуемые проверки |
|---|---|
| `ModuleNotFoundError` или проблемы с импортом | Убедитесь, что перед запуском `python app.py` выполнен `conda activate cellist`. |
| UI отображается без стилей/скриптов | Запустите `npm install` внутри `statics/` и проверьте, что есть `statics/node_modules`. |
| Доступ к MySQL отклонён | Проверьте имя пользователя/пароль в `cellist/utils/constants.py` и режим плагина/аутентификации MySQL. |
| Приложение стартует, но действия модели падают | Проверьте доступность CUDA/GPU; текущие пути ожидают CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Загрузка проходит, но тайлы/модели не появляются | Проверьте, что подкаталоги `data/` существуют и доступны для записи. |
| Ошибки REST/WebSocket-запросов | Убедитесь, что сервер запущен на `http://localhost:8887` и ключи payload совпадают с текущими именами шаблонов. |
| `FileNotFoundError` в `data/` | Запускайте приложение из корня репозитория, чтобы относительные пути стабильно разрешались. |

### Быстрая диагностика

```bash
# Проверить Python-окружение и ключевые импорты
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Проверить, что порт сервера открыт после старта
ss -ltnp | rg 8887

# Проверить подключение к MySQL
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

<a id="-roadmap"></a>
## 🛣️ План развития

Ниже перечислены пункты, сохранённые и сгруппированные по существующей документации/TODO:

- Polygon sample: использовать полигональную разметку вместо прямоугольной.
- Оптимизация модели для поведения `float32` с очень малыми/очень большими значениями.
- Уменьшить размер модели там, где это возможно.
- Повысить устойчивость с подходами, похожими на Transformer/stable-diffusion.
- Добавить варианты базовых моделей (Threshold, Cellpose) и целевых моделей (AIR, Transformer, SD).
- Оптимизация интерфейса (включая множественный выбор).
- Оптимизация backend (включая лучшее управление памятью и кэшем).
- Удобная упаковка с минимальной конфигурацией БД (например, вариант SQLite).

<a id="-contributing"></a>
## 🤝 Участие в проекте

### Внести вклад в Lazeal Cellist

Lazeal Cellist — open-source проект, и мы приветствуем вклад на всех уровнях опыта. Приглашаем изменения, которые:

- Повышают эффективность и производительность алгоритмов
- Улучшают пользовательский интерфейс и UX
- Расширяют документацию и примеры
- Исправляют ошибки и повышают стабильность системы

Перед тем как начать, обсудите запланированные изменения в issue. Это помогает координировать работу и избегать дублирования или конфликтов.

Дополнительная информация о старте в репозитории:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

<a id="-acknowledgements"></a>
## 🙏 Благодарности

- Концепция и реализация Lazeal Cellist во многом опираются на исследования в направлении AIR/SPAIR, перечисленное выше.
- Репозиторий содержит историческую/архивную документацию и команды, намеренно сохранённые для преемственности с прошлой эксплуатацией проекта.

<a id="-support"></a>
## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

Этот проект распространяется под лицензией MIT. Для подробностей см. файл [LICENSE](LICENSE) в репозитории.

Примечание по состоянию репозитория: в текущем checkout отсутствует файл `LICENSE` в корне. Строка выше сохранена из исходной документации как отражение намерения проекта; добавьте локальный `LICENSE` в следующем шаге при необходимости.
