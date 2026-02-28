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

## Lazeal Cellist: Ваша эффективная платформа для 3D-детекции и профилирования клеток

Добро пожаловать в Lazeal Cellist — комплексную и эффективную платформу для детекции, сегментации и профилирования клеток на 3D-микроскопических изображениях.

Наша платформа предназначена для обнаружения клеток с использованием обучения без учителя, методов пороговой обработки и современных алгоритмов, таких как Cellpose. Lazeal Cellist также предоставляет интуитивный интерактивный интерфейс, который позволяет пользователям уточнять результаты детекции. Эти уточненные результаты затем возвращаются в полу-контролируемую обучающую сеть, что непрерывно повышает производительность модели.

Lazeal Cellist выделяется тем, что предлагает эффективную 3D-модель, требующую минимальных усилий для обучения и донастройки, что делает платформу практичным решением для ученых, исследователей и энтузиастов.

---

## 🔍 Обзор

Lazeal Cellist — это веб-платформа на Python/Tornado для рабочих процессов с микроскопическими изображениями, включающая:

- Загрузку через браузер, создание моделей и редактирование аннотаций.
- Инициализацию с помощью алгоритмов (режим ядер в Cellpose).
- Итеративное улучшение с участием человека через действия WebSocket (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Сохранение моделей, срезов изображений и аннотаций в базе данных.

Примечание о текущем поведении: хотя концепция проекта и интерфейс включают 3D-элементы (`/3d`, `templates/cellist_3d.html`), основной поток обучения в коде сейчас в основном построен вокруг 2D-срезов и донастройки модели.

### Кратко

| Область | Текущая реализация |
|---|---|
| Сервер | Tornado (`app.py`) |
| Порт | `8887` |
| База данных | MySQL (`cellist.sql`) |
| Основной ML-стек | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Инициализация инференса | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Ключевые возможности

- **Неконтролируемая 3D-детекция клеток**: обнаружение клеток на 3D-микроскопических изображениях с помощью современных методов машинного обучения.
- **Интерактивный интерфейс уточнения результатов**: улучшение результатов детекции через интуитивный и удобный интерфейс.
- **Эффективная полу-контролируемая обучающая сеть**: постепенное повышение качества модели на основе уточненных результатов.
- **Сегментация и профилирование клеток**: расширенные возможности анализа, выходящие за рамки простой детекции.

Дополнительные особенности текущей реализации:

- Tornado REST + WebSocket сервер (`app.py`) на порту `8887`.
- Автоматическое разбиение изображений на тайлы (`256x256` по умолчанию) для подачи в модель.
- Схема MySQL включена в виде дампа: [`cellist.sql`](cellist.sql).
- Frontend-стек включает Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Структура проекта

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

## ✅ Предварительные требования

| Требование | Примечания |
|---|---|
| ОС | Рекомендуется Linux (команды ниже предполагают поведение Linux shell). |
| Python/Conda | Доступна Conda для создания окружения из [`cellist.yaml`](cellist.yaml). |
| База данных | Запущенный MySQL-сервер на `localhost` с базой `cellist`. |
| GPU | По текущим путям кода настоятельно рекомендуется/ожидается окружение NVIDIA/CUDA. |
| Node.js + npm | Требуется для установки frontend-зависимостей в `statics/node_modules`. |

## 🛠️ Установка

### 1. Клонируйте репозиторий и перейдите в него

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Создайте Python-окружение

Используйте имя файла из репозитория `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Примечание о совместимости из старой документации: ранее использовалось `celist.yaml` (без одной `l`), но в этом репозитории файл называется `cellist.yaml`.

### 3. Установите frontend-зависимости

```bash
cd statics
npm install
cd ..
```

### 4. Подготовьте аутентификацию MySQL (если нужно)

Если аутентификация root основана на socket и блокирует доступ приложения, старые документы проекта предлагают переключиться на аутентификацию по паролю:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Создайте базу данных и восстановите схему/данные

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Пример из устаревшей документации (сохранен):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. Настройте учетные данные MySQL для запуска

Текущий код читает учетные данные из [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` и `mysqlurl`).

В коде по умолчанию сейчас указаны:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Для локальной безопасности обновите эти значения перед запуском в вашем окружении.

### 7. Подготовьте рабочие каталоги данных

Приложение ожидает дерево `data/` (и `.gitignore` уже исключает `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Примечание: [`create_data_folder.py`](create_data_folder.py) существует, но сейчас создает каталоги в текущем рабочем каталоге (не внутри `data/`). Учитывайте это, если будете его использовать.

## 🚀 Использование

### Запуск веб-сервера

```bash
python app.py
```

Устаревшая команда запуска из предыдущей документации (сохранена):

```bash
python app.py -m cellist
```

Маршруты сервера по умолчанию, наблюдаемые в коде:

- Основной UI: `http://localhost:8887/`
- 3D-страница: `http://localhost:8887/3d`

### Типовой рабочий процесс

1. Откройте UI и выполните вход.
2. Загрузите микроскопические изображения из панели Create Model.
3. Выберите базовый алгоритм (`Cellpose`) и создайте модель.
4. Дайте backend разрезать изображения и выполнить инициализацию детекций.
5. Загрузите обрезанные изображения, проверьте/скорректируйте прямоугольные аннотации.
6. Запустите циклы `initialize`, `pretrain` и `train`.
7. Сохраните ручные изменения через действия аннотаций/`Update Model`.

### Встроенные учетные данные входа в UI (текущее поведение шаблонов)

Сейчас frontend проверяет эти статические учетные данные на стороне клиента:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Это поведение прототипа, а не production-аутентификация.

## ⚙️ Конфигурация

### Backend и endpoints

Настраиваются в [`app.py`](app.py):

- Порт: `8887`
- Маршруты:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Поведение модели/данных

- Размер пула потоков: `max_workers=64`.
- Размер тайлов изображения по умолчанию: `256x256`.
- Инициализация Cellpose использует `model_type='nuclei'` и `gpu=True`.
- Обучение и предобучение выполняются асинхронно через действия, запускаемые по WebSocket.

## 🧪 Примеры

### Пример: формат WebSocket-сообщения для создания

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

### Пример: ручное обновление аннотаций по WebSocket

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

## 📚 Вдохновлено исследованиями

Lazeal Cellist вдохновлен передовыми исследованиями в области глубокого обучения, включая:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Эти работы дали ценные идеи, которые направляли разработку алгоритмов и методологий нашей платформы.

(Примечание: для корректного цитирования обращайтесь напрямую к оригинальным статьям.)

## 🧭 Заметки по разработке

- Основные классы моделей находятся в `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Основная логика интерактивного UI встроена напрямую в `templates/cellist.html`.
- SQL-схема и данные в стиле seed находятся в `cellist.sql`.
- Ноутбуки в `notebooks/` и `polygon_sample/` содержат исследовательские материалы.
- В корне репозитория пока нет выделенного набора автоматических тестов и конфигурации CI.

## 🧯 Устранение неполадок

| Симптом | Что проверить |
|---|---|
| `ModuleNotFoundError` или проблемы с импортом | Убедитесь, что перед запуском `python app.py` выполнено `conda activate cellist`. |
| UI рендерится без стилей/скриптов | Выполните `npm install` внутри `statics/` и убедитесь, что существует `statics/node_modules`. |
| Отказ в доступе к MySQL | Проверьте имя пользователя/пароль в `cellist/utils/constants.py` и режим plugin/auth в MySQL. |
| Приложение стартует, но действия модели падают | Проверьте доступность CUDA/GPU; текущие пути предполагают CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Загрузка проходит, но тайлы/модели не появляются | Убедитесь, что подкаталоги `data/` существуют и доступны для записи. |

## 🗺️ Дорожная карта

Следующие пункты сохранены и структурированы из существующей документации проекта/заметок TODO:

- Пример с полигонами: использовать полигональную разметку вместо прямоугольной.
- Оптимизировать модель для поведения `float32` при очень малых/больших значениях.
- По возможности уменьшить размер модели.
- Повысить устойчивость с помощью подходов вроде компонентов, вдохновленных Transformer/stable-diffusion.
- Добавить варианты базовой модели (Threshold, Cellpose) и целевой модели (AIR, Transformer, SD).
- Оптимизация интерфейса (включая множественный выбор).
- Оптимизация backend (включая улучшенную работу с памятью/кэшем).
- Удобная упаковка с минимальной настройкой БД (например, вариант SQLite).

## 🤝 Участие в проекте

### Внесите вклад в Lazeal Cellist

Lazeal Cellist — это проект с открытым исходным кодом, и мы приветствуем вклад от всех, независимо от уровня опыта. Мы приглашаем вклад, который:

- Повышает алгоритмическую эффективность и производительность
- Улучшает интерфейс и пользовательский опыт
- Расширяет документацию и примеры
- Исправляет ошибки и повышает стабильность системы

Перед началом работы, пожалуйста, сначала обсудите желаемое изменение через issue. Это помогает координировать усилия и избегать дублирующейся или конфликтующей работы.

Чтобы начать, прочитайте руководство по участию в проекте.

Дополнительные документы по участию в репозитории:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 Лицензия

Этот проект распространяется по лицензии MIT. Подробнее смотрите файл [LICENSE](https://chat.openai.com/LICENSE) в этом репозитории.

Примечание о состоянии репозитория: в текущем checkout в корне отсутствует файл `LICENSE`. Строка выше сохранена из предыдущего README как каноничное намерение проекта; при желании добавьте локальный файл `LICENSE` отдельным изменением.
