[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>منصتك الفعّالة لاكتشاف الخلايا ثلاثية الأبعاد وتحليلها</strong>
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

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: منصتك الفعّالة لاكتشاف الخلايا ثلاثية الأبعاد وتحليلها

مرحبًا بك في Lazeal Cellist، وهي منصة شاملة وفعّالة لاكتشاف الخلايا وتقسيمها وتحليلها في صور المجهر ثلاثية الأبعاد.

صُمّمت منصتنا لاكتشاف الخلايا باستخدام التعلّم غير الخاضع للإشراف، وتقنيات العتبة (thresholding)، وخوارزميات متقدمة مثل Cellpose. كما توفّر Lazeal Cellist واجهة تفاعلية بديهية تتيح للمستخدمين تحسين نتائج الاكتشاف. بعد ذلك، تُعاد هذه النتائج المُحسّنة إلى شبكة التعلّم شبه الخاضع للإشراف، ما يحسّن أداء النموذج باستمرار.

تتميّز Lazeal Cellist بتقديم نموذج ثلاثي الأبعاد فعّال يتطلب جهدًا محدودًا للتدريب والتحسين، ما يجعلها منصة عملية للعلماء والباحثين والهواة.

> ℹ️ **ملاحظة حول النطاق**
> تتضمن رؤية المشروع والواجهة مفاهيم ثلاثية الأبعاد (`/3d`, `templates/cellist_3d.html`)، بينما يظل تدفّق التدريب الأساسي الحالي في الشيفرة معتمدًا غالبًا على التقطيع ثنائي الأبعاد + التحسين.

---

## جدول المحتويات

- [نظرة عامة](#-نظرة-عامة)
- [الميزات الأساسية](#-الميزات-الأساسية)
- [هيكل المشروع](#-هيكل-المشروع)
- [المتطلبات المسبقة](#-المتطلبات-المسبقة)
- [التثبيت](#-التثبيت)
- [الاستخدام](#-الاستخدام)
- [الإعداد](#-الإعداد)
- [أمثلة](#-أمثلة)
- [مستلهم من الأبحاث](#-مستلهم-من-الأبحاث)
- [ملاحظات التطوير](#-ملاحظات-التطوير)
- [استكشاف الأخطاء وإصلاحها](#-استكشاف-الأخطاء-وإصلاحها)
- [خارطة الطريق](#roadmap)
- [المساهمة](#-المساهمة)
- [الشكر والتقدير](#-الشكر-والتقدير)
- [الدعم](#-support)
- [الترخيص](#-الترخيص)

## 🔍 نظرة عامة

Lazeal Cellist هي منصة ويب Python/Tornado لمسارات عمل صور المجهر، وتتضمن:

- رفع الصور عبر المتصفح، وإنشاء النماذج، وتحرير التعليقات التوضيحية.
- تهيئة بمساعدة الخوارزميات (وضع النوى Cellpose nuclei mode).
- تحسين تكراري بأسلوب human-in-the-loop عبر إجراءات WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- حفظ دائم مدعوم بقاعدة البيانات للنماذج، وشرائح الصور، والتعليقات التوضيحية.

> ℹ️ ملاحظة حول السلوك الحالي: على الرغم من أن رؤية المشروع والواجهة تتضمن مفاهيم ثلاثية الأبعاد (`/3d`, `templates/cellist_3d.html`)، فإن تدفّق التدريب الرئيسي الحالي في الشيفرة يعتمد أساسًا على التقطيع ثنائي الأبعاد + تحسين النموذج.

### لمحة سريعة

| المجال | التطبيق الحالي |
|---|---|
| الخادم | Tornado (`app.py`) |
| المنفذ | `8887` |
| قاعدة البيانات | MySQL (`cellist.sql`) |
| حزمة ML الأساسية | PyTorch + Pyro + Cellpose |
| الواجهة الأمامية | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| تهيئة الاستدلال | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| حالة التحزيم | نموذج أولي بحثي (لا يوجد `pyproject.toml`/`setup.py`) |
| حالة الاختبارات/CI | لا توجد مجموعة اختبارات آلية مخصصة أو إعداد CI في جذر المستودع |

### لغات التوثيق

يتضمن هذا المستودع بالفعل ملفات README متعددة اللغات تحت `i18n/`:

| اللغة | الملف |
|---|---|
| الألمانية | `README.de.md` |
| الإسبانية | `README.es.md` |
| الفرنسية | `README.fr.md` |
| اليابانية | `README.ja.md` |
| الكورية | `README.ko.md` |
| الروسية | `README.ru.md` |
| الفيتنامية | `README.vi.md` |
| الصينية (المبسطة) | `README.zh-Hans.md` |
| الصينية (التقليدية) | `README.zh-Hant.md` |

## ✨ الميزات الأساسية

- **اكتشاف خلايا ثلاثي الأبعاد غير خاضع للإشراف**: تحديد الخلايا في صور المجهر ثلاثية الأبعاد باستخدام تقنيات تعلّم آلة متقدمة.
- **واجهة تفاعلية لتحسين النتائج**: تحسين نتائج الاكتشاف عبر واجهة بديهية وسهلة الاستخدام.
- **شبكة تعلّم شبه خاضع للإشراف فعّالة**: رفع أداء النموذج مع مرور الوقت عبر النتائج المحسّنة.
- **تقسيم الخلايا وتحليلها**: تجاوز الاكتشاف إلى إمكانيات متقدمة للتقسيم والتحليل.

ميزات تنفيذية إضافية موجودة حاليًا:

- خادم Tornado REST + WebSocket (`app.py`) على المنفذ `8887`.
- تقطيع تلقائي للصور (`256x256` افتراضيًا) لإدخال النموذج.
- مخطط MySQL مضمن على شكل dump: [`cellist.sql`](cellist.sql).
- تتضمن حزمة الواجهة الأمامية Bootstrap وjQuery وjQuery UI وThree.js وblueimp-file-upload.
- مهام نموذج غير متزامنة عبر thread pool (`max_workers=64`).

## 🗂️ هيكل المشروع

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

## ✅ المتطلبات المسبقة

| المتطلب | الملاحظات |
|---|---|
| نظام التشغيل | يُنصح بـ Linux (الأوامر أدناه تفترض سلوك shell الخاص بـ Linux). |
| Python/Conda | توفر Conda لإنشاء البيئة من [`cellist.yaml`](cellist.yaml). |
| قاعدة البيانات | خادم MySQL يعمل على `localhost` مع قاعدة بيانات `cellist`. |
| GPU | يُنصح بشدة/يُتوقع توفر بيئة NVIDIA/CUDA وفق مسارات الشيفرة الحالية. |
| Node.js + npm | مطلوبة لتثبيت تبعيات الواجهة الأمامية `statics/node_modules`. |
| صلاحية كتابة على القرص | مطلوبة لبيانات وقت التشغيل تحت `<repo>/data`. |

## 🛠️ التثبيت

### 1. استنسخ المستودع وادخل إليه

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. أنشئ بيئة Python

استخدم اسم ملف المستودع `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

ملاحظة توافق محفوظة من وثائق أقدم: استخدمت وثائق سابقة `celist.yaml` (ينقصه حرف `l`)، لكن الملف الموجود في هذا المستودع هو `cellist.yaml`.

الأمر القديم (محفوظ):

```bash
conda env create -f celist.yaml
```

### 3. ثبّت تبعيات الواجهة الأمامية

```bash
cd statics
npm install
cd ..
```

### 4. جهّز أدلة بيانات وقت التشغيل

يتوقع التطبيق شجرة `data/` (ويستثني `.gitignore` بالفعل `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

ملاحظة: الملف [`create_data_folder.py`](create_data_folder.py) موجود، لكنه ينشئ الأدلة حاليًا في دليل العمل الحالي (وليس تحت `data/`). ضع ذلك في الاعتبار إذا استخدمته.

### 5. جهّز مصادقة MySQL (عند الحاجة)

إذا كانت مصادقة root مبنية على socket وتمنع وصول التطبيق، تقترح وثائق المشروع الأقدم التحويل إلى مصادقة بكلمة مرور:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. أنشئ قاعدة البيانات واستعد المخطط/البيانات

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

مثال الوثائق القديمة (محفوظ):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. اضبط بيانات اعتماد MySQL لوقت التشغيل

تقرأ الشيفرة الحالية بيانات الاعتماد من [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` و`mysqlurl`).

تتضمن قيم الشيفرة الافتراضية الحالية:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

لأمان البيئة المحلية، حدّث هذه القيم قبل التشغيل في بيئتك.

### 8. فحوصات سلامة بيئة اختيارية

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 الاستخدام

### شغّل خادم الويب

```bash
python app.py
```

أمر بدء تشغيل قديم من وثائق سابقة (محفوظ):

```bash
python app.py -m cellist
```

إعدادات المسارات الافتراضية للخادم كما لوحظت في الشيفرة:

- الواجهة الرئيسية: `http://localhost:8887/`
- صفحة 3D: `http://localhost:8887/3d`

### مسار عمل نموذجي

1. افتح الواجهة وسجّل الدخول.
2. ارفع صور المجهر من لوحة Create Model.
3. اختر الخوارزمية الأساسية (`Cellpose`) وأنشئ النموذج.
4. دع الخلفية تقطع الصور وتهيئ عمليات الاكتشاف.
5. حمّل الصور المقصوصة وراجع/عدّل تعليقات المستطيلات.
6. شغّل دورات `initialize` و`pretrain` و`train`.
7. استخدم `Pretrain Stop` / `Stop` (`train-stop`) / `reset` عند الحاجة.
8. احفظ التحديثات اليدوية عبر إجراءات `Update Model`/annotation.

### بيانات تسجيل الدخول المضمنة في الواجهة (سلوك القالب الحالي)

تتحقق الواجهة الأمامية حاليًا من بيانات اعتماد ثابتة على جهة العميل:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

هذا سلوك نموذج أولي وليس مصادقة إنتاجية.

### سطح API/Socket المستخدم حاليًا بواسطة الواجهة

نقاط نهاية HTTP:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

نقطة نهاية WebSocket:

- `ws://localhost:8887/websocket/<ws_uuid>`

رسائل إجراءات `data_type` المتعرّف عليها في معالج WebSocket:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ الإعداد

### الخلفية ونقاط النهاية

مُهيّأة في [`app.py`](app.py):

- المنفذ: `8887`
- المسارات:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### سلوك النموذج/البيانات

- حجم thread pool هو `max_workers=64`.
- مربعات الصور `256x256` افتراضيًا.
- تهيئة Cellpose تستخدم `model_type='nuclei'` و`gpu=True`.
- التدريب وpretraining يعملان بشكل غير متزامن عبر إجراءات تبدأ من WebSocket.
- جذر البيانات يُحلّ من دليل العمل الحالي كـ `<repo>/data`.

### ثوابت قاعدة البيانات/وقت التشغيل

من [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` يتضمن مفاتيح host/user/password
- `mysqlurl` يستهدف قاعدة بيانات باسم `cellist`

### لقطة تبعيات الواجهة الأمامية

من [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### أبرز مكونات بيئة Conda

من [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 أمثلة

### مثال: شكل رسالة create في WebSocket

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

### مثال: تحديث تعليق توضيحي يدوي عبر WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### مثال: طلب تحميل نموذج

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### مثال: تشغيل محلي كامل بحد أدنى

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 مستلهم من الأبحاث

Lazeal Cellist مستلهم من أبحاث رائدة في التعلّم العميق، بما في ذلك:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

تقدم هذه الأعمال رؤى قيّمة وجهت تطوير خوارزميات ومنهجيات منصتنا.

(ملاحظة: للاستشهاد الدقيق، يُرجى الرجوع مباشرةً إلى الأوراق الأصلية.)

## 🧭 ملاحظات التطوير

- فئات النموذج الأساسية موجودة تحت `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- منطق الواجهة التفاعلية الرئيسي مضمن مباشرةً في `templates/cellist.html`.
- مخطط SQL وبيانات نمط seed موجودة في `cellist.sql`.
- دفاتر الملاحظات في `notebooks/` و`polygon_sample/` توفر مراجع استكشافية.
- ملاحظات موسعة للمنصة/النموذج موجودة في [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- لا توجد حاليًا مجموعة اختبارات آلية مخصصة أو إعداد CI في جذر المستودع.

### الافتراضات والقيود الحالية

- يبدو أن هذا المستودع يستهدف أولًا الاستخدام المحلي الموجه للأبحاث.
- بعض مسارات الشيفرة تفترض توفر GPU (`cuda:0`).
- المصادقة وإدارة الأسرار في مستوى النموذج الأولي.
- واجهات 3D موجودة، لكن مسار التدريب المهيمن لا يزال معتمدًا على مربعات 2D.

## 🧯 استكشاف الأخطاء وإصلاحها

| العَرَض | فحوصات مقترحة |
|---|---|
| `ModuleNotFoundError` أو مشاكل استيراد | تأكد من تنفيذ `conda activate cellist` قبل تشغيل `python app.py`. |
| الواجهة تظهر دون تنسيقات/سكريبتات | نفّذ `npm install` داخل `statics/` وتأكد من وجود `statics/node_modules`. |
| رفض وصول MySQL | تحقّق من اسم المستخدم/كلمة المرور في `cellist/utils/constants.py` ووضع plugin/auth في MySQL. |
| التطبيق يعمل لكن إجراءات النموذج تفشل | افحص توفر CUDA/GPU؛ المسارات الحالية تفترض CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| الرفع ينجح لكن لا تظهر tiles/models | تأكد من وجود مجلدات `data/` الفرعية وقابليتها للكتابة. |
| أخطاء طلبات REST/WebSocket | تأكد من أن الخادم يعمل على `http://localhost:8887` وأن مفاتيح الحمولة تطابق أسماء القالب الحالية. |
| `FileNotFoundError` تحت `data/` | شغّل التطبيق من جذر المستودع حتى تُحل المسارات النسبية بشكل متسق. |

### تشخيصات سريعة

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## Roadmap

العناصر التالية محفوظة ومنظمة من وثائق المشروع/ملاحظات TODO الحالية:

- Polygon sample: استخدام polygon بدلًا من rectangle annotation.
- تحسين النموذج لسلوك `float32` مع القيم الصغيرة/الكبيرة جدًا.
- تقليل حجم النموذج حيثما أمكن.
- تحسين المتانة بأساليب مثل مكونات مستوحاة من Transformer/stable-diffusion.
- إضافة خيارات نموذج أساسي (Threshold, Cellpose) وخيارات نموذج هدف (AIR, Transformer, SD).
- تحسين الواجهة (بما في ذلك التحديد المتعدد).
- تحسين الخلفية (بما في ذلك تحسين التعامل مع الذاكرة/الذاكرة المؤقتة).
- تحزيم سهل الاستخدام بأقل إعدادات لقاعدة البيانات (مثل خيار SQLite).

## 🤝 المساهمة

### ساهم في Lazeal Cellist

Lazeal Cellist مشروع مفتوح المصدر، ونرحب بالمساهمات من الجميع بغض النظر عن مستوى الخبرة. ندعو إلى مساهمات:

- تعزز الكفاءة والأداء الخوارزمي
- تحسن واجهة المستخدم وتجربة الاستخدام
- توسع التوثيق والأمثلة
- تصلح الأخطاء وتحسن استقرار النظام

قبل أن تبدأ بالمساهمة، يرجى أولًا مناقشة التغيير الذي تريد تنفيذه عبر issue. يساعد ذلك على تنسيق الجهود وتجنب العمل المكرر أو المتعارض.

لمزيد من المعلومات حول كيفية البدء، يرجى قراءة إرشادات المساهمة.

وثائق مساهمة إضافية في المستودع:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 الشكر والتقدير

- يعتمد مفهوم وتنفيذ Lazeal Cellist بشكل كبير على خط أبحاث AIR/SPAIR المذكور أعلاه.
- يتضمن المستودع وثائق وأوامر تاريخية/قديمة محفوظة عمدًا لضمان الاستمرارية مع استخدامات المشروع السابقة.

## 📄 الترخيص

هذا المشروع مرخّص بموجب رخصة MIT. لمزيد من المعلومات، يرجى الرجوع إلى ملف [LICENSE](LICENSE) في هذا المستودع.

ملاحظة حالة المستودع: لا يوجد حاليًا ملف `LICENSE` في الجذر ضمن هذه النسخة. السطر أعلاه محفوظ من README السابق باعتباره نية المشروع الأساسية؛ يمكنك إضافة ملف `LICENSE` محليًا في تغيير لاحق إذا رغبت.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
