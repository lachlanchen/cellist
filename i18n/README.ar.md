[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>منصتك الفعالة لاكتشاف خلايا ثلاثية الأبعاد وبروفايلها بكفاءة</strong>
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
  <a href="#-نظرة-عامة"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-التثبيت"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-الاستخدام"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-استكشاف-الأخطاء-وإصلاحها"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-المساهمة"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 Preview

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: منصتك الفعالة لاكتشاف الخلايا ثلاثية الأبعاد وبروفايلها

مرحبًا بك في Lazeal Cellist، وهي منصة شاملة وفعالة لاكتشاف الخلايا وتقسيمها وتحليل ملفاتها في صور الميكروسكوب ثلاثية الأبعاد.

صُمّمت منصتنا لاكتشاف الخلايا باستخدام التعلم غير الخاضع للإشراف، وتقنيات العتبة، وخوارزميات حديثة مثل Cellpose. كما توفر Lazeal Cellist واجهة تفاعلية بديهية تسمح لك بتحسين نتائج الاكتشاف. تُعاد هذه النتائج المُحسّنة بعد ذلك إلى شبكة التعلم شبه الخاضعة للإشراف، مما يُحسّن أداء النموذج باستمرار.

تميّز Lazeal Cellist نفسها بنموذج ثلاثي الأبعاد فعّال يحتاج إلى جهد ضئيل للتدريب والتحسين، مما يجعله منصة عملية للباحثين والعلماء والهواة.

> ℹ️ **ملاحظة حول النطاق**
> تتضمن رؤية المشروع والواجهة مفاهيم ثلاثية الأبعاد (`/3d`, `templates/cellist_3d.html`)، بينما يسير مسار التدريب الأساسي حاليًا في الشيفرة نحو تقطيع ثنائي الأبعاد + التحسين.

---

## جدول المحتويات

- [نظرة عامة](#-نظرة-عامة)
- [الميزات الرئيسية](#-الميزات-الرئيسية)
- [هيكل المشروع](#-هيكل-المشروع)
- [المتطلبات المسبقة](#-المتطلبات-المسبقة)
- [التثبيت](#-التثبيت)
- [الاستخدام](#-الاستخدام)
- [الإعداد](#-الإعداد)
- [أمثلة](#-أمثلة)
- [مستوحى من الأبحاث](#-مستوحى-من-الأبحاث)
- [ملاحظات التطوير](#-ملاحظات-التطوير)
- [استكشاف الأخطاء وإصلاحها](#-استكشاف-الأخطاء-وإصلاحها)
- [خارطة الطريق](#-خارطة-الطريق)
- [المساهمة](#-المساهمة)
- [شكر وتقدير](#-شكر-وتقدير)
- [Support](#-support)
- [الترخيص](#-الترخيص)

## 🔍 نظرة عامة

Lazeal Cellist هي منصة ويب مكتوبة بـ Python/Tornado لخطوط سير صور الميكروسكوب وتدعم:

- رفع الصور عبر المتصفح، وإنشاء النماذج، وتحرير التعليقات التوضيحية.
- تهيئة أولية بمساعدة الخوارزميات (وضع النوى في Cellpose).
- تحسين تكراري تفاعلي عبر WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- حفظ قائم على قاعدة البيانات للنماذج وشرائح الصور والتعليقات التوضيحية.

> ℹ️ ملاحظة على السلوك الحالي: على الرغم من أن رؤية المشروع والواجهة تتضمن مفاهيم ثلاثية الأبعاد (`/3d`, `templates/cellist_3d.html`) فإن مسار التدريب الأساسي في الشيفرة الآن يعتمد في الأساس على التقطيع ثنائي الأبعاد + التحسين.

### لمحة سريعة

| المجال | التنفيذ الحالي |
|---|---|
| الخادم | Tornado (`app.py`) |
| المنفذ | `8887` |
| قاعدة البيانات | MySQL (`cellist.sql`) |
| حزمة تعلم الآلة الأساسية | PyTorch + Pyro + Cellpose |
| الواجهة | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| تهيئة الاستدلال | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| حالة التغليف | نموذج أولي بحثي (بدون `pyproject.toml`/`setup.py`) |
| حالة الاختبارات/CI | لا توجد مجموعة اختبار آلية مخصصة أو إعداد CI في جذر المستودع |

### لغات التوثيق

يتضمن هذا المستودع ملفات README متعددة اللغات ضمن `i18n/`:

| اللغة | الملف |
|---|---|
| الألمانية | `README.de.md` |
| الإسبانية | `README.es.md` |
| الفرنسية | `README.fr.md` |
| اليابانية | `README.ja.md` |
| الكورية | `README.ko.md` |
| الروسية | `README.ru.md` |
| الفيتنامية | `README.vi.md` |
| الصينية المبسطة | `README.zh-Hans.md` |
| الصينية التقليدية | `README.zh-Hant.md` |

## ✨ الميزات الرئيسية

- **اكتشاف خلايا ثلاثية الأبعاد غير خاضع للإشراف**: تحديد الخلايا في صور الميكروسكوب ثلاثية الأبعاد باستخدام تقنيات تعلم آلة متقدمة.
- **واجهة تفاعلية لتحسين النتائج**: تنقيح نتائج الاكتشاف عبر واجهة استخدام بديهية وسهلة.
- **شبكة تعلم شبه خاضع للإشراف فعّالة**: تحسين أداء النموذج تدريجيًا بمرور الوقت عبر النتائج المنقحة.
- **تقسيم الخلايا وبروفايلها**: تتجاوز اكتشاف الخلية إلى إمكانيات تقسيم وبروفايل متقدمة.

الميزات الإضافية المتوفرة حاليًا:

- خادم Tornado REST + WebSocket (`app.py`) يعمل على المنفذ `8887`.
- تقطيع تلقائي للصور (`256x256` افتراضيًا) لاستيعاب النموذج.
- مخطط MySQL متضمّن كـ dump: [`cellist.sql`](cellist.sql).
- تتضمن الواجهة الأمامية Bootstrap وjQuery وjQuery UI وThree.js وblueimp-file-upload.
- مهام النموذج تعمل بشكل غير متزامن عبر تجمع مؤشرات التنفيذ (`max_workers=64`).

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
| نظام التشغيل | يفضّل Linux (تُفترض سلوكات shell الخاصة بـ Linux في الأوامر التالية). |
| Python/Conda | يجب أن يكون Conda متاحًا لإنشاء البيئة من [`cellist.yaml`](cellist.yaml). |
| قاعدة البيانات | خادم MySQL يعمل على `localhost` مع قاعدة بيانات `cellist`. |
| GPU | بيئة NVIDIA/CUDA مطلوبة/متوقعة بقوة من مسارات الشيفرة الحالية. |
| Node.js + npm | مطلوبة لتثبيت تبعيات الواجهة الأمامية ضمن `statics/node_modules`. |
| صلاحية الكتابة على القرص | مطلوبة لبيانات التشغيل ضمن `<repo>/data`. |

## 🛠️ التثبيت

### 1. استنساخ المستودع والدخول إليه

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. إنشاء بيئة Python

استخدم الملف المسمّى في المستودع `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

ملاحظة توافقية كما في الوثائق القديمة: كانت الوثائق السابقة تستخدم `celist.yaml` (بلا حرف `l`)، بينما الملف في هذا المستودع هو `cellist.yaml`.

الأمر القديم (محفوظ):

```bash
conda env create -f celist.yaml
```

### 3. تثبيت تبعيات الواجهة الأمامية

```bash
cd statics
npm install
cd ..
```

### 4. تحضير مجلدات بيانات التشغيل

يتوقع التطبيق وجود شجرة `data/` (وقد استُثني `data` بالفعل في `.gitignore`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

ملاحظة: يوجد [`create_data_folder.py`](create_data_folder.py)، لكنه حاليا ينشئ المجلدات في دليل العمل الحالي (وليس تحت `data/`). خذ هذا بعين الاعتبار إذا استُخدم.

### 5. تحضير مصادقة MySQL (عند الحاجة)

إذا كانت مصادقة root تعتمد على socket وتمنع وصول التطبيق، تقترح الوثائق الأقدم التحويل إلى مصادقة بكلمة مرور:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. إنشاء قاعدة البيانات واسترجاع المخطط/البيانات

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

مثال توثيقي قديم (محفوظ):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. ضبط اعتماديات MySQL للتشغيل

تقرا الشيفرة الحالية الاعتماديات من [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` و`mysqlurl`).

تتضمن القيم الافتراضية الحالية:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

لتحسين الأمان المحلي، حدّث هذه القيم قبل التشغيل في بيئتك.

### 8. فحوصات بيئية اختيارية

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 الاستخدام

### تشغيل خادم الويب

```bash
python app.py
```

أمر بدء تشغيل قديم من الوثائق السابقة (محفوظ):

```bash
python app.py -m cellist
```

مسارات السيرفر الافتراضية كما تظهر في الشيفرة:

- الواجهة الأساسية: `http://localhost:8887/`
- صفحة 3D: `http://localhost:8887/3d`

### سير العمل النموذجي

1. افتح الواجهة وسجّل الدخول.
2. ارفع صور الميكروسكوب من لوحة Create Model.
3. اختر الخوارزمية الأساسية (`Cellpose`) وأنشئ النموذج.
4. دع الواجهة الخلفية تقطع الصور وتُهيئ الاكتشافات.
5. حمّل الصور المقصوصة وراجع/اضبط تعليقات المستطيلات.
6. شغّل دورات `initialize` و`pretrain` و`train`.
7. استخدم `Pretrain Stop` / `Stop` (`train-stop`) / `reset` حسب الحاجة.
8. احفظ التحديثات اليدوية عبر إجراءات `Update Model`/annotation.

### بيانات دخول الواجهة المدمجة (السلوك الحالي في القالب)

الواجهة الأمامية تتحقق حاليًا من هذه الاعتماديات الثابتة من جهة العميل:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

هذا سلوك نموذجي وليس نظام مصادقة إنتاجي.

### سطح API/Socket المستخدم حاليًا من الواجهة

نقاط نهاية HTTP:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

نقطة نهاية WebSocket:

- `ws://localhost:8887/websocket/<ws_uuid>`

رسائل `data_type` المعترف بها في معالج WebSocket:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ الإعداد

### الخلفية والنقاط النهائية

مكوّن في [`app.py`](app.py):

- المنفذ: `8887`
- المسارات:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### سلوك النموذج والبيانات

- حجم تجمع الخيوط هو `max_workers=64`.
- بلاطات الصور هي `256x256` افتراضيًا.
- تهيئة Cellpose تستخدم `model_type='nuclei'` و`gpu=True`.
- التدريب وما قبل التدريب يعملان بشكل غير متزامن عبر إجراءات تفعيل WebSocket.
- يُشتق جذر البيانات من دليل العمل الحالي كـ `<repo>/data`.

### ثوابت قاعدة البيانات/زمن التشغيل

من [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` تشمل مفاتيح host/user/password
- `mysqlurl` تستهدف قاعدة بيانات باسم `cellist`

### لقطة اعتمادات الواجهة الأمامية

من [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### لمحة عن بيئة Conda

من [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 أمثلة

### مثال: شكل رسالة WebSocket لإنشاء نموذج

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

### مثال: تحديث تعليقات توضيحية يدوي عبر WebSocket

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

### مثال: تشغيل محلي كامل وبسيط

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 مستوحى من الأبحاث

Lazeal Cellist مستلهم من أبحاث متقدمة في التعلم العميق، بما في ذلك:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

توفر هذه الأعمال رؤى قيمّة ساعدت في توجيه تطوير خوارزميات ومنهجيات المنصة.

(ملاحظة: للاقتباس الدقيق، راجع الأوراق الأصلية مباشرة.)

## 🧭 ملاحظات التطوير

- توجد فئات النموذج الأساسية ضمن `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- منطق الواجهة التفاعلي الرئيسي مضمن مباشرةً في `templates/cellist.html`.
- مخطط SQL والبيانات الشبيهة بالبذور موجود في `cellist.sql`.
- دفاتر الاستكشاف في `notebooks/` و`polygon_sample/` تقدم مراجع استكشافية.
- ملاحظات المنصة/النموذج الموسعة موجودة في [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- لا توجد حاليًا مجموعة اختبار آلية مخصصة أو إعداد CI في جذر المستودع.

### الافتراضات والقيود الحالية

- يبدو أن هذا المستودع يستهدف أولًا الاستخدام المحلي الموجّه للبحث.
- بعض مسارات الشيفرة تفترض توفر GPU (`cuda:0`).
- إدارة المصادقة والأسرار في مستوى النموذج الأولي.
- توجد واجهات 3D، لكن مسار التدريب السائد ما زال قائمًا على مقاطع 2D.

## 🧯 استكشاف الأخطاء وإصلاحها

| العرض | الفحوصات المقترحة |
|---|---|
| `ModuleNotFoundError` أو مشاكل الاستيراد | تأكد من تنفيذ `conda activate cellist` قبل تشغيل `python app.py`. |
| الواجهة تظهر بدون تنسيق/سكريبتات | نفّذ `npm install` داخل `statics/` وتأكد من وجود `statics/node_modules`. |
| رفض وصول MySQL | تحقق من اسم المستخدم/كلمة المرور في [`cellist/utils/constants.py`](cellist/utils/constants.py) ووضعية إضافة المصادقة في MySQL. |
| التطبيق يبدأ لكن تفشل إجراءات النموذج | افحص توافر CUDA/GPU؛ المسارات الحالية تفترض CUDA (`torch.device('cuda:0')`، `Cellpose gpu=True`). |
| الرفع ناجح لكن لا تظهر المربعات/النماذج | تأكد من وجود مجلدات فرعية داخل `data/` وقابليتها للكتابة. |
| أخطاء في طلبات REST/WebSocket | تأكد من أن الخادم يعمل على `http://localhost:8887` وأن مفاتيح الـ payload تتطابق مع أسماء القالب الحالية. |
| `FileNotFoundError` ضمن `data/` | شغّل التطبيق من جذر المستودع كي تُحلّ المسارات النسبية بشكل متسق. |

### تشخيصات سريعة

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ خارطة الطريق

تم الحفاظ على العناصر التالية وتنظيمها من توثيق/TODO المشروع الحالي:

- Polygon sample: استخدام الرسم متعدد الأضلاع بدلًا من التعليق المستطيل.
- تحسين النموذج لسلوك `float32` مع القيم الكبيرة جدًا أو الصغيرة جدًا.
- تقليل حجم النموذج حيثما أمكن.
- زيادة المتانة باستخدام مقاربات مستوحاة من مكوّنات Transformer/stable-diffusion.
- إضافة خيارات النماذج الأساسية (Threshold, Cellpose) وخيارات النماذج المستهدفة (AIR, Transformer, SD).
- تحسين الواجهة (بما في ذلك اختيار متعدد).
- تحسين الخلفية (بما في ذلك تحسين إدارة الذاكرة/التخزين المؤقت).
- تغليف سهل الاستخدام مع أقل إعدادات لقاعدة البيانات (مثل خيار SQLite).

## 🤝 المساهمة

### المساهمة في Lazeal Cellist

Lazeal Cellist مشروع مفتوح المصدر، ونحن نرحب بالمساهمات من الجميع مهما كان مستوى الخبرة.
ندعو مساهمات تساعد على:

- تحسين الكفاءة والأداء الخوارزمي
- تحسين واجهة المستخدم وتجربة الاستخدام
- توسيع التوثيق والأمثلة
- إصلاح الأخطاء وتحسين استقرار النظام

قبل البدء في المساهمة، يُرجى مناقشة التغيير المقترح أولًا عبر issue، مما يساعد على تنسيق الجهود وتجنب تكرار أو تضارب العمل.

للمزيد من المعلومات حول البداية، راجع إرشادات المساهمة.

وثائق إضافية للمساهمة في المستودع:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 الشكر والتقدير

- يعتمد مفهوم وتنفيذ Lazeal Cellist بشكل كبير على خط أبحاث AIR/SPAIR المذكور أعلاه.
- المستودع يضم توثيقًا وأوامرًا تاريخية/وراثية تم حفظها عمدًا للحفاظ على استمرارية استخدام المشروع.

## 📄 الترخيص

هذا المشروع مرخّص برخصة MIT. لمزيد من المعلومات، راجع ملف [LICENSE](LICENSE) في هذا المستودع.

ملاحظة حالة المستودع: لا يوجد حاليًا ملف `LICENSE` في checkout الحالي. السطر أعلاه محفوظ من README الأصلي كـ نية المشروع الأساسية؛ أضف ملف `LICENSE` محليًا في تغيير لاحق إذا رغبت.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
