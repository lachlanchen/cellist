[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


خيارات اللغة: **الإنجليزية (الحالية)** | الترجمات المخطط لها في `i18n/` (المجلد موجود وهو فارغ حاليًا)

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

## Lazeal Cellist: منصتك الفعّالة لاكتشاف الخلايا ثلاثية الأبعاد وتحليلها

مرحبًا بك في Lazeal Cellist، وهي منصة شاملة وفعّالة لاكتشاف الخلايا وتقسيمها وتحليل خصائصها في صور المجهر ثلاثية الأبعاد.

صُممت منصتنا لاكتشاف الخلايا باستخدام التعلم غير المراقب وتقنيات العتبة وخوارزميات متقدمة مثل Cellpose. كما توفّر Lazeal Cellist واجهة تفاعلية سهلة تتيح للمستخدمين تحسين نتائج الاكتشاف. ثم تُعاد هذه النتائج المحسّنة إلى شبكة التعلم شبه المُراقب، ما يؤدي إلى تحسين أداء النموذج بشكل مستمر.

تتميّز Lazeal Cellist بتوفير نموذج ثلاثي الأبعاد فعّال يتطلب جهدًا محدودًا للتدريب والتحسين، ما يجعلها منصة عملية للعلماء والباحثين والهواة.

---

## 🔍 نظرة عامة

Lazeal Cellist منصة ويب مبنية بـ Python/Tornado لسير عمل صور المجهر، وتتضمن:

- رفع الصور عبر المتصفح، وإنشاء النماذج، وتحرير التعليقات التوضيحية.
- تهيئة أولية مدعومة بالخوارزميات (وضع النوى في Cellpose).
- تحسين تكراري ضمن حلقة الإنسان-في-الحلقة عبر إجراءات WebSocket (`initialize`, `pretrain`, `train`, `update`, `reset`).
- حفظًا مستندًا إلى قاعدة بيانات للنماذج ومقاطع الصور والتعليقات التوضيحية.

ملاحظة حول السلوك الحالي: رغم أن رؤية المشروع والواجهة تتضمن مفاهيم ثلاثية الأبعاد (`/3d`, `templates/cellist_3d.html`)، فإن مسار التدريب الرئيسي الحالي في الشيفرة يعتمد أساسًا على التقطيع ثنائي الأبعاد + تحسين النموذج.

### لمحة سريعة

| المجال | التنفيذ الحالي |
|---|---|
| الخادم | Tornado (`app.py`) |
| المنفذ | `8887` |
| قاعدة البيانات | MySQL (`cellist.sql`) |
| حزمة ML الأساسية | PyTorch + Pyro + Cellpose |
| الواجهة الأمامية | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| تهيئة الاستدلال | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ الميزات الأساسية

- **اكتشاف خلايا ثلاثي الأبعاد غير مُراقب**: تحديد الخلايا في صور المجهر ثلاثية الأبعاد باستخدام تقنيات تعلم آلي متقدمة.
- **واجهة تفاعلية لتحسين النتائج**: تحسين نتائج الاكتشاف عبر واجهة بديهية وسهلة الاستخدام.
- **شبكة تعلم شبه مُراقب فعّالة**: رفع أداء النموذج مع الوقت عبر النتائج المُحسّنة.
- **تقسيم الخلايا وتحليل خصائصها**: تجاوز الاكتشاف إلى قدرات متقدمة في التقسيم والتحليل.

ميزات تنفيذ إضافية موجودة حاليًا:

- خادم Tornado REST + WebSocket (`app.py`) على المنفذ `8887`.
- تقطيع الصور تلقائيًا (`256x256` افتراضيًا) لإدخالها إلى النموذج.
- مخطط MySQL مضمّن كملف dump: [`cellist.sql`](cellist.sql).
- حزمة الواجهة الأمامية تتضمن Bootstrap وjQuery وjQuery UI وThree.js وblueimp-file-upload.

## 🗂️ بنية المشروع

```text
cellist/
├── app.py
├── cellist/                     # شيفرة ML/النموذج (PyTorch + Pyro)
├── templates/                   # واجهة HTML (صفحات 2D + 3D)
├── statics/                     # أصول الواجهة الأمامية + تبعيات npm
├── notebooks/                   # تجارب ودفاتر استكشافية
├── polygon_sample/              # استكشاف التعليقات التوضيحية متعددة الأضلاع
├── cellist.sql                  # dump مخطط/بيانات MySQL
├── cellist.yaml                 # بيئة Conda
├── create_data_folder.py        # أداة قديمة لتهيئة المجلدات
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md
└── i18n/                        # موجود حاليًا وهو فارغ
```

## ✅ المتطلبات المسبقة

| المتطلب | ملاحظات |
|---|---|
| نظام التشغيل | يُنصح بـ Linux (الأوامر أدناه تفترض سلوك shell على Linux). |
| Python/Conda | توفر Conda لإنشاء البيئة من [`cellist.yaml`](cellist.yaml). |
| قاعدة البيانات | خادم MySQL يعمل على `localhost` مع قاعدة `cellist`. |
| GPU | يُنصح/يُتوقع بشدة وجود بيئة NVIDIA/CUDA في مسارات الشيفرة الحالية. |
| Node.js + npm | مطلوب لتثبيت تبعيات الواجهة الأمامية في `statics/node_modules`. |

## 🛠️ التثبيت

### 1. استنسخ المستودع وادخل إليه

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. أنشئ بيئة Python

استخدم اسم الملف الموجود في المستودع `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

ملاحظة توافق محفوظة من وثائق أقدم: استخدمت الوثائق السابقة `celist.yaml` (ناقص حرف `l`)، لكن الملف الموجود في هذا المستودع هو `cellist.yaml`.

### 3. ثبّت تبعيات الواجهة الأمامية

```bash
cd statics
npm install
cd ..
```

### 4. جهّز مصادقة MySQL (عند الحاجة)

إذا كانت مصادقة root تعتمد على socket وتمنع وصول التطبيق، تقترح وثائق المشروع الأقدم التحويل إلى مصادقة بكلمة مرور:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. أنشئ قاعدة البيانات واستعد المخطط/البيانات

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

مثال وثائق قديم (محفوظ):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. اضبط بيانات اعتماد MySQL لوقت التشغيل

تقرأ الشيفرة الحالية بيانات الاعتماد من [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` و`mysqlurl`).

القيم الافتراضية في الشيفرة حاليًا تتضمن:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

لأمان البيئة المحلية، حدّث هذه القيم قبل التشغيل في بيئتك.

### 7. جهّز مجلدات بيانات وقت التشغيل

يتوقع التطبيق شجرة `data/` (وملف `.gitignore` يستثني `data` بالفعل).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

ملاحظة: الملف [`create_data_folder.py`](create_data_folder.py) موجود، لكنه ينشئ المجلدات حاليًا في مجلد العمل الحالي (وليس داخل `data/`). ضع ذلك في الاعتبار إذا استخدمته.

## 🚀 الاستخدام

### تشغيل خادم الويب

```bash
python app.py
```

أمر تشغيل قديم من وثائق سابقة (محفوظ):

```bash
python app.py -m cellist
```

مسارات الخادم الافتراضية المرصودة في الشيفرة:

- الواجهة الرئيسية: `http://localhost:8887/`
- صفحة 3D: `http://localhost:8887/3d`

### سير عمل نموذجي

1. افتح الواجهة وسجّل الدخول.
2. ارفع صور المجهر من لوحة Create Model.
3. اختر الخوارزمية الأساسية (`Cellpose`) وأنشئ النموذج.
4. دع الواجهة الخلفية تُقسّم الصور وتهيّئ الاكتشافات.
5. حمّل الصور المقتطعة وراجع/عدّل التعليقات التوضيحية المستطيلة.
6. شغّل دورات `initialize` و`pretrain` و`train`.
7. احفظ التحديثات اليدوية عبر إجراءات `Update Model`/التعليقات التوضيحية.

### بيانات تسجيل الدخول المدمجة في الواجهة (سلوك القالب الحالي)

تتحقق الواجهة الأمامية حاليًا من بيانات اعتماد ثابتة على جانب العميل:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

هذا سلوك تجريبي وليس مصادقة إنتاجية.

## ⚙️ الإعدادات

### الواجهة الخلفية ونقاط النهاية

مُعدّة في [`app.py`](app.py):

- المنفذ: `8887`
- المسارات:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### سلوك النموذج/البيانات

- حجم مجموعة الخيوط هو `max_workers=64`.
- مقاطع الصور تكون `256x256` افتراضيًا.
- تهيئة Cellpose تستخدم `model_type='nuclei'` و`gpu=True`.
- التدريب وما قبل التدريب يعملان بشكل غير متزامن عبر إجراءات تُطلقها WebSocket.

## 🧪 أمثلة

### مثال: بنية رسالة إنشاء عبر WebSocket

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

## 📚 مستوحى من الأبحاث

Lazeal Cellist مستوحى من أبحاث متقدمة في التعلم العميق، ومنها:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

تقدم هذه الأعمال رؤى قيّمة وجّهت تطوير خوارزميات ومنهجيات منصتنا.

(ملاحظة: للحصول على توثيق اقتباس دقيق، يُرجى الرجوع مباشرةً إلى الأوراق الأصلية.)

## 🧭 ملاحظات التطوير

- أصناف النموذج الأساسية موجودة تحت `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- منطق الواجهة التفاعلية الرئيسي مضمن مباشرة في `templates/cellist.html`.
- مخطط SQL وبيانات بأسلوب seed موجودة في `cellist.sql`.
- دفاتر `notebooks/` و`polygon_sample/` توفر مراجع استكشافية.
- لا توجد حاليًا مجموعة اختبارات آلية مخصصة أو إعداد CI في جذر المستودع.

## 🧯 استكشاف الأخطاء وإصلاحها

| العرض | فحوصات مقترحة |
|---|---|
| `ModuleNotFoundError` أو مشاكل استيراد | تأكد من تنفيذ `conda activate cellist` قبل تشغيل `python app.py`. |
| تظهر الواجهة دون تنسيقات/سكربتات | شغّل `npm install` داخل `statics/` وتأكد من وجود `statics/node_modules`. |
| رفض وصول MySQL | تحقق من اسم المستخدم/كلمة المرور في `cellist/utils/constants.py` ووضع plugin/auth في MySQL. |
| يبدأ التطبيق لكن إجراءات النموذج تفشل | تحقق من توفر CUDA/GPU؛ المسارات الحالية تفترض CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| ينجح الرفع لكن لا تظهر مقاطع/نماذج | تأكد من وجود المجلدات الفرعية في `data/` وإمكانية الكتابة إليها. |

## 🗺️ خارطة الطريق

العناصر التالية محفوظة ومنظمة من وثائق المشروع الحالية/ملاحظات TODO:

- عينة polygon: استخدام polygon بدل التعليق التوضيحي المستطيلي.
- تحسين النموذج لسلوك `float32` مع القيم الصغيرة/الكبيرة جدًا.
- تقليل حجم النموذج قدر الإمكان.
- تحسين المتانة بمقاربات مثل مكونات مستوحاة من Transformer/stable-diffusion.
- إضافة خيارات نموذج أساسي (Threshold, Cellpose) وخيارات نموذج مستهدف (AIR, Transformer, SD).
- تحسين الواجهة (بما في ذلك التحديد المتعدد).
- تحسين الواجهة الخلفية (بما في ذلك تحسين التعامل مع الذاكرة/الذاكرة المؤقتة).
- تغليف سهل الاستخدام مع إعداد قاعدة بيانات محدود (مثل خيار SQLite).

## 🤝 المساهمة

### ساهم في Lazeal Cellist

Lazeal Cellist مشروع مفتوح المصدر، ونرحب بمساهمات الجميع بغض النظر عن مستوى الخبرة. ندعو إلى مساهمات:

- تعزز كفاءة الخوارزميات والأداء
- تحسن واجهة المستخدم وتجربة المستخدم
- توسّع التوثيق والأمثلة
- تصلح الأخطاء وترفع استقرار النظام

قبل البدء بالمساهمة، يُرجى مناقشة التغيير الذي ترغب به عبر Issue أولًا. يساعد ذلك في تنسيق الجهود وتجنب العمل المكرر أو المتعارض.

لمزيد من المعلومات حول كيفية البدء، يُرجى قراءة إرشادات المساهمة.

مستندات مساهمة إضافية في المستودع:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 الترخيص

هذا المشروع مرخّص بموجب ترخيص MIT. لمزيد من المعلومات، يُرجى الرجوع إلى ملف [LICENSE](https://chat.openai.com/LICENSE) في هذا المستودع.

ملاحظة حول حالة المستودع: لا يوجد حاليًا ملف `LICENSE` في جذر هذا الإصدار. السطر أعلاه محفوظ من README السابق بوصفه نية المشروع الأساسية؛ يمكنك إضافة ملف `LICENSE` محليًا في تغيير لاحق إذا رغبت.
