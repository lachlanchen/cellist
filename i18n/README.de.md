[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Ihre effiziente Plattform für 3D-Zellerkennung und -profilierung</strong>
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

## 🎬 Vorschau

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Ihre effiziente Plattform für 3D-Zellerkennung und -profilierung

Willkommen bei Lazeal Cellist, einer umfassenden und effizienten Plattform zur Zellenerkennung, Segmentierung und Profilierung in 3D-Mikroskopie-Aufnahmen.

Die Plattform wurde entwickelt, um Zellen mithilfe unüberwachter Methoden, Thresholding-Ansätzen und modernster Algorithmen wie Cellpose zu erkennen. Lazeal Cellist bietet zudem eine intuitive, interaktive Oberfläche, mit der Nutzende die Erkennungsergebnisse nachbearbeiten können. Diese verfeinerten Ergebnisse werden anschließend in ein semi-überwachtes Lernnetzwerk zurückgeführt, wodurch die Modellleistung kontinuierlich verbessert wird.

Lazeal Cellist zeichnet sich dadurch aus, dass es ein effizientes 3D-Modell mit geringem Aufwand für Training und Verfeinerung bietet und damit eine praxistaugliche Plattform für Wissenschaftler:innen, Forschende und Hobbyanwender ist.

> ℹ️ **Hinweis zum Umfang**
> Die Projektvision und die Benutzeroberfläche enthalten 3D-Konzepte (`/3d`, `templates/cellist_3d.html`), während der derzeitige primäre Trainingsfluss im Code überwiegend aus 2D-Slicing + Verfeinerung besteht.

---

## Inhaltsverzeichnis

- [Überblick](#-overview)
- [Kernfunktionen](#-key-features)
- [Projektstruktur](#-project-structure)
- [Voraussetzungen](#-prerequisites)
- [Installation](#-installation)
- [Verwendung](#-usage)
- [Konfiguration](#-configuration)
- [Beispiele](#-examples)
- [Von Forschung inspiriert](#-inspired-by-research)
- [Entwicklungsnotizen](#-development-notes)
- [Fehlerbehebung](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Mitmachen](#-contributing)
- [Danksagung](#-acknowledgements)
- [Support](#-support)
- [Lizenz](#-license)

## 🔍 Überblick

Lazeal Cellist ist eine Python/Tornado-Webplattform für Workflows mit Mikroskopiebildern:

- Browser-basierter Upload, Modellerstellung und Annotationen-Editing.
- Algorithmus-gestützte Initialisierung (Cellpose-Nuclei-Modus).
- Iterative Human-in-the-Loop-Verfeinerung über WebSocket-Aktionen (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Datenbankgestützte Persistenz für Modelle, Bildschnitte und Annotationen.

> ℹ️ Hinweis zum aktuellen Verhalten: Obwohl Projektvision und UI 3D-Konzepte enthalten (`/3d`, `templates/cellist_3d.html`), basiert der aktuelle primäre Trainingsfluss im Code überwiegend auf 2D-Slicing + Modellverfeinerung.

### Schnellüberblick

| Bereich | Aktuelle Implementierung |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Datenbank | MySQL (`cellist.sql`) |
| ML-Core-Stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inferenz-Initialisierung | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Packaging-Status | Forschungsprototyp (kein `pyproject.toml`/`setup.py`) |
| Tests/CI-Status | Keine dedizierte automatisierte Test-Suite oder CI-Konfiguration im Repository-Root |

### Dokumentationssprachen

Dieses Repository enthält bereits mehrsprachige README-Dateien unter `i18n/`:

| Sprache | Datei |
|---|---|
| Arabisch | `README.ar.md` |
| Spanisch | `README.es.md` |
| Französisch | `README.fr.md` |
| Japanisch | `README.ja.md` |
| Koreanisch | `README.ko.md` |
| Russisch | `README.ru.md` |
| Vietnamesisch | `README.vi.md` |
| Chinesisch (Vereinfacht) | `README.zh-Hans.md` |
| Chinesisch (Traditionell) | `README.zh-Hant.md` |

## ✨ Schlüsselmerkmale

- **Unüberwachte 3D-Zellerkennung**: Identifiziert Zellen in 3D-Mikroskopie-Bildern mit fortschrittlichen Machine-Learning-Methoden.
- **Interaktive Oberfläche zur Ergebnisverfeinerung**: Verfeinert Erkennungsergebnisse mit einer intuitiven, benutzerfreundlichen Oberfläche.
- **Effizientes semi-supervised Lernnetzwerk**: Verbessert die Modellleistung über die Zeit anhand verfeinerter Ergebnisse.
- **Zellsegmentierung und -profilierung**: Geht über reine Erkennung hinaus mit fortgeschrittenen Segmentierungs- und Profilierungsfunktionen.

Weitere derzeit implementierte technische Merkmale:

- Tornado REST + WebSocket Server (`app.py`) auf Port `8887`.
- Automatische Bild-Tiling (`256x256` standardmäßig) für die Modelingabe.
- MySQL-Schema als Dump enthalten: [`cellist.sql`](cellist.sql).
- Frontend-Stack enthält Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Asynchrone Modellaufgaben über einen Thread-Pool (`max_workers=64`).

## 🗂️ Projektstruktur

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

## ✅ Voraussetzungen

| Voraussetzung | Hinweise |
|---|---|
| Betriebssystem | Linux empfohlen (nachfolgende Befehle gehen von Linux-Shell-Verhalten aus). |
| Python/Conda | Conda muss verfügbar sein, um die Umgebung aus [`cellist.yaml`](cellist.yaml) zu erstellen. |
| Datenbank | MySQL-Server läuft auf `localhost` mit der Datenbank `cellist`. |
| GPU | NVIDIA/CUDA-Umgebung wird von den aktuellen Codepfaden stark empfohlen/erwartet. |
| Node.js + npm | Erforderlich zur Installation der Frontend-Abhängigkeiten unter `statics/node_modules`. |
| Schreibzugriff auf die Festplatte | Erforderlich für Laufzeitdaten unter `<repo>/data`. |

## 🛠️ Installation

### 1. Repository klonen und betreten

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python-Umgebung erstellen

Verwenden Sie die Repository-Datei `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Kompatibilitätsnotiz aus älteren Dokumenten: Ältere Dokumentation nutzte `celist.yaml` (ohne `l`), aber die Datei in diesem Repository ist `cellist.yaml`.

Legacy-Befehl (beibehalten):

```bash
conda env create -f celist.yaml
```

### 3. Frontend-Abhängigkeiten installieren

```bash
cd statics
npm install
cd ..
```

### 4. Laufzeit-Datenverzeichnisse vorbereiten

Die App erwartet einen `data/`-Baum (und `.gitignore` schließt bereits `data` aus).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Hinweis: [`create_data_folder.py`](create_data_folder.py) existiert, erstellt die Verzeichnisse aber derzeit im aktuellen Arbeitsverzeichnis (nicht unter `data/`). Berücksichtigen Sie das bei der Nutzung.

### 5. MySQL-Authentifizierung vorbereiten (falls erforderlich)

Wenn die Root-Authentifizierung socket-basiert ist und den App-Zugriff blockiert, empfiehlt die ältere Projektdokumentation einen Wechsel auf Passwort-Authentifizierung:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Datenbank erstellen und Schema/Daten wiederherstellen

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Legacy-Dokumentationsbeispiel (beibehalten):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. MySQL-Zugangsdaten für den Laufzeitbetrieb konfigurieren

Der aktuelle Code liest Anmeldeinformationen aus [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` und `mysqlurl`).

Aktuelle Standardwerte sind derzeit:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Aus Sicherheitsgründen aktualisieren Sie diese Werte vor dem Einsatz in Ihrer Umgebung.

### 8. Optionale Umgebungs-Sanity-Checks

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Verwendung

### Webserver starten

```bash
python app.py
```

Legacy-Startbefehl aus früheren Unterlagen (beibehalten):

```bash
python app.py -m cellist
```

Server-Standardrouten im Code:

- Haupt-UI: `http://localhost:8887/`
- 3D-Seite: `http://localhost:8887/3d`

### Typischer Arbeitsablauf

1. Öffnen Sie die Oberfläche und melden Sie sich an.
2. Laden Sie Mikroskopieaufnahmen aus dem Bereich **Create Model** hoch.
3. Wählen Sie den Basisalgorithmus (`Cellpose`) und erstellen Sie das Modell.
4. Lassen Sie das Backend die Bilder in Scheiben schneiden und die Erkennung initialisieren.
5. Laden Sie zugeschnittene Bilder, prüfen und passen Sie Rechteckannotationen an.
6. Führen Sie `initialize`, `pretrain` und `train` Zyklen aus.
7. Nutzen Sie bei Bedarf `Pretrain Stop` / `Stop` (`train-stop`) / `reset`.
8. Speichern Sie manuelle Aktualisierungen über `Update Model`/Annotationsaktionen.

### Eingebaute Login-Zugangsdaten der UI (aktuelles Vorlagenverhalten)

Das Frontend prüft derzeit diese statischen Zugangsdaten clientseitig:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Dies ist Prototyp-Verhalten und keine produktionsreife Authentifizierung.

### API-/Socket-Schnittstelle, die aktuell von der UI verwendet wird

HTTP-Endpunkte:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket-Endpunkt:

- `ws://localhost:8887/websocket/<ws_uuid>`

Erkannte `data_type`-Aktionsnachrichten im WebSocket-Handler:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Konfiguration

### Backend und Endpunkte

Konfiguriert in [`app.py`](app.py):

- Port: `8887`
- Routen:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Modell- und Datenverhalten

- Die Thread-Pool-Größe beträgt `max_workers=64`.
- Bildkacheln sind standardmäßig `256x256` groß.
- Cellpose-Initialisierung nutzt `model_type='nuclei'` und `gpu=True`.
- Training und Pretraining laufen asynchron über WebSocket-gesteuerte Aktionen.
- Datenstammverzeichnis wird aus dem aktuellen Arbeitsverzeichnis als `<repo>/data` aufgelöst.

### Datenbank- und Laufzeitkonstanten

Aus [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` enthält die Schlüssel host/user/password
- `mysqlurl` zielt auf den Datenbanknamen `cellist`

### Frontend-Abhängigkeitsübersicht

Aus [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Conda-Umgebungs-Highlights

Aus [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Beispiele

### Beispiel: WebSocket-Nachrichtenformat für `create`

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

### Beispiel: Manuelle Annotationsaktualisierung über WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Beispiel: Anfrage zum Laden eines Modells

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Beispiel: Minimaler lokaler End-to-End-Start

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 Von Forschung inspiriert

Lazeal Cellist lässt sich durch hochmoderne Forschung im Deep Learning inspirieren, darunter:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Diese Arbeiten liefern wichtige Einblicke, die die Entwicklung der Algorithmen und Methoden unserer Plattform geprägt haben.

(Hinweis: Für eine genaue Zitierung konsultieren Sie bitte direkt die Originalarbeiten.)

## 🧭 Entwicklungsnotizen

- Die Hauptmodellklassen befinden sich unter `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Die zentrale interaktive Logik liegt direkt in `templates/cellist.html`.
- SQL-Schema und Seed-Daten befinden sich in `cellist.sql`.
- Notebooks in `notebooks/` und `polygon_sample/` liefern explorative Referenzen.
- Erweiterte Plattform-/Modellnotizen stehen in [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Im Repository-Root gibt es derzeit keine dedizierte automatisierte Test-Suite oder CI-Konfiguration.

### Annahmen und aktuelle Einschränkungen

- Dieses Repository ist anscheinend primär für lokale, forschungsorientierte Nutzung gedacht.
- Einige Codepfade gehen von GPU-Verfügbarkeit (`cuda:0`) aus.
- Authentifizierung und Secret-Management befinden sich auf Prototyp-Niveau.
- 3D-Oberflächen sind vorhanden, aber der dominante Trainingsworkflow bleibt 2D- und tile-orientiert.

## 🧯 Fehlerbehebung

| Symptom | Empfohlene Prüfung |
|---|---|
| `ModuleNotFoundError` oder Importprobleme | Stellen Sie sicher, dass `conda activate cellist` vor `python app.py` aktiviert wurde. |
| UI rendert ohne Styling/Skripte | Führen Sie `npm install` in `statics/` aus und prüfen Sie, dass `statics/node_modules` existiert. |
| MySQL-Zugriff verweigert | Prüfen Sie Benutzername/Passwort in `cellist/utils/constants.py` und das MySQL-Plugin/Auth-Modus. |
| App startet, aber Modellaktionen schlagen fehl | Prüfen Sie CUDA/GPU-Verfügbarkeit; aktuelle Pfade setzen CUDA voraus (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload gelingt, aber keine Kacheln/Modelle sichtbar | Stellen Sie sicher, dass die Unterverzeichnisse unter `data/` existieren und schreibbar sind. |
| REST/WebSocket-Fehler | Prüfen Sie, ob der Server auf `http://localhost:8887` läuft und Payload-Schlüssel zu aktuellen Vorlagennamen passen. |
| `FileNotFoundError` unter `data/` | Starten Sie die App vom Repository-Root, damit relative Pfade konsistent aufgelöst werden. |

### Kurz-Diagnose

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Roadmap

Die folgenden Punkte werden aus bestehender Projektdokumentation/TODO-Notizen übernommen und geordnet:

- Polygon sample: Nutzung von Polygon statt Rechteckannotation.
- Modell für `float32`-Verhalten mit sehr kleinen/großen Werten optimieren.
- Modellgröße, wo möglich, verkleinern.
- Robustheit mit Ansätzen wie Transformer-/Stable-Diffusion-inspirierten Komponenten verbessern.
- Basismodell-Optionen (Threshold, Cellpose) und Zielmodell-Optionen (AIR, Transformer, SD) ergänzen.
- Interface-Optimierung (inklusive Mehrfachauswahl).
- Backend-Optimierung (inklusive besserem Speicher-/Cache-Handling).
- Einfaches Packaging mit minimaler DB-Konfiguration (z. B. SQLite-Option).

## 🤝 Mitwirken

### Zu Lazeal Cellist beitragen

Lazeal Cellist ist ein Open-Source-Projekt, und Beiträge sind willkommen, unabhängig vom Erfahrungsstand. Wir freuen uns auf Beiträge, die:

- die algorithmische Effizienz und Leistung verbessern
- die Benutzeroberfläche und Nutzererfahrung verbessern
- Dokumentation und Beispiele erweitern
- Fehler beheben und die Systemstabilität erhöhen

Bevor Sie Änderungen beginnen, besprechen Sie den gewünschten Änderungsvorschlag bitte zuerst über ein Issue. Das hilft, Doppelarbeit zu vermeiden und Konflikte zu reduzieren.

Weitere Informationen zum Einstieg finden Sie in den Beitragsrichtlinien.

Weitere Repository-Dokumente zum Beitrag:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Danksagung

- Das Konzept und die Umsetzung von Lazeal Cellist orientieren sich stark an der oben genannten AIR/SPAIR-Forschungslinie.
- Das Repository enthält bewusst historisch/legacy Dokumentation und Befehle, die zur Kontinuität mit früherer Projektnutzung erhalten wurden.

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der Datei [LICENSE](LICENSE) in diesem Repository.

Repository-Status-Hinweis: In diesem Checkout ist zurzeit keine `LICENSE`-Datei im Root-Verzeichnis vorhanden. Die obige Zeile wurde aus der bisherigen README übernommen; fügen Sie bei Bedarf in einem Folge-Schritt eine lokale `LICENSE`-Datei hinzu.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
