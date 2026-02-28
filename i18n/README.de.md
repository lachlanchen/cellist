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
  <img src="https://img.shields.io/badge/database-MySQL-4479A1" alt="Datenbank" />
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey" alt="Plattform" />
  <img src="https://img.shields.io/badge/UI-Bootstrap%20%2B%20jQuery-7952B3" alt="UI" />
  <img src="https://img.shields.io/badge/port-8887-success" alt="Port" />
</p>

<p align="center">
  <a href="#-overview"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Übersicht" /></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-usage"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Nutzung" /></a>
  <a href="#-troubleshooting"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Fehlerbehebung" /></a>
  <a href="#-contributing"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Mitwirken" /></a>
</p>

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Ihre effiziente Plattform für 3D-Zellerkennung und -profilierung

Willkommen bei Lazeal Cellist, einer umfassenden und effizienten Plattform zur Zellerkennung, Segmentierung und Profilierung in 3D-Mikroskopieaufnahmen.

Unsere Plattform ist darauf ausgelegt, Zellen mit unüberwachtem Lernen, Schwellwertverfahren und modernsten Algorithmen wie Cellpose zu erkennen. Lazeal Cellist bietet außerdem eine intuitive, interaktive Oberfläche, mit der Nutzer:innen die Erkennungsergebnisse verfeinern können. Diese verfeinerten Ergebnisse fließen anschließend in das semisupervisierte Lernnetzwerk ein und verbessern die Modellleistung kontinuierlich.

Lazeal Cellist zeichnet sich dadurch aus, dass es ein effizientes 3D-Modell mit minimalem Trainings- und Verfeinerungsaufwand bietet und damit eine praxisnahe Plattform für Wissenschaftler:innen, Forschende und Hobbyprojekte ist.

> ℹ️ **Hinweis zum Umfang**
> Die Projektvision und die Benutzeroberfläche enthalten 3D-Konzepte (`/3d`, `templates/cellist_3d.html`), während der aktuelle primäre Trainingsfluss im Code überwiegend auf 2D-Slicing und Verfeinerung basiert.

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
- [Durch Forschung inspiriert](#-inspired-by-research)
- [Entwicklungsnotizen](#-development-notes)
- [Fehlerbehebung](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Mitwirken](#-contributing)
- [Danksagungen](#-acknowledgements)
- [Support](#-support)
- [Lizenz](#-license)

## 🔍 Overview

Lazeal Cellist ist eine Python/Tornado-Webplattform für Workflows mit Mikroskopie-Bildern mit:

- browserbasierter Upload, Modellerstellung und Bearbeitung von Annotationen.
- Algorithmisch unterstützte Initialisierung (Cellpose-Kernmodus).
- Iterative Mensch-im-Kreis-Verfeinerung über WebSocket-Aktionen (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Datenbankgestützte Persistenz für Modelle, Bildschnitte und Annotationen.

> ℹ️ Hinweis zum aktuellen Verhalten: Obwohl Projektvision und UI 3D-Konzepte enthalten (`/3d`, `templates/cellist_3d.html`), basiert der aktuelle Haupt-Trainingsfluss im Code überwiegend auf 2D-Slicing + Modellverfeinerung.

### Quick At-a-Glance

| Bereich | Aktuelle Implementierung |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Datenbank | MySQL (`cellist.sql`) |
| Kern-Machine-Learning-Stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inferenz-Initialisierung | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Packaging-Status | Forschungsprototyp (kein `pyproject.toml`/`setup.py`) |
| Test/CI-Status | Keine dedizierte automatisierte Test-Suite oder CI-Konfiguration im Repository-Root |

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
| Chinesisch (vereinfacht) | `README.zh-Hans.md` |
| Chinesisch (traditionell) | `README.zh-Hant.md` |

## ✨ Key Features

- **Unüberwachtes 3D-Zellendetektion**: Erkennen von Zellen in 3D-Mikroskopiebildern mit fortschrittlichen Machine-Learning-Techniken.
- **Interaktive Oberfläche zur Ergebnisverfeinerung**: Verfeinern Sie Erkennungsergebnisse mit einer intuitiven, benutzerfreundlichen Benutzeroberfläche.
- **Effizientes semisupervisiertes Lernnetzwerk**: Verbessern Sie die Modellleistung im Laufe der Zeit durch verfeinerte Ergebnisse.
- **Zellsegmentierung und -profilierung**: Gehen Sie über die reine Erkennung hinaus mit fortgeschrittener Segmentierungs- und Profilierungsfunktionalität.

Weitere aktuell vorhandene Implementierungsfunktionen:

- Tornado REST + WebSocket Server (`app.py`) auf Port `8887`.
- Automatisches Image-Tiling (`256x256` standardmäßig) für die Modelleinlesung.
- MySQL-Schema als Dump enthalten: [`cellist.sql`](cellist.sql).
- Frontend-Stack umfasst Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Asynchrone Modellaufgaben über Thread-Pool (`max_workers=64`).

## 🗂️ Project Structure

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

| Anforderung | Hinweise |
|---|---|
| Betriebssystem | Linux empfohlen (die folgenden Befehle gehen von Linux-Shell-Verhalten aus). |
| Python/Conda | Conda muss verfügbar sein, um die Umgebung aus [`cellist.yaml`](cellist.yaml) zu erstellen. |
| Datenbank | MySQL-Server läuft lokal auf `localhost` mit der Datenbank `cellist`. |
| GPU | NVIDIA/CUDA-Umgebung wird von aktuellen Codepfaden stark empfohlen/erwartet. |
| Node.js + npm | Erforderlich zur Installation der Frontend-Abhängigkeiten unter `statics/node_modules`. |
| Schreibzugriff auf Festplatte | Erforderlich für Laufzeitdaten unter `<repo>/data`. |

## 🛠️ Installation

### 1. Repository klonen und öffnen

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

Kompatibilitäts-Hinweis aus älteren Dokumenten: Die frühere Dokumentation verwendete `celist.yaml` (ohne `l`), aber die Datei in diesem Repository ist `cellist.yaml`.

Legacy-Befehl (erhalten):

```bash
conda env create -f celist.yaml
```

### 3. Frontend-Abhängigkeiten installieren

```bash
cd statics
npm install
cd ..
```

### 4. Laufzeitdaten-Verzeichnisse vorbereiten

Die App erwartet einen `data/`-Baum (und `.gitignore` schließt bereits `data` aus).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Hinweis: [`create_data_folder.py`](create_data_folder.py) existiert, erstellt aber derzeit Verzeichnisse im aktuellen Arbeitsverzeichnis (nicht unter `data/`). Berücksichtigen Sie das bei der Nutzung.

### 5. MySQL-Authentifizierung vorbereiten (falls erforderlich)

Wenn die Root-Authentifizierung socketbasiert ist und den App-Zugriff blockiert, empfiehlt ältere Projektdokumentation, auf Passwortauthentifizierung umzuschalten:

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

Legacy-Dokumentationsbeispiel (erhalten):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. MySQL-Anmeldeinformationen für die Laufzeit konfigurieren

Der aktuelle Code liest Anmeldeinformationen aus [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` und `mysqlurl`).

Aktuelle Standardwerte sind derzeit:

- Host: `localhost`
- Benutzer: `root`
- Passwort: `lazeal0626`

Aktualisieren Sie diese Werte aus Sicherheitsgründen vor dem Einsatz in Ihrer Umgebung.

### 8. Optionale Umgebungssanity-Checks

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Usage

### Starten des Webservers

```bash
python app.py
```

Legacy startup-Befehl aus älteren Dokumenten (erhalten):

```bash
python app.py -m cellist
```

Server-Standardrouten im Code:

- Haupt-UI: `http://localhost:8887/`
- 3D-Seite: `http://localhost:8887/3d`

### Typischer Arbeitsablauf

1. Öffnen Sie die Benutzeroberfläche und melden Sie sich an.
2. Laden Sie Mikroskopiebilder aus dem Bereich "Modell erstellen" hoch.
3. Wählen Sie den Basisalgorithmus (`Cellpose`) und erstellen Sie das Modell.
4. Lassen Sie das Backend die Bilder in Scheiben schneiden und Initialerkennungen durchführen.
5. Laden Sie zugeschnittene Bilder, prüfen und passen Sie Rechteckannotation an.
6. Führen Sie `initialize`, `pretrain` und `train`-Zyklen aus.
7. Nutzen Sie `Pretrain Stop` / `Stop` (`train-stop`) / `reset`, wenn nötig.
8. Speichern Sie manuelle Updates über `Update Model`/Annotation-Aktionen.

### Eingebaute Login-Zugangsdaten der UI (aktuelles Template-Verhalten)

Das Frontend prüft aktuell diese statischen Zugangsdaten clientseitig:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Dies ist prototypisches Verhalten und keine Produktionsauthentifizierung.

### Von der UI aktuell genutzte API/WebSocket-Schnittstelle

HTTP-Endpunkte:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

WebSocket-Endpunkt:

- `ws://localhost:8887/websocket/<ws_uuid>`

In WebSocket-Handlern erkannte `data_type`-Aktionsnachrichten:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuration

### Backend und Endpunkte

Konfiguriert in [`app.py`](app.py):

- Port: `8887`
- Routen:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Modell-/Datenverhalten

- Thread-Pool-Größe ist `max_workers=64`.
- Bildkacheln haben standardmäßig `256x256`.
- Cellpose-Initialisierung nutzt `model_type='nuclei'` und `gpu=True`.
- Training und Pretraining laufen asynchron über WebSocket-Trigger-Aktionen.
- Datenwurzel wird aus dem aktuellen Arbeitsverzeichnis als `<repo>/data` aufgelöst.

### Datenbank-/Laufzeitkonstanten

Aus [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` enthält die Schlüssel host/user/password
- `mysqlurl` zielt auf Datenbankname `cellist`

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
- CUDA Toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Examples

### Beispiel: WebSocket create-Nachrichtenstruktur

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

### Beispiel: Manuelle Aktualisierung einer WebSocket-Annotation

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Beispiel: Modellladeanfrage

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Beispiel: Minimaler lokaler Startablauf

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

Lazeal Cellist wurde durch hochmoderne Forschung im Deep Learning inspiriert, unter anderem durch:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Diese Arbeiten liefern wichtige Einblicke, die die Entwicklung der Algorithmen und Methoden unserer Plattform maßgeblich geprägt haben.

(Hinweis: Für eine korrekte Zitierung wenden Sie sich bitte direkt an die Originalarbeiten.)

## 🧭 Development Notes

- Kernmodell-Klassen befinden sich unter `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Die Hauptlogik der interaktiven UI steckt direkt in `templates/cellist.html`.
- SQL-Schema und Seed-Daten befinden sich in `cellist.sql`.
- Notebooks in `notebooks/` und `polygon_sample/` bieten explorative Referenzen.
- Erweiterte Plattform-/Modellnotizen finden Sie in [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Im Repository-Root existiert derzeit keine dedizierte automatisierte Test-Suite oder CI-Konfiguration.

### Annahmen und aktuelle Einschränkungen

- Dieses Repository richtet sich offenbar zunächst auf lokale, forschungsorientierte Nutzung.
- Einige Codepfade gehen von GPU-Verfügbarkeit (`cuda:0`) aus.
- Authentifizierung und Geheimnisverwaltung befinden sich auf Prototyp-Niveau.
- 3D-Oberflächen sind vorhanden, der dominante Trainingsworkflow bleibt jedoch 2D-orientiert und kachelbasiert.

## 🧯 Troubleshooting

| Symptom | Empfohlene Überprüfung |
|---|---|
| `ModuleNotFoundError` oder Importprobleme | Stellen Sie sicher, dass `conda activate cellist` vor dem Start von `python app.py` aktiv ist. |
| UI ohne Styling/Skripte | Führen Sie `npm install` in `statics/` aus und prüfen Sie, dass `statics/node_modules` existiert. |
| MySQL-Zugriff verweigert | Überprüfen Sie den Benutzernamen/Passwort in `cellist/utils/constants.py` und den MySQL-Plugin-/Auth-Modus. |
| App startet, aber Modellaktionen schlagen fehl | Prüfen Sie CUDA/GPU-Verfügbarkeit; aktuelle Pfade gehen von CUDA aus (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload erfolgreich, aber keine Kacheln/Modelle sichtbar | Stellen Sie sicher, dass die Unterverzeichnisse unter `data/` existieren und beschreibbar sind. |
| REST/WebSocket-Anfragefehler | Prüfen Sie, ob der Server auf `http://localhost:8887` läuft und die Payload-Schlüssel den aktuellen Template-Namen entsprechen. |
| `FileNotFoundError` unter `data/` | Starten Sie die App vom Repository-Root aus, damit relative Pfade konsistent aufgelöst werden. |

### Schnelle Diagnostik

```bash
# Python-Umgebung und Schlüsselimporte prüfen
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Sicherstellen, dass Serverport nach dem Start offen ist
ss -ltnp | rg 8887

# MySQL-Konnektivität prüfen
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Roadmap

Die folgenden Punkte wurden aus bestehender Projektdokumentation/TODO-Notizen übernommen und organisiert:

- Polygon Sample: Verwenden Sie Polygonen statt Rechteckannotation.
- Modell für `float32`-Verhalten bei sehr kleinen/großen Werten optimieren.
- Modellgröße wenn möglich verkleinern.
- Robustheit mit Ansätzen wie Transformer-/Stable-Diffusion-inspirierten Komponenten verbessern.
- Basis-Modelloptionen ergänzen (Threshold, Cellpose) und Zielmodell-Optionen (AIR, Transformer, SD).
- Interface-Optimierung (inkl. Mehrfachauswahl).
- Backend-Optimierung (inkl. verbessertem Speicher-/Cache-Handling).
- Einfache Verpackung mit minimaler DB-Konfiguration (z. B. SQLite-Option).

## 🤝 Contributing

### Zu Lazeal Cellist beitragen

Lazeal Cellist ist ein Open-Source-Projekt, und wir freuen uns über Beiträge von allen, unabhängig vom Erfahrungsstand. Wir begrüßen Beiträge, die:

- die algorithmische Effizienz und Leistung verbessern
- die Benutzeroberfläche und Nutzererfahrung verbessern
- die Dokumentation und Beispiele erweitern
- Fehler beheben und die Systemstabilität erhöhen

Bevor Sie mit einem Beitrag beginnen, besprechen Sie die gewünschte Änderung bitte zuerst über ein Issue. Das hilft, Doppelarbeit zu vermeiden und Konflikte zu reduzieren.

Weitere Informationen zum Einstieg finden Sie in den Beitragshinweisen.

Zusätzliche Dokumentation für Beiträge:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Acknowledgements

- Das Konzept und die Umsetzung von Lazeal Cellist beziehen sich stark auf die oben genannten AIR/SPAIR-Forschungsansätze.
- Das Repository enthält historische/Legacy-Dokumentation und Befehle, die bewusst für die Kontinuität mit früherer Projektnutzung beibehalten wurden.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der Datei [LICENSE](LICENSE) in diesem Repository.

Hinweis zum Repository-Status: In diesem Checkout ist derzeit keine `LICENSE`-Datei im Root vorhanden. Die obige Zeile ist aus der bisherigen README übernommen; fügen Sie in einem Folgeschritt eine lokale `LICENSE`-Datei hinzu, falls gewünscht.
