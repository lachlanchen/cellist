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

![Screenshot 2D](../screenshot2d.png)

## Lazeal Cellist: Ihre effiziente Plattform für 3D-Zellerkennung und Profiling

Willkommen bei Lazeal Cellist, einer umfassenden und effizienten Plattform zur Zellerkennung, Segmentierung und Profilierung von 3D-Mikroskopiebildern.

Unsere Plattform ist darauf ausgelegt, Zellen mithilfe von unüberwachtem Lernen, Schwellwertverfahren und modernen Algorithmen wie Cellpose zu erkennen. Lazeal Cellist bietet außerdem eine intuitive, interaktive Oberfläche, mit der Nutzer Erkennungsergebnisse verfeinern können. Diese verfeinerten Ergebnisse werden anschließend in das semisupervidierte Lernnetz zurückgeführt und verbessern die Modellleistung kontinuierlich.

Lazeal Cellist hebt sich durch ein effizientes 3D-Modell hervor, das nur minimalen Aufwand für Training und Verfeinerung erfordert und damit eine praktische Plattform für Wissenschaftler, Forschende und Hobby-Anwender ist.

---

## 🔍 Überblick

Lazeal Cellist ist eine Python/Tornado-Webplattform für Mikroskopie-Bild-Workflows mit:

- Browserbasiertem Upload, Modellerstellung und Bearbeitung von Annotationen.
- Algorithmusgestützter Initialisierung (Cellpose nuclei mode).
- Iterativer Human-in-the-loop-Verfeinerung über WebSocket-Aktionen (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Datenbankgestützter Persistenz für Modelle, Bild-Slices und Annotationen.

Hinweis zum aktuellen Verhalten: Obwohl Projektvision und UI 3D-Konzepte enthalten (`/3d`, `templates/cellist_3d.html`), basiert der aktuelle Haupt-Trainingsfluss im Code primär auf 2D-Slicing plus Modellverfeinerung.

### Kurzüberblick

| Bereich | Aktuelle Implementierung |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Datenbank | MySQL (`cellist.sql`) |
| Core-ML-Stack | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inferenz-Initialisierung | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Hauptfunktionen

- **Unüberwachte 3D-Zellerkennung**: Zellen in 3D-Mikroskopiebildern mit fortschrittlichen Machine-Learning-Techniken identifizieren.
- **Interaktive Oberfläche zur Ergebnisverfeinerung**: Erkennungsergebnisse in einer intuitiven, benutzerfreundlichen Oberfläche verfeinern.
- **Effizientes semisupervidiertes Lernnetz**: Modellleistung mit verfeinerten Ergebnissen fortlaufend verbessern.
- **Zellsegmentierung und Profiling**: Über die Erkennung hinaus mit erweiterten Segmentierungs- und Profiling-Funktionen arbeiten.

Zusätzliche derzeit vorhandene Implementierungsmerkmale:

- Tornado REST + WebSocket-Server (`app.py`) auf Port `8887`.
- Automatisches Image-Tiling (`256x256` standardmäßig) zur Modellverarbeitung.
- MySQL-Schema als Dump enthalten: [`cellist.sql`](../cellist.sql).
- Frontend-Stack enthält Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Projektstruktur

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

## ✅ Voraussetzungen

| Voraussetzung | Hinweise |
|---|---|
| Betriebssystem | Linux empfohlen (die folgenden Befehle setzen Linux-Shell-Verhalten voraus). |
| Python/Conda | Conda verfügbar, um die Umgebung aus [`cellist.yaml`](../cellist.yaml) zu erstellen. |
| Datenbank | MySQL-Server läuft auf `localhost` mit der Datenbank `cellist`. |
| GPU | NVIDIA/CUDA-Umgebung wird von aktuellen Codepfaden stark empfohlen bzw. erwartet. |
| Node.js + npm | Erforderlich, um Frontend-Abhängigkeiten in `statics/node_modules` zu installieren. |

## 🛠️ Installation

### 1. Repository klonen und öffnen

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Python-Umgebung erstellen

Verwenden Sie den Repository-Dateinamen `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Kompatibilitätshinweis aus älteren Dokumenten: Frühere Dokumentation verwendete `celist.yaml` (ein fehlendes `l`), die Datei in diesem Repository heißt jedoch `cellist.yaml`.

### 3. Frontend-Abhängigkeiten installieren

```bash
cd statics
npm install
cd ..
```

### 4. MySQL-Authentifizierung vorbereiten (falls nötig)

Wenn die Root-Authentifizierung socket-basiert ist und den Zugriff der App blockiert, empfehlen ältere Projektdokumente die Umstellung auf Passwortauthentifizierung:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Datenbank erstellen und Schema/Daten wiederherstellen

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Beispiel aus Legacy-Dokumentation (beibehalten):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. MySQL-Zugangsdaten für Laufzeit konfigurieren

Der aktuelle Code liest Zugangsdaten aus [`cellist/utils/constants.py`](../cellist/utils/constants.py) (`mysqlconfig` und `mysqlurl`).

Aktuell enthaltene Standardwerte im Code:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Für lokale Sicherheit aktualisieren Sie diese Werte vor dem Einsatz in Ihrer Umgebung.

### 7. Laufzeit-Datenverzeichnisse vorbereiten

Die App erwartet eine `data/`-Struktur (und `.gitignore` schließt `data` bereits aus).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Hinweis: [`create_data_folder.py`](../create_data_folder.py) ist vorhanden, erstellt Verzeichnisse derzeit jedoch im aktuellen Arbeitsverzeichnis (nicht unter `data/`). Berücksichtigen Sie das bei der Nutzung.

## 🚀 Verwendung

### Webserver starten

```bash
python app.py
```

Legacy-Startbefehl aus früherer Dokumentation (beibehalten):

```bash
python app.py -m cellist
```

Standard-Routen des Servers laut Code:

- Haupt-UI: `http://localhost:8887/`
- 3D-Seite: `http://localhost:8887/3d`

### Typischer Workflow

1. UI öffnen und anmelden.
2. Mikroskopiebilder im Bereich „Create Model“ hochladen.
3. Basisalgorithmus (`Cellpose`) wählen und Modell erstellen.
4. Backend Bilder slicen und Erkennungen initialisieren lassen.
5. Gecroppte Bilder laden und Rechteck-Annotationen prüfen/anpassen.
6. `initialize`-, `pretrain`- und `train`-Zyklen ausführen.
7. Manuelle Aktualisierungen über `Update Model`/Annotation-Aktionen persistieren.

### Integrierte UI-Login-Zugangsdaten (aktuelles Template-Verhalten)

Das Frontend prüft derzeit diese statischen Zugangsdaten clientseitig:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Dies ist Prototyp-Verhalten und keine produktionsreife Authentifizierung.

## ⚙️ Konfiguration

### Backend und Endpunkte

Konfiguriert in [`app.py`](../app.py):

- Port: `8887`
- Routen:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Modell-/Datenverhalten

- Thread-Pool-Größe ist `max_workers=64`.
- Bild-Tiles sind standardmäßig `256x256`.
- Cellpose-Initialisierung verwendet `model_type='nuclei'` und `gpu=True`.
- Training und Pretraining laufen asynchron über WebSocket-getriggerte Aktionen.

## 🧪 Beispiele

### Beispiel: Form einer WebSocket-Create-Nachricht

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

### Beispiel: WebSocket-Update für manuelle Annotation

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Beispiel: Model-Load-Request

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

## 📚 Von Forschung inspiriert

Lazeal Cellist ist von wegweisender Forschung im Deep Learning inspiriert, darunter:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Diese Arbeiten liefern wertvolle Erkenntnisse, die die Entwicklung der Algorithmen und Methodik unserer Plattform geprägt haben.

(Hinweis: Für exakte Zitation bitte direkt auf die Originalarbeiten verweisen.)

## 🧭 Entwicklungshinweise

- Zentrale Modellklassen liegen unter `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Die Hauptlogik der interaktiven UI ist direkt in `templates/cellist.html` eingebettet.
- SQL-Schema und Seed-ähnliche Daten befinden sich in `cellist.sql`.
- Notebooks unter `notebooks/` und `polygon_sample/` dienen als explorative Referenzen.
- Im Repository-Root gibt es derzeit keine dedizierte automatisierte Test-Suite oder CI-Konfiguration.

## 🧯 Fehlerbehebung

| Symptom | Empfohlene Prüfungen |
|---|---|
| `ModuleNotFoundError` oder Importprobleme | Prüfen, ob `conda activate cellist` vor `python app.py` ausgeführt wurde. |
| UI rendert ohne Styling/Skripte | `npm install` in `statics/` ausführen und sicherstellen, dass `statics/node_modules` existiert. |
| MySQL access denied | Benutzername/Passwort in `cellist/utils/constants.py` und MySQL-Plugin/Auth-Modus prüfen. |
| App startet, aber Modellaktionen schlagen fehl | CUDA/GPU-Verfügbarkeit prüfen; aktuelle Pfade setzen CUDA voraus (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload erfolgreich, aber keine Tiles/Modelle sichtbar | Sicherstellen, dass `data/`-Unterverzeichnisse existieren und beschreibbar sind. |

## 🗺️ Roadmap

Die folgenden Punkte sind aus bestehender Projektdokumentation/TODO-Notizen übernommen und neu geordnet:

- Polygon sample: Polygon- statt Rechteck-Annotation verwenden.
- Modell für `float32`-Verhalten bei sehr kleinen/großen Werten optimieren.
- Modellgröße, wo möglich, reduzieren.
- Robustheit mit Ansätzen wie Transformer-/stable-diffusion-inspirierten Komponenten verbessern.
- Basis-Modelloptionen (Threshold, Cellpose) und Ziel-Modelloptionen (AIR, Transformer, SD) ergänzen.
- Interface-Optimierung (einschließlich Mehrfachauswahl).
- Backend-Optimierung (einschließlich verbessertem Speicher-/Cache-Handling).
- Leicht nutzbares Packaging mit minimaler DB-Konfiguration (z. B. SQLite-Option).

## 🤝 Mitwirken

### Zu Lazeal Cellist beitragen

Lazeal Cellist ist ein Open-Source-Projekt, und wir freuen uns über Beiträge von allen, unabhängig vom Erfahrungsniveau. Besonders willkommen sind Beiträge, die:

- Algorithmische Effizienz und Performance verbessern
- Benutzeroberfläche und Nutzererlebnis verbessern
- Dokumentation und Beispiele erweitern
- Bugs beheben und Systemstabilität verbessern

Bevor Sie mit einem Beitrag beginnen, besprechen Sie die gewünschte Änderung bitte zuerst über ein Issue. Das hilft bei der Koordination und vermeidet doppelte oder widersprüchliche Arbeit.

Weitere Informationen für den Einstieg finden Sie in den Contribution-Richtlinien.

Zusätzliche Contribution-Dokumente im Repository:

- [Contribution Guidelines](../CONTRIBUTING.md)
- [Pull Request Template](../PULL_REQUEST_TEMPLATE.md)

## 📄 Lizenz

Dieses Projekt ist unter der MIT License lizenziert. Weitere Informationen finden Sie in der [LICENSE](https://chat.openai.com/LICENSE)-Datei in diesem Repository.

Hinweis zum Repository-Status: In diesem Checkout ist derzeit keine `LICENSE`-Datei im Root vorhanden. Die obige Zeile wird als kanonische Projektabsicht aus der vorherigen README beibehalten; eine lokale `LICENSE`-Datei kann bei Bedarf in einem Folge-Change ergänzt werden.
