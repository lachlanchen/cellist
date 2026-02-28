[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Tu plataforma eficiente de detección y perfilado celular en 3D</strong>
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
  <a href="#-descripción-general"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-instalación"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-uso"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-resolución-de-problemas"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-contribuciones"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 Preview

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Tu plataforma eficiente de detección y perfilado celular en 3D

Bienvenido a Lazeal Cellist, una plataforma completa y eficiente para detectar, segmentar y perfilar células en imágenes de microscopía 3D.

La plataforma está diseñada para detectar células mediante aprendizaje no supervisado, técnicas de umbralización y algoritmos de vanguardia como Cellpose. Lazeal Cellist también ofrece una interfaz intuitiva e interactiva que permite refinar los resultados de detección. Esos resultados refinados se retroalimentan en la red de aprendizaje semi-supervisado, mejorando continuamente el rendimiento del modelo.

Lazeal Cellist se distingue por ofrecer un modelo 3D eficiente que requiere poco esfuerzo para entrenar y refinar, lo que lo convierte en una plataforma práctica para científicas, científicos y aficionados.

> ℹ️ **Nota de alcance**
> La visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), mientras que el flujo principal de entrenamiento actual en el código es principalmente por cortes 2D + refinamiento.

---

## Índice

- [Descripción general](#-descripción-general)
- [Características clave](#-características-clave)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Configuración](#-configuración)
- [Ejemplos](#-ejemplos)
- [Inspirado por investigación](#-inspirado-por-investigación)
- [Notas de desarrollo](#-notas-de-desarrollo)
- [Resolución de problemas](#-resolución-de-problemas)
- [Hoja de ruta](#-hoja-de-ruta)
- [Contribuciones](#-contribuciones)
- [Agradecimientos](#-agradecimientos)
- [Support](#-support)
- [Licencia](#-licencia)

## 🔍 Descripción general

Lazeal Cellist es una plataforma web Python/Tornado para flujos de trabajo con imágenes de microscopía que incluye:

- Subida desde el navegador, creación de modelos y edición de anotaciones.
- Inicialización asistida por algoritmos (modo núcleos de Cellpose).
- Refinamiento iterativo con intervención humana mediante acciones de WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Persistencia con base de datos para modelos, cortes de imagen y anotaciones.

> ℹ️ Nota sobre comportamiento actual: aunque la visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), el flujo principal de entrenamiento en el código es, principalmente, cortes 2D + refinamiento del modelo.

### Resumen rápido

| Área | Implementación actual |
|---|---|
| Servidor | Tornado (`app.py`) |
| Puerto | `8887` |
| Base de datos | MySQL (`cellist.sql`) |
| Stack principal de ML | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inicialización de inferencia | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Estado de empaquetado | Prototipo de investigación (sin `pyproject.toml`/`setup.py`) |
| Estado de pruebas/CI | No existe suite automatizada dedicada ni configuración de CI en la raíz del repositorio |

### Idiomas de la documentación

Este repositorio ya incluye archivos README multilingües en `i18n/`:

| Idioma | Archivo |
|---|---|
| Árabe | `README.ar.md` |
| Alemán | `README.de.md` |
| Francés | `README.fr.md` |
| Japonés | `README.ja.md` |
| Coreano | `README.ko.md` |
| Ruso | `README.ru.md` |
| Vietnamita | `README.vi.md` |
| Chino (Simplificado) | `README.zh-Hans.md` |
| Chino (Tradicional) | `README.zh-Hant.md` |

## ✨ Características clave

- **Detección celular 3D no supervisada**: identifica células en imágenes de microscopía 3D usando técnicas avanzadas de aprendizaje automático.
- **Interfaz interactiva para refinar resultados**: mejora los resultados de detección con una interfaz intuitiva y fácil de usar.
- **Red eficiente de aprendizaje semi-supervisado**: mejora el rendimiento del modelo con el tiempo a partir de resultados refinados.
- **Segmentación y perfilado celular**: va más allá de la detección con capacidades de segmentación y perfilado avanzadas.

Características de implementación adicionales actualmente presentes:

- Servidor Tornado REST + WebSocket (`app.py`) en el puerto `8887`.
- División automática de imágenes en mosaicos (`256x256` de forma predeterminada) para la ingesta del modelo.
- Esquema de MySQL incluido como volcado: [`cellist.sql`](cellist.sql).
- El stack frontend incluye Bootstrap, jQuery, jQuery UI, Three.js y blueimp-file-upload.
- Tareas del modelo de forma asíncrona mediante un pool de hilos (`max_workers=64`).

## 🗂️ Estructura del proyecto

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

## ✅ Requisitos previos

| Requisito | Notas |
|---|---|
| SO | Se recomienda Linux (los comandos siguientes asumen el comportamiento de un shell de Linux). |
| Python/Conda | Conda debe estar disponible para crear el entorno desde [`cellist.yaml`](cellist.yaml). |
| Base de datos | Servidor MySQL ejecutándose en `localhost` con la base de datos `cellist`. |
| GPU | Se recomienda fuertemente/espera un entorno NVIDIA/CUDA según las rutas actuales del código. |
| Node.js + npm | Requerido para instalar dependencias de frontend en `statics/node_modules`. |
| Acceso de escritura en disco | Necesario para los datos en ejecución bajo `<repo>/data`. |

## 🛠️ Instalación

### 1. Clonar y entrar al repositorio

```bash
 git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Crear entorno de Python

Usa el nombre de archivo del repositorio `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Nota de compatibilidad preservada de documentación anterior: documentación previa usaba `celist.yaml` (sin la `l`), pero el archivo de este repositorio es `cellist.yaml`.

Comando legado (preservado):

```bash
conda env create -f celist.yaml
```

### 3. Instalar dependencias del frontend

```bash
cd statics
npm install
cd ..
```

### 4. Preparar directorios de datos de ejecución

La aplicación espera un árbol `data/` (y `.gitignore` ya excluye `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Nota: [`create_data_folder.py`](create_data_folder.py) existe, pero actualmente crea directorios en el directorio de trabajo actual (no dentro de `data/`). Ten esto en cuenta si lo usas.

### 5. Preparar autenticación de MySQL (si es necesario)

Si la autenticación de root se basa en socket y bloquea el acceso de la app, la documentación antigua sugiere cambiar a autenticación por contraseña:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Crear base de datos y restaurar esquema/datos

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Ejemplo de documentación heredada (preservado):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configurar credenciales de MySQL para ejecución

El código actual lee credenciales desde [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` y `mysqlurl`).

Los valores predeterminados actuales incluyen:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Por seguridad local, actualiza estos valores antes de ejecutar en tu entorno.

### 8. Comprobaciones opcionales del entorno

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Uso

### Iniciar el servidor web

```bash
python app.py
```

Comando legado de documentación anterior (preservado):

```bash
python app.py -m cellist
```

Rutas predeterminadas observadas en el código:

- UI principal: `http://localhost:8887/`
- Página 3D: `http://localhost:8887/3d`

### Flujo de trabajo típico

1. Abre la interfaz y autentica sesión.
2. Sube imágenes de microscopía desde el panel **Create Model**.
3. Elige el algoritmo base (`Cellpose`) y crea el modelo.
4. Deja que el backend corte imágenes e inicialice detecciones.
5. Carga imágenes recortadas, revisa y ajusta anotaciones de rectángulos.
6. Ejecuta ciclos de `initialize`, `pretrain` y `train`.
7. Usa `Pretrain Stop` / `Stop` (`train-stop`) / `reset` según sea necesario.
8. Guarda actualizaciones manuales mediante acciones de anotación en **Update Model**.

### Credenciales de inicio de sesión integradas en la UI (comportamiento actual de la plantilla)

El frontend actualmente comprueba estas credenciales estáticas del lado del cliente:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Este comportamiento es de prototipo y no corresponde a autenticación de producción.

### Superficie API/Socket usada actualmente por la UI

Endpoints HTTP:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

Endpoint WebSocket:

- `ws://localhost:8887/websocket/<ws_uuid>`

Mensajes de acción `data_type` reconocidos en el manejador WebSocket:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Configuración

### Backend y endpoints

Configurado en [`app.py`](app.py):

- Puerto: `8887`
- Rutas:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Comportamiento de modelo/datos

- El tamaño del pool de hilos es `max_workers=64`.
- Los mosaicos de imagen son `256x256` de forma predeterminada.
- La inicialización de Cellpose usa `model_type='nuclei'` y `gpu=True`.
- El entrenamiento y preentrenamiento se ejecutan de forma asíncrona mediante acciones disparadas por WebSocket.
- El directorio raíz de datos se resuelve desde el directorio de trabajo actual como `<repo>/data`.

### Constantes de base de datos/tiempo de ejecución

Desde [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` incluye claves de host/usuario/contraseña.
- `mysqlurl` apunta al nombre de base de datos `cellist`

### Instantánea de dependencias del frontend

Desde [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Aspectos destacados del entorno Conda

Desde [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Ejemplos

### Ejemplo: forma de mensaje WebSocket create

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

### Ejemplo: actualización manual de anotaciones por WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Ejemplo: solicitud de carga de modelo

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Ejemplo: inicio local mínimo de extremo a extremo

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 Inspirado por investigación

Lazeal Cellist se inspira en investigación de vanguardia en aprendizaje profundo, incluyendo:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Estos trabajos aportan ideas valiosas que han guiado el desarrollo de los algoritmos y metodologías de la plataforma.

(Nota: para una cita precisa, consulta directamente los artículos originales.)

## 🧭 Notas de desarrollo

- Las clases principales del modelo están bajo `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La lógica interactiva principal de la UI está embebida directamente en `templates/cellist.html`.
- El esquema SQL y datos tipo semilla están en `cellist.sql`.
- Los notebooks en `notebooks/` y `polygon_sample/` proporcionan referencias exploratorias.
- Las notas ampliadas de plataforma/modelo están en [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Actualmente no hay una suite de pruebas automatizada dedicada ni configuración de CI en la raíz del repositorio.

### Supuestos y restricciones actuales

- Este repositorio parece orientado primero a un uso local e investigativo.
- Algunas rutas de código asumen disponibilidad de GPU (`cuda:0`).
- La autenticación y gestión de secretos están a nivel de prototipo.
- Existen interfaces 3D, pero el flujo de entrenamiento dominante sigue orientado a mosaicos 2D.

## 🧯 Resolución de problemas

| Síntoma | Comprobaciones sugeridas |
|---|---|
| `ModuleNotFoundError` o problemas de importación | Confirma que ejecutaste `conda activate cellist` antes de correr `python app.py`. |
| La UI se renderiza sin estilos/scripts | Ejecuta `npm install` dentro de `statics/` y confirma que existe `statics/node_modules`. |
| Acceso denegado en MySQL | Verifica usuario y contraseña en `cellist/utils/constants.py` y el modo de plugin/autenticación de MySQL. |
| La app inicia pero fallan acciones del modelo | Revisa la disponibilidad de CUDA/GPU; las rutas actuales asumen CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| La carga se completa pero no aparecen mosaicos/modelos | Asegura que los subdirectorios de `data/` existan y sean escribibles. |
| Errores en solicitudes REST/WebSocket | Confirma que el servidor está en `http://localhost:8887` y que las claves del payload coinciden con los nombres actuales de la plantilla. |
| `FileNotFoundError` bajo `data/` | Inicia la app desde la raíz del repositorio para que las rutas relativas se resuelvan de forma consistente. |

### Diagnósticos rápidos

```bash
# Verify Python environment and key imports
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirm server port is open after startup
ss -ltnp | rg 8887

# Check MySQL connectivity
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Hoja de ruta

Se preservan y organizan estos elementos desde la documentación/TODOs del proyecto:

- Polygon sample: usar anotación poligonal en lugar de rectángular.
- Optimizar el modelo para el comportamiento `float32` con valores muy pequeños o muy grandes.
- Reducir el tamaño del modelo cuando sea posible.
- Mejorar la robustez con enfoques como componentes inspirados en Transformer/stable-diffusion.
- Añadir opciones de modelo base (Threshold, Cellpose) y opciones de modelo objetivo (AIR, Transformer, SD).
- Optimización de interfaz (incluida selección múltiple).
- Optimización del backend (incluida mejor gestión de memoria/caché).
- Empaquetado fácil de usar con configuración mínima de base de datos (por ejemplo, opción SQLite).

## 🤝 Contribuciones

### Contribuye a Lazeal Cellist

Lazeal Cellist es un proyecto de código abierto y damos la bienvenida a aportes de cualquier persona, sin importar su nivel de experiencia. Invitamos aportes que:

- Mejoren la eficiencia y el rendimiento de los algoritmos.
- Mejoren la interfaz y la experiencia de usuario.
- Amplíen la documentación y los ejemplos.
- Corrijan errores y mejoren la estabilidad del sistema.

Antes de comenzar a contribuir, comenta primero el cambio que deseas hacer mediante un issue. Esto ayuda a coordinar esfuerzos y evitar trabajo duplicado o en conflicto.

Para más información sobre cómo empezar, consulta las guías de contribución.

Documentación adicional de contribución del repositorio:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Agradecimientos

- El concepto y la implementación de Lazeal Cellist se basan fuertemente en la línea de investigación AIR/SPAIR descrita arriba.
- El repositorio incluye documentación y comandos históricos/legados conservados intencionalmente para mantener continuidad con el uso previo del proyecto.

## 📄 Licencia

Este proyecto está licenciado bajo la licencia MIT. Para más información, consulta el archivo [LICENSE](LICENSE) en este repositorio.

Nota sobre estado del repositorio: actualmente no existe un archivo `LICENSE` en este checkout. La línea anterior se conserva desde el README original como intención canónica del proyecto; añade un archivo `LICENSE` en un cambio posterior si lo deseas.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
