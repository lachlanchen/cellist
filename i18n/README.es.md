[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Tu plataforma eficiente para detección y perfilado celular 3D</strong>
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

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Tu plataforma eficiente para detección y perfilado celular 3D

Te damos la bienvenida a Lazeal Cellist, una plataforma integral y eficiente de detección, segmentación y perfilado celular para imágenes de microscopía 3D.

Nuestra plataforma está diseñada para detectar células mediante aprendizaje no supervisado, técnicas de umbralización y algoritmos de vanguardia como Cellpose. Lazeal Cellist también ofrece una interfaz intuitiva e interactiva que permite refinar los resultados de detección. Esos resultados refinados se retroalimentan en la red de aprendizaje semisupervisado, mejorando continuamente el rendimiento del modelo.

Lazeal Cellist destaca por ofrecer un modelo 3D eficiente que requiere un esfuerzo mínimo para entrenarse y refinarse, lo que la convierte en una plataforma práctica para científicas/os, investigadoras/es y entusiastas.

> ℹ️ **Nota de alcance**
> La visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), mientras que el flujo principal de entrenamiento actual en el código es, sobre todo, de cortes 2D + refinamiento.

---

## Tabla de contenidos

- [Descripción general](#-descripción-general)
- [Características clave](#-características-clave)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Configuración](#-configuración)
- [Ejemplos](#-ejemplos)
- [Inspirado en investigación](#-inspirado-en-investigación)
- [Notas de desarrollo](#-notas-de-desarrollo)
- [Resolución de problemas](#-resolución-de-problemas)
- [Hoja de ruta](#-hoja-de-ruta)
- [Contribuciones](#-contribuciones)
- [Agradecimientos](#-agradecimientos)
- [Support](#-support)
- [Licencia](#-licencia)

## 🔍 Descripción general

Lazeal Cellist es una plataforma web Python/Tornado para flujos de trabajo con imágenes de microscopía que incluye:

- Carga desde navegador, creación de modelos y edición de anotaciones.
- Inicialización asistida por algoritmos (modo núcleos de Cellpose).
- Refinamiento iterativo con intervención humana mediante acciones WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Persistencia con base de datos para modelos, cortes de imagen y anotaciones.

> ℹ️ Nota sobre el comportamiento actual: aunque la visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), el flujo principal de entrenamiento en el código es, principalmente, de cortes 2D + refinamiento de modelo.

### Vista rápida

| Área | Implementación actual |
|---|---|
| Servidor | Tornado (`app.py`) |
| Puerto | `8887` |
| Base de datos | MySQL (`cellist.sql`) |
| Stack principal de ML | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inicialización de inferencia | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Estado de empaquetado | Prototipo de investigación (sin `pyproject.toml`/`setup.py`) |
| Estado de pruebas/CI | Sin suite de pruebas automatizadas dedicada ni configuración CI en la raíz del repositorio |

### Idiomas de la documentación

Este repositorio ya incluye archivos README multilingües bajo `i18n/`:

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
- **Interfaz interactiva para refinamiento de resultados**: refina los resultados de detección con una interfaz intuitiva y fácil de usar.
- **Red de aprendizaje semisupervisado eficiente**: mejora el rendimiento del modelo con el tiempo gracias a resultados refinados.
- **Segmentación y perfilado celular**: ve más allá de la detección con capacidades avanzadas de segmentación y perfilado.

Características de implementación adicionales presentes actualmente:

- Servidor Tornado REST + WebSocket (`app.py`) en el puerto `8887`.
- División automática de imágenes en mosaicos (`256x256` por defecto) para la ingesta del modelo.
- Esquema MySQL incluido como volcado: [`cellist.sql`](cellist.sql).
- El stack frontend incluye Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Tareas asíncronas del modelo mediante un pool de hilos (`max_workers=64`).

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
| SO | Se recomienda Linux (los comandos de abajo asumen el comportamiento del shell de Linux). |
| Python/Conda | Conda disponible para crear el entorno desde [`cellist.yaml`](cellist.yaml). |
| Base de datos | Servidor MySQL en ejecución en `localhost` con base de datos `cellist`. |
| GPU | Entorno NVIDIA/CUDA fuertemente recomendado/esperado por las rutas actuales del código. |
| Node.js + npm | Requerido para instalar dependencias frontend en `statics/node_modules`. |
| Permisos de escritura en disco | Necesarios para datos en tiempo de ejecución bajo `<repo>/data`. |

## 🛠️ Instalación

### 1. Clonar y entrar al repositorio

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Crear el entorno de Python

Usa el nombre de archivo del repositorio `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Nota de compatibilidad preservada de documentación anterior: documentación previa usaba `celist.yaml` (falta una `l`), pero el archivo en este repositorio es `cellist.yaml`.

Comando heredado (preservado):

```bash
conda env create -f celist.yaml
```

### 3. Instalar dependencias del frontend

```bash
cd statics
npm install
cd ..
```

### 4. Preparar directorios de datos en tiempo de ejecución

La app espera un árbol `data/` (y `.gitignore` ya excluye `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Nota: existe [`create_data_folder.py`](create_data_folder.py), pero actualmente crea directorios en el directorio de trabajo actual (no bajo `data/`). Tenlo en cuenta si lo usas.

### 5. Preparar autenticación de MySQL (si es necesario)

Si la autenticación de root está basada en socket y bloquea el acceso de la app, la documentación antigua del proyecto sugiere cambiar a autenticación por contraseña:

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

### 7. Configurar credenciales MySQL para ejecución

El código actual lee credenciales desde [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` y `mysqlurl`).

Los valores predeterminados del código incluyen actualmente:

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

Comando de inicio heredado de documentación anterior (preservado):

```bash
python app.py -m cellist
```

Rutas predeterminadas del servidor observadas en el código:

- UI principal: `http://localhost:8887/`
- Página 3D: `http://localhost:8887/3d`

### Flujo de trabajo típico

1. Abre la UI e inicia sesión.
2. Sube imágenes de microscopía desde el panel Create Model.
3. Elige el algoritmo base (`Cellpose`) y crea el modelo.
4. Deja que el backend corte imágenes e inicialice detecciones.
5. Carga imágenes recortadas, revisa/ajusta anotaciones de rectángulos.
6. Ejecuta ciclos de `initialize`, `pretrain` y `train`.
7. Usa `Pretrain Stop` / `Stop` (`train-stop`) / `reset` cuando sea necesario.
8. Guarda actualizaciones manuales mediante acciones de anotación/`Update Model`.

### Credenciales de inicio de sesión integradas en la UI (comportamiento actual de la plantilla)

Actualmente el frontend valida estas credenciales estáticas del lado cliente:

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

Mensajes de acción `data_type` reconocidos en el controlador WebSocket:

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
- Los mosaicos de imagen son `256x256` por defecto.
- La inicialización con Cellpose usa `model_type='nuclei'` y `gpu=True`.
- El entrenamiento y el preentrenamiento se ejecutan de forma asíncrona mediante acciones disparadas por WebSocket.
- La raíz de datos se resuelve desde el directorio de trabajo actual como `<repo>/data`.

### Constantes de base de datos/ejecución

Desde [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` incluye claves host/user/password
- `mysqlurl` apunta a la base de datos `cellist`

### Instantánea de dependencias frontend

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

### Ejemplo: estructura del mensaje WebSocket create

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

## 📚 Inspirado en investigación

Lazeal Cellist está inspirado en investigación de vanguardia en aprendizaje profundo, incluyendo:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Estos trabajos aportan ideas valiosas que han guiado el desarrollo de los algoritmos y metodologías de la plataforma.

(Nota: para una cita precisa, consulta directamente los artículos originales.)

## 🧭 Notas de desarrollo

- Las clases principales del modelo están en `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La lógica principal de la UI interactiva está embebida directamente en `templates/cellist.html`.
- El esquema SQL y los datos tipo semilla están en `cellist.sql`.
- Los cuadernos en `notebooks/` y `polygon_sample/` ofrecen referencias exploratorias.
- Las notas ampliadas de plataforma/modelo están en [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Actualmente no hay una suite de pruebas automatizadas dedicada ni configuración CI en la raíz del repositorio.

### Supuestos y restricciones actuales

- Este repositorio parece orientado primero a uso local y de investigación.
- Algunas rutas de código asumen disponibilidad de GPU (`cuda:0`).
- La autenticación y la gestión de secretos están a nivel de prototipo.
- Existen interfaces 3D, pero el flujo de entrenamiento dominante sigue orientado a mosaicos 2D.

## 🧯 Resolución de problemas

| Síntoma | Comprobaciones sugeridas |
|---|---|
| `ModuleNotFoundError` o problemas de importación | Confirma que `conda activate cellist` se haya aplicado antes de ejecutar `python app.py`. |
| La UI se renderiza sin estilos/scripts | Ejecuta `npm install` dentro de `statics/` y confirma que existe `statics/node_modules`. |
| MySQL access denied | Verifica usuario/contraseña en `cellist/utils/constants.py` y el modo de plugin/autenticación de MySQL. |
| La app inicia pero fallan acciones del modelo | Comprueba la disponibilidad de CUDA/GPU; las rutas actuales asumen CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| La carga funciona pero no aparecen mosaicos/modelos | Asegura que los subdirectorios de `data/` existan y tengan permisos de escritura. |
| Errores en solicitudes REST/WebSocket | Confirma que el servidor corre en `http://localhost:8887` y que las claves del payload coinciden con los nombres actuales de la plantilla. |
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

Los siguientes elementos se preservan y organizan a partir de la documentación/notas TODO existentes del proyecto:

- Polygon sample: usar anotación poligonal en lugar de rectángulos.
- Optimizar el modelo para comportamiento `float32` con valores muy pequeños/grandes.
- Reducir el tamaño del modelo donde sea posible.
- Mejorar la robustez con enfoques como componentes inspirados en Transformer/stable-diffusion.
- Añadir opciones de modelo base (Threshold, Cellpose) y opciones de modelo objetivo (AIR, Transformer, SD).
- Optimización de interfaz (incluida selección múltiple).
- Optimización de backend (incluida mejora de manejo de memoria/caché).
- Empaquetado fácil de usar con configuración mínima de BD (p. ej., opción SQLite).

## 🤝 Contribuciones

### Contribuye a Lazeal Cellist

Lazeal Cellist es un proyecto de código abierto, y damos la bienvenida a contribuciones de cualquier persona, sin importar su nivel de experiencia. Invitamos contribuciones que:

- Mejoren la eficiencia y el rendimiento algorítmico.
- Mejoren la interfaz y la experiencia de usuario.
- Amplíen la documentación y los ejemplos.
- Corrijan errores y mejoren la estabilidad del sistema.

Antes de empezar a contribuir, primero comenta el cambio que deseas realizar mediante un issue. Esto ayuda a coordinar esfuerzos y evitar trabajo duplicado o en conflicto.

Para más información sobre cómo empezar, consulta las guías de contribución.

Documentación adicional de contribución del repositorio:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Agradecimientos

- El concepto e implementación de Lazeal Cellist se apoyan ampliamente en la línea de investigación AIR/SPAIR listada arriba.
- El repositorio incluye documentación y comandos históricos/heredados, preservados intencionalmente para mantener continuidad con usos previos del proyecto.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 Licencia

Este proyecto está licenciado bajo la licencia MIT. Para más información, consulta el archivo [LICENSE](LICENSE) en este repositorio.

Nota de estado del repositorio: actualmente no hay un archivo `LICENSE` en la raíz de esta copia. La línea anterior se preserva del README previo como intención canónica del proyecto; si lo deseas, añade un archivo `LICENSE` local en un cambio posterior.
