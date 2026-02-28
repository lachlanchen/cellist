[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Tu plataforma eficiente de detección y perfilado de células 3D</strong>
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
  <a href="#-resumen"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-instalacion"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-uso"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-resolucion-de-problemas"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-contribucion"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 Vista previa

![Screenshot 2D](screenshot2d.png)

## Lazeal Cellist: Tu plataforma eficiente de detección y perfilado de células 3D

Bienvenido a Lazeal Cellist, una plataforma integral y eficiente de detección, segmentación y perfilado celular para imágenes de microscopía 3D.

Nuestra plataforma está diseñada para detectar células mediante aprendizaje no supervisado, técnicas de umbral y algoritmos de vanguardia como Cellpose. Lazeal Cellist también ofrece una interfaz interactiva e intuitiva que permite a los usuarios refinar los resultados de detección. Esos resultados refinados luego se devuelven a la red de aprendizaje semi-supervisado, mejorando de forma continua el rendimiento del modelo.

Lazeal Cellist se distingue por ofrecer un modelo 3D eficiente que requiere poco esfuerzo para entrenar y refinar, convirtiéndolo en una plataforma práctica para científicos, investigadores y entusiastas.

> ℹ️ **Nota de alcance**
> La visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), mientras que el flujo de entrenamiento principal en el código es actualmente principalmente por cortes 2D + refinamiento.

---

## Tabla de contenidos

- [Resumen](#-resumen)
- [Características clave](#-caracteristicas-clave)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Prerrequisitos](#-prerrequisitos)
- [Instalación](#-instalacion)
- [Uso](#-uso)
- [Configuración](#-configuracion)
- [Ejemplos](#-ejemplos)
- [Inspirado por la investigación](#-inspirado-por-la-investigacion)
- [Notas de desarrollo](#-notas-de-desarrollo)
- [Resolucion de problemas](#-resolucion-de-problemas)
- [Hoja de ruta](#-hoja-de-ruta)
- [Contribucion](#-contribucion)
- [Agradecimientos](#-agradecimientos)
- [Support](#-support)
- [Licencia](#-licencia)

## 🔍 Resumen

Lazeal Cellist es una plataforma web en Python/Tornado para flujos de trabajo de imágenes de microscopía con:

- Carga desde el navegador, creación de modelos y edición de anotaciones.
- Inicialización asistida por algoritmos (modo núcleos de Cellpose).
- Refinamiento iterativo tipo human-in-the-loop mediante acciones WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Persistencia en base de datos para modelos, cortes de imagen y anotaciones.

> ℹ️ Nota sobre el comportamiento actual: aunque la visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), el flujo de entrenamiento principal en el código actual se basa principalmente en cortes 2D + refinamiento del modelo.

### Visión rápida

| Área | Implementación actual |
|---|---|
| Servidor | Tornado (`app.py`) |
| Puerto | `8887` |
| Base de datos | MySQL (`cellist.sql`) |
| Stack ML principal | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inicialización de inferencia | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Estado de empaquetado | Prototipo de investigación (sin `pyproject.toml`/`setup.py`) |
| Estado de pruebas/CI | Sin suite de tests automatizada dedicada ni configuración de CI en la raíz del repositorio |

### Idiomas de documentación

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
| Chino simplificado | `README.zh-Hans.md` |
| Chino tradicional | `README.zh-Hant.md` |

## ✨ Características clave

- **Detección celular 3D no supervisada**: Identifica células en imágenes de microscopía 3D usando técnicas avanzadas de aprendizaje automático.
- **Interfaz interactiva de refinamiento de resultados**: Refina resultados de detección con una interfaz intuitiva y fácil de usar.
- **Red de aprendizaje semi-supervisado eficiente**: Mejora el rendimiento del modelo con el tiempo usando resultados refinados.
- **Segmentación y perfilado celular**: Va más allá de la detección con capacidades avanzadas de segmentación y perfilado.

Características de implementación adicionales disponibles actualmente:

- Servidor REST + WebSocket de Tornado (`app.py`) en el puerto `8887`.
- Mosaico automático de imágenes (`256x256` por defecto) para la ingestión del modelo.
- Esquema de MySQL incluido como volcado: [`cellist.sql`](cellist.sql).
- Frontend incluye Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Tareas asíncronas del modelo mediante un pool de hilos (`max_workers=64`).

## 🗂️ Estructura del proyecto

```text
cellist/
├── app.py                               # Servidor principal de Tornado + handlers REST/WebSocket
├── cellist/                             # Código central de ML/modelo
│   ├── model_init.py                    # Clase principal de modelo 2D y flujo de entrenamiento
│   ├── model_pretrain.py                # Variante de preentrenamiento
│   ├── model_2d_components.py           # Componentes Encoder/Decoder/SPAIR
│   ├── model_2d_utilities.py            # Metadatos del modelo con respaldo de BD + transformaciones
│   ├── image_preprocessing.py           # Utilidades de corte/unión de imágenes
│   └── utils/constants.py               # Rutas runtime + configuración MySQL
├── templates/
│   ├── cellist.html                     # UI 2D principal
│   └── cellist_3d.html                  # Variante/prototipo de UI 3D
├── statics/                             # Activos frontend y dependencias npm
│   ├── package.json
│   └── node_modules/
├── i18n/                                # Archivos README traducidos
├── notebooks/                           # Notebooks exploratorios
├── polygon_sample/                      # Experimentos de anotación poligonal
├── figs/                                # Activos de marca
├── cellist.sql                          # Esquema y datos de MySQL
├── cellist.yaml                         # Especificación de entorno Conda
├── create_data_folder.py                # Helper heredado de creación de carpeta de datos
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Notas extendidas de arquitectura/TODO
└── README.md
```

## ✅ Prerrequisitos

| Requisito | Notas |
|---|---|
| Sistema operativo | Linux recomendado (los comandos abajo asumen comportamiento de shell de Linux). |
| Python/Conda | Conda disponible para crear el entorno desde [`cellist.yaml`](cellist.yaml). |
| Base de datos | Servidor MySQL ejecutándose en `localhost` con base de datos `cellist`. |
| GPU | Entorno NVIDIA/CUDA fuertemente recomendado/supone las rutas de código actuales. |
| Node.js + npm | Requerido para instalar dependencias frontend de `statics/node_modules`. |
| Acceso de escritura en disco | Necesario para datos en runtime bajo `<repo>/data`. |

## 🛠️ Instalación

### 1. Clonar y entrar al repositorio

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Crear entorno de Python

Usa el archivo de repositorio `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Nota de compatibilidad preservada de documentación antigua: documentación previa usaba `celist.yaml` (faltaba una `l`), pero el archivo en este repositorio es `cellist.yaml`.

Comando heredado (preservado):

```bash
conda env create -f celist.yaml
```

### 3. Instalar dependencias frontend

```bash
cd statics
npm install
cd ..
```

### 4. Preparar directorios de datos en runtime

La app espera un árbol `data/` (y `.gitignore` ya excluye `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Nota: [`create_data_folder.py`](create_data_folder.py) existe, pero actualmente crea directorios en el directorio de trabajo (no bajo `data/`). Tenlo en cuenta si lo usas.

### 5. Preparar autenticación de MySQL (si hace falta)

Si la autenticación de root se basa en socket y bloquea el acceso de la app, documentación anterior sugiere cambiar a auth por contraseña:

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

Ejemplo de documentación histórica (preservado):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Configurar credenciales de MySQL para runtime

El código actual lee las credenciales desde [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` y `mysqlurl`).

Valores por defecto actuales en el código:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Por seguridad local, actualiza estos valores antes de ejecutar tu entorno.

### 8. Verificaciones opcionales de entorno

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Uso

### Iniciar servidor web

```bash
python app.py
```

Comando de inicio histórico de documentación anterior (preservado):

```bash
python app.py -m cellist
```

Rutas de servidor observadas en el código:

- UI principal: `http://localhost:8887/`
- Página 3D: `http://localhost:8887/3d`

### Flujo de trabajo típico

1. Abrir la interfaz y autenticarse.
2. Subir imágenes de microscopía desde el panel Create Model.
3. Elegir algoritmo base (`Cellpose`) y crear el modelo.
4. Dejar que el backend corte las imágenes e inicialice detecciones.
5. Cargar imágenes recortadas, revisar/ajustar anotaciones rectangulares.
6. Ejecutar ciclos `initialize`, `pretrain` y `train`.
7. Usar `Pretrain Stop` / `Stop` (`train-stop`) / `reset` según sea necesario.
8. Persistir actualizaciones manuales mediante `Update Model`/acciones de anotación.

### Credenciales de inicio de sesión integradas en la UI (comportamiento actual del template)

El frontend actualmente verifica estas credenciales estáticas del lado cliente:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Este es un comportamiento de prototipo y no una autenticación de producción.

### Superficie API/WebSocket usada actualmente por la UI

Endpoints HTTP:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

Endpoint WebSocket:

- `ws://localhost:8887/websocket/<ws_uuid>`

Mensajes `data_type` reconocidos en el handler WebSocket:

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

### Comportamiento del modelo/datos

- El pool de hilos está en `max_workers=64`.
- Los mosaicos de imagen son de `256x256` por defecto.
- La inicialización con Cellpose usa `model_type='nuclei'` y `gpu=True`.
- Entrenamiento y preentrenamiento se ejecutan de forma asíncrona mediante acciones disparadas por WebSocket.
- La raíz de datos se resuelve desde el directorio de trabajo actual como `<repo>/data`.

### Constantes runtime/base de datos

Desde [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` incluye claves host/user/password
- `mysqlurl` apunta al nombre de base de datos `cellist`

### Snapshot de dependencias frontend

Desde [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Puntos principales del entorno Conda

Desde [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Ejemplos

### Ejemplo: formato de mensaje WebSocket create

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

### Ejemplo: actualización manual de anotación por WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Ejemplo: petición de carga de modelo

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

## 📚 Inspirado en la investigación

Lazeal Cellist se inspira en investigación de frontera en deep learning, incluyendo:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Estos trabajos ofrecen ideas valiosas que han guiado el desarrollo de los algoritmos y metodologías de nuestra plataforma.

(Nota: Para una cita precisa, consulta directamente los documentos originales.)

## 🧭 Notas de desarrollo

- Las clases principales del modelo están en `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La lógica principal de la UI interactiva está embebida directamente en `templates/cellist.html`.
- El esquema SQL y datos tipo seed están en `cellist.sql`.
- Los notebooks en `notebooks/` y `polygon_sample/` proporcionan referencias exploratorias.
- Las notas extendidas de plataforma/modelo están en [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Actualmente no existe una suite de tests automatizados ni configuración de CI dedicada en la raíz del repositorio.

### Suposiciones y restricciones actuales

- Este repositorio parece orientado primero al uso local de investigación.
- Algunas rutas de código asumen disponibilidad de GPU (`cuda:0`).
- La autenticación y gestión de secretos están a nivel de prototipo.
- Las interfaces 3D existen, pero el flujo de entrenamiento dominante sigue siendo orientado a cortes 2D.

## 🧯 Resolucion de problemas

| Síntoma | Chequeos sugeridos |
|---|---|
| `ModuleNotFoundError` o problemas de importación | Confirma que `conda activate cellist` se aplicó antes de ejecutar `python app.py`. |
| La UI se renderiza sin estilos/scripts | Ejecuta `npm install` dentro de `statics/` y confirma que `statics/node_modules` existe. |
| Acceso denegado a MySQL | Verifica usuario/contraseña en `cellist/utils/constants.py` y el modo de plugin/auth de MySQL. |
| La app inicia pero las acciones del modelo fallan | Revisa disponibilidad de CUDA/GPU; las rutas actuales asumen CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| La carga sube pero no aparecen tiles/modelos | Asegúrate de que los subdirectorios de `data/` existan y tengan escritura. |
| Errores de solicitud REST/WebSocket | Confirma que el servidor está ejecutándose en `http://localhost:8887` y que las claves del payload coinciden con los nombres actuales del template. |
| `FileNotFoundError` bajo `data/` | Inicia la app desde la raíz del repositorio para que las rutas relativas se resuelvan de forma consistente. |

### Diagnósticos rápidos

```bash
# Verificar el entorno Python y las importaciones clave
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Confirmar que el puerto del servidor quedó abierto tras el arranque
ss -ltnp | rg 8887

# Comprobar conectividad de MySQL
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Hoja de ruta

Los siguientes elementos se conservan y organizan a partir de la documentación/TODO existente del proyecto:

- Muestra de polígono: usar polígono en lugar de anotación rectangular.
- Optimizar el modelo para comportamiento `float32` con valores muy pequeños/grandes.
- Reducir tamaño del modelo cuando sea posible.
- Mejorar robustez con enfoques como componentes inspirados en Transformer/stable-diffusion.
- Añadir opciones de modelo base (Threshold, Cellpose) y de modelo objetivo (AIR, Transformer, SD).
- Optimización de interfaz (incluyendo selección múltiple).
- Optimización de backend (incluyendo mejor manejo de memoria/caché).
- Empaquetado fácil de usar con configuración mínima de BD (por ejemplo, opción SQLite).

## 🤝 Contribución

### Contribuir a Lazeal Cellist

Lazeal Cellist es un proyecto open source, y damos la bienvenida a contribuciones de todas las personas, sin importar el nivel de experiencia. Invitamos aportes que:

- Mejoren la eficiencia y rendimiento algorítmico
- Mejoren la interfaz de usuario y experiencia de usuario
- Amplíen la documentación y ejemplos
- Corrijan errores y mejoren la estabilidad del sistema

Antes de empezar a contribuir, primero conversa el cambio que deseas hacer a través de una issue. Esto ayuda a coordinar esfuerzos y evitar trabajo duplicado o conflictivo.

Para más información sobre cómo empezar, consulta las guías de contribución.

Documentación adicional del repositorio:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Agradecimientos

- El concepto y la implementación de Lazeal Cellist se basan en gran medida en la línea de investigación AIR/SPAIR mencionada arriba.
- El repositorio incluye documentación y comandos históricos/legacy preservados intencionalmente para mantener continuidad con el uso previo del proyecto.

## 📄 Licencia

Este proyecto está licenciado bajo MIT License. Para más información, consulta el archivo [LICENSE](LICENSE) de este repositorio.

Nota de estado del repositorio: actualmente no hay un archivo `LICENSE` en la raíz en este checkout. La línea anterior se conserva del README original como intención canónica del proyecto; añade un archivo `LICENSE` local en un cambio posterior si lo consideras necesario.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
