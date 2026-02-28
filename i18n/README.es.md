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

## Lazeal Cellist: Tu plataforma eficiente de detección y perfilado celular 3D

Bienvenido a Lazeal Cellist, una plataforma integral y eficiente para detección, segmentación y perfilado celular en imágenes de microscopía 3D.

Nuestra plataforma está diseñada para detectar células mediante aprendizaje no supervisado, técnicas de umbralización y algoritmos de última generación como Cellpose. Lazeal Cellist también ofrece una interfaz intuitiva e interactiva que permite a los usuarios refinar los resultados de detección. Estos resultados refinados se reincorporan a la red de aprendizaje semisupervisado, mejorando continuamente el rendimiento del modelo.

Lazeal Cellist destaca por ofrecer un modelo 3D eficiente que requiere un esfuerzo mínimo para entrenar y refinar, lo que la convierte en una plataforma práctica para científicos, investigadores y aficionados.

---

## 🔍 Descripción general

Lazeal Cellist es una plataforma web en Python/Tornado para flujos de trabajo de imágenes de microscopía con:

- Carga en navegador, creación de modelos y edición de anotaciones.
- Inicialización asistida por algoritmos (modo núcleos de Cellpose).
- Refinamiento iterativo con intervención humana mediante acciones WebSocket (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Persistencia respaldada por base de datos para modelos, cortes de imagen y anotaciones.

Nota sobre el comportamiento actual: aunque la visión del proyecto y la interfaz incluyen conceptos 3D (`/3d`, `templates/cellist_3d.html`), el flujo principal de entrenamiento en el código es principalmente de cortes 2D + refinamiento del modelo.

### Resumen rápido

| Área | Implementación actual |
|---|---|
| Servidor | Tornado (`app.py`) |
| Puerto | `8887` |
| Base de datos | MySQL (`cellist.sql`) |
| Stack ML principal | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Inicialización de inferencia | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Funcionalidades clave

- **Detección celular 3D no supervisada**: Identifica células en imágenes de microscopía 3D mediante técnicas avanzadas de aprendizaje automático.
- **Interfaz interactiva de refinamiento de resultados**: Refina resultados de detección con una interfaz intuitiva y fácil de usar.
- **Red eficiente de aprendizaje semisupervisado**: Mejora el rendimiento del modelo con el tiempo a partir de resultados refinados.
- **Segmentación y perfilado celular**: Va más allá de la detección con capacidades avanzadas de segmentación y perfilado.

Características de implementación adicionales presentes actualmente:

- Servidor Tornado REST + WebSocket (`app.py`) en el puerto `8887`.
- Teselado automático de imágenes (`256x256` por defecto) para la ingesta del modelo.
- Esquema MySQL incluido como volcado: [`cellist.sql`](cellist.sql).
- El stack frontend incluye Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Estructura del proyecto

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

## ✅ Requisitos previos

| Requisito | Notas |
|---|---|
| SO | Linux recomendado (los comandos siguientes asumen comportamiento de shell Linux). |
| Python/Conda | Conda disponible para crear el entorno desde [`cellist.yaml`](cellist.yaml). |
| Base de datos | Servidor MySQL ejecutándose en `localhost` con la base de datos `cellist`. |
| GPU | Se recomienda/espera fuertemente un entorno NVIDIA/CUDA según las rutas actuales del código. |
| Node.js + npm | Necesario para instalar dependencias frontend en `statics/node_modules`. |

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

Nota de compatibilidad preservada de documentación antigua: la documentación previa usaba `celist.yaml` (falta una `l`), pero el archivo en este repositorio es `cellist.yaml`.

### 3. Instalar dependencias frontend

```bash
cd statics
npm install
cd ..
```

### 4. Preparar autenticación MySQL (si es necesario)

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

### 5. Crear base de datos y restaurar esquema/datos

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Ejemplo de documentación heredada (preservado):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. Configurar credenciales MySQL para ejecución

El código actual lee las credenciales desde [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` y `mysqlurl`).

Los valores por defecto del código incluyen actualmente:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Por seguridad local, actualiza estos valores antes de ejecutar en tu entorno.

### 7. Preparar directorios de datos de ejecución

La app espera un árbol `data/` (y `.gitignore` ya excluye `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Nota: existe [`create_data_folder.py`](create_data_folder.py), pero actualmente crea directorios en el directorio de trabajo actual (no bajo `data/`). Tenlo en cuenta si lo usas.

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
4. Deja que el backend corte las imágenes e inicialice detecciones.
5. Carga imágenes recortadas, revisa/ajusta anotaciones de rectángulos.
6. Ejecuta ciclos de `initialize`, `pretrain` y `train`.
7. Persiste actualizaciones manuales mediante acciones de anotación/`Update Model`.

### Credenciales de inicio de sesión integradas en la UI (comportamiento actual de plantillas)

El frontend actualmente valida estas credenciales estáticas en el cliente:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Este comportamiento es de prototipo y no es autenticación de producción.

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
- Los tiles de imagen son `256x256` por defecto.
- La inicialización de Cellpose usa `model_type='nuclei'` y `gpu=True`.
- El entrenamiento y preentrenamiento se ejecutan de forma asíncrona mediante acciones activadas por WebSocket.

## 🧪 Ejemplos

### Ejemplo: forma del mensaje WebSocket de creación

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

## 📚 Inspirado en investigación

Lazeal Cellist está inspirado en investigación de vanguardia en aprendizaje profundo, incluyendo:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Estos trabajos aportan ideas valiosas que han guiado el desarrollo de los algoritmos y metodologías de nuestra plataforma.

(Nota: para una cita precisa, consulta directamente los artículos originales.)

## 🧭 Notas de desarrollo

- Las clases principales del modelo están en `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- La lógica principal de UI interactiva está embebida directamente en `templates/cellist.html`.
- El esquema SQL y datos estilo semilla están en `cellist.sql`.
- Los notebooks en `notebooks/` y `polygon_sample/` proporcionan referencias exploratorias.
- Actualmente no hay una suite de pruebas automatizadas dedicada ni configuración de CI en la raíz del repositorio.

## 🧯 Solución de problemas

| Síntoma | Verificaciones sugeridas |
|---|---|
| `ModuleNotFoundError` o problemas de importación | Confirma que `conda activate cellist` se aplicó antes de ejecutar `python app.py`. |
| La UI se renderiza sin estilos/scripts | Ejecuta `npm install` dentro de `statics/` y confirma que existe `statics/node_modules`. |
| Acceso MySQL denegado | Verifica usuario/contraseña en `cellist/utils/constants.py` y el modo plugin/auth de MySQL. |
| La app inicia pero fallan las acciones del modelo | Revisa disponibilidad de CUDA/GPU; las rutas actuales asumen CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| La carga funciona pero no aparecen tiles/modelos | Asegura que los subdirectorios `data/` existen y tienen permisos de escritura. |

## 🗺️ Hoja de ruta

Los siguientes ítems se conservan y organizan desde documentación existente del proyecto/notas TODO:

- Muestra de polígonos: usar polígonos en lugar de anotación rectangular.
- Optimizar el modelo para comportamiento `float32` con valores muy pequeños/grandes.
- Reducir el tamaño del modelo donde sea posible.
- Mejorar robustez con enfoques como componentes inspirados en Transformer/stable-diffusion.
- Añadir opciones de modelo base (Threshold, Cellpose) y opciones de modelo objetivo (AIR, Transformer, SD).
- Optimización de interfaz (incluyendo selección múltiple).
- Optimización de backend (incluyendo mejor manejo de memoria/caché).
- Empaquetado fácil de usar con configuración mínima de BD (por ejemplo, opción SQLite).

## 🤝 Contribuir

### Contribuye a Lazeal Cellist

Lazeal Cellist es un proyecto de código abierto, y damos la bienvenida a contribuciones de cualquier persona, sin importar su nivel de experiencia. Invitamos contribuciones que:

- Mejoren la eficiencia y rendimiento algorítmico
- Mejoren la interfaz y experiencia de usuario
- Amplíen documentación y ejemplos
- Corrijan errores y mejoren la estabilidad del sistema

Antes de empezar a contribuir, por favor comenta primero el cambio que quieres hacer mediante un issue. Esto ayuda a coordinar esfuerzos y evitar trabajo duplicado o conflictivo.

Para más información sobre cómo empezar, consulta las pautas de contribución.

Documentos adicionales de contribución del repositorio:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Para más información, consulta el archivo [LICENSE](https://chat.openai.com/LICENSE) en este repositorio.

Nota sobre el estado del repositorio: actualmente no hay un archivo `LICENSE` en la raíz en este checkout. La línea anterior se conserva del README previo como intención canónica del proyecto; añade un archivo `LICENSE` local en un cambio posterior si lo deseas.
