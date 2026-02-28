[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)




[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

<h1 align="center">Lazeal Cellist</h1>

<p align="center">
  <strong>Nền tảng Phát hiện và tạo hồ sơ Tế bào 3D hiệu quả của bạn</strong>
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
  <a href="#-overview"><img src="https://img.shields.io/badge/Read-Overview-0EA5E9?style=flat-square" alt="Overview" /></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Setup-Installation-10B981?style=flat-square" alt="Installation" /></a>
  <a href="#-usage"><img src="https://img.shields.io/badge/Run-Usage-F59E0B?style=flat-square" alt="Usage" /></a>
  <a href="#-troubleshooting"><img src="https://img.shields.io/badge/Fix-Troubleshooting-E11D48?style=flat-square" alt="Troubleshooting" /></a>
  <a href="#-contributing"><img src="https://img.shields.io/badge/Build-Contributing-6366F1?style=flat-square" alt="Contributing" /></a>
</p>

## 🎬 Xem trước

![Ảnh chụp màn hình 2D](screenshot2d.png)

## Lazeal Cellist: Nền tảng phát hiện và tạo hồ sơ tế bào 3D hiệu quả của bạn

Chào mừng đến với Lazeal Cellist, nền tảng toàn diện và hiệu quả để phát hiện, phân đoạn và tạo hồ sơ tế bào cho ảnh hiển vi 3D.

Nền tảng của chúng tôi được thiết kế để phát hiện tế bào bằng học không giám sát, kỹ thuật ngưỡng và các thuật toán tiên tiến như Cellpose. Lazeal Cellist còn cung cấp giao diện tương tác trực quan giúp người dùng tinh chỉnh kết quả phát hiện. Những kết quả đã tinh chỉnh này sau đó được phản hồi vào mạng học bán giám sát, giúp cải thiện hiệu năng mô hình liên tục.

Lazeal Cellist nổi bật ở điểm cung cấp mô hình 3D hiệu quả, đòi hỏi rất ít công sức huấn luyện và tinh chỉnh, nên trở thành nền tảng thực tiễn cho nhà khoa học, nhà nghiên cứu và cả người dùng đam mê.

> ℹ️ **Ghi chú phạm vi**
> Tầm nhìn dự án và giao diện UI bao gồm các khái niệm 3D (`/3d`, `templates/cellist_3d.html`), trong khi luồng huấn luyện chính hiện tại trong mã chủ yếu là cắt lát 2D + tinh chỉnh.

---

## Mục lục

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Inspired by Research](#-inspired-by-research)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)
- [Support](#-support)
- [License](#-license)

## 🔍 Tổng quan

Lazeal Cellist là nền tảng web Python/Tornado cho luồng làm việc ảnh hiển vi với các khả năng:

- Tải lên, tạo mô hình và chỉnh sửa nhãn trực tiếp trên trình duyệt.
- Khởi tạo ban đầu có hỗ trợ thuật toán (Cellpose nuclei mode).
- Tinh chỉnh lặp lại theo kiểu human-in-the-loop qua các thao tác WebSocket (`create`, `initialize`, `pretrain`, `pretrain-stop`, `train`, `train-stop`, `update`, `reset`).
- Lưu trữ bền vững bằng cơ sở dữ liệu cho mô hình, lát cắt ảnh và nhãn.

> ℹ️ Ghi chú về hành vi hiện tại: mặc dù tầm nhìn dự án và UI có các khái niệm 3D (`/3d`, `templates/cellist_3d.html`), luồng huấn luyện chính trong mã hiện tại chủ yếu là lát cắt 2D + tinh chỉnh mô hình.

### Nhìn nhanh

| Khu vực | Triển khai hiện tại |
|---|---|
| Server | Tornado (`app.py`) |
| Port | `8887` |
| Cơ sở dữ liệu | MySQL (`cellist.sql`) |
| Ngăn xếp ML cốt lõi | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Khởi tạo suy luận | Cellpose (`model_type='nuclei'`, `gpu=True`) |
| Trạng thái đóng gói | Prototype nghiên cứu (không có `pyproject.toml`/`setup.py`) |
| Trạng thái kiểm thử/CI | Không có test tự động hay cấu hình CI riêng trong thư mục gốc repository |

### Ngôn ngữ tài liệu

Repository này đã có sẵn README đa ngôn ngữ trong `i18n/`:

| Ngôn ngữ | Tệp |
|---|---|
| Tiếng Ả Rập | `README.ar.md` |
| Tiếng Đức | `README.de.md` |
| Tiếng Tây Ban Nha | `README.es.md` |
| Tiếng Pháp | `README.fr.md` |
| Tiếng Nhật | `README.ja.md` |
| Tiếng Hàn | `README.ko.md` |
| Tiếng Nga | `README.ru.md` |
| Tiếng Trung giản thể | `README.zh-Hans.md` |
| Tiếng Trung phồn thể | `README.zh-Hant.md` |

## ✨ Tính năng nổi bật

- **Phát hiện tế bào 3D không giám sát**: Xác định tế bào trong ảnh hiển vi 3D bằng kỹ thuật học máy tiên tiến.
- **Giao diện tinh chỉnh kết quả tương tác**: Làm sạch và sửa kết quả phát hiện qua giao diện trực quan, dễ dùng.
- **Mạng học bán giám sát hiệu quả**: Nâng cao hiệu năng mô hình theo thời gian nhờ các kết quả đã được tinh chỉnh.
- **Phân đoạn và tạo hồ sơ tế bào**: Không chỉ dừng ở phát hiện, mà còn có khả năng phân đoạn và tạo hồ sơ nâng cao.

Các tính năng triển khai bổ sung hiện có:

- Máy chủ REST + WebSocket Tornado (`app.py`) chạy trên cổng `8887`.
- Tự động chia ảnh thành lưới (`256x256` mặc định) cho bước nạp mô hình.
- Schema MySQL được đóng gói dưới dạng dump: [`cellist.sql`](cellist.sql).
- Frontend sử dụng Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.
- Công việc mô hình bất đồng bộ qua thread pool (`max_workers=64`).

## 🗂️ Cấu trúc dự án

```text
cellist/
├── app.py                               # Máy chủ Tornado chính + xử lý REST/WebSocket
├── cellist/                             # Mã ML/mô hình lõi
│   ├── model_init.py                    # Lớp mô hình 2D chính và luồng huấn luyện
│   ├── model_pretrain.py                # Biến thể tiền huấn luyện
│   ├── model_2d_components.py           # Các thành phần Encoder/Decoder/SPAIR
│   ├── model_2d_utilities.py            # Metadata mô hình dựa trên DB + biến đổi
│   ├── image_preprocessing.py           # Công cụ chia/lắp ảnh
│   └── utils/constants.py               # Đường dẫn runtime + cấu hình MySQL
├── templates/
│   ├── cellist.html                     # UI 2D chính
│   └── cellist_3d.html                  # Biến thể/prototype UI 3D
├── statics/                             # Tài nguyên frontend và phụ thuộc npm
│   ├── package.json
│   └── node_modules/
├── i18n/                                # README đã được dịch
├── notebooks/                           # Notebook khám phá
├── polygon_sample/                      # Thử nghiệm chú thích polygon
├── figs/                                # Tài nguyên nhận diện thương hiệu
├── cellist.sql                          # Dump schema/dữ liệu MySQL
├── cellist.yaml                         # Định nghĩa môi trường Conda
├── create_data_folder.py                # Tiện ích tạo thư mục dữ liệu kế thừa
├── CONTRIBUTING.md
├── PULL_REQUEST_TEMPLATE.md
├── LazealCellist Documentation.md       # Kiến trúc mở rộng và ghi chú TODO
└── README.md
```

## ✅ Điều kiện tiên quyết

| Yêu cầu | Ghi chú |
|---|---|
| Hệ điều hành | Linux được khuyến nghị (các lệnh dưới đây giả định hành vi shell của Linux). |
| Python/Conda | Conda cần có sẵn để tạo môi trường từ [`cellist.yaml`](cellist.yaml). |
| Cơ sở dữ liệu | Máy chủ MySQL chạy trên `localhost` với cơ sở dữ liệu `cellist`. |
| GPU | Môi trường NVIDIA/CUDA được khuyến nghị/được giả định bởi luồng mã hiện tại. |
| Node.js + npm | Cần để cài đặt phụ thuộc frontend trong `statics/node_modules`. |
| Quyền ghi đĩa | Cần để lưu dữ liệu runtime trong `<repo>/data`. |

## 🛠️ Cài đặt

### 1. Sao chép và vào repository

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Tạo môi trường Python

Sử dụng tên tệp trong repository là `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Lưu ý tương thích giữ nguyên từ tài liệu cũ: tài liệu trước đây dùng `celist.yaml` (thiếu chữ `l`), nhưng tệp trong repository này là `cellist.yaml`.

Lệnh kế thừa (được giữ nguyên):

```bash
conda env create -f celist.yaml
```

### 3. Cài đặt phụ thuộc frontend

```bash
cd statics
npm install
cd ..
```

### 4. Chuẩn bị cây thư mục dữ liệu runtime

Ứng dụng kỳ vọng có cây `data/` (và `.gitignore` đã loại trừ `data`):

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Lưu ý: [`create_data_folder.py`](create_data_folder.py) tồn tại, nhưng hiện tại nó tạo thư mục trong thư mục làm việc hiện tại (không nằm dưới `data/`). Hãy lưu ý nếu bạn dùng nó.

### 5. Chuẩn bị xác thực MySQL (nếu cần)

Nếu xác thực root dùng socket và gây cản trở truy cập app, tài liệu cũ của dự án gợi ý chuyển sang xác thực mật khẩu:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 6. Tạo database và khôi phục schema/dữ liệu

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Ví dụ tài liệu lịch sử (được giữ nguyên):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 7. Cấu hình thông tin đăng nhập MySQL cho runtime

Mã hiện tại đọc cấu hình từ [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` và `mysqlurl`).

Giá trị mặc định hiện tại gồm:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Vì an toàn cục bộ, hãy cập nhật các giá trị này trước khi chạy trong môi trường của bạn.

### 8. Kiểm tra môi trường tùy chọn

```bash
python -V
python -c "import torch, pyro, tornado, pymysql; print('core imports OK')"
node -v
npm -v
```

## 🚀 Sử dụng

### Khởi chạy máy chủ web

```bash
python app.py
```

Lệnh khởi động kế thừa từ tài liệu trước (được giữ nguyên):

```bash
python app.py -m cellist
```

Các route mặc định quan sát được trong mã:

- UI chính: `http://localhost:8887/`
- Trang 3D: `http://localhost:8887/3d`

### Quy trình làm việc điển hình

1. Mở UI và đăng nhập.
2. Tải ảnh hiển vi từ bảng Create Model.
3. Chọn thuật toán cơ sở (`Cellpose`) rồi tạo mô hình.
4. Để backend cắt ảnh và khởi tạo phát hiện.
5. Tải ảnh đã cắt, xem lại/điều chỉnh nhãn hình chữ nhật.
6. Chạy các chu kỳ `initialize`, `pretrain`, và `train`.
7. Dùng `Pretrain Stop` / `Stop` (`train-stop`) / `reset` khi cần.
8. Lưu cập nhật thủ công qua các hành động `Update Model`/nhãn.

### Thông tin đăng nhập đăng nhập UI có sẵn (hành vi template hiện tại)

Frontend hiện kiểm tra các thông tin đăng nhập tĩnh sau ở phía client:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Đây là hành vi prototype và không phải xác thực production.

### Bề mặt API/WebSocket hiện tại được UI sử dụng

Endpoint HTTP:

- `GET /`
- `GET /3d`
- `POST /upload/<ws_uuid>`
- `POST /load_model/<ws_uuid>`

Endpoint WebSocket:

- `ws://localhost:8887/websocket/<ws_uuid>`

Các thông điệp `data_type` được nhận trong handler WebSocket:

- `create`
- `update`
- `initialize`
- `pretrain`
- `pretrain-stop`
- `train`
- `train-stop`
- `reset`

## ⚙️ Cấu hình

### Backend và endpoints

Được cấu hình trong [`app.py`](app.py):

- Cổng: `8887`
- Routes:
  - `/`
  - `/3d`
  - `/upload/(.*)`
  - `/load_model/.*`
  - `/websocket/(.*)`

### Hành vi mô hình/dữ liệu

- Kích thước thread pool là `max_workers=64`.
- Mảnh ảnh mặc định là `256x256`.
- Khởi tạo Cellpose dùng `model_type='nuclei'` và `gpu=True`.
- Huấn luyện và tiền huấn luyện chạy không đồng bộ qua các action được kích hoạt bởi WebSocket.
- Đường dẫn dữ liệu gốc được giải quyết từ thư mục làm việc hiện tại dưới dạng `<repo>/data`.

### Hằng số runtime/cơ sở dữ liệu

Từ [`cellist/utils/constants.py`](cellist/utils/constants.py):

- `dataroot = os.path.join(curdir, "data")`
- `mysqlconfig` bao gồm các khóa host/user/password
- `mysqlurl` trỏ tới cơ sở dữ liệu `cellist`

### Tóm tắt phụ thuộc frontend

Từ [`statics/package.json`](statics/package.json):

- `bootstrap`
- `bootstrap-icons`
- `jquery`
- `jquery-ui` / `jquery-ui-dist`
- `three`
- `blueimp-file-upload`

### Điểm nổi bật môi trường Conda

Từ [`cellist.yaml`](cellist.yaml):

- Python `3.8.12`
- PyTorch `1.12.0`
- CUDA toolkit `11.3.1`
- Tornado `6.1`
- Cellpose `0.7.2` (pip)
- Pyro (`pyro-ppl==1.8.1`)
- PyMySQL + SQLAlchemy

## 🧪 Ví dụ

### Ví dụ: Hình dạng thông điệp WebSocket tạo

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

### Ví dụ: Cập nhật nhãn thủ công qua WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Ví dụ: Yêu cầu tải mô hình

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

### Ví dụ: Khởi động local end-to-end tối thiểu

```bash
conda env create -f cellist.yaml
conda activate cellist
cd statics && npm install && cd ..
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
python app.py
```

## 📚 Được truyền cảm hứng từ nghiên cứu

Lazeal Cellist được truyền cảm hứng bởi các nghiên cứu đột phá trong học sâu, gồm:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Các công trình này cung cấp những hiểu biết quý giá đã định hướng cho các thuật toán và phương pháp luận phát triển nền tảng.

(Lưu ý: để trích dẫn chính xác, vui lòng tham khảo trực tiếp tài liệu gốc.)

## 🧭 Ghi chú phát triển

- Các lớp mô hình lõi nằm trong `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Luồng UI tương tác chính được nhúng trực tiếp trong `templates/cellist.html`.
- Schema SQL và dữ liệu dạng seed nằm trong `cellist.sql`.
- Notebook trong `notebooks/` và `polygon_sample/` là tài liệu tham khảo thăm dò.
- Ghi chú kiến trúc/mô hình mở rộng nằm trong [`LazealCellist Documentation.md`](LazealCellist%20Documentation.md).
- Hiện chưa có bộ test tự động hoặc cấu hình CI riêng trong thư mục gốc repository.

### Giả định và hạn chế hiện tại

- Repository này dường như ưu tiên sử dụng cục bộ theo hướng nghiên cứu trước.
- Một số đường dẫn mã giả định có sẵn GPU (`cuda:0`).
- Xác thực và quản lý bí mật đang ở mức prototype.
- Giao diện 3D đã tồn tại, nhưng luồng huấn luyện chiếm ưu thế vẫn là định hướng theo tile 2D.

## 🧯 Khắc phục sự cố

| Triệu chứng | Kiểm tra gợi ý |
|---|---|
| `ModuleNotFoundError` hoặc lỗi import | Kiểm tra đã chạy `conda activate cellist` trước khi `python app.py`. |
| UI hiển thị thiếu style/script | Chạy `npm install` trong `statics/` và xác nhận `statics/node_modules` tồn tại. |
| Truy cập MySQL bị từ chối | Kiểm tra username/password trong `cellist/utils/constants.py` và plugin/chế độ auth MySQL. |
| App khởi động nhưng hành động mô hình lỗi | Kiểm tra sự sẵn sàng của CUDA/GPU; đường dẫn hiện tại giả định CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload thành công nhưng không thấy tile/mô hình | Đảm bảo thư mục con `data/` tồn tại và có quyền ghi. |
| Lỗi REST/WebSocket request | Xác nhận server đang chạy tại `http://localhost:8887` và key payload khớp với template hiện tại. |
| `FileNotFoundError` trong `data/` | Khởi chạy app từ root repository để đường dẫn tương đối giải quyết nhất quán. |

### Chẩn đoán nhanh

```bash
# Xác thực môi trường Python và các import quan trọng
python -c "import torch, pyro, tornado, pymysql; print('imports ok')"

# Xác nhận cổng server mở sau khi khởi động
ss -ltnp | rg 8887

# Kiểm tra kết nối MySQL
mysql -u root -p -e "SHOW DATABASES LIKE 'cellist';"
```

## 🛣️ Lộ trình

Các mục sau được giữ nguyên và tổ chức từ tài liệu dự án/TODO hiện có:

- Mẫu polygon: dùng polygon thay vì chú thích hình chữ nhật.
- Tối ưu mô hình cho hành vi `float32` với giá trị rất nhỏ/rất lớn.
- Giảm kích thước mô hình khi có thể.
- Tăng tính bền vững với các hướng tiếp cận như Transformer/thành phần truyền cảm hứng từ stable diffusion.
- Thêm tùy chọn mô hình cơ sở (Threshold, Cellpose) và mô hình đích (AIR, Transformer, SD).
- Tối ưu giao diện (bao gồm chọn nhiều phần tử cùng lúc).
- Tối ưu backend (bao gồm xử lý bộ nhớ/cache tốt hơn).
- Đóng gói dễ dùng với cấu hình DB tối thiểu (ví dụ tùy chọn SQLite).

## 🤝 Đóng góp

### Đóng góp cho Lazeal Cellist

Lazeal Cellist là một dự án mã nguồn mở, và chúng tôi hoan nghênh đóng góp từ mọi người, bất kể trình độ.

Chúng tôi chào đón các đóng góp giúp:

- Nâng cao hiệu năng và độ hiệu quả thuật toán
- Cải thiện giao diện người dùng và trải nghiệm người dùng
- Mở rộng tài liệu và ví dụ
- Sửa lỗi và tăng tính ổn định của hệ thống

Trước khi bắt đầu đóng góp, xin hãy trao đổi trước thay đổi bạn muốn thực hiện thông qua issue. Điều này giúp phối hợp công việc và tránh trùng lặp hoặc xung đột.

Để biết thêm cách bắt đầu, đọc hướng dẫn đóng góp.

Tài liệu đóng góp bổ sung của repository:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 🙏 Lời cảm ơn

- Ý tưởng và triển khai Lazeal Cellist phần lớn dựa nhiều vào dòng nghiên cứu AIR/SPAIR như đã nêu ở trên.
- Repository bao gồm tài liệu và lệnh lịch sử/legacy được giữ nguyên có chủ đích để duy trì liên tục với cách dùng dự án trước đây.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

Dự án này được cấp phép theo MIT License. Để biết thêm chi tiết, xem file [LICENSE](LICENSE) trong repository.

Lưu ý trạng thái repository: hiện chưa có file `LICENSE` ở thư mục gốc trong checkout này. Dòng trên được giữ nguyên theo ý đồ chính thống của README trước đó; thêm file `LICENSE` tại local trong lần cập nhật tiếp theo nếu cần.
