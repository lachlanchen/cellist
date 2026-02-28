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

## Lazeal Cellist: Nền tảng phát hiện và lập hồ sơ tế bào 3D hiệu quả của bạn

Chào mừng bạn đến với Lazeal Cellist, một nền tảng toàn diện và hiệu quả cho phát hiện, phân đoạn và lập hồ sơ tế bào trên ảnh hiển vi 3D.

Nền tảng của chúng tôi được thiết kế để phát hiện tế bào bằng học không giám sát, các kỹ thuật ngưỡng hóa và các thuật toán tiên tiến như Cellpose. Lazeal Cellist cũng cung cấp một giao diện trực quan, tương tác để người dùng tinh chỉnh kết quả phát hiện. Các kết quả tinh chỉnh này sau đó được đưa ngược lại vào mạng học bán giám sát, giúp liên tục cải thiện hiệu năng mô hình.

Lazeal Cellist nổi bật nhờ cung cấp một mô hình 3D hiệu quả, cần rất ít công sức để huấn luyện và tinh chỉnh, khiến nó trở thành một nền tảng thực tiễn cho nhà khoa học, nhà nghiên cứu và người đam mê.

---

## 🔍 Tổng quan

Lazeal Cellist là một nền tảng web Python/Tornado cho quy trình làm việc ảnh hiển vi với:

- Tải lên qua trình duyệt, tạo mô hình và chỉnh sửa chú thích.
- Khởi tạo có hỗ trợ thuật toán (Cellpose chế độ nuclei).
- Tinh chỉnh lặp có con người trong vòng lặp thông qua các hành động WebSocket (`initialize`, `pretrain`, `train`, `update`, `reset`).
- Lưu trữ bền vững dựa trên cơ sở dữ liệu cho mô hình, lát ảnh và chú thích.

Lưu ý về hành vi hiện tại: dù tầm nhìn dự án và UI có bao gồm các khái niệm 3D (`/3d`, `templates/cellist_3d.html`), luồng huấn luyện chính trong mã hiện tại chủ yếu là cắt lát 2D + tinh chỉnh mô hình.

### Tóm tắt nhanh

| Khu vực | Triển khai hiện tại |
|---|---|
| Máy chủ | Tornado (`app.py`) |
| Cổng | `8887` |
| Cơ sở dữ liệu | MySQL (`cellist.sql`) |
| Stack ML cốt lõi | PyTorch + Pyro + Cellpose |
| Frontend | Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload |
| Khởi tạo suy luận | Cellpose (`model_type='nuclei'`, `gpu=True`) |

## ✨ Tính năng chính

- **Phát hiện tế bào 3D không giám sát**: Xác định tế bào trong ảnh hiển vi 3D bằng các kỹ thuật học máy tiên tiến.
- **Giao diện tương tác tinh chỉnh kết quả**: Tinh chỉnh kết quả phát hiện với giao diện trực quan, thân thiện.
- **Mạng học bán giám sát hiệu quả**: Nâng cao hiệu năng mô hình theo thời gian với các kết quả đã tinh chỉnh.
- **Phân đoạn và lập hồ sơ tế bào**: Vượt ra ngoài phát hiện với khả năng phân đoạn và lập hồ sơ nâng cao.

Các tính năng triển khai bổ sung hiện đang có:

- Máy chủ Tornado REST + WebSocket (`app.py`) trên cổng `8887`.
- Tự động chia ô ảnh (`256x256` mặc định) để nạp vào mô hình.
- Lược đồ MySQL đi kèm dưới dạng bản dump: [`cellist.sql`](cellist.sql).
- Stack frontend bao gồm Bootstrap, jQuery, jQuery UI, Three.js, blueimp-file-upload.

## 🗂️ Cấu trúc dự án

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

## ✅ Điều kiện tiên quyết

| Yêu cầu | Ghi chú |
|---|---|
| OS | Khuyến nghị Linux (các lệnh bên dưới giả định hành vi shell Linux). |
| Python/Conda | Có sẵn Conda để tạo môi trường từ [`cellist.yaml`](cellist.yaml). |
| Cơ sở dữ liệu | Máy chủ MySQL chạy trên `localhost` với cơ sở dữ liệu `cellist`. |
| GPU | Môi trường NVIDIA/CUDA được khuyến nghị mạnh/mặc định kỳ vọng theo các luồng mã hiện tại. |
| Node.js + npm | Cần để cài dependency frontend trong `statics/node_modules`. |

## 🛠️ Cài đặt

### 1. Clone và vào thư mục repository

```bash
git clone <your-fork-or-upstream-url>
cd cellist
```

### 2. Tạo môi trường Python

Dùng tên tệp trong repository là `cellist.yaml`:

```bash
conda env create -f cellist.yaml
conda activate cellist
```

Ghi chú tương thích được giữ lại từ tài liệu cũ: tài liệu trước đây dùng `celist.yaml` (thiếu một chữ `l`), nhưng tệp trong repository này là `cellist.yaml`.

### 3. Cài dependency frontend

```bash
cd statics
npm install
cd ..
```

### 4. Chuẩn bị xác thực MySQL (nếu cần)

Nếu xác thực root dựa trên socket và chặn truy cập của ứng dụng, tài liệu dự án cũ gợi ý chuyển sang xác thực bằng mật khẩu:

```bash
sudo mysql
```

```sql
SELECT user,authentication_string,plugin,host FROM mysql.user;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Tạo cơ sở dữ liệu và phục hồi lược đồ/dữ liệu

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS cellist;"
mysql -u root -p cellist < cellist.sql
```

Ví dụ tài liệu kế thừa (được giữ lại):

```sql
mysql -u root -p cellist < /home/user/cellist.sql
```

### 6. Cấu hình thông tin MySQL khi chạy

Mã hiện tại đọc thông tin xác thực từ [`cellist/utils/constants.py`](cellist/utils/constants.py) (`mysqlconfig` và `mysqlurl`).

Giá trị mặc định trong mã hiện bao gồm:

- host: `localhost`
- user: `root`
- password: `lazeal0626`

Vì bảo mật cục bộ, hãy cập nhật các giá trị này trước khi chạy trong môi trường của bạn.

### 7. Chuẩn bị thư mục dữ liệu thời gian chạy

Ứng dụng cần cây thư mục `data/` (và `.gitignore` đã loại trừ `data`).

```bash
mkdir -p data/{annotation_algorithm,annotation_manual,cropped,dataset,images,models,models_backup,temp,uploads}
```

Lưu ý: có tệp [`create_data_folder.py`](create_data_folder.py), nhưng hiện tại nó tạo thư mục trong thư mục làm việc hiện tại (không nằm dưới `data/`). Hãy lưu ý điều này nếu bạn dùng nó.

## 🚀 Sử dụng

### Khởi động web server

```bash
python app.py
```

Lệnh khởi động kế thừa từ tài liệu trước đó (được giữ lại):

```bash
python app.py -m cellist
```

Các route máy chủ mặc định quan sát được trong mã:

- UI chính: `http://localhost:8887/`
- Trang 3D: `http://localhost:8887/3d`

### Quy trình làm việc điển hình

1. Mở UI và đăng nhập.
2. Tải ảnh hiển vi lên từ bảng Create Model.
3. Chọn thuật toán nền (`Cellpose`) và tạo mô hình.
4. Để backend cắt lát ảnh và khởi tạo phát hiện.
5. Tải ảnh đã crop, xem lại/chỉnh chú thích hình chữ nhật.
6. Chạy các chu kỳ `initialize`, `pretrain`, và `train`.
7. Lưu các cập nhật thủ công qua hành động chú thích/`Update Model`.

### Tài khoản đăng nhập tích hợp sẵn trên UI (hành vi template hiện tại)

Frontend hiện kiểm tra các thông tin đăng nhập tĩnh này ở phía client:

- `admin` / `admin`
- `lachlan` / `lachlan`
- `yanjun` / `yanjun`

Đây là hành vi prototype và không phải xác thực production.

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
- Ô ảnh mặc định là `256x256`.
- Khởi tạo Cellpose dùng `model_type='nuclei'` và `gpu=True`.
- Huấn luyện và pretraining chạy bất đồng bộ thông qua các hành động kích hoạt bởi WebSocket.

## 🧪 Ví dụ

### Ví dụ: định dạng message WebSocket tạo mô hình

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

### Ví dụ: cập nhật chú thích thủ công qua WebSocket

```json
{
  "data_type": "update",
  "username": "lachlan",
  "model_id": "<model_id>",
  "image_uuid": "<cropped_id>",
  "z_where": [[0.1, 0.1, 0.2, -0.1]]
}
```

### Ví dụ: yêu cầu tải mô hình

```bash
curl -X POST http://localhost:8887/load_model/any \
  -d "model_id=<model_id>" \
  -d "cursor=0"
```

## 📚 Lấy cảm hứng từ nghiên cứu

Lazeal Cellist được lấy cảm hứng từ các nghiên cứu học sâu tiên phong, bao gồm:

1. "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
2. "Spatially Invariant Attend, Infer, Repeat"
3. "Faster Attend-Infer-Repeat with Tractable Probabilistic Models"

Những công trình này cung cấp các góc nhìn giá trị, định hướng cho việc phát triển thuật toán và phương pháp của nền tảng.

(Lưu ý: để trích dẫn chính xác, vui lòng tham khảo trực tiếp các bài báo gốc.)

## 🧭 Ghi chú phát triển

- Các lớp mô hình cốt lõi nằm trong `cellist/` (`ModelD2Init`, `ModelD2Pretrain`).
- Logic UI tương tác chính được nhúng trực tiếp trong `templates/cellist.html`.
- Lược đồ SQL và dữ liệu kiểu seed nằm trong `cellist.sql`.
- Notebook trong `notebooks/` và `polygon_sample/` cung cấp tài liệu tham chiếu khám phá.
- Hiện chưa có bộ kiểm thử tự động chuyên biệt hoặc cấu hình CI ở thư mục gốc repository.

## 🧯 Khắc phục sự cố

| Triệu chứng | Kiểm tra được đề xuất |
|---|---|
| `ModuleNotFoundError` hoặc lỗi import | Xác nhận đã chạy `conda activate cellist` trước khi chạy `python app.py`. |
| UI hiển thị nhưng thiếu style/script | Chạy `npm install` trong `statics/` và xác nhận `statics/node_modules` tồn tại. |
| MySQL báo access denied | Kiểm tra username/password trong `cellist/utils/constants.py` và chế độ plugin/auth của MySQL. |
| Ứng dụng khởi chạy nhưng thao tác model bị lỗi | Kiểm tra khả dụng CUDA/GPU; các luồng hiện tại giả định CUDA (`torch.device('cuda:0')`, Cellpose `gpu=True`). |
| Upload thành công nhưng không thấy tile/model | Đảm bảo các thư mục con trong `data/` tồn tại và có quyền ghi. |

## 🗺️ Lộ trình

Các hạng mục sau được giữ lại và sắp xếp từ tài liệu dự án hiện có/ghi chú TODO:

- Polygon sample: dùng polygon thay cho chú thích hình chữ nhật.
- Tối ưu mô hình cho hành vi `float32` với các giá trị rất nhỏ/rất lớn.
- Thu nhỏ kích thước mô hình khi có thể.
- Cải thiện độ bền vững bằng các hướng tiếp cận như thành phần lấy cảm hứng từ Transformer/stable-diffusion.
- Thêm tùy chọn mô hình nền (Threshold, Cellpose) và mô hình đích (AIR, Transformer, SD).
- Tối ưu giao diện (bao gồm chọn nhiều).
- Tối ưu backend (bao gồm cải thiện quản lý bộ nhớ/cache).
- Đóng gói dễ dùng với cấu hình DB tối thiểu (ví dụ: tùy chọn SQLite).

## 🤝 Đóng góp

### Đóng góp cho Lazeal Cellist

Lazeal Cellist là dự án mã nguồn mở, và chúng tôi hoan nghênh đóng góp từ mọi người, bất kể mức kinh nghiệm. Chúng tôi chào đón các đóng góp:

- Nâng cao hiệu quả và hiệu năng thuật toán
- Cải thiện giao diện và trải nghiệm người dùng
- Mở rộng tài liệu và ví dụ
- Sửa lỗi và tăng độ ổn định hệ thống

Trước khi bắt đầu đóng góp, vui lòng thảo luận thay đổi bạn muốn thực hiện thông qua một issue. Điều này giúp phối hợp nỗ lực và tránh công việc trùng lặp hoặc xung đột.

Để biết thêm thông tin về cách bắt đầu, vui lòng đọc hướng dẫn đóng góp.

Tài liệu đóng góp bổ sung trong repository:

- [Contribution Guidelines](CONTRIBUTING.md)
- [Pull Request Template](PULL_REQUEST_TEMPLATE.md)

## 📄 Giấy phép

Dự án này được cấp phép theo MIT License. Để biết thêm thông tin, vui lòng tham khảo tệp [LICENSE](https://chat.openai.com/LICENSE) trong repository này.

Lưu ý trạng thái repository: hiện chưa có tệp `LICENSE` ở thư mục gốc trong bản checkout này. Dòng trên được giữ lại từ README trước đó như ý định chính thức của dự án; bạn có thể thêm tệp `LICENSE` cục bộ trong thay đổi tiếp theo nếu muốn.
