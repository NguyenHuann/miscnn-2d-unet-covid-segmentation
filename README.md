# COVID-19 CT Scan Lesion Segmentation using U-Net 2D

Dự án này triển khai mô hình mạng nơ-ron tích chập **U-Net 2D** kết hợp với framework **MIScnn** để tự động phân đoạn (khoanh vùng) các vết tổn thương phổi do COVID-19 trên ảnh chụp CT.

## Tính năng nổi bật

* **Custom Data Generator:** Tối ưu hóa bộ nhớ, hỗ trợ training trên dataset lớn mà không bị tràn RAM.
* **Architecture:** Sử dụng kiến trúc U-Net kinh điển với các lớp Conv2D, MaxPooling và Transpose Conv.
* **Auto-fix Class Inversion:** Tự động phát hiện và đảo ngược mask dự đoán nếu mô hình học nhầm giữa nền và bệnh.
* **Visualization:**
    * Xuất ảnh so sánh trực quan (Input - Ground Truth - Prediction).
    * Minh họa quá trình biến đổi đặc trưng qua từng lớp của U-Net (Feature Map Visualization).
* **Metrics:** Đánh giá chính xác bằng Dice Score và IoU Score.

## Cài đặt môi trường

### 1. Yêu cầu hệ thống
* Python 3.8+
* TensorFlow 2.x
* Git LFS (Để tải dataset)

### 2. Clone Repository và Cài đặt thư viện
```bash
# Clone dự án
git clone https://github.com/NguyenHuann/miscnn-2d-unet-covid-segmentation.git
cd miscnn-2d-unet-covid-segmentation

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

```

### 3. Tải và Giải nén Dữ liệu (Quan trọng)

Dự án sử dụng **Git LFS** để lưu trữ file nén dataset. Hãy đảm bảo bạn đã tải file `dataset.rar` về máy.

```bash
# Cài đặt Git LFS (nếu chưa có)
git lfs install

# Kéo file dataset về
git lfs pull

```

**BẮT BUỘC:** Sau khi tải xong, bạn cần giải nén file `dataset.rar` vào thư mục gốc của dự án sao cho cấu trúc thư mục trông như sau:

```text
miscnn-2d-unet-covid-segmentation/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── train.py
├── inference.py
├── evaluate.py
├── ...

```

---

## Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)

Chạy file `train.py` để bắt đầu quá trình huấn luyện. Code sẽ tự động chia dữ liệu dựa trên thư mục `train/val` và lưu model tốt nhất vào `best_model_2d.hdf5`.

```bash
python train.py

```

* **Cấu hình:** Epochs=50, Batch_size=4, Optimizer=Adam (1e-4).
* **Cơ chế:** Early Stopping (dừng nếu không cải thiện sau 10 epoch) và Model Checkpoint.

### 2. Đánh giá mô hình (Evaluation)

Để tính toán độ chính xác trung bình (Dice Score, IoU) trên toàn bộ tập Validation/Test:

```bash
python evaluate.py

```

### 3. Dự đoán và Trực quan hóa (Inference)

Chạy file `inference.py` để dự đoán trên một ảnh cụ thể, xuất ra mask đen trắng và các biểu đồ minh họa.

```bash
python inference.py

```

**Kết quả đầu ra:**

* File ảnh mask đen trắng sẽ được lưu vào thư mục `output/`.
* Biểu đồ so sánh (Figure 1) và biểu đồ kiến trúc mạng (Figure 2) cũng được lưu tại `output/`.
* Các lớp trung gian (Feature Maps) được lưu chi tiết trong thư mục `unet_output/`.

---

## Kết quả minh họa

### Figure 1: Kết quả phân đoạn

So sánh giữa Ảnh gốc, Mask chuẩn của bác sĩ và Kết quả AI dự đoán.
![Figure 1](/images/Figure_1.png)

### Figure 2: Kiến trúc U-Net trực quan

Minh họa quá trình nén ảnh (Encoder) và khôi phục ảnh (Decoder) qua các tầng mạng.
![Figure 2](/images/Figure_2.jpg)

---

## Cấu trúc dự án

* `train.py`: Mã nguồn chính để huấn luyện mô hình.
* `inference.py`: Mã nguồn dự đoán, xử lý lỗi ngược lớp và vẽ biểu đồ minh họa.
* `evaluate.py`: Mã nguồn đánh giá độ chính xác trên tập dữ liệu lớn.
* `best_model_2d.hdf5`: File trọng số mô hình đã được train (Best weights).
* `dataset.rar`: File nén chứa dữ liệu ảnh CT và Mask.
* `output/`: Thư mục chứa kết quả dự đoán (Mask, Figure).
* `unet_output/`: Thư mục chứa ảnh feature map từng lớp của U-Net.

---

## Tác giả

* **Thực hiện:** Nhóm 2 - 23CN1
* **Nghiên cứu:** Phân đoạn tổn thương phổi Covid-19 trên ảnh CT sử dụng Framework MIScnn và kiến trúc U-net.
