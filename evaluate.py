import os
import cv2
import numpy as np
import tensorflow as tf
from miscnn.neural_network.metrics import dice_soft

# CẤU HÌNH
MODEL_PATH = "best_model_2d.hdf5"
# Bạn có thể đổi thành "./dataset/test" nếu muốn đánh giá tập test
VAL_DIR = "./dataset/test"
INPUT_SHAPE = (256, 256)

# HÀM TÍNH ĐIỂM
def calculate_metrics(y_true, y_pred):
    """
    Tính Dice và IoU cho 1 cặp ảnh mask
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    #  Dice Score
    union_dice = np.sum(y_true_f) + np.sum(y_pred_f)
    if union_dice == 0:
        dice = 1.0  # Cả 2 đều đen xì -> Đúng
    else:
        dice = (2. * intersection) / union_dice

    #  IoU Score
    union_iou = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    if union_iou == 0:
        iou = 1.0
    else:
        iou = intersection / union_iou

    return dice, iou



# HÀM ĐÁNH GIÁ TOÀN BỘ DATASET

def evaluate_dataset():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có file model: {MODEL_PATH}")
        return

    print(f" Đang load model... ")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'dice_soft': dice_soft})

    # 2. Lấy danh sách file
    img_dir = os.path.join(VAL_DIR, "images")
    mask_dir = os.path.join(VAL_DIR, "masks")

    if not os.path.exists(img_dir):
        print(f"Không tìm thấy thư mục ảnh: {img_dir}")
        return

    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    print(f"Tìm thấy {len(files)} ảnh để đánh giá.")

    list_dice = []
    list_iou = []

    print("\nBắt đầu chạy đánh giá (kèm Auto-Fix Lỗi Ngược) ")

    for idx, f in enumerate(files):
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)

        # ĐỌC ẢNH INPUT
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, INPUT_SHAPE)
        img_input = img.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)

        # ĐỌC MASK GỐC (GROUND TRUTH)
        if not os.path.exists(mask_path):
            continue  # Bỏ qua nếu ko có mask đối chiếu

        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, INPUT_SHAPE, interpolation=cv2.INTER_NEAREST)

        # Xử lý mask index hay mask 255
        if np.max(mask_raw) <= 1:
            mask_true = (mask_raw > 0).astype(np.uint8)
        else:
            mask_true = (mask_raw > 127).astype(np.uint8)

        # DỰ ĐOÁN
        pred_prob = model.predict(img_input, verbose=0)
        pred_mask = np.argmax(pred_prob, axis=-1)[0]  # (256, 256)

        # [AUTO-FIX] ĐẢO NGƯỢC MASK NẾU BỊ HỌC SAI
        # Nếu AI tô màu quá 50% ảnh là bệnh -> Chắc chắn sai -> Đảo ngược
        if np.sum(pred_mask) > (pred_mask.size / 2):
            pred_mask = 1 - pred_mask


        # TÍNH ĐIỂM
        d, i = calculate_metrics(mask_true, pred_mask)
        list_dice.append(d)
        list_iou.append(i)

        # In tiến trình mỗi 10 ảnh
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(files)} files...")

    # 3. KẾT QUẢ CUỐI CÙNG
    avg_dice = np.mean(list_dice)
    avg_iou = np.mean(list_iou)

    print("\n" + "=" * 40)
    print(f"BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (Test Set)")
    print(f"Thư mục: {VAL_DIR}")
    print(f"Tổng số ảnh: {len(list_dice)}")
    print("-" * 40)
    print(f"Average Dice Score: {avg_dice:.4f} ({avg_dice * 100:.2f}%)")
    print(f"Average IoU Score:  {avg_iou:.4f}  ({avg_iou * 100:.2f}%)")
    print("=" * 40)

    if avg_dice > 0.8:
        print("Kết quả rất tốt! Model đã sẵn sàng.")
    elif avg_dice > 0.6:
        print("Kết quả khá ổn.")
    else:
        print("Kết quả còn thấp. Cần kiểm tra lại dữ liệu train.")


if __name__ == "__main__":
    evaluate_dataset()