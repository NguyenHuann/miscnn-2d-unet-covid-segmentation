import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from miscnn.neural_network.metrics import dice_soft

# CẤU HÌNH

MODEL_PATH = "best_model_2d.hdf5"
TEST_IMAGE_PATH = "./dataset/test/images/1638.png"
INPUT_SHAPE = (256, 256)

# Tên thư mục output
OUTPUT_DIR = "output"

# HÀM TÍNH ĐIỂM

def calculate_metrics(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union_dice = np.sum(y_true_f) + np.sum(y_pred_f)
    if union_dice == 0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union_dice

    union_iou = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    if union_iou == 0:
        iou = 1.0
    else:
        iou = intersection / union_iou
    return dice, iou

# HÀM DỰ ĐOÁN, LƯU MASK VÀ HIỂN THỊ

def predict_and_viz(image_path, model):
    # 1. LOAD ẢNH
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    mask_path = image_path.replace("images", "masks")
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None: return

    # Resize & Normalize
    img_resized = cv2.resize(original_img, INPUT_SHAPE)
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    # 2. LOAD MASK GỐC (để so sánh)
    mask_true = None
    if os.path.exists(mask_path):
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, INPUT_SHAPE, interpolation=cv2.INTER_NEAREST)
        if np.max(mask_raw) <= 1:
            mask_true = (mask_raw > 0).astype(np.uint8)
        else:
            mask_true = (mask_raw > 127).astype(np.uint8)

    # 3. DỰ ĐOÁN
    print("Đang dự đoán...")
    prediction = model.predict(img_input)
    pred_mask = np.argmax(prediction, axis=-1)[0]

    # Fix lỗi ngược lớp
    pixel_count = pred_mask.size
    ones_count = np.sum(pred_mask)
    if ones_count > (pixel_count / 2):
        print("Phát hiện lỗi ngược lớp -> Đảo ngược mask.")
        pred_mask = 1 - pred_mask

    # In điểm số (nếu có mask gốc)
    if mask_true is not None:
        d, i = calculate_metrics(mask_true, pred_mask)
        print(f"Dice: {d:.4f} | IoU: {i:.4f}")


    # [QUAN TRỌNG] LƯU FILE MASK ĐEN TRẮNG RA OUTPUT
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mask_to_save = (pred_mask * 255).astype(np.uint8)

    base_name = os.path.basename(image_path)
    save_name = f"mask_pred_{base_name}"  # Ví dụ: mask_pred_1638.png
    save_path = os.path.join(OUTPUT_DIR, save_name)

    cv2.imwrite(save_path, mask_to_save)
    print(f"Đã lưu ảnh Mask tại: {save_path}")

    # 4. HIỂN THỊ TRỰC QUAN (Vẫn giữ lại để bạn xem trên màn hình)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1);
    plt.title("Ảnh gốc");
    plt.axis('off')
    plt.imshow(img_resized, cmap='gray')

    plt.subplot(1, 3, 2);
    plt.title("Mask Gốc");
    plt.axis('off')
    if mask_true is not None: plt.imshow(mask_true, cmap='gray')

    plt.subplot(1, 3, 3);
    plt.title(f"AI Mask (Output) \nDice: {d:.4f} | IoU: {i:.4f}");

    plt.axis('off')
    # Hiển thị đúng cái mask vừa lưu (đen trắng)
    plt.imshow(pred_mask, cmap='gray')

    plt.show()


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Load model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'dice_soft': dice_soft})
        predict_and_viz(TEST_IMAGE_PATH, model)
    else:
        print("Chưa có model.")