import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model  # Cần thêm cái này
from miscnn.neural_network.metrics import dice_soft


# CẤU HÌNH

MODEL_PATH = "best_model_2d.hdf5"
TEST_IMAGE_PATH = "./dataset/test/images/0896.png"  # Đảm bảo file này tồn tại
INPUT_SHAPE = (256, 256)

# Thư mục lưu kết quả cuối cùng (Mask)
OUTPUT_DIR = "output"
# [MỚI] Thư mục lưu ảnh từng bước của U-Net
UNET_STEPS_DIR = "unet_output"



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



# [MỚI] HÀM LƯU CÁC BƯỚC TRUNG GIAN (FEATURE MAPS)

def save_intermediate_layers(model, img_input, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Đã tạo thư mục: {save_dir}")

    # 1. Chọn các lớp muốn xem (Conv, Pool, Transpose, Concat)
    # Bỏ qua Input layer hoặc Dropout để đỡ rối
    layer_outputs = [layer.output for layer in model.layers
                     if 'conv' in layer.name or 'pool' in layer.name or 'gath' in layer.name]

    # Lấy tên lớp để đặt tên file
    layer_names = [layer.name for layer in model.layers
                   if 'conv' in layer.name or 'pool' in layer.name or 'gath' in layer.name]

    # 2. Tạo model trung gian
    # Input: Ảnh gốc -> Output: Danh sách các feature map của từng lớp
    visualization_model = Model(inputs=model.input, outputs=layer_outputs)

    # 3. Dự đoán (Lấy feature maps)
    # activations là một list, mỗi phần tử là kết quả của 1 lớp
    activations = visualization_model.predict(img_input)

    print(f"--- Đang xuất {len(activations)} lớp trung gian vào {save_dir} ---")

    for i, activation in enumerate(activations):
        # activation có shape: (1, height, width, channels)

        # Lấy kích thước hiện tại (để in log xem nó resize thế nào)
        h, w = activation.shape[1], activation.shape[2]
        num_channels = activation.shape[3]

        # Xử lý để thành ảnh 2D:
        # Cách 1: Lấy trung bình cộng của tất cả các kênh (Heatmap tổng quát)
        feature_map = np.mean(activation[0, :, :, :], axis=-1)

        # Cách 2: (Tùy chọn) Chỉ lấy kênh đầu tiên nếu muốn nhìn chi tiết vân
        # feature_map = activation[0, :, :, 0]

        # Chuẩn hóa về 0-255 để lưu ảnh
        feature_map -= feature_map.min()
        if feature_map.max() > 0:
            feature_map /= feature_map.max()
        feature_map *= 255
        feature_map = feature_map.astype(np.uint8)

        # Lưu ảnh
        # Đặt tên file có số thứ tự để sort theo đúng quy trình U-Net
        file_name = f"{i:03d}_{layer_names[i]}_{h}x{w}.png"
        save_path = os.path.join(save_dir, file_name)

        # Dùng colormap 'jet' hoặc 'viridis' để nhìn cho đẹp (như ảnh nhiệt),
        # hoặc lưu đen trắng. Ở đây lưu đen trắng cho đúng yêu cầu "ảnh".
        cv2.imwrite(save_path, feature_map)

    print(f"Đã lưu xong toàn bộ các bước vào thư mục: {save_dir}")



# HÀM DỰ ĐOÁN CHÍNH

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

    # 2. LOAD MASK GỐC
    mask_true = None
    if os.path.exists(mask_path):
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, INPUT_SHAPE, interpolation=cv2.INTER_NEAREST)
        if np.max(mask_raw) <= 1:
            mask_true = (mask_raw > 0).astype(np.uint8)
        else:
            mask_true = (mask_raw > 127).astype(np.uint8)

    # 3. DỰ ĐOÁN MASK CUỐI CÙNG
    print("Đang dự đoán Mask...")
    prediction = model.predict(img_input)
    pred_mask = np.argmax(prediction, axis=-1)[0]

    # Fix lỗi ngược lớp
    pixel_count = pred_mask.size
    ones_count = np.sum(pred_mask)
    if ones_count > (pixel_count / 2):
        print("Phát hiện lỗi ngược lớp -> Đảo ngược mask.")
        pred_mask = 1 - pred_mask


    save_intermediate_layers(model, img_input, UNET_STEPS_DIR)


    # Tính điểm
    d, i = 0.0, 0.0
    if mask_true is not None:
        d, i = calculate_metrics(mask_true, pred_mask)
        print(f"Dice: {d:.4f} | IoU: {i:.4f}")

    # 4. LƯU MASK RA OUTPUT
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mask_to_save = (pred_mask * 255).astype(np.uint8)
    base_name = os.path.basename(image_path)
    save_name = f"mask_pred_{base_name}"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    cv2.imwrite(save_path, mask_to_save)
    print(f"Đã lưu ảnh Mask cuối cùng tại: {save_path}")

    # 5. HIỂN THỊ
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Ảnh gốc")
    plt.axis('off')
    plt.imshow(img_resized, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Mask Gốc")
    plt.axis('off')
    if mask_true is not None: plt.imshow(mask_true, cmap='gray')

    plt.subplot(1, 3, 3)
    title_viz = f"AI Mask (Output)"
    if mask_true is not None:
        title_viz += f"\nDice: {d:.4f} | IoU: {i:.4f}"
    plt.title(title_viz)
    plt.axis('off')
    plt.imshow(pred_mask, cmap='gray')

    plt.show()



# MAIN

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Load model: {MODEL_PATH}")
        # Cần custom_objects để load model có loss lạ
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'dice_soft': dice_soft})
        predict_and_viz(TEST_IMAGE_PATH, model)
    else:
        print("Chưa có model.")