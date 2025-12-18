import os
import shutil
import random

# --- CẤU HÌNH ---
# Đường dẫn dữ liệu gốc (Input)
SRC_IMAGES_DIR = "dataset_/images"
SRC_MASKS_DIR = "dataset_/masks"

# Đường dẫn dữ liệu đích sau khi chia (Output)
OUTPUT_DIR = "dataset"

# Số lượng file test cố định
TEST_SIZE_FIXED = 10
# Tỷ lệ tập Validation (ví dụ 0.2 là 20% của phần còn lại sau khi tách test)
VAL_RATIO = 0.2


def split_dataset_only():
    print(">>> BẮT ĐẦU QUÁ TRÌNH CHIA DỮ LIỆU...")

    # 1. Kiểm tra thư mục nguồn
    if not os.path.exists(SRC_IMAGES_DIR) or not os.path.exists(SRC_MASKS_DIR):
        print(f"LỖI: Không tìm thấy thư mục nguồn.")
        print(f"Vui lòng đảm bảo cấu trúc: {SRC_IMAGES_DIR} và {SRC_MASKS_DIR} tồn tại.")
        return

    # 2. Xóa thư mục đích cũ nếu đã tồn tại để làm sạch (Optional)
    if os.path.exists(OUTPUT_DIR):
        print(f"-> Phát hiện thư mục '{OUTPUT_DIR}' cũ. Đang xóa để tạo mới...")
        shutil.rmtree(OUTPUT_DIR)

    # 3. Tạo cấu trúc thư mục mới
    print(f"-> Đang tạo cấu trúc thư mục tại '{OUTPUT_DIR}'...")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

    # 4. Quét và ghép cặp file
    all_images = os.listdir(SRC_IMAGES_DIR)
    valid_pairs = []

    print("-> Đang kiểm tra tính toàn vẹn của dữ liệu (khớp ảnh và mask)...")
    for filename in all_images:
        mask_path = os.path.join(SRC_MASKS_DIR, filename)
        if os.path.isfile(mask_path):
            valid_pairs.append(filename)
        else:
            print(f"   [Cảnh báo] Bỏ qua '{filename}' vì không tìm thấy mask tương ứng.")

    total_files = len(valid_pairs)
    print(f"-> Tổng số cặp dữ liệu hợp lệ: {total_files}")

    if total_files <= TEST_SIZE_FIXED:
        print("LỖI: Số lượng dữ liệu quá ít để chia tập Test (cần > 10 file).")
        return

    # 5. Trộn và chia dữ liệu
    random.shuffle(valid_pairs)

    # Tách Test
    test_files = valid_pairs[:TEST_SIZE_FIXED]
    remaining = valid_pairs[TEST_SIZE_FIXED:]

    # Tách Val và Train
    num_val = int(len(remaining) * VAL_RATIO)
    val_files = remaining[:num_val]
    train_files = remaining[num_val:]

    splits = {
        'test': test_files,
        'val': val_files,
        'train': train_files
    }

    # 6. Copy file vào thư mục đích
    print("-> Đang sao chép file...")
    for split_name, files in splits.items():
        print(f"   + Đang xử lý tập '{split_name}': {len(files)} file")

        dest_img_path = os.path.join(OUTPUT_DIR, split_name, 'images')
        dest_mask_path = os.path.join(OUTPUT_DIR, split_name, 'masks')

        for filename in files:
            # Copy Image
            shutil.copy(os.path.join(SRC_IMAGES_DIR, filename),
                        os.path.join(dest_img_path, filename))
            # Copy Mask
            shutil.copy(os.path.join(SRC_MASKS_DIR, filename),
                        os.path.join(dest_mask_path, filename))

    print("\n>>> HOÀN TẤT! Dữ liệu đã sẵn sàng tại thư mục:", OUTPUT_DIR)
    print(f"    Cấu trúc: {OUTPUT_DIR}/train, {OUTPUT_DIR}/val, {OUTPUT_DIR}/test")


if __name__ == "__main__":
    split_dataset_only()