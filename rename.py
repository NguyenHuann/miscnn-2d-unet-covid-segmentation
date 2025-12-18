import os

# CONFIG
IMAGE_DIR = "dataset_/images"
MASK_DIR  = "dataset_/masks"

EXT = ".png"
START_INDEX = 1
DIGITS = 4

DRY_RUN = False

# RENAME FUNCTION
def rename_pairs(image_dir, mask_dir):
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(EXT)]
    )
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.lower().endswith(EXT)]
    )

    if len(image_files) != len(mask_files):
        raise RuntimeError(
            f"Số image ({len(image_files)}) ≠ số mask ({len(mask_files)})"
        )

    print(f"✔ Found {len(image_files)} image-mask pairs")

    for idx, (img, msk) in enumerate(zip(image_files, mask_files), start=START_INDEX):
        new_name = f"{str(idx).zfill(DIGITS)}{EXT}"

        img_old = os.path.join(image_dir, img)
        img_new = os.path.join(image_dir, new_name)

        msk_old = os.path.join(mask_dir, msk)
        msk_new = os.path.join(mask_dir, new_name)

        if DRY_RUN:
            print(f"[DRY] {img} -> {new_name}")
            print(f"[DRY] {msk} -> {new_name}")
        else:
            os.rename(img_old, img_new)
            os.rename(msk_old, msk_new)

    print("Rename completed successfully!")


# MAIN
if __name__ == "__main__":
    print("=== Dataset Rename Tool ===")
    print(f"Images dir: {IMAGE_DIR}")
    print(f"Masks  dir: {MASK_DIR}")
    print(f"Extension: {EXT}")
    print(f"Dry run : {DRY_RUN}")
    print("===========================")

    rename_pairs(IMAGE_DIR, MASK_DIR)
