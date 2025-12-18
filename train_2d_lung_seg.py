import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, \
    Activation
from tensorflow.keras.models import Model

from miscnn.data_loading.interfaces.abstract_io import Abstract_IO
from miscnn.neural_network.metrics import dice_soft


# INTERFACE (Tự động tìm file trong train/val)

class FlatFolderInterface(Abstract_IO):
    def __init__(self, classes, channels=1, three_dim=False):
        self.classes = classes
        self.channels = channels
        self.three_dim = three_dim
        self.data_path = None
        self.file_map = {}  # Lưu vị trí file (train hay val)

    def initialize(self, data_path):
        self.data_path = data_path
        self.file_map = {}
        all_files = []

        # Quét cả 2 thư mục
        subsets = ['train', 'val']

        for subset in subsets:
            img_dir = os.path.join(data_path, subset, "images")
            if not os.path.exists(img_dir):
                continue

            files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
            for f in files:
                # Ghi nhớ file này thuộc folder nào
                self.file_map[f] = os.path.join(data_path, subset)
                all_files.append(f)

        all_files.sort()
        return all_files

    def load_image(self, image_id):
        subset_path = self.file_map.get(image_id)
        path = os.path.join(subset_path, "images", image_id)

        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img_data.ndim == 2:
            img_data = np.expand_dims(img_data, axis=-1)

        return img_data, {"spacing": (1.0, 1.0, 1.0)}

    def load_segmentation(self, image_id):
        subset_path = self.file_map.get(image_id)
        path = os.path.join(subset_path, "masks", image_id)

        mask_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Binary mask (0 và 1)
        mask_data = (mask_data > 127).astype(np.uint8)

        if mask_data.ndim == 2:
            mask_data = np.expand_dims(mask_data, axis=-1)

        return mask_data

    def load_prediction(self, image_id):
        return self.load_segmentation(image_id)

    def save_prediction(self, prediction, image_id):
        pass

    def check_file_termination(self, file):
        return file.endswith(".png")



# CUSTOM GENERATOR (Dùng tf.keras.utils.Sequence có sẵn)

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sample_list, interface, batch_size=4, input_shape=(256, 256), n_classes=2, shuffle=True):
        self.sample_list = sample_list
        self.interface = interface
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.sample_list))
        self.on_epoch_end()

    def __len__(self):
        # Tính số bước (steps) trong 1 epoch
        return int(np.floor(len(self.sample_list) / self.batch_size))

    def __getitem__(self, index):
        # Lấy danh sách ID cho batch này
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.sample_list[k] for k in indexes]

        # Sinh dữ liệu
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.input_shape, 1))
        # Output cho Softmax 2 lớp: (Batch, H, W, 2)
        y = np.empty((self.batch_size, *self.input_shape, self.n_classes))

        for i, ID in enumerate(list_IDs_temp):
            # 1. Load ảnh & Mask từ Interface
            img = self.interface.load_image(ID)[0]  # load_image trả về (data, meta)
            mask = self.interface.load_segmentation(ID)

            # 2. Resize
            if img.shape[0] != self.input_shape[0] or img.shape[1] != self.input_shape[1]:
                img = cv2.resize(img, self.input_shape)
                mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)

                if img.ndim == 2: img = np.expand_dims(img, axis=-1)
                if mask.ndim == 2: mask = np.expand_dims(mask, axis=-1)

            # 3. Normalize (0-1)
            img = img.astype('float32') / 255.0

            # 4. One-Hot Encoding cho Mask (Thủ công, không cần import thêm)
            # Mask gốc: (H,W,1) giá trị 0 hoặc 1
            # Tạo channel background (giá trị 1 nếu mask=0)
            bg = (mask == 0).astype(np.float32)
            # Tạo channel foreground (giá trị 1 nếu mask=1)
            fg = (mask == 1).astype(np.float32)

            # Gộp lại thành (H,W,2)
            mask_onehot = np.concatenate([bg, fg], axis=-1)

            X[i,] = img
            y[i,] = mask_onehot

        return X, y



# KIẾN TRÚC U-NET 2D

class Architecture2D:
    def __init__(self, n_filters=32, n_labels=2):
        self.n_filters = n_filters
        self.n_labels = n_labels

    def create_model_2D(self, input_shape, activation='softmax', **kwargs):
        inputs = Input(input_shape)

        c1 = self.conv_block(inputs, self.n_filters)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = self.conv_block(p1, self.n_filters * 2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = self.conv_block(p2, self.n_filters * 4)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = self.conv_block(p3, self.n_filters * 8)

        u3 = Conv2DTranspose(self.n_filters * 4, (2, 2), strides=(2, 2), padding='same')(c4)
        u3 = concatenate([u3, c3])
        c5 = self.conv_block(u3, self.n_filters * 4)

        u2 = Conv2DTranspose(self.n_filters * 2, (2, 2), strides=(2, 2), padding='same')(c5)
        u2 = concatenate([u2, c2])
        c6 = self.conv_block(u2, self.n_filters * 2)

        u1 = Conv2DTranspose(self.n_filters, (2, 2), strides=(2, 2), padding='same')(c6)
        u1 = concatenate([u1, c1])
        c7 = self.conv_block(u1, self.n_filters)

        outputs = Conv2D(self.n_labels, (1, 1), activation=activation)(c7)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def conv_block(self, input_tensor, filters):
        x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x



# MAIN - SETUP VÀ TRAIN

if __name__ == "__main__":
    DATA_PATH = "./dataset"
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    EPOCHS = 100
    INPUT_SHAPE = (256, 256)

    # 1. Interface & Load danh sách file
    print("Khởi tạo Interface")
    my_interface = FlatFolderInterface(classes=NUM_CLASSES)
    all_files = my_interface.initialize(DATA_PATH)

    # 2. Tách Train / Val dựa trên thư mục đã lưu trong file_map
    train_ids = [f for f in all_files if "train" in my_interface.file_map[f]]
    val_ids = [f for f in all_files if "val" in my_interface.file_map[f]]

    print(f"Train files: {len(train_ids)}")
    print(f"Val files:   {len(val_ids)}")

    if len(train_ids) == 0:
        print("LỖI: Không tìm thấy ảnh train. Kiểm tra lại folder dataset/train/images")
        sys.exit()

    # 3. Tạo Data Generators
    print("Tạo Generators")
    train_gen = CustomDataGenerator(train_ids, my_interface, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE,
                                    shuffle=True)
    val_gen = CustomDataGenerator(val_ids, my_interface, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, shuffle=False)

    # 4. Build Model
    print("Build Model")
    arch = Architecture2D(n_filters=32, n_labels=NUM_CLASSES)
    model = arch.create_model_2D(input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=dice_soft,
                  metrics=['accuracy', dice_soft])

    # 5. Callbacks (Dừng thông minh)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("best_model_2d.hdf5", monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    ]

    # 6. Train
    print("\nBắt đầu Training")
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=EPOCHS,
              callbacks=callbacks)

    print("\nHOÀN TẤT")