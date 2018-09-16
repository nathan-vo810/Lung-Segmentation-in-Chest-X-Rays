
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from src.callbacks import Metrics
from src.utils import *
import os
import cv2
from scipy import misc
import pickle as pkl


DIR = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(DIR, '../weight/deep_model/')
weight_11_file = '11_weight_deep_cnn_val_0.771_loss_0.451.hdf5'
weight_32_file = '32_weight_deep_cnn_val_0.859_loss_0.274.hdf5'
weight_99_file = '99_weight_deep_cnn_val_0.942_loss_0.123.hdf5'

TEST_DIR = os.path.join(DIR, '../data/test_data/')
RESULT_DIR = os.path.join(DIR, '../data/v1-99/')
LOG_DIR = os.path.join(DIR, '../log')
WINDOW_SIZE = 11
BIG_WINDOW_SIZE = 99


class DeepCNN:
    def __init__(self, load_weight=False, batch_size=32, epochs=100, lr=1e-4):
        self.load_weight = load_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # self.model_11, self.model_99 = self.prepare_models()
        # self.model = self.build_model_window_99(load_weight=True)

    def build_model_window_11(self, load_weight=False):
        '''
        Window size = 11 -> Model: Conv1 -> Conv2 -> Conv3 -> Pool3 -> Conv4 -> Pool4 -> Conv5 -> FC1 -> Sigmoid
        '''
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',
                         input_shape=(11, 11, 3), name='Conv1'))

        model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2'))

        model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv3'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv4'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv5'))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        if load_weight:
            print("Loading weight...")
            model.load_weights(WEIGHT_DIR + weight_11_file)
            print("Weight loaded.")

        return model

    def build_model_window_99(self, load_weight=False):
        '''
        Window size = 32,99 -> Model: (Conv -> Pool) x 4 -> Conv5 -> FC1 -> Sigmoid
        '''
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',
                         input_shape=(BIG_WINDOW_SIZE, BIG_WINDOW_SIZE, 3), name='Conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv4'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv5'))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        if load_weight:
            print("Loading weight...")
            model.load_weights(WEIGHT_DIR + weight_99_file)
            print("Weight loaded.")

        return model

    def fit(self, x_train, y_train, x_val, y_val, window_size):
        dice_coeff = Metrics()
        if window_size == 11:
            model = self.build_model_window_11(load_weight=False)
        else:
            model = self.build_model_window_99(load_weight=False)
        print("Training...")

        weight_file = str(window_size) + '_weight_deep_cnn_val_{val_acc:.3f}_loss_{loss:.3f}.hdf5'
        model_checkpoint = ModelCheckpoint(WEIGHT_DIR + weight_file, monitor='val_acc', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=LOG_DIR + 'train_deep_log_99', histogram_freq=0, write_graph=True, write_images=False)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                            callbacks=[model_checkpoint, tensorboard, dice_coeff])

        return history

    def prepare_models(self):
        model_11 = self.build_model_window_11(load_weight=True)
        model_99 = self.build_model_window_99(load_weight=True)
        return model_11, model_99

    def predict(self, image_path, window_size, threshold=0.5):

        print("Loading test image...")
        test_image = cv2.imread(os.path.join(TEST_DIR,image_path))
        self.save(test_image, window_size)
        windows = self.load(window_size)
        print("Loaded.")

        print("Predicting...")
        predict_value = self.build_model_window_99(True).predict(windows, verbose=1)
        result_image = process_predict_value(predict_value, threshold, test_image, window_size)
        print("Predicted.")

        print("Saving...")
        result_path = os.path.join(RESULT_DIR, image_path)
        misc.imsave(result_path, result_image)
        print("Saved.")

    def save(self, test_image, window_size):
        windows = get_test_data(test_image, window_size)
        with open('windows_{}.pkl'.format(str(window_size)), 'wb') as file:
            pkl.dump(windows, file, pkl.HIGHEST_PROTOCOL)

    def load(self, window_size):
        with open('windows_{}.pkl'.format(str(window_size)), 'rb') as file:
            windows = pkl.load(file)
        return windows

    def predict_custom(self, image_path):


        print("Loading test image...")
        test_image = cv2.imread(os.path.join(TEST_DIR,image_path))
        self.save(test_image, 99)
        windows_99 = self.load(99)
        # windows_99 = self.load(99)
        print("Loaded.")

        print("Predicting...")

        predict_value_99 = self.model_99.predict(windows_99, verbose=1)

        height, width = test_image.shape[:2]
        n_cols, n_rows = int(width / BIG_WINDOW_SIZE), int(height / BIG_WINDOW_SIZE)

        result_image = np.array([]).reshape(0, n_cols * BIG_WINDOW_SIZE)
        row = np.array([]).reshape((BIG_WINDOW_SIZE, 0))

        alpha = 0.6
        beta = 0.8
        threshold = 0.6

        for i in range(len(windows_99)):
            windows_11 = get_test_data(windows_99[i], window_size=11)
            predict_value_11 = self.model_11.predict(windows_11, verbose=1)
            predict_value = alpha * predict_value_99[i] + beta * predict_value_11
            predicted_window = process_predict_value(predict_value, threshold, windows_99[i], 11)
            row = np.hstack((row, predicted_window))
            if (i + 1) % n_cols == 0:
                result_image = np.vstack((result_image, row))
                row = np.array([]).reshape((BIG_WINDOW_SIZE, 0))

        print("Predicted.")

        print("Saving...")
        result_path = os.path.join(RESULT_DIR, image_path)
        misc.imsave(result_path, result_image)
        # misc.imsave(TEST_DIR + '{:2f}_{:2f}_{:2f}.jpg'.format(alpha, beta, threshold), result_image)
        print("Saved.")


def load_all_data(window_size):
    print("Loading training data...")
    x_train, y_train = load_data('generate', window_size)
    print("Loading validate data...")
    x_val, y_val = load_data('validate', window_size)
    print("Data loaded.")

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    network = DeepCNN()

    x_train, y_train, x_val, y_val = load_all_data(window_size=99)
    network.fit(x_train, y_train, x_val, y_val, window_size=99)

    network.predict(TEST_DIR, window_size=11, threshold=0.5)
    # Predict
    images = os.listdir(TEST_DIR)
    images = [image for image in images if image.endswith('.jpg')]

    for i, image in enumerate(images):
        print("Predicting {}/{}".format(i + 1, len(images)))
        network.predict(image, 99)
