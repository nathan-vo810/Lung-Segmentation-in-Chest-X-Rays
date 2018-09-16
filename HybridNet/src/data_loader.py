import numpy as np
import cv2
import os
import pickle

DIR = os.path.dirname(__file__)


class DataLoader:
    def __init__(self, data_path, window_size):
        self.image_type = 'jpg'
        self.window_size = window_size
        self.data_path = os.path.join(DIR, '../data/' + str(window_size) + '/' + data_path + '/')

    def get_images(self, is_lung):
        folder = '1/' if is_lung else '0/'
        images = os.listdir(self.data_path + folder)
        images = [image for image in images if image.endswith(self.image_type)]
        return sorted(images[:])

    def append_images_to_array(self, array, images, is_lung, index):
        i = index
        folder = '1/' if is_lung else '0/'
        for file in images:
            image = cv2.imread(self.data_path + folder + file)
            array[i] = image
            i += 1

    def generate_labels_array(self, no_lung_images, no_non_lung_images):
        labels_array = np.ndarray((no_lung_images + no_non_lung_images, 1))
        for i in range(no_lung_images):
            labels_array[i] = 1
        for i in range(no_lung_images, no_lung_images + no_non_lung_images):
            labels_array[i] = 0
        return labels_array

    def load_pkl(self, pkl_file):
        with open(pkl_file, 'rb') as file:
            locations = pickle.load(file)
        return locations

    def load_train_data_hybrid(self):
        print('Load lung samples...')
        lung_images = self.get_images(is_lung=True)
        lung_images_location = self.load_pkl(self.data_path + '1/location.pkl')
        print('Load non lung samples...')
        non_lung_images = self.get_images(is_lung=False)
        non_lung_images_location = self.load_pkl(self.data_path + '0/location.pkl')

        locations = np.concatenate((lung_images_location, non_lung_images_location), axis=0)
        locations = np.reshape(locations, (locations.shape[0], -1, 1))

        no_lung_images = len(lung_images)
        no_non_lung_images = len(non_lung_images)

        train_data = np.ndarray((no_lung_images + no_non_lung_images, self.window_size, self.window_size, 3))
        self.append_images_to_array(array=train_data, images=lung_images, is_lung=True, index=0)
        self.append_images_to_array(array=train_data, images=non_lung_images, is_lung=False, index=no_lung_images)

        label_data = self.generate_labels_array(no_lung_images, no_non_lung_images)

        return [train_data, locations], label_data

    def load_train_data(self):
        print('Load lung samples...')
        lung_images = self.get_images(is_lung=True)

        print('Load non lung samples...')
        non_lung_images = self.get_images(is_lung=False)

        no_lung_images = len(lung_images)
        no_non_lung_images = len(non_lung_images)

        train_data = np.ndarray((no_lung_images + no_non_lung_images, self.window_size, self.window_size, 3))
        self.append_images_to_array(array=train_data, images=lung_images, is_lung=True, index=0)
        self.append_images_to_array(array=train_data, images=non_lung_images, is_lung=False, index=no_lung_images)

        label_data = self.generate_labels_array(no_lung_images, no_non_lung_images)

        return train_data, label_data
