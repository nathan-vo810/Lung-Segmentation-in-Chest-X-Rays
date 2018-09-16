import os
import sys
import time

from src.load_data import dataProcess
from src.UNet_model import UNet

DIR = os.path.dirname(__file__)

WIDTH = 512
HEIGHT = 512

model = UNet()
data_loader = dataProcess(WIDTH, HEIGHT)

# Train
# images_train, labels_train = data_loader.loadTrainData()
# model.fit(images_train, labels_train)
# model.save(MODEL_PATH)

# Predict
model.load()
images_test, images_info = data_loader.loadTestData()
start = time.process_time()
model.predict(images_test, images_info)
print('Time eslapsed: {}'.format(time.process_time() - start))
