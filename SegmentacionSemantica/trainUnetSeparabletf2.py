import os

import cv2
import modelUnetSeparabletf2 as mu
import numpy as np
import pandas as pd
import tensorflow as tf
from config import clases
from imageAugment import DataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


IMAGES_PATH = "../testImage"  # img
LABLES_PATH = "../testLabel"  # json
WEIGHTS_FILE_PATH = "pesosunetseparable256final.h5"


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split(".")[0])


image_paths = [os.path.join(IMAGES_PATH, x) for x in sorted_fns(IMAGES_PATH)]
annot_paths = [os.path.join(LABLES_PATH, x) for x in sorted_fns(LABLES_PATH)]

tg = DataGenerator(
    image_paths=image_paths, annot_paths=annot_paths, batch_size=1, augment=False
)


# Create a callback that saves the model's weights every 5 epochs
checkpoint = ModelCheckpoint(
    WEIGHTS_FILE_PATH, monitor="dice", verbose=1, save_best_only=True, mode="max"
)

modelo = mu.modelUnet()

# model = load_model('modelPrebatch512v2.h5', custom_objects={'dice': mu.dice})
modelo.load_weights(WEIGHTS_FILE_PATH)
history = modelo.fit_generator(
    tg, steps_per_epoch=1, callbacks=[checkpoint], epochs=10000
)


modelo.save("modelunetseparable256finalv1.h5")
modelo.save_weights("pesosunetseparable256finalv1.h5")
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = "historyUnetSeparable.json"
with open(hist_json_file, mode="w") as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = "historyUnetSeparable.csv"
with open(hist_csv_file, mode="w") as f:
    hist_df.to_csv(f)


"""
for i,batch in enumerate(image_generator):
    if(i >= num_batch):
        break
for i,batch in enumerate(mask_generator):
    if(i >= num_batch):
        break
"""
"""
train_generator = zip(image_generator, mask_generator)

modelo.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=20)
modelo.save("modelnormalizado20i.h5")
modelo.save_weights("pesosnormalizado20i.h5")
"""
