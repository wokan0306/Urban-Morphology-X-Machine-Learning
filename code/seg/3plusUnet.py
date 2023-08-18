# -*- coding: utf-8 -*-
"""Copy of Retrieving_Urban_Morphology_Without_Cloud.ipynb

# UROP 1100: Retrieving Urban Morphology from Satellite Images and Machine Learning

**Original Author:** Edwin Ka Ho NG<br>
**Modified by** Stan Wan On KAN<br>
**Supervisor:** Professor Jimmy Chi Hung FUNG <br>
**Last modified:** 21/08/2022

Based on
* [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/)
    * **By:** [fchollet](https://twitter.com/fchollet)
* [Winning solution of SpaceNet2](https://github.com/SpaceNetChallenge/BuildingDetectors_Round2/tree/master/1-XD_XD)
    * **By:** XD_XD

# Import Dependencies
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from keras import backend as K

import datetime

import os, shutil, sys, humanize, psutil, GPUtil
from pathlib import Path

import pprint

from tensorflow.keras import layers
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout, Concatenate,
    Activation, BatchNormalization, Conv2DTranspose, SeparableConv2D)
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import regularizers
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from PIL import ImageOps

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

"""## Utility Functions"""

def store_in_dict(*args):
    return {key: globals()[key] for key in args}

from dateutil import tz

# import pytz
# print(pytz.common_timezones)

def get_datetime_now():
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Asia/Hong_Kong')

    utc = datetime.datetime.utcnow()

    # Tell the datetime object that it's in UTC time zone since
    # datetime objects are 'naive' by default
    utc = utc.replace(tzinfo=from_zone)

    # Convert time zone
    return utc.astimezone(to_zone)

"""To check if two list contains duplicates.

Useful for double-checking training data and validation data do not intersect.
"""

# Modified from https://stackoverflow.com/questions/68468710/common-elements-in-two-lists-preserving-duplicates
from collections import Counter

def check_duplicate(list_1, list_2):
    counter_1 =Counter(list_1)
    counter_2 =Counter(list_2)

    res=[]

    for i in set(list_1).intersection(set(list_2)):
        res.extend([i] * min(counter_2[i], counter_1[i]))


    if(len(res) > 0):
        print(res)
        raise AssertionError
    else:
        print("No leaks")

"""To check the usage of GPU RAM"""

def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

mem_report()

"""## Mount Private Google Drive

### Declare Data Name

# Set Hyperparameters

## Dataset-specific Formats

### Declare input and target directory
"""

data_name = "HK Map from Google with Unet 3+"

# Path
input_dir = "./train"
input_ext = ".png"
target_dir = "./target"
target_ext = ".png"

# Format
img_size = (256, 256)
input_band_count = 3    # only support 3 (RGB) for now

"""## With AND Without Transform"""

# Training
batch_size = 10

train_sample_index = slice(None, int(0.75*len(os.listdir(input_dir))))  # (None, 100)
valid_sample_index = slice(int(0.75*len(os.listdir(input_dir))), None)  # (100, 180)
monitor_function = 'val_jaccard_coef_int' # val_loss
early_stop_patience = 15
reduced_patience = 3
epochs = 100
lr_start = 0.0001

# Visualize
preview_valid_sample_index = slice(0, 12)  # Without Transform Note: start from valid_sample_index_start (AFTER SHUFFLE)
                                            #  10, 22  # Transform
custom_gif_text = ""

# For serialization
training_params = store_in_dict(
    "batch_size", "train_sample_index", "valid_sample_index",
    "monitor_function", "early_stop_patience", "epochs", "preview_valid_sample_index", "custom_gif_text"
)

"""# Load Data

## Without Transform

### Build data paths
"""

import os

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(input_ext)
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(target_ext) and not fname.startswith(".")
    ]
)

paths_shuffled = False

print("Total number of samples:", len(input_img_paths))

print("")

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

import random

"""### Build (Batch) Data Loader (`Sequence`)"""

class ImagesSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):  # no. of batches
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (input_band_count,), dtype="float32") #concatnation of tuples; (3,) stands for 1-element tuple

        for j, path in enumerate(batch_input_img_paths):
            if input_band_count == 3:
                img = load_img(path, target_size=self.img_size)
                x[j] = img
                x[j] = x[j] /  255 # Normalize
            else:
                raise NotImplementedError
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, -1)/255  # Grayscale doesn't have last dimension
        return x, y

"""#### Split train and validation data"""

# Fix random seed for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

if not paths_shuffled:
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    paths_shuffled = True

train_input_img_paths   = input_img_paths[train_sample_index]
train_target_img_paths  = target_img_paths[train_sample_index]
val_input_img_paths     = input_img_paths[valid_sample_index]
val_target_img_paths    = target_img_paths[valid_sample_index]

# Instantiate data Sequences for each split
train_gen = ImagesSequence(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = ImagesSequence(batch_size, img_size, val_input_img_paths, val_target_img_paths)

print(f"Number of batches of training dataset: {len(train_gen)}")
print(f"Number of batches of validate dataset: {len(val_gen)}")
check_duplicate(train_input_img_paths, val_input_img_paths)
train_batch_count = len(train_gen)
valid_batch_count = len(val_gen)

"""#### Preview Input and Target Output (Images)"""

img_width = img_size[0]
img_height = img_size[1]

preview_valid_sample_index_start = preview_valid_sample_index.start
preview_valid_sample_index_end = preview_valid_sample_index.stop
preview_valid_sample_index_count = preview_valid_sample_index_end - preview_valid_sample_index_start

"""#### Preview Input and Target Output (Batch Array)"""

print("Number of batches:", len(train_gen))
print("Shape of each input batch:", np.shape(train_gen.__getitem__(0)[0]))
print("An input image segment:", train_gen.__getitem__(0)[0][0, 0:10, 0:10, 0], sep="\n")
print("Shape of each target batch:", np.shape(train_gen.__getitem__(0)[1]))
print("A target image segment:", train_gen.__getitem__(0)[1][0, 0:10, 0:10, 0], sep="\n")


"""# Build Models

## Metrics

### Jaccard Coefficient
"""

# Modified from XD_XD's v9s.py
# Minimize Jaccard distance := 1-Jaccard coeff
# where coeff is intersection / union

# y_true.shape: (None, None, None, None); y_pred.shape: (None, 256, 256, 1)


def jaccard_coef_int(y_true, y_pred, smooth = 1e-12):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[-1, -2, 0])
    sum_ = K.sum(y_true + y_pred_pos, axis=[-1, -2, 0])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - K.mean(jac)) * smooth

"""## Models
"""
from keras_unet_collection import models

model_name = "Unet 3+"
model_params = dict(
    convolution_activation = "relu", last_activation = "sigmoid",#
    optimizer = "Adam", learning_rate = 0.001,
    loss='binary_crossentropy',
    metrics=['accuracy', jaccard_coef_int]
)
positive_class_index = 0

optimizer = keras.optimizers.Adam(learning_rate=lr_start)

model = models.unet_3plus_2d(input_size=(256,256,3), filter_num_down=[64, 128, 256, 512], n_labels=1)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=['accuracy', jaccard_coef_int]
)
model.summary()


"""## Callbacks

### Save intermediate images to Progress and TensorBoard
"""

# Modified from https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

from keras.callbacks import Callback
import shutil
from PIL import Image as PILImage
import datetime

class IntermediateImageCallback(Callback):
    def __init__(self, log_dir, progress_image_dir, valid_data = None, positive_class_index = 0):
        super(IntermediateImageCallback, self).__init__()

        self.progress_image_dir = progress_image_dir
        Path(self.progress_image_dir).mkdir(parents=True, exist_ok=True)

        self.writer = tf.summary.create_file_writer(log_dir)    # 2.0 API of filewriter

        # preview_count = min(preview_count, valid_data.__len__())

        self.valid_data = valid_data                            # self.validation_data is deprecated. Therefore, manually passing the valid_data as a parameter.
        self.positive_class_index = positive_class_index

        # NO need to print the same ground truth in the loop of epoches:
        with self.writer.as_default():
            input_array = (self.valid_data.__getitem__(0)[0][preview_valid_sample_index_start:preview_valid_sample_index_end, :, :, :] * 255).astype(np.uint8)
            ground_array = (self.valid_data.__getitem__(0)[1][preview_valid_sample_index_start:preview_valid_sample_index_end, :, :, :] * 255).astype(np.uint8)  # prevent uint8 overflow

            tf.summary.image("1_Input", input_array, step=0, max_outputs=preview_valid_sample_index_end - preview_valid_sample_index_start)
            tf.summary.image("2_Ground", ground_array, step=0, max_outputs=preview_valid_sample_index_end - preview_valid_sample_index_start)

            for i, arr in enumerate(input_array):
                if input_band_count == 1:
                    image = PILImage.fromarray(arr[:, :, 0])     #arr[:, :, ::-1]) # BGR to RGB
                elif input_band_count == 3:
                    image = PILImage.fromarray(arr)     #arr[:, :, ::-1]) # BGR to RGB
                else:
                    raise NotImplementedError
                image.save(str(Path(self.progress_image_dir) / Path(f"1_Input_id{i}.png")))

            for i, arr in enumerate(ground_array):
                image = PILImage.fromarray(arr[:, :, 0])
                image.save(str(Path(self.progress_image_dir) / Path(f"2_Ground_id{i}.png")))
            # autocontrast on nparray
            # ⚠ assumed 32 batch size, 8 valid augmentation
            # Bug: .astype(np.uint8) must be applied at last after ALL array operations. Otherwise (e.g. arrayop → astype → arrayop) the output is a blank black image.


    # A callback has access to its associated model through the class property self.model.
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}

        # 1.0 API requires
        # * make_image: convert array to PIL.Image to string
        # * summary_str.append
        # 2.0 API directly works with images:
        with self.writer.as_default():

            quick_preds = model(
                self.valid_data.__getitem__(0)[0][preview_valid_sample_index_start:preview_valid_sample_index_end, :, :, :]
            ) # tf.Tensor (4, 256, 256, 2)

            quick_preds_probability_array = (np.array(quick_preds)[:,:,:,self.positive_class_index] * 255)
            quick_preds_probability_array = np.expand_dims(quick_preds_probability_array, axis=-1).astype(np.uint8)
            tf.summary.image("3_Probability", quick_preds_probability_array, step=epoch, max_outputs=preview_valid_sample_index_end - preview_valid_sample_index_start)

            if quick_preds.shape[-1] == 1:      # single-channel output
                quick_pred_mask_array = (np.array(quick_preds) > 0.5).astype(np.uint8)
            else:                               # multi-channel output
                quick_pred_mask_array = np.argmax(quick_preds, axis=-1)
                quick_pred_mask_array = np.expand_dims(quick_pred_mask_array, axis=-1)
            quick_pred_mask_array = (quick_pred_mask_array * 255).astype(np.uint8)
            tf.summary.image("4_Prediction", quick_pred_mask_array, step=epoch, max_outputs=preview_valid_sample_index_end - preview_valid_sample_index_start)

            for i, arr in enumerate(quick_preds_probability_array):
                image = PILImage.fromarray(arr[:, :, 0])
                image.save(str(Path(self.progress_image_dir, f"3_Probability_epoch{epoch}_id{i}.png")))

            for i, arr in enumerate(quick_pred_mask_array):
                image = PILImage.fromarray(arr[:, :, 0])
                image.save(str(Path(self.progress_image_dir, f"4_Prediction_epoch{epoch}_id{i}.png")))


"""# Train Models

The progress folder stores custom records for each run, including:
* the best model and the latest model
* the backups of intermediate images shown in Tensorboard
* the loss / val_loss history as a csv
"""

fit_comments = data_name
fit_timestamp = get_datetime_now().strftime("%Y%m%d-%H%M%S")

##############################

logdir = "logs"                        # /content/logs
progress_dir = Path(data_name+" Progress", fit_timestamp)
progress_image_dir = Path(progress_dir, "images")
train_model_filename = Path(progress_dir, f"model_train.h5")
best_model_filename = Path(progress_dir, f"model_best.h5")
history_csv_filename = Path(progress_dir, f"loss_history.csv")
params_txt_filename = Path(progress_dir, f"params.txt")
model_txt_filename = Path(progress_dir, f"model_summary.txt")

progress_dir.mkdir(parents=True, exist_ok=True)
with open(params_txt_filename, 'w') as f:
    f.write(pprint.pformat(store_in_dict(
        "fit_timestamp", "data_name", "fit_comments", "training_params", "model_name", "model_params"
    )))
with open(model_txt_filename, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

tensorboard_image_callback = IntermediateImageCallback(str(logdir), str(progress_image_dir), valid_data = val_gen, positive_class_index = positive_class_index)
train_checkpoint_callback = keras.callbacks.ModelCheckpoint(train_model_filename, monitor=monitor_function, save_best_only=False)
best_checkpoint_callback = keras.callbacks.ModelCheckpoint(best_model_filename, monitor=monitor_function, save_best_only=True, mode="min", verbose=1)
logger_callback = keras.callbacks.CSVLogger(history_csv_filename, append=True)  # more fault-tolerant than output of model.fit()
earlystopping_callback = keras.callbacks.EarlyStopping(
    monitor=monitor_function,
    mode="min",
    patience=early_stop_patience,
    verbose=0)
image_callback = IntermediateImageCallback(str(logdir), str(progress_image_dir), valid_data = val_gen, positive_class_index = positive_class_index)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.2, mode="min",
                              patience=reduced_patience, min_lr=0.00000001)

callbacks = [
    train_checkpoint_callback,
    best_checkpoint_callback,
    logger_callback,
    earlystopping_callback,
    image_callback,
    reduce_lr
]

# Fix random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

print(f"Current Run: {fit_timestamp}")
# Train the model, doing validation at the end of each epoch.
mem_report()
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
mem_report()
