import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
import argparse

from models.ultrafast import UltraFastNet
from utils import losses, metrics
from utils.datasets import llamas_dataset, labelme_dataset


parser = argparse.ArgumentParser(description='Train Ultrafast Net')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-images', type=int, default=None)
parser.add_argument('--prefetch-size', type=int, default=256)
parser.add_argument('--resnet-weights', default=None)
parser.add_argument('--base-path', default='')
parser.add_argument('--llamas-path', default='llamas/')
parser.add_argument('--culane-path', default='culane/')
parser.add_argument('--labelme-path', default=[], action='append')
parser.add_argument('--model-name', default='ultrafast')

args = parser.parse_args()
print(args)
#sys.exit()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PREFETCH_SIZE = args.prefetch_size

NUM_IMAGES = args.max_images
VAL_SPLIT = 0.3

NUM_LANES = 2
CLS_SHAPE = (100, 20, NUM_LANES)
IMAGE_SHAPE = (288, 800, 3)

RESNET_WEIGHTS = args.resnet_weights

BASE_PATH = args.base_path
LLAMAS_PATH = os.path.join(BASE_PATH, args.llamas_path)
CULANE_PATH = os.path.join(BASE_PATH, args.culane_path)
LABELME_PATHS = [os.path.join(BASE_PATH, path) for path in args.labelme_path]

LOG_DIR = './logs_ultrafast'
CHECKPOINT_DIR = './checkpoints_ultrafast'
MODEL_DIR = './trained/'

MODEL_NAME = '%s.tf' % (args.model_name)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

print('Preparing datasets')
llamas_train_ds = llamas_dataset(os.path.join(LLAMAS_PATH, 'labels', 'train', '*', '*.json'), CLS_SHAPE, IMAGE_SHAPE)
llamas_valid_ds = llamas_dataset(os.path.join(LLAMAS_PATH, 'labels', 'train', '*', '*.json'), CLS_SHAPE, IMAGE_SHAPE)

labelme_ds = None
for path in LABELME_PATHS:
    ds = llamas_dataset(os.path.join(LLAMAS_PATH, '*.json'), CLS_SHAPE, IMAGE_SHAPE)
    if labelme_ds is None:
        labelme_ds = ds
    else:
        labelme_ds = labelme_ds.concatenate(ds)
        
train_ds = llamas_train_ds
valid_ds = llamas_valid_ds
if labelme_ds is not None:
    train_ds = train_ds.concatenate(labelme_ds)
    
train_ds = train_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)

print('Preparing model')
model = UltraFastNet(num_lanes=NUM_LANES, 
                     size=IMAGE_SHAPE[0:2],
                     cls_dim=CLS_SHAPE, 
                     use_aux=False, 
                     resnet_weights=RESNET_WEIGHTS)

adam = tf.keras.optimizers.Adam(
   # 4e-4
)
model.compile(optimizer=adam, loss=losses.ultrafast_loss, metrics=['accuracy', metrics.ultrafast_accuracy])
model.summary()


shutil.rmtree(LOG_DIR, ignore_errors=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'chkpt')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True)

print('Training model')
history = model.fit(train_ds, 
                    epochs=EPOCHS,
                    validation_data=valid_ds,
                    callbacks=[tensorboard_callback, model_checkpoint_callback])

model.save(MODEL_PATH)
