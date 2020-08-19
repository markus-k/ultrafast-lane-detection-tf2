import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
import argparse

from models.resnet import resnet_18


parser = argparse.ArgumentParser(description='Train ResNet18')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-images', type=int, default=None)
parser.add_argument('--prefetch-size', type=int, default=256)

args = parser.parse_args()

INPUT_SHAPE = (256,256,3)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PREFETCH_SIZE = args.prefetch_size

NUM_CLASSES = 1000

NUM_IMAGES = args.max_images
VAL_SPLIT = 0.3

LOG_DIR = './logs_resnet'
CHECKPOINT_DIR = './checkpoints_resnet'

model = resnet_18()
model.build(input_shape=(None, *INPUT_SHAPE))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print('Preparing imagenet data')
imagenet = tfds.load('imagenet2012', data_dir='/media/mldata/imagenet', as_supervised=True)

def resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) / 255
    image = tf.image.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label

train_ds = imagenet['train']
val_ds = imagenet['validation']

if NUM_IMAGES:
    train_ds = train_ds.take(NUM_IMAGES)
    val_ds = val_ds.take(int(NUM_IMAGES * VAL_SPLIT))

train_ds = train_ds.map(resize_image)
val_ds = val_ds.map(resize_image)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)

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
model.fit(train_ds, 
          epochs=EPOCHS,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, model_checkpoint_callback])

model.save('resnet18.tf')
