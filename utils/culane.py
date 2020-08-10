import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


BASE_PATH = 'culane/'

INPUT_SHAPE = (590, 1640, 3)
IMAGE_SHAPE = (512, 1024, 3)
MASK_SHAPE = IMAGE_SHAPE


def read_list(name):
    with open(os.path.join(BASE_PATH, 'list', name)) as fd:
        list_str = fd.read()

    img_list = list_str.splitlines()
    train_meta = dict()
    for item in img_list:
        img_name, seg_name, l0, l1, l2, l3 = item.split(' ')
        yield dict(
            img_name=img_name,
            seg_name=seg_name,
            lanes=[l0 == '1',
                  l1 == '1',
                  l2 == '1',
                  l3 == '1']
        )


def get_path(item):
    return os.path.join(BASE_PATH, item['img_name'][1:])


def resize_img(img, shape):
    w = img.shape[1]
    h = img.shape[0]
    
    ratio = shape[1] / shape[0]
    
    #print(0+(w-(h*2))/2, (h*2) + (w-(h*2))/2)
    
    tgt = img[0:h, 0+(w-int(h*ratio))//2 : int(h*ratio) + (w-int(h*ratio))//2]
    
    #print(tgt.shape)
    tgt = cv2.resize(tgt, (shape[1], shape[0]))
    return tgt


def get_lanes(item):
    lines_file = get_path(item).replace('.jpg', '.lines.txt')
    with open(lines_file, 'r') as fd:
        meta = fd.read()
    
    lanes = []
    for line in meta.splitlines():
        coords = []
        split = line.split(' ')
        for x,y in zip(split[0::2], split[1::2]):
            coords.append([float(x),float(y)])

        coords = np.array(coords)
        lanes.append(coords)
    
    return lanes


def render_lanes(img, item, enabled_lanes = None):
    lanes = get_lanes(item)
    lane_i = 0
    colors = [(0,255,0),(0,0,255),(255,0,0),(255,255,0)]
    
    if enabled_lanes is None:
        enabled_lanes = [False,True,True,False]
    
    for i in range(len(item['lanes'])):
        if item['lanes'][i]:
            if enabled_lanes[i]:
                cv2.polylines(img, np.int32([lanes[lane_i]]), isClosed=False, color=colors[i], thickness=10, lineType=cv2.LINE_8)
            lane_i += 1

            
def read_image(item):
    file_path = get_path(item)
    img = cv2.imread(file_path)
    if img is None:
        print('Reading %s failed' % file_path)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return resize_img(img, shape=IMAGE_SHAPE)


def generate_mask(item):
    img = np.zeros(INPUT_SHAPE)
    render_lanes(img, item)
    return resize_img(img, shape=MASK_SHAPE)


def generate(list_file):
    img_list = read_list(list_file)
    
    def gen():
        for item in img_list:
            img = read_image(item)
            if img is None:
                continue
            
            mask = generate_mask(item)

            yield (img, mask)
            
    return gen


def process_image(file):
    img = tf.py_function(read_image, [file], Tout=tf.uint8)
    mask = tf.py_function(generate_mask, [file], Tout=tf.uint8)
    
    img = tf.reshape(img, shape=IMAGE_SHAPE)
    mask = tf.reshape(mask, shape=MASK_SHAPE)
    return img, mask
