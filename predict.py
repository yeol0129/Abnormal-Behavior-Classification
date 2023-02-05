import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import json
import random
import os.path as osp
import cv2
from PIL import Image
import sklearn
import tensorflow as tf
import sklearn.model_selection
from pandas.io.json import json_normalize
from tensorflow import keras
from keras import models
from keras import layers
from keras import preprocessing
from keras import optimizers
from keras import callbacks
from keras import applications
from sklearn.metrics import confusion_matrix,precision_score , recall_score , accuracy_score, f1_score
from tensorflow import keras
from keras import models
from keras import layers
from keras import preprocessing
from keras import optimizers
from keras import callbacks
from keras import applications

Efficientnet_model = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape = (224,224,3),
                                                    include_top=False)
for layer in Efficientnet_model.layers:
    layer.trainable = False

model = tf.keras.Sequential([
   Efficientnet_model,
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(8, activation='softmax')
])
new_model = tf.keras.models.load_model('/Users/siyeolkim/PycharmProjects/pythonProject4/Efficientnet_model/10-0.0338.hdf5')
labels=["assault","child","escalator_fall","person","public_intoxication","spy_camera","surrounding_fall","theft"]

im=cv2.imread('/Users/siyeolkim/Downloads/r0_276_1292_1005_w1200_h678_fmax.jpg')
print(im.shape)
im.resize(15,224,224,3)
print(im.shape)
r=new_model.predict(im,batch_size=32, verbose=1)

print(r)
res = r[0]
for i, acc in enumerate(res) :
    print(labels[i], "=", acc*100)
print("---")
print("result = " , labels[res.argmax()])