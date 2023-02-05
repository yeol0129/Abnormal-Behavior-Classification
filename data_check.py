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
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow import keras
from keras import models
from keras import layers
from keras import preprocessing
from keras import optimizers
from keras import callbacks
from keras import applications

path='/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1'
file_list = os.listdir(path)
json_file_list= [file for file in file_list if file.endswith('.json')]
print(json_file_list)
print(len(json_file_list))

with open("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1/"+json_file_list[37]) as f:
    d = json.load(f)
frame = json_normalize(d['frames'])
print(frame)

list_df=list()

for i in range(len(json_file_list)):
    with open("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1/"+json_file_list[i]) as f:
        d = json.load(f)
    frame=json_normalize(d['frames'])
    list_df.append(frame)
df=pd.concat(list_df,ignore_index=True)
print(df)

images=os.path.join("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[원천]폭행_1/1")
annotations=os.path.join("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1")

image_pre=[]
label_pre=[]
code_pre=[]
for i in range(len(df)):
    for j in df["annotations"][i]:
        p=df["image"][i]
        f=j["label"]
        image_pre.append(p)
        label_pre.append(f)

for i in range(len(df)):
    for j in df["annotations"][i]:
        code=j["category"]["code"]
        code_pre.append(code)

image_df=pd.DataFrame(image_pre)
image_df.columns=['image']
label_df=pd.DataFrame(label_pre)
code_df=pd.DataFrame(code_pre)
code_df.columns=['code']
frame_pre=pd.concat([image_df,label_df],axis=1)
df2=pd.concat([frame_pre,code_df],axis=1)
print(df2)
print(df2['code'].value_counts())

#sample image
plt.figure(figsize=(12,12))
sample_img = Image.open(os.path.join(images, df2['image'][6900]))
plt.imshow(sample_img, cmap='gray')


def plot_img(image_name):
    fig, ax = plt.subplots(1, 2, figsize=(14, 14))
    ax = ax.flatten()

    bbox = df2[df2['image'] == image_name]
    img_path = os.path.join(images, image_name)

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    image /= 255.0
    image2 = image

    ax[0].set_title('Original Image')
    ax[0].imshow(image)

    for idx, row in bbox.iterrows():
        x = row['x']
        y = row['y']
        w = row['width']
        h = row['height']
        label = row['code']
        cv2.rectangle(image2, (int(x), int(y), int(w), int(h)), (255, 0, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image2, label, (int(x), int(y - 10)), font, 3, (255, 0, 0), 4)

    ax[1].set_title('Image with Boundary Box')
    ax[1].imshow(image2)

    plt.show()

plot_img("frame_4223 복사본36.jpg")