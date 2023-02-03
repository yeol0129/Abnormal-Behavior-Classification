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


path = '/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1'
file_list = os.listdir(path)
json_file_list = [file for file in file_list if file.endswith('.json')]
len(json_file_list)


with open("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1/"+json_file_list[37]) as f:
    d = json.load(f)

frame = pd.json_normalize(d['frames'])
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
for i in range(len(df)):
    for j in df["annotations"][i]:
        p=df["image"][i]
        f=j["label"]
        image_pre.append(p)
        label_pre.append(f)

code_pre=[]
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

x=[]
y=[]
for i in range(len(df2)):
    arr=[]
    for item in df2.iloc[i]:
        arr.append(item)
    img = cv2.imread(os.path.join(images, arr[0]))
    img = img[arr[2]:(arr[2]+arr[4]),arr[1]:(arr[1]+arr[3])]
    try:
        img = cv2.resize(img, (224, 224))
    except:
        continue
    x.append(img)
    y.append(arr[5])


x_act=[]
for i in range(len(x)):
    x_act.append(x[i])

x_train=np.array(x_act)
from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
y_train = lr.fit_transform(y)

y_train = tf.keras.utils.to_categorical(y_train)


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
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])



x_train,x_te,y_train,y_te=sklearn.model_selection.train_test_split(x_train,y_train,test_size=0.4,random_state=0)
x_val,x_test,y_val,y_test=sklearn.model_selection.train_test_split(x_te,y_te,test_size=0.5,random_state=0)
train_datagen = preprocessing.image.ImageDataGenerator(rescale = 1/255,rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2,
                                   shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip =True)
test_datagen = preprocessing.image.ImageDataGenerator(rescale = 1/255)
train_datagen.fit(x_train)
test_datagen.fit(x_val)


MODEL_DIR='./Efficientnet_model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./Efficientnet_model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer=callbacks.ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)
early_stopping_callback=callbacks.EarlyStopping(monitor='val_loss',patience=3)

hist = model.fit(x_train, y_train,batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpointer,early_stopping_callback])

print(hist.history)
print(hist.history['loss'])
print(hist.history['accuracy'])
print(hist.history['val_loss'])
print(hist.history['val_accuracy'])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
precision = precision_score(y_test, y_pred,average= "macro")
recall = recall_score(y_test, y_pred,average= "macro")
f1= f1_score(y_test, y_pred,average= "macro")
cm=confusion_matrix(y_test, y_pred)
print(cm)
print('precision: {0:.4f}, recall: {1:.4f}, f1: {2:.4f}'.format(precision, recall,f1))

plt.plot(hist.history['accuracy'],'y')
plt.plot(hist.history['val_accuracy'],'r')
plt.plot(hist.history['loss'],'g')
plt.plot(hist.history['val_loss'],'b')
plt.legend(['train_acc', 'test_acc','train_loss', 'test_loss'], loc='center right')
plt.show()