# Abnormal Behavior Classification

***
1. Resnet50V2 model [code](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/main.py), [output](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/Resnet_output.out)
2. EfficientNetB0 model [code](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/efficient.py), [output](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/Efficientnet_output.out)
***
## Data (by [AIHub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=174))
> * labeled data (The number of json file is 157)
> * image data (The number of image is 21,136)
> ### Labeled Data sample (json file)
> 
> <details>
> <summary>open</summary>
> <img src="https://user-images.githubusercontent.com/111839344/216802059-378c31f1-14a0-4127-8cdf-5ef276f004f3.png" width="350" height="400">
>
> ```
> 'annotation_2245757.json',
> 'annotation_2250768.json',
> 'annotation_2250172.json',
> 'annotation_2076491.json',
> 'annotation_2250177.json',
> ...
> ```
> 
> </details>
> 
> ### Image Data sample
> <details>
> <summary>open</summary>
> <img src="https://user-images.githubusercontent.com/111839344/216803164-7229af7d-90cb-4f5b-b74f-f303ec3e0a48.png" width="300" height="300">
>
> 
> </details>

## Check Data
Data Frame
<details>
<summary>open code</summary>
    
```python
for i in range(len(json_file_list)):
    with open("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1/"+json_file_list[i]) as f:
        d = json.load(f)
    frame=json_normalize(d['frames'])
    list_df.append(frame)
df=pd.concat(list_df,ignore_index=True)
print(df)
```
</details>
output sample: 

number|image|annotations
---|---|---|
13|frame_13.jpg|[{'label': {'x': 1976, 'y': 43, 'width': 23, '...
...|...|...|

Annotation data must be separated to import bounding boxes(x,y,width,height).
<details>
<summary>open code</summary>
    
```python
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

```
</details>
I created a new data frame with only the image name, bounding box, and code left.

df2 sample : 
image|x|y|width|height|code
---|---|---|---|---|---|
frame_13.jpg|1976|43|23|82|child
frame_13.jpg|2045|19|22|43|person
frame_4124복103.jpg|2889|481|275|403|assault

Check the contents of the code.
```
[41174 rows x 6 columns]
assault                30158
person                 10895
theft                     65
spy_camera                20
child                     12
public_intoxication       12
surrounding_fall          10
escalator_fall             2
Name: code, dtype: int64
```

## Show bounding box
<details>
<summary>open code</summary>    

```python
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
```
</details>
output : 

<img src="https://user-images.githubusercontent.com/111839344/216853336-db10eb1e-4536-4d2a-993a-9c51b21cdaa3.png" width="450" height="200">

## Data Preprocessing
Put the bounding box image in x and code in y.
<details>
<summary>open code</summary>

```python
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
 ```
    
 </details>
 
x,y sample

<img src="https://user-images.githubusercontent.com/111839344/216856224-63c361e1-2e64-4e38-8b70-51fcf9f10176.png" width="250" height="300">

One-hot encoding

```python
lr = LabelEncoder()
y_train = lr.fit_transform(y)
y_train = tf.keras.utils.to_categorical(y_train)
```

Separate data into training, validation, and test data.

```python
x_train,x_te,y_train,y_te=sklearn.model_selection.train_test_split(x_train,y_train,test_size=0.4,random_state=0)
x_val,x_test,y_val,y_test=sklearn.model_selection.train_test_split(x_te,y_te,test_size=0.5,random_state=0)
```

Use ImageDataGenerator to transform training data.

```python
train_datagen = preprocessing.image.ImageDataGenerator(rescale = 1/255,rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2,
                                   shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip =True)
test_datagen = preprocessing.image.ImageDataGenerator(rescale = 1/255)
train_datagen.fit(x_train)
test_datagen.fit(x_val)
```

## Modeling
* EfficientNetB0
> <details>
> <summary>open code</summary>
>
> ```python
> Efficientnet_model = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape = (224,224,3),
>                                                     include_top=False)
> ```
> </details>

* ResNet50V2
> <details>
> <summary>open code</summary>
> 
> ```python
> Resnet_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape = (224,224,3),
>                                                     include_top=False)
> ```
> </details>

model compile
<details>
<summary>open code</summary>

```python
model = tf.keras.Sequential([
   Efficientnet_model,
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(8, activation='softmax')
])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])
```    
    
</details>

### ResNet50 Result
<img src="https://user-images.githubusercontent.com/111839344/216874403-6146acee-7d3c-49af-9adf-e3f5b6175236.png" width="300" height="300">

```
confusion matrix : 
[[5750    0    0  315    0    0    0    0]
 [   1    0    0    0    0    0    0    0]
 [   1    0    0    0    0    0    0    0]
 [ 832    1    0 1311    0    0    0    0]
 [   0    0    0    1    0    0    0    0]
 [   0    0    0    5    0    0    0    0]
 [   1    0    0    1    0    0    0    0]
 [   7    1    1    7    0    0    0    0]]
 
precision: 0.2090
recall: 0.1949
f1: 0.2002
```

### EfficientNetB0 Result
<img src="https://user-images.githubusercontent.com/111839344/216874958-8090d16e-ac03-4cf3-9b77-787cb134c208.png" width="300" height="300">

```
confusion matrix :
[[5746    0    0  315    1    0    0    3]
 [   0    1    0    0    0    0    0    0]
 [   1    0    0    0    0    0    0    0]
 [ 152    0    0 1983    0    1    6    2]
 [   0    0    0    1    0    0    0    0]
 [   0    0    0    2    0    3    0    0]
 [   0    0    0    2    0    0    0    0]
 [   2    0    0    0    0    0    0   14]]
 
precision: 0.5402
recall: 0.5434
f1: 0.5399
```
