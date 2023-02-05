# Abnormal Behavior Classification
***
1. Resnet50V2 model [code](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/main.py), [output](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/Resnet_output.out)
2. EfficientNetB0 model [code](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/efficient.py), [output](https://github.com/yeol0129/AbnormalBehavior-Classification/blob/master/Efficientnet_output.out)
***
## Data (by [AIHub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=174))
> * labeled data
> * image data
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
> The number of json file is 157
> </details>
> 
> ### Image Data sample
> <details>
> <summary>open</summary>
> <img src="https://user-images.githubusercontent.com/111839344/216803164-7229af7d-90cb-4f5b-b74f-f303ec3e0a48.png" width="300" height="300">
>
> The number of image file is 21,136
> </details>

## Check Data
```python
for i in range(len(json_file_list)):
    with open("/Volumes/Siyeol_ssd/jupyter/지하철 역사 내 CCTV 이상행동 영상/Training/폭행/[라벨]폭행_1/"+json_file_list[i]) as f:
        d = json.load(f)
    frame=json_normalize(d['frames'])
    list_df.append(frame)
df=pd.concat(list_df,ignore_index=True)
print(df)
```
output : 

number|image|annotations
---|---|---|
13|frame_13.jpg|[{'label': {'x': 1976, 'y': 43, 'width': 23, '...
...|...|...|


