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
> ``` {
>    "id": ...,
>    "file": "video....mp4",
>    "metadata": {
>        "width": 3840,
>        "height": 2160,
>        "duration": 313.7,
>        "fps": 30,
>        "frames": 9411,
>        "created": "..."
>    },
>    "events": [
>        {
>            "name": "폭행",
>            "start_time": 152.9,
>            "duration": 4.95
>        }
>    ],
>    "frames": [
>        {
>            "number": ...,
>            "image": "frame....jpg",
>            "annotations": [
>                {
>                    "label": {
>                        "x": 1679,
>                        "y": 938,
>                        "width": 246,
>                        "height": 760
>                    },
>                    "category": {
>                        "code": "theft", ... ```
> </details>
