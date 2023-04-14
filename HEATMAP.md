# Representation with heatmap

## Origin data shape(dive48)

: pickle file하나로 원본 데이터셋 모두 저장(metadata, keypoint..)

- label mapping file,
- *******keypoint score****** : 좌표 신뢰도인지 뭔지 모르겠음…
    
    hrnet에서 뽑아내는거면 좌표 신뢰도가 맞음.(중요한 feature인가????)
    
- **data shape**: [person Num x total frame x keypoint Num x coordinates Num]
- **ksl_data(KETI)** shape: [frame x keypoint Num x coordinates Num)
    
    —> x, y, score??????
    

+) removal  / Keypoints-removal : 비디오와 keypoint(npy) 차이

## vis_heatmap~~~ heatmap 생성과 visualize

‘pipelines’ : preproc에 필요한 여러가지 함수, 클래스 정의 ~ register module로 지정하고,

각 클래스, 함수들을 dict ~ config파일로 바꾸어 Compose 함수에 인수로 넘김

~~ preprocessing!

—> 

```
dict(type='PoseDecode'),#pose_related
dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),#augmentation
dict(type='Resize', scale=(-1, 64)),#augmentation
dict(type='CenterCrop', crop_size=64),#augmentation
dict(type='GeneratePoseTarget', with_kp=True, with_limb=False)#heatmap_generate
```

uniformsample은 안함…

frame에 대해서 했다는거 보면 UniformSampleFrames 적용한듯

### Compose.py에서 각 function 적용 부분

```python
def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
```

### heatmap 생성 - heatmap_related.py

** Compose type-GeneratePoseTarget

```jsx
def get_an_aug():
	data : [person Num x total frame x keypoint Num x coordinates Num]
	num1 = total_frame
	
	for i in num1:#per frames
		generate_heatmap(data[i]):
			data2 = data[i]: [person Num x keypoint Num x coordinates Num]
			num2 = keypoint Num
			
			for t in num2:#per keypoints
				generate_a_heatmap(data2[t])
```

### heatmap score J 생성

![image](https://user-images.githubusercontent.com/82634312/231936531-0c77a528-e9b5-407a-9e73-6c8b900832c2.png)


```jsx
for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)
```

—> 최종적으로 들어온(anno)에 key ‘imgs’ 로 heatmap 결과값 넣어서 return

함수만 다르고 limb도 똑같다…..

## Korean에 heatmap generate 적용시키기

- 알맞은 데이터 형태 생성(pkl파일)

![image](https://user-images.githubusercontent.com/82634312/231936479-aaf81fff-0b25-4780-acb0-0a13a59fb692.png)


위와 같은 형태로 생성하는 함수



### 필요한 것

- label index mapping
- data npy axis 0 추가 (num_person =1)
- KETI config file

train을 실행하면 preprocessing까지 함께 들어가는 구조,

pyskl을 쓸거라면 따로 heatmap만 생성하는 함수 필요 없음

heatmap만 생성하고 다른 모델애 넣을거라면 약간 수정해서 heatmap 생성함수 만들기

### 해야 할 것

1. ~~KETI_annotation.pkl 생성~~
    
    → 라벨 인덱스 맵핑 파일
    
    → train, test 분리해놓은 json
    
    → 키포인트 27개로 줄여서 생성
    

→ 위치: /users/neuron2/pyskl/mypyskl/workdir/test/KETI_annotation.pkl

→ 생성 코드: /users/neuron2/pyskl/mypyskl/tools/annotation.py : 함수 파일

1. PYSKL에 KETI train 
    
    → 프레임 통일시키는 곳 어디?
    
    → 학습오류 ~ config file []_per_gpu 숫자 줄이기
    
    → none 데이터… (1 x 109(avg frame) x 27 x 2} / (1 x 109 x 27) —> np.zeros()
    
    학습 중…. 14일 1am~~ 
    
    24 에폭 느린 속도
    
2. ~~heatmap만 만드는 함수 생성~~
    
    → 133노드 / 27노드 각각
    
    1frmae img,  동영상
    

+) vis_heatmap 결과 해석

한 이미지 픽셀에대한 joint 값들(17개) ...
x y c -> t, k, h, w(frmae keypoint h w)
confidence score ~ 이 픽셀에서 제일 정확하게 나타난 joint로. 그 픽셀의 keypoint를 대표.
--> 한 이미지 픽셀에서 joint값 최대
