## SL-GCN preprocessing  
(1) 움직임이 없는 프레임을 판별하고 잘라내는 전처리  
(2) 전처리 이후 영상 프레임 분포 분석  
두가지 기능 구현에 대한 설명입니다.  
  
  
실행  

```
cd /users/suhyeon/GitHub/NIL-st-gcn/simple-chal-slr/analization/code_file

#npy 파일에서 움직임 없는 프레임을 제거
python npy_frame_remove.py  

#비디오 파일에서 움직임 없는 프레임을 제거
python video_frame_remove.py 
```
video_frame_remove.py는 프레임 제거 방법이 유효한지 확인하기 위하여 작성되었습니다.  
비디오 파일을 가져오고 손동작이 없는 프레임을 제거한 영상을 구성합니다.  
  
  
</br>
npy_frame_remove.py는 본격적인 학습을 위해, 기존에 만들어져 있던 키포인트 npy 파일에서 움직임이 없는 프레임을 지우기 위해 작성되었습니다.  
다만 video_frame_remove.py실행 과정에서 전체 영상이 아닌 소수의 영상에 대해서만 프레임이 제거된 영상을 만들었습니다.  
따라서 npy 파일 제거 방법이 (1) 프레임이 제거된 영상이 있음 (2) 프레임이 제거된 영상이 없음 이 두 경우를 고려하여 구현되어야 했습니다.  
npu_frame_remove.py는 자동으로 이를 고려하여 프레임을 제거한 npy파일을 생성합니다.  
자세한 설명은 아래 문서를 참고해 주세요.  
</br></br>

### Details  
  
  

#### 움직임 없는 프레임을 분별하기 위한 기준 선별  
  

##### 1. npy 파일의 keypoint들을 영상에 다시 표시하고 각자의 정확도를 살펴보았습니다.  

(1) mmpose를 활용  
![KETI_SL_0000000023](https://user-images.githubusercontent.com/82634312/192952244-5b54f73d-70f9-43bf-b196-0e2118272be3.gif)  


(2) opencv를 활용  
![KETI_SL_0000000007](https://user-images.githubusercontent.com/82634312/192951982-e8595da4-9dad-4110-a07f-01e2d6cb2160.gif)  



</br>
둘 모두 어느정도의 정확도를 보이나, frame을 구분하기 위한 기준을 세우기에는 근거가 부족하였습니다.  

coco-whole body keypoints들의 hand keypoints들을 프레임에 표시해보거나, 신뢰도도 함께 비교해 보았으나,  
마찬가지로 손동작이 발생했다고 판단할만한 근거로 삼기 어려웠습니다.  

</br>

##### 2. mmpose의 hand detection, hand estimation 활용  
whole body의 hand 부분 keypoitns들과는 별개로, mmpose의 hand detection과 estimation을 사용하였습니다.  
*실제로 실행 결과 정확도 측면에서 차이를 보였습니다.  
</br>


(1) hand detection  
detection은 손이 잡히면 화면에 박스를 그려주나, 손동작이 잡히지 않으면 화면에 랜덤하게 박스를 그립니다.  
![KETI_SL_0000000002 avi](https://user-images.githubusercontent.com/82634312/192951660-44ec1246-854d-4d02-90b3-71e2194daa46.gif)  


(2) hand estimation  
inference_top_down_pose_model 은 손 keypoint에 대한 일정 신뢰도 이하는 빈 리스트를 반환하고,  
일정 손동작이 발생하면 keypoint 리스트인 hand_result를 반환합니다.  
위의 근거로 0001~3000까지의 KETI 데이터와 AUTSL 데이터의 손동작이 발생하지 않는 프레임 분포를 확인하였습니다.  
  
  

확인 방법은 단순하게 각 데이터 별 최대 프레임 길이로, 0으로 채워진 list를 준비합니다. (AUTSL 156, KETI 432)  
영상마다 프레임별로 hand result가 빈 list로 반환되면 해당 프레임 index의 숫자에 +1 합니다.  


```
def check_noMove:
	whole_stop = []
	# AUTSL은 len(whole_stop) = 156
	# KETI len(whole_stop) = 432

	for i, path in enumerate(videos):
       		video = read_video

        	for frame_id, cur_frame in enumerate(video):
            	hand_result = hand_estimate(video)

            	if not hand_results:
                		whole_stop[frame_id] += 1

```  


그 결과, 손동작이 추정되지 않은 프레임의 분포는 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/82634312/192951089-a36c7022-1a33-499f-b5bf-f3a291f0d8f9.png)  


</br>

실제 영상은    
** 대기자세(손동작 없음) - 수어 동작 - 수어 완료 후 손을 내리면서 영상이 종료 ** 의 구성을 가지고 있습니다.  
따라서 손동작이 없는 frame은 영상의 앞 부분에 분포해 있으며, ** 첫 프레임 ~ hand_result가 발생한 순간까지의 frame ** 을 잘라내면 움직임 없는 frame을 제거 할수 있을것이라고 판단하였습니다.  
</br></br>

#### 움직임 없는 frame 제거  
##### 영상  

실제로 움직임이 없는 frame을 제거한 영상입니다. (저장 경로: /dataset/KETI_SignLanguage/removal)  
*위:원본 영상, 아래: frame 제거 영상*  
  
  

https://user-images.githubusercontent.com/82634312/192956184-268997ef-862c-40bf-9ecf-6d01141bb662.mp4  

</br>

https://user-images.githubusercontent.com/82634312/192956223-03b24ef4-99c5-4602-acfb-c00ea6f71182.mp4




 
 

</br></br>
신뢰할 만한 결과를 바탕으로 이번에는 영상이 아니라, 기존에 estimation 된 npy파일 에서 필요없는 frame 제거를 진행하였습니다.  
  
##### npy 파일  

npy_frame_remove.py는 두가지 경우의 frame 제거를 실행합니다. (frame 제거를 모든 영상에 대해 실행하지 않았기 때문입니다)  
</br> 

**(공통)**  
/dataset/KETI_SignLanguage/Keypoints-MMPOSE에 저장된 npy 파일을 불러옴  
</br>
  
**(1) npy 파일의 원본 영상이 /dataset/KETI_SignLanguage/removal 에 존재**  
npy파일을 로드하고, frame이 제거된 영상의 프레임 수만큼 뒤에서부터 슬라이싱합니다.  

```
def video_exist:
    videos = read_video_list('removal')
    for i, video in enumerate(videos):
        exist_npy = read_npy()

        start_frame = len(video)
        exist_npy = exist_npy[-start_frame:]
        np.save('Keypoints-removal/')
```
</br> 

**(2) /removal에 영상이 저장되지 않은 경우**  
   
   
원본 영상을 불러와 hand estimation을 진행하고,  
hand_result가 발생한 frame부터 npy 파일을 슬라이싱합니다.  

```
def detect_remove:
    videos = read_video_list('removal')
    for i, video_path in enumerate(videos):
        exist_npy = read_npy()

        video = read_video(video_path)
        start_frame = 0

        for frame_id, cur_frame in enumerate(video):

            hand_results = hand_estimation()

            if hand_results:
                start_frame = frame_id
                break

        exist_npy = exist_npy[start_frame:]
        np.save('Keypoints-removal/')
```
###### 실제 구현 코드(npy_frame_remove.py)  
```
for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            print(' ({} / {})'.format(i + 1, len(videos)), end=' ')
            mmdet_results = inference_detector(det_model, cur_frame)
            detect_results = process_mmdet_results(mmdet_results, args_det_cat_id)

            #mmpose hand estimation을 이용한 손이 보이는지 여부 판정
            #inference_top_down_pose_model은 hand detect 영역(detect_results)안에서 hand keypoint를 반환함
            #만약 손이 검출되지 않아 제대로 된 hand keypoint를 얻지 못했을 경우, 빈 리스트를 반환
            hand_results, returned_outputs = inference_top_down_pose_model(
            hand_model,
            cur_frame,
            detect_results,
            bbox_thr = args_bbox_thr,
            format='xyxy',
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            return_heatmap=False,
            outputs=None)

            #hand_results가 존재한다는 것은 유효한 hand keypoint를 반환 받았음을 의미 = 손이 보인다고 판정
            #손동작이 시작된 프레임 저장
            if hand_results:
                start_frame = frame_id
                break
        
        #유효한 손동작이 보인 프레임부터 새 npy 파일 생성
        exist_npy = exist_npy[start_frame : ]
        np.save(output_dir, exist_npy)
```

실행 결과 /dataset/KETI_SignLanguage/Keypoints-removal에 총 33517개 영상에 대한 npy 파일이 저장되어 있습니다.  
</br>


#### frame 제거 후 프레임 길이 분포  
프레임을 제거 후 KETI 데이터의 프레임 분포는 다음과 같습니다.  

```
[DATASET: KETI]
 AVERAGE: 109.335
 VARIACE: 814.731
 STANDARD DEVIATION:28.543
 MAX:283.000
```
</br>

![image](https://user-images.githubusercontent.com/82634312/192951336-6c13cff8-d274-4401-9db5-36caa9dac9dd.png)

  
AUTSL과의 비교 분포도입니다.  
![image](https://user-images.githubusercontent.com/82634312/192947075-9efaef7b-3a61-4d4f-82d0-f6df277ef948.png)  

