# NIL-st-gcn
Sign Language Recognition을 위한 **ST-GCN**과 **ST-GCN-SL** 모델을 업로드한 리포지토리 입니다.  
  
  
###### 원본 리포지토리 링크  
[ST-GCN](https://github.com/yysijie/st-gcn)  
[ST-GCN-SL](https://github.com/amorim-cleison/st-gcn-sl)  

###### 각 모델별 문서
[ST-GCN](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN_README.md)  
[ST-GCN-SL](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN-SL_README.md)  
[SL-GCN](https://github.com/LimSuH/NIL-st-gcn/blob/main/README.md#sl-gcn-preprocessing)
   
###### 한국 수어 데이터 세트
```
cd /dataset  
```
KETI, AIHub, ASLLVD 데이터 세트  
  
###### 디렉토리 위치
```
cd /users/suhyeon/GitHub/NIL-st-gcn 
```
    
## Abstract  
Linear 모델은 모든 피처 맵을 1차원의 매트릭스로 이어, Convolution을 실행한다.
이후 CNN이 등장했다. CNN은 그리드 형태의 데이터를 가지고 ConVolution 연산을 실행한다.
이미지는 픽셀값이 그리드 형태로 정렬한 데이터로, CNN은 이미지 관련 학습, 또 데이터가 정적이므로 Locality 찾기, 인풋의 특성을 찾아내는것에 특화되어있다.  

그러나 인간의 움직임은 동적이며, 연속적이다.  
이미지로는 그 특성을 반영하기 어렵고, 일반화가 어려우며 사람 주위의 배경에 영향을 많이 받는 단점이 있었다.  
따라서 인간 동작 데이터는 스켈레톤을 이용한 그래프로써 나타내진다. 이는 이미지와 같이 그리드 형태의 데이터가 아니므로, 기존의 CNN에서의 연산을 기대하기 어려웠기 때문에 본 모델들은 CNN이 아닌 GCN을 활용하였다.  

또한 인간의 움직임은  
1. 연속된 동작이므로 시간에 따른 패턴 분석이 필요하며
2. 관절들의 연결의 움직임이므로 공간적 요소도 고려해야 한다.  

우리의 task인 수어도 인간의 동작을 이용한 언어라는 점에서, 인간 동작 인식에 대한 연구가 수어 인식에도 적용될 수 있을거라 예측해 볼 수 있다.  
최종적으로 자동으로 시간의 흐름에 따라 관절들의 공간적 구성을 학습하는 신경망, **spatial-Temporal Graph Convolutional Network**를 활용하였다.
  
  

## Pipe Line  
![stgcnpipeline](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fmcha0%2FbtqxECHKZR6%2FhZtIxTUkxSP4eU6KL84CUK%2Fimg.png)
ST-GCN의 동작 순서는 다음과 같다.  
1. 비디오 형태의 데이터를 입력한다.  
2. 인간의 관절을 노드로, 신체구조에 따른 자연스러운 연결을 엣지로 그래프를 구성한다.  
3. 영상의 프레임 별로 신체 그래프를 형성하고, 시간 순서(프레임 순서)로 늘어놓아 이를 엣지로 연결해준다. (위의 그림에서 대각선 방향이 곧 시간순서에 따른 전개 방향임을 알수 있다)  
4. 그래프 컨볼루션 연산을 한다.  
    --> 중심 노드 a와 직접 연결된 노드들의 피처에 weight를 곱하여 더한다. 이 값은 다음 레이어의 중심노드 a의 피처값으로 업데이트된다.

5. 모든 컨볼루션 연산이 끝나면, 최종적으로 얻은 피처 맵으로 softMax classifier을 이용해 각 카테고리별 점수를 얻는다. 최고로 높은 점수를 가진 라벨로 동작을 분류한다.

  
  
</br></br></br>
## SL-GCN preprocessing  
실행  

```
cd /users/suhyeon/GitHub/NIL-st-gcn/simple-chal-slr/analization
python remove_npy.py  
```


### Details
SL-GCN 전처리는 기존에 estimation 된 npy 파일에서 움직임이 없는 frame을 제거하는 방식으로 진행되었습니다.  

</br>

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

remove_npy.py는 두가지 경우의 frame 제거를 실행합니다. (frame 제거를 모든 영상에 대해 실행하지 않았기 때문입니다)  
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

</br></br></br></br>
