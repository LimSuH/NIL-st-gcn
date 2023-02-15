# 연구 수행 과정  

#### 각도 visualization 순서 변경  
```
			#시청자 기준 왼쪽(수어 화자 기준 오른쪽)
			(7, 0, 1), (0, 7, 9), (7, 9, 11),
                        (9, 113, 117),
                        (117, 113, 118),(118, 113, 122), (122, 113, 126),(126, 113, 130),
                        (113, 118, 121), (113, 122, 125), (113, 126, 129), (113, 130, 133),

			#시청자 기준 오른쪽(수어 화자 기준 왼쪽)
                        (1, 0, 6), (8, 6, 0), (92, 8, 6),
                        (96, 92, 8),
                        (97, 92, 96), (101, 92, 97), (105, 92, 101), (109, 92, 105),
                        (100, 97, 92), (104, 101, 92), (108, 105, 92), (112, 109, 92)
```  
  
  
##### 결과  
![image](https://user-images.githubusercontent.com/82634312/204615619-4b50ab63-af4d-44a9-be25-28759bc7844a.png)  

#### 노이즈 제거 - 뒷부분 손가락이 보이지 않는 프레임 삭제  
기존 openmmlab detection 이용한 방식은 같으나(npy_frame_remove), 중간 부분을 자르지 않기 위해 영상 프레임 리스트 순서를 뒤집음  
mmcv. 로 영상을 불러옴 ~~ type 확인 필요~~ : <class 'mmcv.video.io.VideoReader'>  
list와 같은 취급 가능함 확인  
영상과 npy 뒤집어 기존 프레임 제거방법과 같은 원리로 프레임 제거함
  
단, 영상 마지막 부분에 나타나는 노이즈는 손가락이 안보이는 문제 때문이 아닌것 같음... keypoint estimation 과정에서 좌표값 자체에 노이즈가 섞인듯  
--> 끝부분이 아니더라도 노이즈가 섞임을 보임 (KETI_SL_0000000396.mp4)  
그렇다고 그런 부분들이 score 값이 낮은것도 아님 그들은 이미 확신을 하고 있음(애초에 score값을 크게 기대할수 없는듯  
score값 - 정상적인 좌표는 0.79~ 0.8 정도의 값을 가지고, noise는 0.17~ 0.1 이하의 값을 가짐 그러나 위의 이유 때문에 score로 판단할 순 없다
score로 지우면 좌표는 무엇으로 대체되는가? 0? 그럼 그건 또 다른 noise?  
  
#### angle data encoding  
참고: https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network  
라디안을 구하고(기존 구현) -> cos sin을 이용한 값으로 변경  
기존 데이터 구조 [frame, angle] -> [frame, cos, sin]  
npy까지 만들어놓나 / 모델 넣기 직전에 변환을 하는가  

#### 1dcnn  
1dcnn input data 형태 주의 2d/3d 까지 가능
이미지 텐서 [batch, channel, width, height] -> 형태 변환 필요...  
[frame, cos, sin] ~> batch로 들어가면 4d가 되는데 cos, sin을 묶어서 한 차원으로? [frame, (cos, sin)]  
list 구조만 같게 해서 테스트 필요

#### 손이 보이지 않는 frame check  
기존 histogram 코드(simple_chal-slr) 이용하여 plot을 그림, /users/suhyeon/GitHub/NIL-st-gcn/ksl_angle/no_hand_frame.py  
실행 ~ cuda 문제 발생,  
RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW  

Failed to initialize NVML: Driver/library version mismatch


</br></br>
#### Custom Dataset/DataLoader  
```
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
``` 
수어연구 에서 데이터와 label을 불러오는법  
annotation(csv 파일) -> 주요 정보를 추출하여 json 파일로 따로 저장 -> json 파일로부터 데이터 디렉토리, label을 읽어들임  

프로젝트 json파일 생성이라는 과정을 한 번 더 수정해야 하는가?  ***<- 전체 주석에서 날짜만 모아서 이 과정을 거친 것***  
사용자로부터 원하는 속성명을 받아서 사용하도록 교체  

+) simple-chal-slr 안의 SL-GCN 중복 

