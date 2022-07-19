# [ST-GCN-SL](https://github.com/amorim-cleison/st-gcn-sl)
ST-GCN을 수어 인식에 활용한 모델  
[논문 링크](https://arxiv.org/abs/1901.11164)  
  
## Dataset  
#### American SignLanguage Lexicon video Dataset(ASLLVD-skeleton)
[ASLLVD 웹사이트](http://www.cin.ufpe.br/~cca5/asllvd-skeleton/)를 통해 ST-GCN-SL의 전처리 단계별 출력물을 다운로드 할 수 있습니다.  
#### ASLLVD-skeleton-20  
ASLLVD의 하위 집합 데이터 세트로, ASLLVD를 다음 20개의 수어에 대해서 구성한 데이터 세트입니다.  
  
|Label|
|-------|
|adopt, again, all, awkward, baseball, behavior, can, chat, cheap, cheat, church, coat, conflict, court, deposit, depressed, doctor, don’t want, dress, enough|

ST-GCN-SL 모델은 ASLLVD-skeleton-20을 사용하여 학습을 진행하였습니다.  
<br/><br/>
전처리 과정 설명과 실제 진행 상황은 다음을 참고해 주세요.  
- [데이터 전처리](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC)  
- [실제 진행 상황](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-%EC%8B%A4%EC%A0%9C-%EC%8B%A4%ED%96%89-%EC%83%81%ED%99%A9)
#



### 1. 데이터 전처리  
ST-GCN-SL 모델은 다음과 같은 전처리 단계를 거칩니다.  
![st-gcn-slpreproc](https://www.cin.ufpe.br/~cca5/img/dataset_preprocessing.png)  


###### (1) Download  
asllvd 데이터를 다운받습니다. 
asllvd 데이터는 연속적인 수어 대화 영상과 메타 데이터 파일로 이루어져 있습니다.  
![image](https://user-images.githubusercontent.com/82634312/179709829-6fa41ae2-d5bb-4060-82f4-1d12be64d417.png)  
메타데이터 파일은 main gloss, 수어 발화자, gloss를 포함하는 동영상 파일 이름, 영상 확장자 등을 항목으로 가집니다.  
*연속적인 대화 영상이므로, 한 영상에 여러 gloss가 존재해 영상 기준이 아닌 gloss 기준으로 메타데이터 파일을 서술한것 같습니다.*  
<br/>

###### (2) Segment  
gloss별로 영상을 분할합니다. 또한 동시에 분할한 영상별 gloss를 기록한 파일과, 전체 gloss 파일을 생성합니다.  
자세한 형태는 [이곳](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-2-%ED%98%84-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC-%EC%83%81%ED%99%A9)을 확인해 주세요.  
<br/>
###### (3) Estimate Skeleton  
openpose를 이용해 영상에 스켈레톤을 적용합니다.  openpose는 Neuron3에 다운되어 있으며, 위치는 '/home/lab/openpose/build'입니다. config 파일에서 자동으로 실행하므로 크게 중요하지 않을것 같습니다.  모든 키포인트 좌표는 json 파일로 저장됩니다.
![keypoint_json](https://www.cin.ufpe.br/~cca5/img/openpose_coordinates.PNG)  
<br/>
###### (4) Filter Keypoint  
![keypoint](https://www.cin.ufpe.br/~cca5/img/filtered_keypoints_hand.png)
모든 스켈레톤 키포인트를 사용하지 않으므로, 몸톤 5개, 각 손 11개로 총 27개의 키포인트 좌표를 추출합니다.  
(3)의 출력물과 유사한 형태지만 좌표가 더 적습니다.  
<br/>
###### (5) Split Dataset  
(4)에서 얻은 json파일을 train과 test 데이터 세트로 분류합니다.  
**train-80%, test-20%** 로 구성됩니다.  
<br/>
###### (6) Normalize & Serialize  
마지막으로, 정규화와 직렬화를 진행합니다.
정규화는 모든 데이터의 길이를 일정하게 만드는것을 의미합니다.  
st-gcn-sl 모델은 한 영상 당 63의 프레임 수를 사용하도록 했습니다. 원본 영상 데이터의 fps가 30이므로, 이는 약 2초정도의 길이입니다.  
  
직렬화는 모델이 돌아가는 파이썬 코드가 데이터를 읽어들일 수 있도록 파이썬 파일로 변환하는것을 의미합니다.
키포인트 좌표는 npy(넘파이 파일)로, label은 pkl파일로 저장됩니다.  
  
아래에서 모든 전처리 단계가 완료된 데이터를 확인할 수 있습니다.  
```
cd /users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/st-gcn/data/asllvd-skeleton-20/normalized
```


<br/>

### 1-2. 데이터 전처리 코드



#
### 2. 실제 실행 상황
###### 전처리 단계를 진행하려면 다음 과정을 실행합니다.  
```
cd ST-GCN-SL/st-gcn-sl/asllvd-skeleton-preproc
```  

전처리를 위해 필요한 라이브러리를 설치합니다.  
```
bash setup.sh 
```
전처리 과정을 실행합니다.  
```
python main.py preprocessing -c config/preproc-27-save.yaml [--work_dir <work folder>]
```
<br/>
preproc-27-save.yaml파일과 preproc-27.yaml 두가지가 있는데, preproc-27.yaml파일은 한국 수어 데이터 적용을위해 segment단계까지 전처리 과정을 생략시킨 파일입니다.  
이 파일로 실행시 오류가 일어나므로, 원본 파일인 preproc-27-save.yaml을 사용해 주세요.  
또한 제가 이용한 work_dir 위치는 **/users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/workdir** 입니다. 
<br/>  

#
<br/>  

#### 2-1. 전처리 실행 결과  
  
전처리를 실행하시면 workdir 디렉토리 안에 각 단계별로 디렉토리가 생성되어 단계별 결과물이 저장됩니다.  
asllvd를 사용한 전체 전처리 단계도 진행한 바 있으나, 현재는 한국수어데이터 적용을 위해 단계별 디렉토리는 삭제되었으며, segmented 디렉토리만 남아있는 상태입니다.  
  
![segmented](https://user-images.githubusercontent.com/82634312/179699877-cd73f387-81e4-4b4d-912c-16aed3cf3717.png)  <br/><br/>

#### 2-2. 현 디렉토리 상황
한국 수어 데이터 적용을 위해 preproc-27.yaml파일로 전처리를 진행하였으므로, Neuron3 서버의 디렉토리 상황은 preproc-27.yaml실행 기준으로 서술하였습니다.  
앞서 말한 segmented 디렉토리에는 segment단계에서 생성되는 label 파일들을 임의로 생성해놓았습니다.  

![label_files](https://user-images.githubusercontent.com/82634312/179699762-ee6a3c85-549e-4f27-af68-5cbee6dd2f21.png)  

파일명에서 알수 있듯이 모두 KETI 데이터 세트이며, /dataset 위치에 있는 KETI 데이터와는 별개로 **/users/suhyeon/GitHub/ST-GCN-SL/KETI** 위치에 해당 비디오셋들이 따로 위치해 있습니다.  
따라서 현재 전처리 단계는 **/users/suhyeon/GitHub/ST-GCN-SL/KETI**위치의 비디오를 가지고 진행되도록 설정되어 있습니다.  
다른 비디오 세트로 전처리를 실행하고 싶다면, 사용할 preproc-config 파일에서 경로를 수정해야 합니다.  

![image](https://user-images.githubusercontent.com/82634312/179703735-e5e18762-de9d-4b89-85d7-f8abb75e5a52.png)  

preproc-27-save.yaml 파일입니다.  
전처리 단계 별 설정사항과 입력/결과 디렉토리를 확인할 수 있습니다.  
여기서는 원시 비디오 세트의 저장 위치가 /users/suhyeon/GitHub/ST-GCN-SL/original로 설정되어 있습니다.  
이는 asllvd 세트가 저장되어있는 위치입니다.(역시 /dataset에 있는 asllvd와는 별개입니다)  

<br/><br/><br/><br/>
