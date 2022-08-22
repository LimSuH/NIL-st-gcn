# [ST-GCN-SL](https://github.com/amorim-cleison/st-gcn-sl)
ST-GCN을 수어 인식에 활용한 모델  
[논문 링크](https://arxiv.org/abs/1901.11164)  

[Dataset](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#dataset)  
[Training](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#training)
<br/><br/><br/>
  
  
# Dataset  
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
- [1. 데이터 전처리](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC)  
  - [1-1. 전처리 단계](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#1-1-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%8B%A8%EA%B3%84)
  - [1-2. 전처리 실행](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-1-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%8B%A4%ED%96%89)  
    - [전처리 코드](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#%EA%B0%81-%EB%8B%A8%EA%B3%84%EB%B3%84-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%BD%94%EB%93%9C)  
  
- [2. 실제 진행 상황](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-%EC%8B%A4%EC%A0%9C-%EC%8B%A4%ED%96%89-%EC%83%81%ED%99%A9)  
  - [2-1. 전처리 결과](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-2-%EC%A0%84%EC%B2%98%EB%A6%AC-%EA%B2%B0%EA%B3%BC)
  - [2-2. 현 디렉토리 상황](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-3-%ED%98%84-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC-%EC%83%81%ED%99%A9)
<br/><br/><br/>




## 1. 데이터 전처리  

ST-GCN-SL 모델은 다음과 같은 전처리 단계를 거칩니다.  
![st-gcn-slpreproc](https://www.cin.ufpe.br/~cca5/img/dataset_preprocessing.png)   
<br/>
<br/>

## 1-1. 전처리 단계  

#### (1) Download  
asllvd 데이터를 다운받습니다. 
asllvd 데이터는 연속적인 수어 대화 영상과 메타 데이터 파일로 이루어져 있습니다.  
  
  
![image](https://user-images.githubusercontent.com/82634312/179709829-6fa41ae2-d5bb-4060-82f4-1d12be64d417.png)  
  
  
메타데이터 파일은 main gloss, 수어 발화자, gloss를 포함하는 동영상 파일 이름, 영상 확장자 등을 항목으로 가집니다.  
*연속적인 대화 영상이므로, 한 영상에 여러 gloss가 존재해 영상 기준이 아닌 gloss 기준으로 메타데이터 파일을 서술한것 같습니다.*  
<br/>

#### (2) Segment  
gloss별로 영상을 분할합니다. 또한 동시에 분할한 영상별 gloss를 기록한 파일과, 전체 gloss 파일을 생성합니다.  
자세한 형태는 [이곳](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#2-2-%ED%98%84-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC-%EC%83%81%ED%99%A9)을 확인해 주세요.  
<br/>
#### (3) Estimate Skeleton  
openpose를 이용해 영상에 스켈레톤을 적용합니다.  openpose는 Neuron3에 이미 다운되어 있으며, 위치는 '/home/lab/openpose/build'입니다. config 파일에서 자동으로 실행하므로 openpose의 위치는 크게 중요하지 않습니다.  
  
  
![keypoint_json](https://www.cin.ufpe.br/~cca5/img/openpose_coordinates.PNG)  
  
  
모든 키포인트 좌표는 json 파일로 저장됩니다.  
<br/>
#### (4) Filter Keypoint  
![keypoint](https://www.cin.ufpe.br/~cca5/img/filtered_keypoints_hand.png)  
  
  
모든 스켈레톤 키포인트를 사용하지 않으므로, 몸톤 5개, 각 손 11개로 총 27개의 키포인트 좌표를 추출합니다.  
(3)의 출력물과 유사한 형태지만 좌표가 더 적습니다.  
<br/>
#### (5) Split Dataset  
(4)에서 얻은 json파일을 train과 test 데이터 세트로 분류합니다.  
**train-80%, test-20%** 로 구성됩니다.  
<br/>
#### (6) Normalize & Serialize  
마지막으로, 정규화와 직렬화를 진행합니다.
정규화는 모든 데이터의 길이를 일정하게 만드는것을 의미합니다.  
st-gcn-sl 모델은 한 영상 당 63의 프레임 수를 사용하도록 했습니다. 원본 영상 데이터의 fps가 30이므로, 이는 약 2초정도의 길이입니다.  
  
직렬화는 모델이 돌아가는 파이썬 코드가 데이터를 읽어들일 수 있도록 파이썬 파일로 변환하는것을 의미합니다.
키포인트 좌표는 npy(넘파이 파일)로, label은 pkl파일로 저장됩니다.  
  
아래에서 모든 전처리 단계가 완료된 데이터를 확인할 수 있습니다.  
```
cd /users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/st-gcn/data/asllvd-skeleton-20/normalized
```
<br/><br/>




## 1-2. 전처리 실행  
#### 전처리 단계를 진행하려면 다음 과정을 실행합니다.  
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
asllvd-skeleton-preproc/config/ 디렉토리에는 preproc-27-save.yaml 파일과 preproc-27.yaml 두가지가 있습니다.  
preproc-27.yaml 파일은 한국 수어 데이터 적용을위해 segment단계까지 전처리 과정을 생략시킨 파일입니다.  
  
  
이 파일로 실행시 오류가 일어나므로, 전처리 실행 시 원본 파일인 preproc-27-save.yaml을 사용해 주세요.  
preproc-27-save.yaml은 원본대로 asllvd 데이터셋으로 전처리를 진행합니다.  
또한 제가 이용한 work_dir 위치는 **/users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/workdir** 입니다.  
<br/><br/>  

추가적으로, video_preprocessor.py도 66~69번째 줄까지의 주석을 해제해야 합니다.  
```
download=import_class(
                'processor.sl.preprocessor.downloader.Downloader_Preprocessor'),
segment=import_class(
                'processor.sl.preprocessor.splitter.Splitter_Preprocessor'),
```
<br/>  

#
전처리 코드가 내부적으로 실행되는 과정을 설명하겠습니다.  
main.py를 실행하면 /processor/sl/video_preprocessor.py가 호출됩니다.  
video_preprocessor.py는 config 파일에 적힌 phases 인자를 읽어들여, 해당하는 전처리 코드를 실행합니다.  
각 단계별 전처리 코드는 /processor/sl/preprocessor/ 에 위치합니다.  
  
  *preproc-27-save.yaml 파일에 적힌 Phases*
```
phases:
  download,
  segment,
  skeleton,
  filter,
  split,
  normalize
```

*video-preprocessor.py의 각 phase별 불리는 전처리 코드 파일*  
```
 def get_phases(self):
        return dict(
            download=import_class(
                'processor.sl.preprocessor.downloader.Downloader_Preprocessor'),
            segment=import_class(
                'processor.sl.preprocessor.splitter.Splitter_Preprocessor'),
            skeleton=import_class(
                'processor.sl.preprocessor.openpose.OpenPose_Preprocessor'),
            filter=import_class(
                'processor.sl.preprocessor.keypoint.Keypoint_Preprocessor'),
            split=import_class(
                'processor.sl.preprocessor.holdout.Holdout_Preprocessor'),
            normalize=import_class(
                'processor.sl.preprocessor.gendata.Gendata_Preprocessor')
        )
```
  
  
 <br/><br/> 
### 각 단계별 전처리 코드
#
#### (0) preprocessor.py, io.py, gendata_feeder.py  
각 단계별 사용되는전처리 class는 모두 preprocessor class를 상속하여 구성됩니다.    
즉, 각 단계들이 공통적으로 사용하는 함수들을 가진 파일입니다.  
preprocessor 클래스가 가진 함수들은 같은 위치의 io.py 파일을 불러와 구현됩니다.  
파일 입출력, 디렉토리 접근, 인자 호출, progress 바 표시 등을 포함합니다.  
  
  
gendata_feeder는 normalize 단계에서 사용됩니다. 정규화를 위해 frame을 가공합니다.  
<br/>

#### (1) Download  
/processor/sl/preprocessor/downloader.py 코드가 실행됩니다.  
다운로드 url, 디렉토리 위치를 argument로 preprocessor.py를 호출합니다.  
  
  
preprocessor.py는 사전에 설정된 url로부터 데이터를 다운받아 저장하고, 메타데이터로부터 인자를 저장합니다.  
이때 저장한 인자는 segmentation 등 다음 단계에서 활용됩니다.  

결과물은 ST-GCN-SL/st-gcn-sl/workdir/original 에 저장됩니다.
<br/>

#### (2) Segment   
/processor/sl/preprocessor/**splitter**.py 코드가 실행됩니다.  
프레임, ['Main New Gloss.1', 'Session', 'Scene', 'Start', 'End'] 과 같은 메타데이터 인자를 받아 영상 분할을 실행합니다.

결과물은 ST-GCN-SL/st-gcn-sl/workdir/segmented 에 저장됩니다.
<br/>

#### (3) Estimate Skeleton  
/processor/sl/preprocessor/openpose.py 코드가 실행됩니다.  
openpose 라이브러리를 불러와 스켈레톤을 적용하고 프레임별 좌표값을 json파일로 저장합니다.
결과물은 ST-GCN-SL/st-gcn-sl/workdir/skeleton 에 저장됩니다.
<br/>

#### (4) Filter Keypoint  
/processor/sl/preprocessor/keypoint.py 코드가 실행됩니다.  
__get_keypoints(self, arg)함수는 preproc-27.yaml 파일로부터 넘겨받은 'points' 인자로 사용할 키 포인트를 저장합니다.  
![image](https://user-images.githubusercontent.com/82634312/179737452-df274ce9-3133-47a6-8450-c987620eae11.png)  
결과물은 (3)과 마찬가지로 json파일로 저장됩니다.  

결과물은 ST-GCN-SL/st-gcn-sl/workdir/filtered 에 저장됩니다.
<br/>

#### (5) Split Dataset  
/processor/sl/preprocessor/**holdout**.py 코드가 실행됩니다.  
preproc-27.yaml 파일로부터 넘겨받은 'test', 'val' 인자로 데이터들의 비율을 조정합니다.  
st-gcn-sl 모델은 validation 세트 없이, 20%의 test세트와 나머지 80%의 training set로 구성되어있습니다.  
![image](https://user-images.githubusercontent.com/82634312/179738609-0ac625b8-77cd-422e-ab17-d7b4a350f2e3.png)  

결과물은 ST-GCN-SL/st-gcn-sl/workdir/splitted 에 저장됩니다.
<br/>

#### (6) Normalize & Serialize  
/processor/sl/preprocessor/normalize.py 코드가 실행됩니다.  
gendata-feeder을 호출해 데이터를 정규화 하고, 차이썬에서 읽어들이기 좋은 확장자로 변환합니다.  
json 파일로 저장된 keypoint는 npy파일로, 라벨은 pkl 파일로 저장합니다.
결과물은 ST-GCN-SL/st-gcn-sl/workdir/normalized 에 저장됩니다.
<br/><br/><br/>




## 2. 실제 실행 상황   

## 2-1. 전처리 결과  

전처리를 실행하시면 workdir 디렉토리 안에 각 단계별로 디렉토리가 생성되어 단계별 결과물이 저장됩니다.  
asllvd를 사용한 전체 전처리 단계도 진행한 바 있으나, 현재는 한국수어데이터 적용을 위해 단계별 디렉토리는 삭제되었으며, segmented 디렉토리만 남아있는 상태입니다.  
  
![segmented](https://user-images.githubusercontent.com/82634312/179699877-cd73f387-81e4-4b4d-912c-16aed3cf3717.png)  <br/><br/>

## 2-2. 현 디렉토리 상황  

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

<br/><br/><br/><br/><br/><br/><br/>

# Training  
ST-GCN-SL 모델 실행 과정을 설명하였습니다.  
모델이 가진 오류를 해결하고, 학습 진행을 위해 거친 과정을 서술하였습니다.  
<br/>
## 1. 오류 해결  
st-gcn-sl 모델이 가진 오류를 해결하였습니다.  

  
![image](https://user-images.githubusercontent.com/82634312/179758301-05c2046d-5daf-4479-9c05-cb395e39be32.png)  

pyyaml의 load 함수가 더이상 지원되지 않아 생긴 오류로, pyyaml 6.0.0을 삭제하고 낮은 버전으로 다시 설치하여 해결하였습니다.  
  
  
그러나 pyyaml 오류를 고치기 위해 torchlight까지 함께 삭제 해야하는 과정이 있었습니다.  
pip 로 다시 설치했지만, 각 모델들이 torchlight 라이브러리를 인식하지 못하고 import_class 를 불러오지 못하는 새로운 오류가 나타났습니다.  
이는 ST-GCN-SL의 원본인 ST-GCN 모델에서도 똑같이 나타는 현상으로, 아래 링크를 참고하여 해걀하였습니다.  


모델 실행 과정에서 같은 오류가 계속해서 나타나는 문제가 있었습니다. 실행에서 나타는 오류는 모두 해결했으나, 같은 오류가 또 나타난다면 [이곳](https://github.com/limaosen0/AS-GCN/issues/4) 을 참고하여 주세요.  
<br/><br/><br/>

## 2. 데이터 다운로드  
trining을 확인하기 위해서라면, 굳이 전처리 과정을 실행하지 않고 이미 전처리가 된 데이터를 받아 진행할 수 도 있습니다.  
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fB4BMmewTI5-eNDI83mcOmIETPjVg56N' -O adllvd-skeleton-20.zip  
unzip adllvd-skeleton-20.zip
```
Neuron 3에는 /users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/st-gcn/data/asllvd-skeleton-20/normalized 에 저장되어 있습니다.   
  
   
## 3. 학습 진행 확인  
```
cd /users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/st-gcn  
python main.py recognition -c config/sl/train-asllvd-skeleton-20.yaml --work_dir config/sl/
```  
위 명령은 새로운 모델을 training 합니다.
실행 시 지정한 work_dir에 10epoch마다 모델 parameter을 저장합니다. 또한 work_dir에 config.yaml 파일이 생성됩니다.  
이config 파일은 위의 명령으로 실행된 training 정보를 담고 있어, 다음에 같은 조건으로 학습을 진행할때 활용할 수 있습니다.  
  
  
모델 깃허브 페이지에 따르면, test.yaml을 통해 모델 평가도 진행할 수 있다고 나와있으나 test.yaml 파일이 존재하지 않는 문제가 있습니다.  
ST-GCN 모델에는 해당 파일이 존재하므로 참고해서 해결할 수 있을것으로 보입니다.  

