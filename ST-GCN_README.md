# [ST-GCN](https://github.com/yysijie/st-gcn)
Spatial-Temporal graph convolution을 human action recognition에 사용한 모델  
[논문 링크](https://github.com/yysijie/st-gcn)  

[Dataset](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN_README.md#dataset)  
[Training](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN_README.md#training)  
[Pipeline](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN_README.md#pipeline)
<br/><br/><br/>


# Dataset
ST-GCN은 Kinetics와 NTU-RGB+D 두가지 데이터를 사용합니다.  

### Kinetics  
실제 영상 데이터가 아닌, URL 모음입니다.  
따라서 ST-GCN에서는 kinetics에 골격 데이터를 추출한 Kinetics-skeleton을 제공합니다.  

### NTU-RGB+D  
NTU의 경우 skeleton 데이터를 포함하고 있습니다.  
[NTU-RGB+D 웹사이트에서 다운받으실 수 있습니다.](https://rose1.ntu.edu.sg/)  
리포지토리에 기록된 링크로 가시면 이용하실 수 없습니다.  
또한 해당 사이트에서 제공하는 데이터를 이용하기 위해선 가입과 데이터 이용 요청이 필요합니다.  
모든 계정은 연구실 소속이어야 하므로 학교 메일 계정을 이용하셔야 가입이 승인됩니다.  
  
  
![image](https://user-images.githubusercontent.com/82634312/180881999-45f1d7c4-60ff-4d7a-9fa9-f90a11d2505f.png)  
이외에도 이용 요청에 추가적인 정보가 필요해서, AiHub에서 했던것 보다는 다른 정보가 필요할 것으로 보입니다.  
(아직 다운을 진행하진 않았습니다)  
  


두 데이터 모두 다운과 압축 해제 후, 전처리 과정을 거쳐야 합니다.  

### KETI
st-gcn-master/data에 KETI 데이터 디렉토리를 만들었습니다!  
디렉토리 구조는 다음과 같습니다.  
</br>
![image](https://user-images.githubusercontent.com/82634312/185132197-38e6d346-7e02-4ab8-9aed-6b01843907f9.png)  


  
</br></br>
## 원시 데이터  
우선 kinetics 데이터셋을 기준으로 서술하였습니다.  
</br>
ST-GCN의 전처리 과정이나 인풋 형식은 ST-GCN-SL에 비해 훨씬 간단합니다.  
이는 ST-GCN의 인풋 데이터가 이미 1차적인 openpose 추출 과정을 거쳐서 들어오는데 이유가 있습니다.  
ST-GCN레포지토리가 제공하는 raw 데이터도 별도의 메타데이터 파일이 존재하지 않으며, 모두 좌표와 라벨을 기록한 json 파일만이 인풋으로 사용됩니다.  
</br>
![image](https://user-images.githubusercontent.com/82634312/180869950-467718ea-d817-412a-aac1-486a075f8ef9.png)  
kinetics-skeleton의 raw 데이터를 다운받아 압축 해제한 모습입니다.  
데이터가 train과 validation 세트로 구분되어 있으며, 각각 좌표와 라벨 파일로 나누어진 것을 볼 수 있습니다.  
  
### label 파일
![image](https://user-images.githubusercontent.com/82634312/180870395-63025cc0-046f-4f5f-9331-493b8a511c09.png)  
kinetics-train-label.json입니다. 이는  
```
{
    "유튜브동영상id": {
        "has_skeleton": [true/false], 
        "label": "라벨이름", 
        "label_index": 라벨의 인덱스
    }, 
}  
```  

구조로 이루어진 형태입니다.  
```
https://www.youtube.com/watch?v="유튜브동영상id"
```  
를 입력하시면 실제 유튜브 동영상을 확인할 수 있습니다.    
또한 개중 id 앞에 -가 붙어있는 것은 더이상 이용할 수 없는 영상입니다.  
  
학습에 이용된 모든 라벨 종류와 인덱스는 /resource/kinetics_skeleton/lable_name.txt 에서 확인하실 수 있습니다.  
인덱스 순서대로 라벨들이 line by line으로 정리되어 있습니다. 깃허브에서 라인 숫자와 함께 보시면 확인이 쉽습니다. (단, 데이터셋의 라벨 인덱스는 0부터 시작하므로 한줄씩 밀려있는것을 유념해 주세요!)  

</br>

### 데이터 파일  

kinetics_train 디렉토리로 들어가면, 다음과 같은 파일들이 있습니다.  
  
  
![image](https://user-images.githubusercontent.com/82634312/180872317-04ebc3df-8408-4afe-a686-e3ead865254f.png)  
  
이 json 파일들은 각 동영상 별로 keypoint 좌표를 두었습니다.  
각 파일의 이름은 역시 동영상 id로, 라벨과 같이  
```
https://www.youtube.com/watch?v="유튜브동영상id"
```  
를 입력하시면 실제 유튜브 동영상을 확인할 수 있습니다.  
  
  
![image](https://user-images.githubusercontent.com/82634312/180872876-412c19d3-7feb-47e0-bd43-b86d6a42f835.png)  
각 json 파일을 확인하면 위와 같습니다. 각 동영상들로부터 프레임 별로 pose(좌표), 좌표의 score, label, label_index를 기록해 놓습니다.  
validation도 이와 같은 형식을 가지고 있습니다.  
**ST-GCN-SL보다 훨씬 갖추기 쉬운 형식으로, 전에 만들었던 mediapipe를 이용한 skeleton 적용 코드를 응용하여 이와 같은 형태를 만드는 코드 구현이 가능해 보입니다.**  
  
또한 NTU-RGB+D도 kinetics와 비슷한 형태를 가지고 있을것이라 예상되나, 현재 Neuron3에 있는 NTU 데이터셋을 전처리가 완료된, pkl과 npy 파일 뿐입니다.  
ST-GCN 레포지토리에 올려진 NTU raw 데이터 다운로드 링크는 현재 사용이 불가하여, 다른 사이트를 찾아 놓으려 합니다.

</br></br>
## 데이터 전처리  

[데이터이름]_gendata.py_ 는 kinetics와, NTU마다 다른 방법으로 파일을 pkl, npy 파일로 구성합니다.  
그러나 각 데이터에 맞추어 세부 사항이 다를 뿐, 대략적인 진행 방법은 같다고 볼 수 있습니다.  

```
#Kinetics-skeleton
python tools/kinetics_gendata.py --data_path data/kinetics-skeleton  
  

#NTU  
python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons(Neuron3에는 없습니다)>
```
  
  
위의 과정을 거치지 않고 전처리가 이미 완료된 데이터도 제공됩니다.  
https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view  
전처리가 완료된 데이터는 좌표는 pkl, 라벨은 npy 형태로 제공됩니다.  
  
각 데이터셋 별 전처리 코드에 대해 설명하겠습니다.  
  
  

### kinetics  
kinetics_getdata.py는 data/kinetics-skeleton로부터 kinetics_{train/val}_label.json_ 과 kinetics_skeleton 디렉토리의 각 동영상별 좌표 파일을 불러와 전처리를 실행합니다.  
과정은 아주 단순합니다.  kinetics_{train/val}_label.json_ 로부터는 라벨들을 sample_label 변수에 저장한뒤, 한번에 pkl 파일로 저장하고, kinetics_skeleton 디렉토리의 파일들로부터는 좌표를 불러와 sample_name 변수에 모두 저장하고 open_memmap 함수로 npy파일로 저장합니다.  

  
데이터 전처리는 kinetics_gendata.py -> feeder_kinetics 의 getitem()에서 이루어집니다.  
getitem은 좌표가 저장된 json 파일에서 데이터들을 읽어들입니다.  
이후 다음과 같은 shape의 0으로 초기화 된 넘파이 배열을선언합니다.

각 채널별로(3개 - x, y, score)  
>각 프레임 별로(총300프레임)  
> >각 joint 별로(18개)  
> > >영상의 사람 수 만큼

</br>
이제 위의 넘파이 배열에 불러온 좌표값을 저장합니다.(이를 그래프 구성이라 볼수 있을것 같습니다.)  
영상에 복수의 사람이 등장한다면 최대 4명까지에 대해서만 다음을 진행합니다.  


```
data_numpy[0, frame_index, :, m] = pose[0::2]
data_numpy[1, frame_index, :, m] = pose[1::2]
data_numpy[2, frame_index, :, m] = score
```
</br>
n번째 프레임에 대해
1. m번째 사람의 모든x좌표를 가져와 저장합니다
2. m번째 사람의 모든y좌표를 가져와 저장합니다
3. m번째 사람의 모든 score를 가져와 저장합니다
모든 사람에 대해 마쳤다면 다음 프레임으로 넘어가 반복합니다  

이후는 data augmentation을 진행합니다.  
이후 각 좌표값의 score가 제일 높은 순으로 각 사람들의 좌표를 정렬합니다.  
이후 모든 사람의 데이터를 넘기는것이 아니라, 가장 신뢰도 높은사람 2명의 데이터만 사용하게 됩니다.  
이렇게 정제된 데이터는 다시 kinetics_gendata.py로 넘어와, npy와 pkl 파일이 되어 학습에 사용됩니다.  
</br></br>


### NTU RGB+D  
NTU도 kinetics와 방식이 다를 뿐 같은 결과물을 만들어 냅니다.  
단 NTU는 별도로 라벨 데이터가 있는것이 아니라, 데이터의 파일 이름에 라벨 인덱스가 적혀있어 이를 통해 데이터 전처리를 진행합니다. (자세한것은 NTU Raw데이터를 확인해보려 합니다)   
이를 sample_label과 sample_name 리스트에 한번에 저장한 뒤, pkl은 pickle.dump(), npy파일은 open_memmap함수로 변환해 학습에서 사용되는 데이터를 만듭니다.

![image](https://user-images.githubusercontent.com/82634312/180882420-30541104-96a6-4708-af80-5b6ff7e072af.png)  
위와 같은 과정을 거쳐, 유저에게는 툴바로 전처리 진행도가 보여집니다.  
진행에 약간의 시간이 소요됩니다....  
  
  
모든 과정이 끝나면 kinetics의 경우 **/data/Kinetics/kinetics-skeleton**, NTU는 **/data/NTU-RGB-D** 에 저장됩니다.  
</br></br>

### KETI
한국어 수어 데이터셋 전처리를 구현하였습니다.  
구조는 다음과 같습니다.  
</br></br>


1.다음을 입력하여 전처리를 실행합니다.  
</br>

##### (1) label 파일과 skeleton json 파일 생성
lab 계정으로 접속합니다.  
가상환경을 활성화 합니다. 반드시 잊지 말고 가상환경을 활성화 해주세요! openpose가 작동하지 않을 수 있습니다.  
```
conda activate pytorch
```

전처리를 실행합니다.  
```
python tools/ksl_pose_estimation.py --data data/KETI
```  

2. 엑셀 파일로부터 label.json 파일을 생성합니다.  
  - making_annotation.py 실행
  - makine_lable.py 실행  
</br>

3. openpose initialize  

4. def making_data 호출  

5. def get_label 호출  

6. label index를 오름차순으로 정렬합니다.  

7. label_train(test).json파일로부터 영상 목록과 label index를 불러와 dictionary로 저장합니다.  

8. 영상 목록으로 수어 영상 데이터를 불러와 pose estimation을 진행합니다  

9. def making_json 호출 - skeleton과 label, label index로 json 파일을 만듭니다. 

</br>


현재는 6001~8280 까지의 subset으로만 전처리를 진행하였습니다.  
영상마다 길이가 워낙 달라서, 데이터에 따라 전처리 시간이 크게 달라지고 있습니다.  
making_annotation, lable.py 파일은 tools/utils에 있습니다.  
</br>

학습을 위해서는 데이터셋의 label index가 0,1,2.... 이렇게 순서대로 구성되어야 합니다.  
현재는 label index를 순서대로 가져오도록 했으므로 위의 과정이 생략되었지만, ksl_gendata.py에 포함되어 있습니다.  
사용 시 주석 해제하여 실행하시면 됩니다.  
```
 mapping = 0
 for i in range(data_num):
     file_list.append(ground_truth[i][0])
     label.append(mapping)

     if ground_truth[i][0] < ground_truth[i + 1][0]:
         mapping += 1
```

</br></br></br>

# Training  
위의 한국어 수어 데이터 전처리를 거쳐, 학습까지 진행하였습니다.  
학습 실행을 위해서는 다음과 같은 과정을 거쳤습니다.  
</br>
1. KETI 데이터 학습을 위한 yaml 파일을 생성합니다.  
위치는 config/st_gcn/KETI/train.yaml 입니다.  
구성은 kinetics의 config 파일과 같으나, 경로나 값을 KETI 데이터, 그리고 학습 환경에 맞추어 변경하였습니다.  
특히 parameter은 그때의 학습 환경에 맞추어 재설정 해야 합니다.  
</br>

### 다음과 같은 항목을 환경에 맞추어 필수적으로 재설정 해야 합니다:  
경로 전반, num_class, device, batch_size, test_batch_size
</br>

#### num_class  
학습 데이터의 class 종류 갯수입니다.  
현재 저장된 data.npy는 총 20개의 class로 이루어져 있습니다.  
</br>

#### device  
gpu 갯수입니다.  
st-gcn의 저자들은 4개의 gpu를 사용했으나, Neuron3는 2개의 gpu를 사용합니다.
</br>

#### batch_size, test_batch_size  
사용하는 데이터의 갯수에 따라 맞아떨어지도록 batch 사이즈를 조정합니다.
</br>
이외의 parameter는 원하는 대로 조정이 가능합니다.  

</br></br>
2. 다음과 같이 학습을 진행합니다.
lab 계정으로 접속합니다.  
다음을 실행합니다.  
</br>

```
python main.py recognition -c config/st_gcn/KETI/train.yaml
```
![image](https://user-images.githubusercontent.com/82634312/186068924-a4f58638-a180-4c7b-90c6-e44b8219d44f.png)  
입력시 다음과 같이 학습이 진행됩니다.  
학습이 진행되면서 출력된 학습 현황과 model이 저장됩니다.  
st-gcn-master/work_dir/recognition/KETI/ST_GCN 에서 모델, 학습 진행 로그(log.txt), parameter 설정(config.yaml)을 확인하실 수 있습니다.  
</br></br></br></br>



# pipeline  
### preprocessing pipeline  
</br>

![image](https://user-images.githubusercontent.com/82634312/182273786-be09d992-eaa9-4f12-b9d4-c580a45deae3.png)  

</br></br>


### Training pipeline  
</br>

![image](https://user-images.githubusercontent.com/82634312/182276257-14297e1a-c529-458f-972d-7162fc099845.png)  
</br></br>

### demo pipeline  
</br>

![image](https://user-images.githubusercontent.com/82634312/182292719-fd869f85-6036-46e0-8849-ac75f5b5b29e.png)  
</br></br>






