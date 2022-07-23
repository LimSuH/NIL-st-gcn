# [ST-GCN](https://github.com/yysijie/st-gcn)
Spatial-Temporal graph convolution을 human action recognition에 사용한 모델  
[논문 링크](https://github.com/yysijie/st-gcn)  

[Dataset](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#dataset)  
[Training](https://github.com/LimSuH/NIL-st-gcn/edit/main/ST-GCN-SL_README.md#training)
<br/><br/><br/>


# Dataset
ST-GCN은 Kinetics와 NTU-RGB+D 두가지 데이터를 사용합니다.  

### Kinetics  
실제 영상 데이터가 아닌, URL 모음입니다.  
따라서 ST-GCN에서는 kinetics에 골격 데이터를 추출한 Kinetics-skeleton을 제공합니다.  

### NTU-RGB+D  
NTU의 경우 skeleton 데이터를 포함하고 있습니다. ST-GCN이 제공하는 구글 드라이브에서 다운받을 수 있습니다.  
  
    
두 데이터 모두 다운과 압축 해제 후, 전처리 과정을 거쳐야 합니다.  
```
#Kinetics-skeleton
python tools/kinetics_gendata.py --data_path <path to kinetics-skeleton>  
  

#NTU  
python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```
  
  
위의 과정을 거치지 않고 전처리가 이미 완료된 데이터도 제공됩니다. https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view  
[데이터이름]_gendata.py_ 는 다른 메타데이터를 사용하지 않고, 데이터의 파일 이름으로 라벨을 생성합니다.  
이를 리스트에 한번에 저장한 뒤, pkl 파일로 만들어 학습에서 사용되는 라벨 파일을 만듭니다.
sample_label과 sample_name이 생성되는데, sample_label 은 모든 라벨 모음, sample_name은 동영상의 라벨을 가리킵니다.  
이후 데이터들을 training과 validation 셋트로 나눕니다.
