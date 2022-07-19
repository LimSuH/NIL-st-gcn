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
***


#### 데이터 전처리  
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
preproc-27-save.yaml파일과 preproc-27.yaml 두가지가 있는데, preproc-27.yaml파일은 한국 수어 데이터 적용을위해 segment단계까지 전처리 과정을 생략시킨 파일입니다.  
이 파일로 실행시 오류가 일어나므로, 원본 파일인 preproc-27-save.yaml을 사용해 주세요.  
또한 제가 이용한 work_dir 위치는 **/users/suhyeon/GitHub/ST-GCN-SL/st-gcn-sl/workdir** 입니다. 
  
전처리를 실행하시면 workdir 디렉토리 안에 각 단계별로 디렉토리가 생성되어 단계별 결과물이 저장됩니다.  
asllvd를 사용한 전체 전처리 단계도 진행한 바 있으나, 현재는 한국수어데이터 적용을 위해 단계별 디렉토리는 삭제되었으며, segmented 디렉토리만 남아있는 상태입니다.  
  
이 segmented 디렉토리에는 segment단계에서 생성되는 label 파일들을 임의로 생성해놓았습니다.
![label_files]()

ST-GCN-SL 모델은 다음과 같은 전처리 단계를 거칩니다.
![st-gcn-slpreproc](https://www.cin.ufpe.br/~cca5/img/dataset_preprocessing.png)  


###### 1. 
