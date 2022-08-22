# NIL-st-gcn
Sign Language Recognition을 위한 **ST-GCN**과 **ST-GCN-SL** 모델을 업로드한 리포지토리 입니다.  
  
  
###### 원본 리포지토리 링크  
[ST-GCN](https://github.com/yysijie/st-gcn)  
[ST-GCN-SL](https://github.com/amorim-cleison/st-gcn-sl)  

###### 각 모델별 문서
[ST-GCN](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN_README.md)  
[ST-GCN-SL](https://github.com/LimSuH/NIL-st-gcn/blob/main/ST-GCN-SL_README.md)
   
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
