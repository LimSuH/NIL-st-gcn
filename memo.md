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
  
   
 ```
 sample_data, sample_label= mydataset[0]
 print(sample_data.shape, sample_label)
 ```
 output:  
 (71, 24) test  
    
    
 ```
 trainset = DataLoader(mydataset, batch_size=batch_size)
 sample, label = next(iter(trainset))
 print(sample.shape, label)
 ```
 output:  
 torch.Size([1, 71, 24]) ('test',)  
    
 -> 둘의 차이는 dataLoader로 감쌌기 때문인가  
 타입도 달라지고 지정한 batch_size때문에 차원이 하나 더 생기는  
   
    
 각도 데이터 넣었을때 shape 관련 오류  
 차원이 아니라 frame 때문이었음  
 각 영상마다 frame 길이가 다르므로 미리 데이터 형태를 지정x  
 
~> (1)transformer에서 일정 frame길이로 잘라내기, 덧붙이기: frame 길이 뽑은것의 평균/최빈값/....  
   (2)outchannel을 알아내서 계산?
   
**1dcnn을 이용한 동작 분류 예제:**  
https://st4ndup.tistory.com/13

**작성한 코드 in Neuron3**  
<details>
<summary>펼치기</summary>
<div markdown="1">

```
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#custom dataset
class CustomDataset(Dataset):
    def __init__(self, annotation_file, data_dir, transform=None, target_transform=None, col_name="file_name", col_label="label"):
        self.label = pd.read_csv(annotation_file, names=[col_name, col_label], skiprows=[0])
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.label.iloc[index, 0])
        data = np.load(data_path)
        label = np.zeros(15, dtype = np.float32)
        label[self.label.iloc[index, 1]] = 1.0
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        
        return data, label

#custom transfomer
class myTransformer(object):
    def __call__(self, sample):
        sample = sample.astype(np.float32)
        sample = np.transpose(sample)     
        sample = torch.from_numpy(sample)
        #print(sample.shape)   
        return sample

# model
class CustomNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.one_stack = nn.Sequential(
            nn.Conv1d(24, 34, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(34, 14, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(14*67, 15)
        )
        

    def forward(self, x):
        #x = self.flatten(x)
        x = self.one_stack(x)
        return x

#trainig & test function
def training(dataloader, model, criterion, optim):
    total_loss = 0
    accuracy = 0

    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        #prediction(forward)
        pred = model(data)
        loss = criterion(pred, label)
        
        #backward
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()


        if pred.argmax(1) == label.argmax(1): 
            accuracy += 1
    accuracy = accuracy/len(dataloader)*100
    total_loss /= len(dataloader)

    return accuracy, total_loss

def test(dataloader, model, criterion, optim):
    total_loss = 0
    accuracy = 0
    f1Score = 0
    
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            #prediction(forward)
            pred = model(data)# pred 결과가 1, 15(14, 15)x
            loss = criterion(pred, label)

            total_loss += loss.item()
            if pred.argmax(1) == label.argmax(1): 
                accuracy += 1
   
    accuracy = accuracy/len(dataloader)*100
    total_loss /= len(dataloader)
    
    return accuracy, total_loss


def F1Score(testLoader, model):
    conf_matrix = np.zeros((15, 15), dtype=int)
    model.eval()
    precision = []
    recall = []

    with torch.no_grad():
        for data, label in testLoader:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            conf_matrix[pred.argmax(1)][label.argmax(1)] +=1

        print(conf_matrix)
        #precision
        for t in range(conf_matrix.shape[1]):
    
            p = conf_matrix[t][t] /conf_matrix[t].sum()
            precision.append(p)

            r = conf_matrix[t][t] / conf_matrix[:][t].sum()
            recall.append(r)
        
    avgPrecision = sum(precision) / len(conf_matrix[0])
    avgRecall = sum(recall) / len(conf_matrix[0])
    f1Score = 2 * ((avgPrecision * avgRecall) / (avgPrecision + avgRecall))

    return avgPrecision, avgRecall, f1Score   


#prediction
def prediction(sample, label, model):
    model.eval()
    
    with torch.no_grad():
        pred = model(sample)
        print(f"pred: {pred.argmax(1)}, actual: {label.argmax(1)}")
        if pred.argmax(1) == label.argmax(1):
            print("The predicton is correct.")
        else:
            print("The prediction is wrong.")


if __name__ == "__main__":
    #check gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")
print(device)

annotation_file = "test_annotation.csv"
data_dir = "./"
batch_size = 1
    
#dataset
mydataset = CustomDataset(annotation_file , data_dir, transform=myTransformer())
trainLoader = DataLoader(mydataset, batch_size=batch_size)
sample, label = next(iter(trainLoader))
print("next dataloader", sample.shape, label.shape)

#model
myModel = CustomNeuralNetwork().to(device)
# print(myModel)

#training
epochs = 100
criterion = nn.CrossEntropyLoss()
optimization = torch.optim.SGD(myModel.parameters(), lr=0.01)
best_loss = 100
    
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch_num in range(epochs):
    accuracy, loss = training(trainLoader, myModel, criterion, optimization)
    train_loss.append(loss)
    train_accuracy.append(accuracy)

    print(f"\nEpoch: {epoch_num} / {epochs - 1}, Accuarcy: {accuracy} Loss: {loss}")

    accuracy, loss = test(trainLoader, myModel, criterion, optimization)
    test_loss.append(loss)
    test_accuracy.append(accuracy)
    print(f"Evaluation: [Accuracy: {accuracy}, Loss: {loss}]")
        
    if loss < best_loss:
        best_epoc = epoch_num
        best_model = myModel.state_dict()
        print("The best model state updated.")

torch.save(best_model, "./save/epoch_" + str(best_epoc) + ".pth")

#draw plot with best model
best_model = CustomNeuralNetwork()
state_path = os.path.join("./save", os.listdir("./save")[-1])
best_model.load_state_dict(torch.load(state_path))
prediction(sample, label, best_model)


xmin = 0
xmax = max(len(train_loss), len(test_loss))
ymin = min(min(train_loss), min(test_loss))
ymax = max(max(train_loss), max(test_loss))


plt.title("LOSS")
plt.axis([0, xmax, ymin, ymax])
plt.plot(train_loss, label="train", color='red')
plt.plot(test_loss, label="validation", color="blue")
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.title("ACCURACY")
plt.axis([0, xmax, min(min(train_accuracy), min(test_accuracy)), max(max(train_accuracy), max(test_accuracy))])
plt.plot(train_accuracy, label="train", color='red')
plt.plot(test_accuracy, label="validation", color="blue")
plt.legend()
plt.savefig('loss.png')
plt.show()

precision, recall, f1 = F1Score(trainLoader, best_model)
print(f"Average precision:{precision}, Average recall:{recall}, F1 Score:{f1}")



```


</div>
</details>  
</br>

##### F1Score #####
[참고한 링크](https://leedakyeong.tistory.com/entry/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-Confusion-Matrix%EB%9E%80-%EC%A0%95%ED%99%95%EB%8F%84Accuracy-%EC%A0%95%EB%B0%80%EB%8F%84Precision-%EC%9E%AC%ED%98%84%EB%8F%84Recall-F1-Score)  

아직 데이터를 하나밖에 넣지 않아 제대로 동작하지 않음(분모 0 에러)  
~~ 만들어둔 각도 데이터 프레임 분포 분석으로 데이터 길이를 통일하고, 실제 label 불러와 csv 파일 적용시켜 학습 돌려보기  
/users/suhyeon/GitHub/NIL-st-gcn/simple-chal-slr/analization/jupyter/about_frame.ipynb --> Keypoints-removal 데이터를 기준으로 frame 분포 정보를 확인한 jupyter notebook  
살펴보고 어떤 값을 기준으로 프레임 길이를 정할지 결정  
**프레임 패딩 방법**
</br>

##### 이번주 목표(2.22~) #####  
**1. angle 기반 데이터셋 생성**
- /dataset/KETI_SignLanguage/Video/0001~3000 로 subset 형성
- 사전에 만들어놓은 각도 생성 코드 활용(visualization 확인해보기)
- 일단 전체 데이터 생성

**2. angle 데이터 analyzation**
- 마찬가지로 기존에 만들어 놓은 analyzation 코드 활용, 프레임 분포 알아내기
- 프레임 자르기/패딩의 기준 확립
- 자르기 : 최솟값, 최빈값, 최대값 
- 패딩 : 제로 패딩, 마지막 프레임 연장, 영상 반복
- 자르기와 패딩에 있어... 중간 프레임을 솎아내거나 추가하는 방식  
  자르기의 경우 n번째마다의 프레임을 삭제
  패딩의 경우 n번째마다 앞 혹은 뒤의 프레임을 복사해 추가

**3. preprocessing code(with argparse)**
- 데이터 subset 생성
- 자르기와 패딩방법을 조합하여 총 6개의 데이터 셋을 만들어냄

**4. 데이터 학습**
- 6 가지 데이터셋을 학습하고 결과 비교

###### preprocessing pipeline ######  
![image](https://user-images.githubusercontent.com/82634312/221242840-2e3bbbdd-7335-4d62-968e-5b2a592ccf52.png)

