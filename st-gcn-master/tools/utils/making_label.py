# video 폴더 한 개에 대한 json 파일을 생성하는 코드
import json
import pandas as pd
import os

# pd.options.display.max_columns = None
# pd.options.display.width = None

'''
dir_xlsx: KETI-2017-2018.xlsx 파일 위치  ex) ../KETI/KETI-2017-2018.xlsx
dir_video: video 폴더 위치  ex) ../KETI/6001~8280
dir_json: 저장하고자 하는 json 파일 위치 (.json 미포함)  ex) ../KETI/label
'''
dir_xlsx, dir_video, dir_json = input("please type path: ").split()

df = pd.read_excel(dir_xlsx)  # 방향-타입(단어/문장)-파일명-한국어 구성

df = df.drop(columns='Unnamed: 0')  # drop Unnamed: 0
df = df.drop(columns='방향')  # drop 방향


files = os.listdir(dir_video)  # 폴더 내 파일명 저장

fold = pd.DataFrame(files, columns=['파일명'])  # 특정 폴더 내 파일명들
fold['파일명'] = fold['파일명'].str.split(".").str.get(0)


fdf = pd.merge(df, fold, how='right')  # 타입(단어/문장)-파일명-한국어 구성

A = fdf[fdf['타입(단어/문장)'] == '문장'].index  # 단어만 사용
fdf.drop(A, axis='index', inplace=True)
fdf = fdf.drop(columns='타입(단어/문장)')  # drop 타입(단어/문장)

fdf['파일명'] = fdf['파일명'].astype(str)  # 문자열 변환
fdf['한국어'] = fdf['한국어'].astype(str)

label = fdf['한국어'].to_list()  # '한국어'만 불러옴

set1 = set(label)  # 중복값 제거
label = list(set1)

label.sort()  # 정렬

label = pd.DataFrame({'label': label})  # 폴더 내 영상의 한국어 단어 의미 중복 제거 후 정렬
label['label_index'] = label.index  # label-label_index 구성
# label.to_excel(dir_xlsx + "_wlabel.xlsx")

fdf = pd.merge(fdf, label, how='left', left_on='한국어', right_on='label')
fdf = fdf.drop(columns='한국어')
fdf.insert(1, "has_skeleton", False)  # 파일명-has_skeleton-label-label_index 구성
# fdf.to_excel(dir_xlsx + "_result.xlsx")

dic = fdf.set_index('파일명').T.to_dict()  # dictionary로 변환

with open(dir_json + '.json', 'w') as f:
    json.dump(dic, f, ensure_ascii=False, indent=4)  # json 파일로 저장s