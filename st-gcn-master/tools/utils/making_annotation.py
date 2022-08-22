# 2017, 2018 annotation xlsx 파일을 하나로 합치는 코드
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

'''
dir_2017: 2017 annotation xlsx 파일 위치  ex) ../KETI/KETI-2017-SL-Annotation-v2_1.xlsx
dir_2018: 2018 annotation xlsx 파일 위치  ex) ../KETI/KETI-2018-SL-Annotation-v1.xlsx
dir_xlsx: 하나로 합쳐진 annotation 파일을 저장하고자 하는 위치 (.xlsx 미포함)  ex) ../KETI/KETI-2017-2018
'''
dir_2017, dir_2018, dir_xlsx = input("please type path: ").split()

df1 = pd.read_excel(dir_2017)
df2_sheet1 = pd.read_excel(dir_2018, sheet_name="KETI-2018-수어데이터-학습용-Annotation")
df2_sheet2 = pd.read_excel(dir_2018, sheet_name="KETI-2018-수어데이터-응답용-Anotation")

df1 = df1.drop(columns="번호")
df1 = df1.drop(columns="언어 제공자 ID")
df1 = df1.drop(columns="취득연도")
df1 = df1.drop(columns="Unnamed: 7")
df1 = df1.dropna()

df2_sheet1 = df2_sheet1.drop(columns="번호")
df2_sheet1 = df2_sheet1.drop(columns="언어 제공자 ID")
df2_sheet1 = df2_sheet1.drop(columns="취득연도")
df2_sheet1 = df2_sheet1.drop(columns="Unnamed: 5")
df2_sheet1 = df2_sheet1.drop(columns="Unnamed: 8")
df2_sheet1 = df2_sheet1.dropna()

df2_sheet2 = df2_sheet2.drop(columns="번호")
df2_sheet2 = df2_sheet2.drop(columns="언어 제공자 ID")
df2_sheet2 = df2_sheet2.drop(columns="취득연도")
df2_sheet2 = df2_sheet2.dropna()

# 2017 파일명 재설정
df1["파일명"] = df1["파일명"].str.split(".").str.get(0)

result = pd.concat([df1, df2_sheet1, df2_sheet2], ignore_index=True)
result.to_excel(dir_xlsx + '.xlsx')  # xlsx 파일로 저장
