# :books:CNN Traffic Classification Module

## :book:개요

네트워크 트래픽 특징 벡터 정보(예: KDD cup 99, CICIDS2017)를 이미지로 변환하여 입력으로 사용하는, CNN기반 네트워크 트래픽 공격 분류 모델을 학습하고 검증(3 fold cross validation)하는 기능을 지원하는 모듈.

## :book:License

이 코드는 재배포, 재발행, 미러링 될수 있습니다. 다만, 다음의 논문을 꼭 인용해주시기 바랍니다.

You may redistribute, republish, and mirror this code in any form. However, any use or redistribution must include a citation to the following paper.

> Sungwoong Yeom, Van-Quyet Nguyen and Kyungbaek Kim, Assessing Convolutional Neural Network based Malicious Network Traffic Detection Methods, KNOM REVIEW, Vol. 22, No. 1, pp. 20-29, August, 2019.


## :book:Model Process
### :heavy_check_mark:입력 파라메터
```
'-d','--dataset', type=str, help='Training Data Path (necessary)'
'-n','--outputname', type=str, default="", help='Output File Prefix'
'-c','--conv', type=int, default=1, help='Convolution Layer Number'
'-p','--pool', type=int, default=1, help='Max Pooling Layer Number'
'-f','--fcl', type=int, default=1, help='Fully Connected Layer Number'
'--hidden', type=int, default=512, help='Hidden Layer Number'
'--units', type=int, default=100, help='N Hidden Units Value'
'-l','--learning', type=float, default=0.00001, help='Learning Rate Value'
'-i','--iters', type=int, default=10500, help='Training Iters Value'
'-b','--batch', type=int, default=50, help='Batch Size Value'
'--display', type=float, default=10, help='Display Step'
'--dropout', type=float, default=.5, help='Dropout Value'
```
  - 입력 파일: Sample.csv (CICIDS2017 Friday Morning)
  - 출력 파일: Precision, Recall, F1score, Confusion Matrix 정보를 담은 결과 

### :heavy_check_mark:데이터 전처리
1. 데이터셋에서 공백문자 제거 및 Oulier(16777216=2^8*2^8*2^8 이상의 값)을 16777216으로 변환
2. 3-Fold Validation을 수행
2. CNN 모델의 입력을 위해 통계적으로 추출된 악성 트래픽 데이터 셋을 nxn 형태의 Metrix로 변환
  - 전처리 후 생성되는 데이터: 9x9 Input Metrix
  
### :heavy_check_mark:악성 트래픽 분류
<img src="/img/Architecture of CNN Traffic Classification Model.PNG">

  - 인공지능 모델: CNN 기반 악성 트래픽 분류
  - 입력: 78개의 통계적 특징 9x9 Input Metrix
  - 출력: 정상(BENIGN), 악성(Bot) 분류

## :book:Dataset Summary
### :heavy_check_mark:CICIDS2017 Friday Morning
<img src="/img/DataFormat of CICIDS2017.PNG">

  - 지속시간, 패킷 수, 바이트 수, 패킷 길이 등과 같은 통계적 특성 78가지 column을 가짐
  - 3988개의 row로 구성 (BENIGN: 2022, Bot: 1966)
  - CIC IDS2017 dataset details : https://www.unb.ca/cic/datasets/ids-2017.html 
