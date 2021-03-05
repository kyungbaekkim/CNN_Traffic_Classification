# :books:CNN Traffic Classification Module

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

### :heavy_check_mark:데이터 전처리
CNN 모델의 입력을 위해 통계적으로 추출된 악성 트래픽 데이터 셋을 nxn 형태의 Metrix로 변환
전처리 후 생성되는 데이터
  - 9x9 Input Metrix
  
### :heavy_check_mark:악성 트래픽 분류
<img src="/img/Architecture of CNN Traffic Classification Model.PNG">
  - 인공지능 모델: CNN 기반 악성 트래픽 분류
  - 입력 데이터: 78개의 통계적 특징을 9x9 Input Metrix 형태로 구성
  - 출력 데이터: 악성 트래픽 유무

## :book:Dataset Summary
### :heavy_check_mark:CICIDS2017
