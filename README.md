# 캡슐내시경 병변 검출
## mAP 0.7 달성 소스 코드

**Discussion:** https://dacon.io/competitions/official/235855/overview/description

**사용한 모델** 
* Faster_RCNN
* DetectoRS
* CenterNet

**submissions**
* Faster_RCNN의 결과와 DetectoRS의 결과와 CenterNet의 결과를 앙상블한 submission 결과 mAP 0.70125

**모델 훈련 환경**
* CPU Intel(R) Core(TM) i7-4790K 4.00GHz
* GPU RTX 3060 12GB
* python 3.8.11
* mmdetection 2.18.0
* mmcv 1.3.11
* torch 1.9.0+cu111
* ensemble_boxes 1.0.7
* cuda 10.1

**모델 학습 단계**
* 1단계: 학습데이터 시각화
* 2단계: 원본 데이터 셋 분포와 동일한 분포를 갖게끔 데이터 셋 분할
* 3단계: json 파일들을 Coco Format Json 파일로 컨버트
* 4단계: 모델 학습
* 5단계: 각 모델의 inference 진행
* 6단계: 각 모델의 inference 결과를 앙상블 실행
