## 1. 대회 개요

### 한국어 음성인식 모델 역량 평가

주요 목표 : 네이버 클로버의 현안 문제인 `회의 음성 인식` 에 강건한 모델을 개발하는 것

## 2. 대회의 특징

![image](https://github.com/jungsiroo/jungsiroo.github.io/assets/54366260/322345dc-cf9f-4672-b3fd-c0bd21dc4e8d)

- 한국어 자유대화 학습용 데이터 **200,000개를 활용**하여 모델 학습
    - Format : PCM(음성 파일 중 header 가 없는 raw data)
    - Channel : Mono 채널
    - Label : 각 음성 파일의 대한 csv 전사 데이터
- 최종적으로 음성 파일에 대한 **Speech-To-Text 태스크를 학습**한 모델 개발
- 데이터 접근 불가 → 오디오 음성이나 그에 해당하는 라벨 파일이 접근 불가
    - 오디오 파일은 접근 자체가 불가했고 라벨 파일을 로그로 저장하여 분석
- 평가 지표로는 CER(Character Error Rate)를 활용
    - 실제 라벨 대비 잘못 대체, 삭제, 추가된 음절 수를 계산
    - 낮은 지표값을 획득할 수록 강건한 모델

## 3. 진행 과정에서의 챌린지

- 팀원 모두 처음 다뤄보는 오디오 데이터 모델링
    - 관련 논문 / 레퍼런스 스터디를 진행하였고 각자 역할 분담
- 데이터 접근 불가 문제
    - 낯선 오디오 데이터 + 데이터 분석 불가한 상태이기에 모델과 Augmentation에 집중하기로 결정
    - 특히 차용한 모델인 Deep Speech 2 모델의 논문에서 활용한 Data Augmentation과 트릭들을 적용

## 4. 대회 수행 내용

### Data Transfer & Augmentation

- 대회에서 다루는 데이터는 **비주기적인 음성 데이터** 라는 점을 주목
- raw 오디오에서는 고차원의 Sampling Rate 때문에 유의미한 Feature를 뽑아내기 쉽지 않음
- 이를 시간 도메인의 신호를 주파수 도메인으로 변환을 통해 쉽게 해석 가능해짐 (**Spectrogram**)
- Spectrogram에 인간이 저주파수에 민감한 특성을 적용한 **Mel-Spectrogram** 으로 변환
- 노이즈에 강건한 모델을 만들기 위해 주파수를 이미지처럼 다루어 랜덤 마스킹을 하는 **SpecAugmentation** 적용
    - 원본 데이터 : Augmented 데이터  = 2 : 1 (오버피팅 문제를 언더피팅 문제로 변환)
    - 시간 축과 주파수 축에 랜덤 마스킹 적용 (아래 이미지 참조)

![image](https://github.com/jungsiroo/jungsiroo.github.io/assets/54366260/34a3bac5-e8c1-41d9-880d-b3364f5075f4)

### 5. 회고

> 최종 결과 CER : 0.269 (본선 9등)
> 

**성장한 점**

- Task Dependency data 분석 및 구현
- 처음 다루는 분야이더라도 레퍼런스 분석과 이해 능력
    - 다양한 NLP Task 경험
- Pytorch 를 이용한 Speech 데이터 핸들링

**아쉬운 점**

- 학기 병행으로 인한 Deep Dive 불가능
    - 모델링 참여의 아쉬움
- 하이퍼 파라미터 튜닝에 너무 많은 집중
    - 리더보드에 쫓기듯 하며 점수 올리기에 연연
    - 조금 더 여유롭게 본질을 이해하고 했다하면 하는 점
