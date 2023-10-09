# 객체 밀집 알람 프로그램

호서대 컴퓨터공학과 컴퓨터 비전 강의 팀 프로젝트입니다.

참여인원: 김민수_팀장(발표 및 개발), 이수빈(기획 및 개발), 우원봉(보고서 작성 및 개발)

프로젝트 진행 기간은 2022.10월 중순 ~ 12월 초 입니다.

## 🎬 프로젝트 개요

객체 검출 딥러닝의 한 방법인 yolo를 사용하여 한 프레임 안에 있는 객체의 수를 카운트하며,
검출된 객체가 한 구역에 많이 밀집하여 일정 수준을 넘어서면 경고를 알리는 알람 서비스를 제공.
객체는 사람, 자동차 등으로 사용자가 설정할 수 있도록 구현

## ⚙️ 개발환경

OS : Window 10, 11

RAM : 16 GB

UI : Qt Designer, pyQt(ver.5.15.7)

Python : ver.3.10.6, Pycham,IDLE

Yolo : ver.3

기타 : OpenCV-python(ver.4.6.0), Numpy(ver.1.23.5), pyinstaller(ver.5.6.2)

## 🧱 인터페이스 구성 구조도

<img width="549" alt="스크린샷 2023-10-09 오후 10 43 49" src="https://github.com/Pushedsu/WithPet/assets/109027302/acfb5e7d-d5b3-4eaf-8a0c-5c2608c60b4d">

## 기능 구조도

<img width="557" alt="스크린샷 2023-10-09 오후 10 47 05" src="https://github.com/Pushedsu/WithPet/assets/109027302/a332c43b-5ce5-423c-a523-186402323f59">

## ⛏️ 주요 기능

### 전처리

전처리 과정을 거쳐 보다 더 경계를 잘 찾아낼 수 있도록 양방향 필터를 통해 잡음을 제거한 후 하이패스 필터를 통해 경계값을 증폭

### 객체 검출

weight 훈련 모델 파일과 cfg 알고리즘에 관한 설정이 담긴 파일, 객체 이름에 대한 names 파일의 데이터를 토대로 객체를 검출한다.

### 모자이크

OpenCV의 cv2.blur를 이용

### 경고문

필터 사람(Person)을 사용했을 때 인식된 객체가 13명 이상일 경우 경고문을 출력한다.

## 📺 실행화면

<img width="572" alt="스크린샷 2023-10-09 오후 11 07 43" src="https://github.com/Pushedsu/WithPet/assets/109027302/f8b43474-6414-4b73-8b58-85758bb98c28">

<img width="571" alt="스크린샷 2023-10-09 오후 11 08 57" src="https://github.com/Pushedsu/WithPet/assets/109027302/888aaef0-92cb-4741-8720-40c731c24d95">
