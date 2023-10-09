import sys
import cv2
import numpy as np
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import *
import os
from PyQt5.QtGui import QPixmap, QImage
import time
from playsound import playsound
class MySignal(QObject):
    signal1 = pyqtSignal()

    def run(self):
        self.signal1.emit()
#시그널이란 위젯의 상태가 바뀌었을 때, 특정 행동을 하게 하는 코드 -> Qt Designer 프레임 워크에서 설정한 오브젝트들의 이벤트 시그널을 받아오는 코드
class MySignal2(QObject):
    signal2 = pyqtSignal()

    def run(self):
        self.signal2.emit()

class MySignal3(QObject):
    signal3 = pyqtSignal()

    def run(self):
        self.signal3.emit()

class MySignal4(QObject):
    signal4 = pyqtSignal()

    def run(self):
        self.signal4.emit()

class MySignal5(QObject):
    signal5 = pyqtSignal()

    def run(self):
        self.signal5.emit()

class MySignal6(QObject):
    signal6 = pyqtSignal()

    def run(self):
        self.signal6.emit()

class MySignal7(QObject):
    signal7 = pyqtSignal()

    def run(self):
        self.signal7.emit()

class MySignal8(QObject):
    signal8 = pyqtSignal()

    def run(self):
        self.signal8.emit()

class MySignal9(QObject):
    signal9 = pyqtSignal()

    def run(self):
        self.signal9.emit()

class MySignal10(QObject):
    signal10 = pyqtSignal()

    def run(self):
        self.signal10.emit()

class warnSignal(QObject):
    warnsignal = pyqtSignal()

    def run(self):
        self.warnsignal.emit()

#영상을 불러오기 위한 클래스를 정의한 코드입니다.
class thread_camera(QThread):
    def __init__(self, fname, ws, cap, moving, mysignal4, mysignal5, mysignal6, mysignal7, mysignal8, mysignal9, mysignal10, label_c, label_n):
        super(thread_camera, self).__init__()
        self.fname = fname
        self.ws = ws
        self.moving = moving
        self.mysignal4 = mysignal4
        self.mysignal5 = mysignal5
        self.mysignal6 = mysignal6
        self.mysignal7 = mysignal7
        self.mysignal8 = mysignal8
        self.mysignal9 = mysignal9
        self.mysignal10 = mysignal10
        self.label_c = label_c
        self.label_n = label_n

        self.cap = cap
        self.mysignal4.signal4.connect(self.signal4_emitted)
        self.mysignal5.signal5.connect(self.signal5_emitted)
        self.mysignal6.signal6.connect(self.signal6_emitted)
        self.mysignal7.signal7.connect(self.signal7_emitted)
        self.mysignal8.signal8.connect(self.signal8_emitted)
        self.mysignal9.signal9.connect(self.signal9_emitted)
        self.mysignal10.signal10.connect(self.signal10_emitted)
        self.yolo_display = 0
        self.blur_display = 0
        self.cnt = 0
        self.cnt_2 = 0
        self.choice = ""
        self.flag = False

    @pyqtSlot()
    def signal4_emitted(self):
        if self.cnt % 2 == 0:
            self.yolo_display = 1
        else:
            self.yolo_display = 0
        self.cnt = self.cnt + 1
        print("py signal yolo")

    @pyqtSlot()
    def signal5_emitted(self):
        if self.cnt_2 % 2 == 0:
            self.blur_display = 1
        else:
            self.blur_display = 0
        self.cnt_2 = self.cnt_2 + 1
        print("py signal mosaic")

    @pyqtSlot()
    def signal6_emitted(self):
        self.terminate()
        self.wait(1000)

    @pyqtSlot()
    def signal7_emitted(self):
        self.choice = "person"

    @pyqtSlot()
    def signal8_emitted(self):
        self.choice = "car"

    @pyqtSlot()
    def signal9_emitted(self):
        self.choice = "bus"

    @pyqtSlot()
    def signal10_emitted(self):
        self.choice = "all"

    def img_filter(self, frame):
        #필터 마스크 행렬 설정(고주파 필터 마스크)
        sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #  OpenCV에서 제공하는 양방향 필터 함수 cv2.bilateralFilter(입력 영상,  필터링에 사용될 이웃 픽셀의 거리(지름) 음수(-1)를 입력하면 sigmaSpace 값에 의해 자동 결정, 색 공간에서 필터의 표준 편차, 좌표 공간에서 필터의 표준 편차)
        b_filter = cv2.bilateralFilter(frame, -1, 10, 5)
        # OpenCV에서 제공하는 필터링 함수 filter2D(입력 영상, 출력 영상 데이터 타입, 필터 마스크 행렬)
        dst = cv2.filter2D(b_filter, -1, sharpening_mask)
        return dst

    def img_blur(self, frame, x, y, h, w):
        # 박스 처리된 위치의 이미지 범위 깊은 복사
        copy_img = frame[y:y + h, x:x + w].copy()
        # 복사된 이미지 범위만큼 평균 블러링 20X20 범위내 이웃 픽셀의 평균을 이미지의 픽셀값으로 치환
        frame[y:y + h, x:x + w] = cv2.blur(copy_img, (20, 20))
        return frame

    def run(self):
        # Yolo 로드 Weights : 훈련된 model, Cfg file : 구성파일. 알고리즘에 관한 모든 설정
        self.YOLO_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        # YOLO NETWORK 재구성
        classes = []
        #알고리즘이 감지할 수 있는 객체의 이름
        with open("yolov3.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = self.YOLO_net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.YOLO_net.getUnconnectedOutLayers()]
        while True:
            self.pCount = 0
            try:
                ret, frame = self.cap.read()
                frame = self.img_filter(frame)

                if self.yolo_display == 1:
                    #Blob은 이미지에서 특징을 잡아내고 크기를 조정 YOLO가 허용하는 세가지 크기 중 하나인 416, 416
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                    self.YOLO_net.setInput(blob)
                    #outs: 탐지 결과
                    outs = self.YOLO_net.forward(output_layers)

                    # 정보를 화면에 표시
                    class_ids = []
                    confidences = []
                    boxes = []

                    for out in outs:

                        for detection in out:

                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]

                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * w)
                                center_y = int(detection[1] * h)
                                dw = int(detection[2] * w)
                                dh = int(detection[3] * h)
                                # Rectangle coordinate 좌표
                                x = int(center_x - dw / 2)
                                y = int(center_y - dh / 2)
                                boxes.append([x, y, dw, dh])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    #같은 물체에 대한 박스가 많은것을 제거
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                    for i in range(len(boxes)):
                        if i in indexes:
                            #Box : 감지된 개체를 둘러싼 사각형의 좌표
                            x, y, w, h = boxes[i]
                            #Label : 감지된 물체의 이름
                            label = str(classes[class_ids[i]])
                            #Confidence: 0에서 1까지의 탐지에 대한 신뢰도
                            score = confidences[i]

                            if label == "person" and self.choice == "person":
                            # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif label == "car" and self.choice == "car":
                            # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif label == "bus" and self.choice == "bus":
                            # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif self.choice == "all":
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            if self.blur_display == 1:
                                frame = self.img_blur(frame, x, y, h, w)
            except:
                continue


            self.label_c.setText("{}".format(self.pCount))
            self.label_c.show()
            self.label_n.setText("{}".format(self.choice))
            self.label_n.show()

            h, w, c = frame.shape
            qImg = QImage(frame.data, w, h, w * c, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qImg)

            self.moving.setPixmap(pixmap)
            self.moving.show()
            time.sleep(0.04)
            if self.pCount <= 1:
                self.flag = True
            #객체 카운팅 변수 pCount 13이상이면 사람만 알림 경고창이 뜨도록 설정
            if self.pCount > 12 and self.flag == True and self.choice == "person":
                self.ws.run()
                self.flag = False

#경고 메시지 ui 클래스
class warn(QDialog):
    def __init__(self, parent):
        super(warn, self).__init__(parent)
        uic.loadUi('warning.ui', self)
        self.show()

class thread_video(QThread):
    def __init__(self, fname, ws, cap, moving, mysignal, mysignal2, mysignal3, mysignal7, mysignal8, mysignal9, mysignal10, label_c_2, label_n_2):
        super(thread_video, self).__init__()
        self.fname = fname
        self.ws = ws
        self.moving = moving
        self.mysignal = mysignal
        self.mysignal2 = mysignal2
        self.mysignal3 = mysignal3
        self.mysignal7 = mysignal7
        self.mysignal8 = mysignal8
        self.mysignal9 = mysignal9
        self.mysignal10 = mysignal10

        self.cap = cap
        self.mysignal.signal1.connect(self.signal1_emitted)
        self.mysignal2.signal2.connect(self.signal2_emitted)
        self.mysignal3.signal3.connect(self.signal3_emitted)
        self.mysignal7.signal7.connect(self.signal7_emitted)
        self.mysignal8.signal8.connect(self.signal8_emitted)
        self.mysignal9.signal9.connect(self.signal9_emitted)
        self.mysignal10.signal10.connect(self.signal10_emitted)
        self.yolo_display = 0
        self.blur_display = 0
        self.cnt = 0
        self.cnt_2 = 0
        self.choice = " "
        self.label_c_2 = label_c_2
        self.label_n_2 = label_n_2
        self.Flag = False


    @pyqtSlot()
    def signal1_emitted(self):
        if self.cnt % 2 == 0:
            self.yolo_display = 1
        else:
            self.yolo_display = 0
        self.cnt = self.cnt + 1
        print("py signal yolo")


    @pyqtSlot()
    def signal2_emitted(self):
        if self.cnt_2 % 2 == 0:
            self.blur_display = 1
        else:
            self.blur_display = 0
        self.cnt_2 = self.cnt_2 + 1
        print("py signal mosaic")

    @pyqtSlot()
    def signal7_emitted(self):
        self.choice = "person"

    @pyqtSlot()
    def signal8_emitted(self):
        self.choice = "car"

    @pyqtSlot()
    def signal9_emitted(self):
        self.choice = "bus"

    @pyqtSlot()
    def signal10_emitted(self):
        self.choice = "all"

    @pyqtSlot()
    def signal3_emitted(self):
        #self.wait(3000)
        self.terminate()
        self.wait(1000)

    def img_filter(self, frame):
        sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        b_filter = cv2.bilateralFilter(frame, -1, 10, 5)
        dst = cv2.filter2D(b_filter, -1, sharpening_mask)
        return dst

    def img_blur(self, frame, x, y, h, w):
        copy_img = frame[y:y + h, x:x + w].copy()
        frame[y:y + h, x:x + w] = cv2.blur(copy_img, (20, 20))
        return frame

    def run(self):

        self.YOLO_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        # YOLO NETWORK 재구성
        classes = []
        with open("yolov3.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = self.YOLO_net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.YOLO_net.getUnconnectedOutLayers()]
        while True:
            self.pCount = 0
            try:
                ret, frame = self.cap.read()
                frame = self.img_filter(frame)

                if self.yolo_display == 1:
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                    self.YOLO_net.setInput(blob)
                    outs = self.YOLO_net.forward(output_layers)

                    class_ids = []
                    confidences = []
                    boxes = []

                    for out in outs:

                        for detection in out:

                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]

                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * w)
                                center_y = int(detection[1] * h)
                                dw = int(detection[2] * w)
                                dh = int(detection[3] * h)
                                # Rectangle coordinate
                                x = int(center_x - dw / 2)
                                y = int(center_y - dh / 2)
                                boxes.append([x, y, dw, dh])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            score = confidences[i]

                            if label == "person" and self.choice == "person":
                                # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif label == "car" and self.choice == "car":
                                # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif label == "bus" and self.choice == "bus":
                                # 경계상자와 클래스 정보 이미지에 입력
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            elif self.choice == "all":
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                                self.pCount = self.pCount + 1

                            if self.blur_display == 1:
                                try:
                                    frame = self.img_blur(frame, x, y, h, w)
                                except:
                                    continue
            except:
                continue


            self.label_c_2.setText("{}".format(self.pCount))
            self.label_c_2.show()
            self.label_n_2.setText("{}".format(self.choice))
            self.label_n_2.show()

            h, w, c = frame.shape
            qImg = QImage(frame.data, w, h, w * c, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qImg)

            self.moving.setPixmap(pixmap)
            self.moving.show()

            if self.pCount <= 1:
                self.flag = True

            if self.pCount > 12 and self.flag == True and self.choice == "person":
                self.ws.run()
                self.flag = False

            time.sleep(0.04)

#화면을 띄우는데 사용되는 Class 선언
class team_yolo(QMainWindow):
    def __init__(self):
        super(team_yolo, self).__init__()
        #.ui파일과 연결
        uic.loadUi("C:/Users/alstn_bl71ud5/PycharmProjects/pythonProject1/team_yolo.ui", self)
        self.show()

        self.Pic.triggered.connect(self.picture) #왼쪽 상단 버튼들
        self.Cam.triggered.connect(self.camera)
        self.Person.triggered.connect(self.person)
        self.Car.triggered.connect(self.car)
        self.Bus.triggered.connect(self.bus)

        self.sMenu.setCurrentIndex(0)
        self.sView.setCurrentIndex(0)

        self.mOpen.clicked.connect(self.mopen)
        self.mPlay.clicked.connect(self.mplay)
        self.mYolo.clicked.connect(self.myolo)
        self.mMosaic.clicked.connect(self.mmosaic)
        self.mClose.clicked.connect(self.mclose)

        self.wOpen.clicked.connect(self.wopen)
        self.wPlay.clicked.connect(self.wplay)
        self.wYolo.clicked.connect(self.wyolo)
        self.wMosaic.clicked.connect(self.wmosaic)
        self.wClose.clicked.connect(self.wclose)

        self.mysignal = MySignal()
        self.mysignal2 = MySignal2()
        self.mysignal3 = MySignal3()
        self.mysignal4 = MySignal4()
        self.mysignal5 = MySignal5()
        self.mysignal6 = MySignal6()
        self.mysignal7 = MySignal7()
        self.mysignal8 = MySignal8()
        self.mysignal9 = MySignal9()
        self.mysignal10 = MySignal10()
        self.ws = warnSignal()
        self.ws.warnsignal.connect(self.warns)

    #시그널에 연결된 함수들
    def warns(self):
        warn(self)

    def mopen(self):
        self.sMenu.setCurrentIndex(1)
        self.sView.setCurrentIndex(1)
        self.fname = QFileDialog.getOpenFileName(self)
        print(self.fname[0])
        self.cap = cv2.VideoCapture(self.fname[0])

    def mplay(self):
        self.sMenu.setCurrentIndex(1)
        self.sView.setCurrentIndex(1)
        self.t = thread_video(self.fname[0], self.ws, self.cap, self.moving, self.mysignal, self.mysignal2, self.mysignal3, self.mysignal7, self.mysignal8, self.mysignal9, self.mysignal10, self.label_c_2, self.label_n_2)
        self.t.start()

    def myolo(self):
        self.sMenu.setCurrentIndex(1)
        self.sView.setCurrentIndex(1)
        self.mysignal.run()
        self.mysignal10.run()

    def mmosaic(self):
        self.sMenu.setCurrentIndex(1)
        self.sView.setCurrentIndex(1)
        self.mysignal2.run()

    def mclose(self):
        self.sMenu.setCurrentIndex(0)
        self.sView.setCurrentIndex(0)
        self.mysignal3.run()

    def wopen(self):
        self.sMenu.setCurrentIndex(2)
        self.sView.setCurrentIndex(2)
        self.cap1 = cv2.VideoCapture(0)

    def wplay(self):
        self.sMenu.setCurrentIndex(2)
        self.sView.setCurrentIndex(2)
        self.t1 = thread_camera("hello", self.ws, self.cap1, self.cam, self.mysignal4, self.mysignal5, self.mysignal6, self.mysignal7, self.mysignal8, self.mysignal9, self.mysignal10, self.label_c, self.label_n)
        self.t1.start()

    def wyolo(self):
        self.sMenu.setCurrentIndex(2)
        self.sView.setCurrentIndex(2)
        self.mysignal4.run()
        self.mysignal10.run()

    def wmosaic(self):
        self.sMenu.setCurrentIndex(2)
        self.sView.setCurrentIndex(2)
        self.mysignal5.run()

    def wclose(self):
        self.sMenu.setCurrentIndex(0)
        self.sView.setCurrentIndex(0)
        self.mysignal6.run()

    def picture(self):
        self.sMenu.setCurrentIndex(1)
        self.sView.setCurrentIndex(1)

    def camera(self):
        self.sMenu.setCurrentIndex(2)
        self.sView.setCurrentIndex(2)

    def person(self):
        self.mysignal7.run()

    def car(self):
        self.mysignal8.run()

    def bus(self):
        self.mysignal9.run()

#QApplication : 프로그램을 실행시켜주는 클래스
app = QApplication(sys.argv)
#team_yolo의 인스턴스 생성
window = team_yolo()
#프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
app.exec_()
