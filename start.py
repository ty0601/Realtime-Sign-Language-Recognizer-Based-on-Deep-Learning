import cv2
import numpy as np
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random
import speech_recognition as sr
import logging

import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qt_material import *
from UI import Ui_MainWindow
from chatWidget import Prompt, ChatBrowser


class MainWindow_controller(QMainWindow):
    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.pushButton_clicked)
        self.ui.pushButton_2.clicked.connect(self.pushButton2_clicked)

    def __init__(self, parent=None):
        # super().__init__()
        super(MainWindow_controller, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.ui.pushButton.setCheckable(True)
        self.ui.pushButton_2.setCheckable(True)
        self.thread_a = QThread()

        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture(0)

        ###
        self.CAM_NUM = 0
        self.timer_camera.timeout.connect(self.show_camera)

        # sign_lang
        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
        self.DATA_PATH = os.path.join('MP_Data')  # create data path/folder
        self.actions = np.array(
            ['at', 'dworry', 'hi', 'hospital', 'school', 'thank', 'train_station', 'where'])  # action create
        self.threshold = np.array([0.76, 0.2, 0.2, 0.6, 0.6, 0.1, 0.8, 0.9999])
        self.words = {'at': '在', 'dworry': '沒關係', 'hi': '你好', 'hospital': '醫院', 'school': '學校', 'thank': '謝謝',
                 'train_station': '火車站', 'where': '哪裡'}

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        # res = [0.7, 0.2, 0.1]

        # self.model.load_weights('checkpoint/best_model11.h5')
        self.model = keras.models.load_model('checkpoint/best_model11.h5', compile=False)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.sequence = []
        self.sentence = []
        self.predictions = []

        self.font_file = "font/NotoSansTC-Bold.otf"
        self.font_size = 28
        self.font = ImageFont.truetype(self.font_file, self.font_size)

        self.text = ""

        # chat
        self.__prompt = Prompt()
        # self.__textEdit = self.__prompt.getTextEdit()
        # self.__textEdit.setPlaceholderText('Write some text...')
        # self.__textEdit.returnPressed.connect(self.__chat)
        self.__browser = ChatBrowser()

        lay = QVBoxLayout()
        lay.addWidget(self.__browser)
        lay.addWidget(self.__prompt)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        # mainWidget = QWidget()
        self.ui.widget.setLayout(lay)
        # self.widget.set
        # self.resize(600, 400)
        # self.__browser.showText('你好學校在哪裡 ', True)
        # self.__browser.showText('直走右轉就到了', False)
        # self.ui.label.setText("謝謝")
        # self.ui.label.setText('直走右轉就到了')



    # def __chat(self):
    #     self.__browser.showText(self.__textEdit.toPlainText(), True)
    #     self.__browser.showText(f'You said "{self.__textEdit.toPlainText()}"', False)
    #     self.__textEdit.clear()

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

    def draw_styled_landmarks(self, image, results):
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, lh, rh])

    def stt(self):
        logging.basicConfig(level=logging.DEBUG)
        while self.ui.pushButton.isChecked():
            r = sr.Recognizer()
            # Mic
            mic = sr.Microphone()
            logging.info('message enter')
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            logging.info('message end and recognize')
            try:
                test = r.recognize_google(audio, language='zh-tw')
                print(test)
                self.text = self.text + test
                self.ui.label.setText(self.text)
                QApplication.processEvents()
            except sr.UnknownValueError:
                print("Can't Understand!!!")
            logging.info('end')
        print("End STT")

    def pushButton_clicked(self):
        if self.ui.pushButton.isChecked():
            self.thread_a.run = self.stt
            self.thread_a.start()
        else:
            self.thread_a.terminate()
            if self.text:
                self.__browser.showText(self.text, False)
                self.text = ""
                self.ui.label.clear()

    def show_camera(self):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            ret, image = self.cap.read()

            # show = cv2.resize(image, (800, 600))
            # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            # showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            # self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))

            image, results = self.mediapipe_detection(image, holistic)
            self.draw_styled_landmarks(image, results)
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            if len(self.sequence) >= 30:
                self.sequence = self.sequence[-30:]
                res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                print(f"{self.actions[np.argmax(res)]}, {res[np.argmax(res)]}")
                # predictions.append(np.argmax(res))
                # predictions = predictions[-10:]
                # if np.argmax(np.bincount(predictions)) == np.argmax(res):
                # if len(sentence) != 0 and words[actions[np.argmax(res)]] == sentence[-1]:
                #     sequence = []
                if res[np.argmax(res)] > self.threshold[np.argmax(res)]:
                    self.sequence = []
                    if len(self.sentence) > 0:
                        if self.words[self.actions[np.argmax(res)]] != self.sentence[-1]:
                            self.sentence.append(self.words[self.actions[np.argmax(res)]])
                            self.text = self.text + self.words[self.actions[np.argmax(res)]]
                    elif len(self.sentence) == 0:
                        self.sentence.append(self.words[self.actions[np.argmax(res)]])
                        self.text = self.text + self.words[self.actions[np.argmax(res)]]
                else:
                    self.sequence = self.sequence[-29:]
                if len(self.sentence) > 8:
                    self.sentence = self.sentence[-8:]

            cv2.rectangle(image, (0, 0), (640, 50), (216, 152, 83), -1)
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 4), ' '.join(self.sentence), font=self.font, fill=(255,255,255))

            self.ui.label.setText(self.text)

            cv2.namedWindow('SIGN LANGUAGE RECOGNIZE', 0)
            cv2.resizeWindow('SIGN LANGUAGE RECOGNIZE',1200, 900)
            cv2.imshow('SIGN LANGUAGE RECOGNIZE', np.array(pil_img))


    def pushButton2_clicked(self):
        if self.timer_camera.isActive() == False:
            self.openCamera()
        else:
            self.closeCamera()
            if self.text:
                self.__browser.showText(self.text, True)
                self.sentence = []
            self.text = ""
            self.ui.label.clear()

    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'Warning',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
        else:
            self.timer_camera.start(30)

    def closeCamera(self):
        self.timer_camera.stop()
        self.cap.release()
        # self.ui.label_2.clear()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    MainWindow = MainWindow_controller()

    # apply_stylesheet(app, theme='light_blue.xml')

    MainWindow.show()
    sys.exit(app.exec_())
