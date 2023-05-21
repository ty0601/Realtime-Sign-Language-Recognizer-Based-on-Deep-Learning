from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui


class ChatBrowser(QScrollArea):
    def __init__(self):
        super().__init__()
        self.__initUi()

    def __initUi(self):
        lay = QVBoxLayout()
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(10)
        lay.setContentsMargins(15, 15, 15, 15)
        widget = QWidget()
        widget.setLayout(lay)
        self.setWidget(widget)
        self.setWidgetResizable(True)

    # def showText(self, text, user_f):
    #     chatLbl = QLabel(text)
    #     # chatLbl.setFrameShape(QFrame.Box)
    #     # chatLbl.setFrameShadow(QFrame.Raised)
    #     # chatLbl.setLineWidth(20)
    #     # chatLbl.setStyleSheet('background-color: #3598DB;border: #EAEAEF; padding: 1em ')
    #     font = QtGui.QFont()
    #     font.setFamily("微軟正黑體")
    #     font.setPointSize(16)
    #     font.setBold(True)
    #     font.setWeight(75)
    #     chatLbl.setFont(font)
    #     chatLbl.setWordWrap(True)
    #     chatLbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    #     chatLbl.adjustSize()
    #     chatLbl.setMaximumWidth(chatLbl.width())
    #
    #     if user_f:
    #         chatLbl.setStyleSheet('QLabel { background-color: #3598DB; padding: 1em }')
    #         chatLbl.setAlignment(Qt.AlignRight)
    #         self.widget().layout().addWidget(chatLbl, alignment=Qt.AlignRight)
    #     else:
    #         chatLbl.setStyleSheet('QLabel { background-color: #FFFFFF;padding: 1em }')
    #         chatLbl.setAlignment(Qt.AlignLeft)
    #         self.widget().layout().addWidget(chatLbl, alignment=Qt.AlignLeft)
    #     # self.widget().layout().addStretch()
    #     # self.widget().layout().addWidget(chatLbl, Qt.AlignRight)
    #     # text.size().weight()
    #     # chatLbl.setMaximumHeight(int(text.size().weight() + text.documentMargin()))
    def showText(self, text, user_f):
        chatWidget = QWidget()
        chatLayout = QHBoxLayout(chatWidget)
        chatLayout.setSpacing(15)

        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)

        if user_f:
            chatLbl = QLabel(text)
            chatLbl.setStyleSheet('background-color: #3598DB; padding: 1em; border-radius: 20px;color: white;')
            chatLbl.setAlignment(Qt.AlignLeft)
            chatLbl.setFont(font)
            chatLbl.setWordWrap(True)

            iconLabel = QLabel()
            iconPixmap = QPixmap("icon/user1.png").scaled(60, 60)
            iconLabel.setPixmap(iconPixmap)

            chatLayout.addWidget(chatLbl)
            chatLayout.addWidget(iconLabel)
            chatLayout.setAlignment(iconLabel, Qt.AlignRight)
        else:
            chatLbl = QLabel(text)
            chatLbl.setStyleSheet('background-color: #FFFFFF; padding: 1em; border-radius: 20px;')
            chatLbl.setAlignment(Qt.AlignLeft)
            chatLbl.setFont(font)
            chatLbl.setWordWrap(True)

            iconLabel = QLabel()
            iconPixmap = QPixmap("icon/user2.png").scaled(60, 60)
            iconLabel.setPixmap(iconPixmap)

            chatLayout.addWidget(iconLabel)
            chatLayout.addWidget(chatLbl)
            chatLayout.setAlignment(iconLabel, Qt.AlignLeft)

        fontMetrics = QtGui.QFontMetrics(font)
        textWidth = fontMetrics.width(text)

        chatWidget.setMaximumWidth(max(textWidth + 40, 500))
        self.widget().layout().addWidget(chatWidget, alignment=Qt.AlignRight if user_f else Qt.AlignLeft)

    def event(self, e):
        if e.type() == 43:
            self.verticalScrollBar().setSliderPosition(self.verticalScrollBar().maximum())
        return super().event(e)


class TextEditPrompt(QLabel):
    returnPressed = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initUi()

    def __initUi(self):
        self.setText("123")
        # self.setStyleSheet('QTextEdit { border: 1px solid #AAA; } ')

    # def keyPressEvent(self, e):
    #     if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
    #         if e.modifiers() == Qt.ShiftModifier:
    #             return super().keyPressEvent(e)
    #         else:
    #             self.returnPressed.emit()
    #     else:
    #         return super().keyPressEvent(e)


class Prompt(QWidget):
    def __init__(self):
        super().__init__()
        self.__initUi()

    def __initUi(self):
        # self.__textEdit = TextEditPrompt()
        # self.__textEdit.textChanged.connect(self.updateHeight)
        lay = QHBoxLayout()
        # lay.addWidget(self.__textEdit)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lay)
        # self.updateHeight()

    # def updateHeight(self):
    #     document = self.__textEdit.document()
    #     height = document.size().height()
    #     self.setMaximumHeight(int(height+document.documentMargin()))

    def getTextEdit(self):
        return self.__textEdit
