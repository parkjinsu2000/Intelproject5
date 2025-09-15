import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

class GamePlayViewPanelPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 윈도우 기본 설정
        self.setWindowTitle('이중 화면 레이아웃')
        self.setGeometry(100, 100, 800, 600)

        # 메인 수평 레이아웃 생성
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10) # 패널 사이 간격 설정
        main_layout.setContentsMargins(10, 10, 10, 10) # 창 가장자리 여백

        # 왼쪽 패널 위젯
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #6daee3;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignCenter)
        left_label = QLabel("선택한 Ref 영상\n플레이")
        self.set_label_style(left_label)
        left_layout.addWidget(left_label)

        # 오른쪽 패널 위젯
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #6daee3;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignCenter)
        right_label = QLabel("내 영상\n웹캠")
        self.set_label_style(right_label)
        right_layout.addWidget(right_label)

        # 메인 레이아웃에 두 패널 추가
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.setLayout(main_layout)

    def set_label_style(self, label):
        """QLabel의 폰트와 색상 스타일을 설정하는 헬퍼 함수"""
        font = QFont('나눔고딕', 20)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        label.setPalette(palette)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GamePlayViewPanelPage()
    ex.show()
    sys.exit(app.exec_())