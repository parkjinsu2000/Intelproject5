import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

class AvatarSelectViewPanelPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 윈도우 기본 설정
        self.setWindowTitle('4분할 화면 레이아웃')
        self.setGeometry(100, 100, 1000, 700) # 창 크기 설정

        # 메인 수평 레이아웃 (전체 화면을 가로로 분할)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10) # 패널 사이 가로 간격
        main_layout.setContentsMargins(10, 10, 10, 10) # 창 가장자리 여백

        # 각 패널 위젯 생성 및 레이아웃 설정
        panel_texts = ["아바타 1\n예시", "아바타 2\n예시", "아바타 3\n예시", "아바타 4\n예시"]

        for text in panel_texts:
            panel = QWidget()
            panel.setStyleSheet("background-color: #6daee3;")
            
            panel_layout = QVBoxLayout(panel)
            panel_layout.setAlignment(Qt.AlignCenter)
            
            label = QLabel(text)
            self.set_label_style(label)
            
            panel_layout.addWidget(label)
            
            main_layout.addWidget(panel)

        self.setLayout(main_layout)

    def set_label_style(self, label):
        """QLabel의 폰트와 색상 스타일을 설정하는 헬퍼 함수"""
        font = QFont('나눔고딕', 20)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255)) # 폰트 색상을 흰색으로
        label.setPalette(palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AvatarSelectViewPanelPage()
    ex.show()
    sys.exit(app.exec_())