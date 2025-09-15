import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap

class ScoreViewPanelPage(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        
        # 메인 레이아웃 설정
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 이미지를 표시할 라벨 생성
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        
        # 이미지 경로 설정
        self.pixmap = QPixmap(image_path)
        
        # QPixmap이 유효한지 확인
        if self.pixmap.isNull():
            print(f"Error: Unable to load image from {image_path}")
            # 로드 실패 시 대체 텍스트 표시
            self.image_label.setText("Image Not Found")
        else:
            self.image_label.setPixmap(self.pixmap)
        
        self.main_layout.addWidget(self.image_label)
        
        # 창 크기 변경 시 이미지도 함께 리사이즈되도록 이벤트 연결
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        """
        Resize the pixmap to fit the new window size while maintaining aspect ratio.
        """
        if not self.pixmap.isNull():
            self.image_label.setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 현재 스크립트 파일의 디렉토리 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 상대 경로를 절대 경로로 변환
    image_file = os.path.join(current_dir, "../../resource/control_title.png")

    window = ScoreViewPanelPage(image_path=image_file)
    window.setWindowTitle("Score View Panel")
    window.show() # showMaximized() 대신 show()를 사용합니다.
    
    sys.exit(app.exec_())
