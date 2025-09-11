import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# QWidget 클래스를 상속받는 VideoSelectViewPannelPage 클래스 정의
class VideoSelectViewPannelPage(QWidget):
    def __init__(self):
        super().__init__()
        
        # 윈도우 기본 설정
        self.setWindowTitle(' ')
        self.setGeometry(100, 100, 800, 600)
        
        # 레이아웃 생성
        main_layout = QVBoxLayout()
        # 레이아웃의 여백과 위젯 간의 간격을 0으로 설정
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.setLayout(main_layout)
        
        # 제목 이미지를 표시할 라벨
        self.title_image_label = QLabel()
        self.title_image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_image_label)
        
        # QVideoWidget을 사용하여 비디오 재생 영역 추가
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: lightblue;")
        main_layout.addWidget(self.video_widget)
        
        # QMediaPlayer 인스턴스 생성
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        
        # 비디오 재생 (테스트용)
        # 여기에 비디오 파일 경로를 설정하세요.
        # 예: self.player.setMedia(QMediaContent(QUrl.fromLocalFile('videos/video.mp4')))

    def showEvent(self, event):
        # 현재 스크립트 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 이미지 파일의 절대 경로 생성
        image_path = os.path.join(current_dir, '..', '..', 'resource', 'title.png')
        
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            # 이미지 로드 실패 시, 터미널에 에러 메시지 출력
            print(f"Error: Could not load image from '{image_path}'")
        else:
            # 창 크기에 맞게 이미지 비율을 유지하며 확장
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.title_image_label.setPixmap(scaled_pixmap)
        
        super().showEvent(event)

if __name__ == '__main__':
    # QApplication 인스턴스 생성
    app = QApplication(sys.argv)
    
    # VideoSelectViewPannelPage 클래스의 인스턴스 생성
    window = VideoSelectViewPannelPage()
    
    # 윈도우 화면에 표시
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())
