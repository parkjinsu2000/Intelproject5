import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QStackedWidget, QSplitter
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QTimer, Qt, QUrl, QFileInfo, QSize, QEvent, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont
import sys

# QVideoWidget 상속 → sizeHint 무시 및 최소 크기 설정
class MyVideoWidget(QVideoWidget):
    """QVideoWidget을 상속받아 sizeHint를 오버라이드하여 레이아웃 내에서 유연하게 크기 조절"""
    def sizeHint(self):
        # 최소 크기를 반환하여 위젯이 0으로 수축되는 것을 방지
        return QSize(1, 1)

class PoseScoreApp(QWidget):
    """
    포즈 교정 앱의 메인 위젯 클래스:
    왼쪽(참고 영상)과 오른쪽(웹캠) 화면을 QSplitter로 반반 분할하여 관리
    """
    def __init__(self, args, model=None, use_half=False):
        super().__init__()
        self.args = args
        self.model = model
        self.use_half = use_half
        self.setMinimumSize(400, 300) # 윈도우 최소 크기 설정
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- 1. 왼쪽 화면 (미리보기/재생 전환) 설정 ---
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setScaledContents(True)
        self.preview_label.setMinimumSize(1, 1)

        self.video_widget = MyVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setMinimumSize(1, 1)

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.player.setVolume(100)

        self.video_stack = QStackedWidget()
        self.video_stack.setContentsMargins(0, 0, 0, 0)
        self.video_stack.addWidget(self.preview_label)
        self.video_stack.addWidget(self.video_widget)
        self.video_stack.setCurrentWidget(self.preview_label)

        left_container = QWidget()
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.video_stack)

        # --- 2. 오른쪽 화면 (웹캠) 설정 ---
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cam_label.setScaledContents(True)
        self.cam_label.setMinimumSize(1, 1)

        # 카운트다운 오버레이
        self.overlay_label = QLabel("", self.cam_label)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: red; font-weight: bold;")
        self.overlay_label.setFont(QFont("Arial", 96))
        self.overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.cam_label)

        # --- 3. Splitter 및 전체 레이아웃 설정 ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        # 초기 비율 설정 (1:1)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.splitter)

        # --- 4. 초기 동작 및 타이머 설정 ---
        self.show_preview_frame(self.args.ref)

        self.cap = cv2.VideoCapture(self.args.cam)
        if not self.cap.isOpened():
            print(f"Error: 웹캠 ({self.args.cam})을 열 수 없습니다. 다른 프로그램이 사용 중이거나 인덱스가 잘못되었을 수 있습니다.")
            self.cam_label.setText("웹캠을 찾을 수 없습니다.")
            self.cam_label.setStyleSheet("font-size: 20px; color: red;")
            self.frame_timer = None
        else:
            self.frame_timer = QTimer(self)
            self.frame_timer.timeout.connect(self.update_frame)
            self.frame_timer.start(30)

        self.count = 4
        self.count_timer = QTimer(self)
        self.count_timer.timeout.connect(self.update_countdown)
        self.count_timer.start(1000)

        # 초기 반반 세팅
        QTimer.singleShot(0, self.equalize_splitter)

    # --- 비율 및 이벤트 처리 메서드 ---

    def equalize_splitter(self):
        """Splitter의 위젯 크기를 정확히 1:1로 설정"""
        w = max(2, self.splitter.width())
        # QSplitter의 sizes를 직접 설정하여 1:1 비율 유지
        self.splitter.setSizes([w // 2, w - (w // 2)])

    def show_preview_frame(self, video_path):
        """참고 영상의 첫 프레임을 미리보기에 표시"""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(),
                             Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
            )

    def update_frame(self):
        """웹캠 프레임을 읽어와 cam_label에 표시하고 오버레이 위치를 업데이트"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.cam_label.setPixmap(
                    pixmap.scaled(self.cam_label.size(),
                                 Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
                )
                self.update_overlay_size_and_position() # 프레임이 업데이트될 때마다 오버레이 위치 갱신

    def update_countdown(self):
        """카운트다운 로직 실행 및 영상 재생 시작"""
        self.count -= 1
        if self.count > 0:
            self.overlay_label.setText(str(self.count))
        elif self.count == 0:
            self.overlay_label.setText("START")
        else:
            self.overlay_label.hide()
            self.count_timer.stop()
            # 비디오 재생 위젯으로 전환
            self.video_stack.setCurrentWidget(self.video_widget)
            # 전환 후 레이아웃 비율 재조정 (안정성 확보)
            QTimer.singleShot(0, self.equalize_splitter)
            self.play_video()

    def play_video(self):
        """참고 영상 재생"""
        abs_path = QFileInfo(self.args.ref).absoluteFilePath()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self.player.play()

    # 전체 화면 토글 기능은 제거되었습니다.
    def toggle_fullscreen(self):
        pass

    def update_overlay_size_and_position(self):
        """
        오버레이 라벨의 크기와 위치를 실제 영상 영역에 맞춰 조정하고 글자 크기 동적 조절
        """
        # pixmap이 아직 로드되지 않았다면 함수 종료
        if not self.cam_label.pixmap():
            return

        # 웹캠 화면 위젯의 크기
        label_size = self.cam_label.rect().size()
        # 현재 표시되는 pixmap의 원본 크기
        pixmap_size = self.cam_label.pixmap().size()

        # pixmap이 label에 맞춰 비율을 유지하며 scaled될 때의 실제 크기 계산
        if pixmap_size.width() / pixmap_size.height() > label_size.width() / label_size.height():
            # 와이드한 영상: 상하 여백 발생 (Letterbox)
            scaled_width = label_size.width()
            scaled_height = int(scaled_width * pixmap_size.height() / pixmap_size.width())
            x_offset = 0
            y_offset = (label_size.height() - scaled_height) / 2
        else:
            # 세로가 긴 영상: 좌우 여백 발생 (Pillarbox)
            scaled_height = label_size.height()
            scaled_width = int(scaled_height * pixmap_size.width() / pixmap_size.height())
            x_offset = (label_size.width() - scaled_width) / 2
            y_offset = 0

        # 오버레이 라벨의 위치와 크기를 실제 영상 영역에 맞게 설정
        self.overlay_label.setGeometry(
            QRect(
                int(x_offset),
                int(y_offset),
                int(scaled_width),
                int(scaled_height)
            )
        )

        # 웹캠 화면 높이에 비례하여 폰트 크기 계산 (이제는 실제 영상 높이를 사용)
        font_size = max(12, int(scaled_height / 8))
        font = QFont("Arial", font_size)
        self.overlay_label.setFont(font)


    def changeEvent(self, event):
        """윈도우 상태 변경 감지 (최대화/복원)"""
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            # 상태 변경 후 비율 재조정
            QTimer.singleShot(0, self.equalize_splitter)

    def resizeEvent(self, event):
        """위젯 리사이즈 시 비율 재조정"""
        # 리사이즈 후 비율 재조정
        QTimer.singleShot(0, self.equalize_splitter)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """위젯 종료 시 웹캠 및 타이머 해제"""
        if self.frame_timer:
            self.frame_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)
