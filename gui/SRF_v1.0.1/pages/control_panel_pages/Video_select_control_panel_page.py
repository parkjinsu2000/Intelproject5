import sys
import os
import shutil
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel


class VideoSelectControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()
      
        # 메인 레이아웃 (수직)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 동영상 기능을 가진 패널을 포함시킵니다.
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.main_layout.addWidget(self.video_panel)
        
        # 단일 이미지 경로
        thumbnail_image_path = "/home/ubuntu/Qt/SRF_v1.0.0/resource/button1.png"
        name_button_image_path = "/home/ubuntu/Qt/SRF_v1.0.0/resource/button.png"
        
        # 버튼들을 비디오 위젯 위에 직접 배치
        self.buttons = []
        rows = 2
        cols = 3
        
        for i in range(rows * cols):
            # 썸네일 버튼 스타일 (CSS로 이미지 적용)
            thumbnail_style = f"""
                QPushButton {{
                    background-image: url({thumbnail_image_path});
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 100% 100%; /* 버튼 크기에 맞게 이미지 늘리기 */
                    border: none;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: rgba(169, 208, 230, 0.9);
                }}
            """
            
            # 썸네일 버튼
            thumb_w, thumb_h = 500, 400
            thumbnail_btn = QPushButton('', self.video_panel.video_widget)
            thumbnail_btn.setFixedSize(thumb_w, thumb_h)
            thumbnail_btn.setStyleSheet(thumbnail_style)
            self.buttons.append(thumbnail_btn)
            
            # 이름 버튼 스타일 (CSS로 이미지 적용)
            name_style = f"""
                QPushButton {{
                    background-image: url({name_button_image_path});
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 100% 100%; /* 버튼 크기에 맞게 이미지 늘리기 */
                    border: none;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: rgba(70, 130, 180, 0.5);
                }}
            """

        # 영상 패널
        self.video_panel = BackgroundVideoPlayControlPannel(video_path, screen_index)
        self.video_panel.setStyleSheet("background-color: black;")

        # 오버레이 위젯 (버튼들 담을 투명 레이어)
        self.overlay = QWidget(self.video_panel.video_widget)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.overlay.setStyleSheet("background-color: transparent;")

        # 오버레이 레이아웃
        self.overlay_layout = QVBoxLayout(self.overlay)
        self.overlay_layout.setContentsMargins(20, 20, 20, 20)
        self.overlay_layout.setSpacing(20)
        self.overlay_layout.setAlignment(Qt.AlignTop)

        # 뒤로가기 버튼
        self.back_button = QPushButton("← 메인")
        self.back_button.setFixedSize(120, 40)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.6);
                color: white;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.9);
            }
        """)
        self.back_button.clicked.connect(self.on_back_clicked)
        self.overlay_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)

        # 썸네일 그리드
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setAlignment(Qt.AlignCenter)

        self.overlay_layout.addWidget(self.grid_widget, alignment=Qt.AlignCenter)

        # 시작 버튼
        self.start_button = QPushButton("Start")
        self.start_button.setFixedSize(150, 50)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: royalblue;
                color: white;
                font-size: 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: dodgerblue;
            }
        """)
        self.overlay_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # 전체 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_panel)

        # 동영상 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        videos_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'resource', 'videos'))

        self.video_files = []
        if os.path.exists(videos_dir):
            all_videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
            self.video_files = all_videos[:6]
        else:
            print("Error: 'videos' directory not found.")

        # 썸네일 버튼 생성
        for idx, video_file in enumerate(self.video_files):
            video_full_path = os.path.join(videos_dir, video_file)
            thumbnail_file = os.path.splitext(video_file)[0] + '.png'
            thumbnail_path = os.path.join(videos_dir, thumbnail_file)

            # 썸네일 없으면 FFmpeg로 생성
            if not os.path.exists(thumbnail_path):
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path:
                    try:
                        subprocess.run([
                            ffmpeg_path, "-y",
                            "-i", video_full_path,
                            "-ss", "00:00:05",
                            "-vframes", "1",
                            "-s", "300x240",
                            thumbnail_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as e:
                        print(f"FFmpeg thumbnail generation failed for {video_file}: {e}")
                else:
                    print("⚠️ FFmpeg not found. Please install FFmpeg.")

            # 썸네일 버튼
            thumb_btn = QPushButton()
            thumb_btn.setFixedSize(300, 240)
            thumb_btn.setCursor(Qt.PointingHandCursor)

            if os.path.exists(thumbnail_path):
                icon = QIcon(thumbnail_path)
                thumb_btn.setIcon(icon)
                thumb_btn.setIconSize(QSize(300, 240))
                thumb_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background-color: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255, 255, 255, 0.1);
                    }
                """)
            else:
                thumb_btn.setText("No Thumbnail")
                thumb_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(93, 153, 198, 200);
                        color: white;
                        border-radius: 5px;
                        font-size: 18px;
                    }
                    QPushButton:hover {
                        background-color: rgba(74, 122, 159, 200);
                    }
                """)

            thumb_btn.clicked.connect(lambda _, path=video_full_path: self.on_item_clicked(path))

            # 이름 버튼
            name_btn = QPushButton(os.path.splitext(video_file)[0])
            name_btn.setFixedSize(300, 60)
            name_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(70, 130, 180, 0.5);
                    color: white;
                    border-radius: 5px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: rgba(70, 130, 180, 0.9);
                }
            """)
            name_btn.clicked.connect(lambda _, path=video_full_path: self.on_item_clicked(path))

            # 버튼 수직 정렬
            cell = QVBoxLayout()
            cell.setSpacing(10)
            cell.addWidget(thumb_btn)
            cell.addWidget(name_btn)

            container = QWidget()
            container.setLayout(cell)

            row = idx // 3
            col = idx % 3
            self.grid_layout.addWidget(container, row, col)

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.video_panel.video_widget.rect())
        super().resizeEvent(event)

    def on_item_clicked(self, video_path):
        print(f"Clicked on video: {video_path}")
        # 동영상 재생 로직 추가 가능

    def on_back_clicked(self):
        print("뒤로가기 버튼 클릭됨")
        self.close()

    def set_screen(self, screen_index):
        screens = QApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("No monitor found at the specified index. Using the default monitor.")

if __name__ == '__main__':
    app = QApplication(sys.argv)

<<<<<<< HEAD:gui/SRF_v1.0.1/pages/control_pannel_pages/Video_select_control_pannel_page.py
    video_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_small.mp4"
    window = VideoSelectControlPannelPage(video_path=video_file, screen_index=1)
    window.set_screen(1)
    window.show()
=======
    window = VideoSelectControlPanelPage(video_path=video_file, screen_index=0)
    window.showMaximized()
>>>>>>> ad490cb (메인 컨트롤러를 만들어서 두 개의 창을 띄우도록 하는 main.py를 만들었음):gui/SRF_v1.0.1/pages/control_panel_pages/Video_select_control_panel_page.py

    sys.exit(app.exec_())
