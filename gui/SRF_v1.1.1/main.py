import sys
import os
import subprocess

# PyQt5를 먼저 임포트하고 환경 변수를 설정합니다.
import PyQt5
if hasattr(PyQt5, 'QtCore'):
    pyqt_plugins_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugins_path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication, QKeyEvent, QImage
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, pyqtSlot, QVariant, Qt, QMetaObject, QEvent, QThread, QGenericArgument

# 게임 관련 모듈 임포트
import torch
from ultralytics import YOLO
from argparse import Namespace

# --- 추가 임포트 ---
from avatar_qt import MannequinRenderer
import cv2
import numpy as np
# -----------------

# merge_test 폴더를 모듈 검색 경로에 추가
sys.path.insert(0, os.path.abspath('merge_test'))
from pages.Single_Player_app import SinglePlayerApp

# ⌨️ 전역 키 이벤트 필터: 'q' 키를 누르면 앱 종료
class AppEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Q:
            print("'q' key pressed. Terminating application.")
            QGuiApplication.instance().quit()
            return True
        return super().eventFilter(obj, event)

# 🔔 시그널 브리지: QML에서 Python으로 데이터를 전달하고, 다시 다른 QML로 명령을 보냅니다.
class SignalBridge(QObject):
    videoSelected = pyqtSignal(str)

    def __init__(self, main_view_window, parent=None):
        super().__init__(parent)
        self.main_view_window = main_view_window
        self.videoSelected.connect(self.onVideoSelected)

    @pyqtSlot(str)
    def onVideoSelected(self, videoPath):
        print(f"🎬 시그널 수신 → 영상 변경: {videoPath}")
        self.main_view_window.playVideo(videoPath)


# --- 아바타 변환 작업자 ---
class ConversionWorker(QObject):
    finished = pyqtSignal()
    totalProgress = pyqtSignal(int) # 전체 진행률 (0-100)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = None

    @pyqtSlot()
    def run(self):
        """Long-running task for avatar conversion."""
        try:
            # Stage 1: Video to JSON (0% -> 10%)
            self.totalProgress.emit(0)
            video_in = "resource/output.mp4"
            json_out = "resource/output.json"
            model_path = "merge_test/yolov8l-pose.pt"
            
            self.log.emit("Starting video to JSON conversion...")
            cmd = [
                sys.executable, "merge_test/tools/video_to_json.py",
                "--video_path", video_in,
                "--output_json", json_out,
                "--model_path", model_path
            ]
            
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.abspath("merge_test") + os.pathsep + env.get("PYTHONPATH", "")

            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            for line in iter(process.stdout.readline, ''):
                self.log.emit(line.strip())
            process.wait()
            self.log.emit("Video to JSON conversion finished.")

            if process.returncode != 0:
                raise RuntimeError("video_to_json.py failed")
            self.totalProgress.emit(10)

            # Stage 2: Render frames (10% -> 60%)
            self.log.emit("Loading assets and rendering frames...")
            assets_dir = "dady_parts"
            self.renderer = MannequinRenderer(
                json_path=json_out,
                assets_dir=assets_dir,
                stride=3 # 3프레임 단위로 렌더링
            )
            self.renderer.log.connect(self.log.emit)
            self.renderer.error.connect(self.log.emit)
            self.renderer.progress.connect(self.onRenderProgress)
            self.renderer.playReady.connect(self.write_video)
            
            self.renderer.run()

        except Exception as e:
            self.log.emit(f"Error during conversion: {e}")
        finally:
            self.finished.emit()

    @pyqtSlot(int)
    def onRenderProgress(self, value):
        # 렌더링 진행률(0-100)을 전체 진행률의 10-60% 범위로 매핑
        total_progress = 10 + int(value * 0.5)
        self.totalProgress.emit(total_progress)

    def write_video(self, qframes, fps):
        # Stage 2 is done, we are at 60%.
        self.totalProgress.emit(60)
        if not qframes:
            self.log.emit("No frames to write.")
            return

        video_out = "resource/output_character.mp4"
        
        try:
            first_frame = qframes[0]
            height, width = first_frame.height(), first_frame.width()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

            self.log.emit(f"Writing video to {video_out}...")
            total_frames = len(qframes)
            for i, qframe in enumerate(qframes):
                img = qframe.convertToFormat(QImage.Format.Format_RGB888)
                ptr = img.constBits()
                ptr.setsize(img.sizeInBytes())
                arr = np.array(ptr).reshape(height, width, 3)  # RGB
                bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
                
                # Stage 3: Writing video (60% -> 100% of total progress)
                video_progress = int((i + 1) * 100 / total_frames)
                total_progress = 60 + int(video_progress * 0.4)
                self.totalProgress.emit(total_progress)

            writer.release()
            self.log.emit("Finished writing video.")
        except Exception as e:
            self.log.emit(f"Error writing video: {e}")

# 🎮 컨트롤 브리지: 버튼 클릭 시 화면 전환 신호를 보냅니다.
class ControlBridge(QObject):
    showVideoSelect = pyqtSignal()
    gameStarted = pyqtSignal()
    gameFinished = pyqtSignal()
    showRank = pyqtSignal(int)
    showMainMenu = pyqtSignal()
    showAvatarScreen = pyqtSignal()
    conversionStarted = pyqtSignal()
    conversionFinishedForControl = pyqtSignal()

    def __init__(self, screens, signalBridge, model_data, view_window, parent=None):
        super().__init__(parent)
        self.screens = screens
        self.signalBridge = signalBridge
        self.model = model_data['model']
        self.device = model_data['device']
        self.use_half = model_data['use_half']
        self.view_window = view_window
        self.game_window = None
        self.last_video_path = None
        self.conversion_thread = None
        self.conversion_worker = None

    @pyqtSlot(str)
    def selectVideo(self, videoPath):
        print(f"🎬 QML에서 영상 선택: {videoPath}")
        self.signalBridge.videoSelected.emit(videoPath)

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 버튼 클릭됨: Video Select 화면으로 전환 신호 전송")
        self.showVideoSelect.emit()
        
    @pyqtSlot()
    def avatarButtonClicked(self):
        print("🎬 아바타 버튼 클릭됨: 아바타 화면으로 전환 신호 전송")
        self.showAvatarScreen.emit()
        
    @pyqtSlot()
    def startAvatarConversion(self):
        print("🔄 아바타 변환 시작 신호 수신")
        if self.conversion_thread and self.conversion_thread.isRunning():
            print("❗ Conversion is already in progress.")
            return

        # 컨트롤 UI를 "변환 중" 상태로 변경
        self.conversionStarted.emit()
        # 메인 뷰를 로딩 화면으로 변경
        QMetaObject.invokeMethod(self.view_window, "showAvatarLoading", Qt.QueuedConnection)

        self.conversion_thread = QThread()
        self.conversion_worker = ConversionWorker()
        self.conversion_worker.moveToThread(self.conversion_thread)

        self.conversion_thread.started.connect(self.conversion_worker.run)
        self.conversion_worker.finished.connect(self.conversion_thread.quit)
        self.conversion_worker.finished.connect(self.conversion_worker.deleteLater)
        self.conversion_thread.finished.connect(self.conversion_thread.deleteLater)
        
        self.conversion_worker.totalProgress.connect(self.updateConversionProgress)
        self.conversion_worker.finished.connect(self.onConversionFinished)
        self.conversion_worker.log.connect(lambda msg: print(f"[CONVERSION]: {msg}"))

        self.conversion_thread.start()

    @pyqtSlot(int)
    def updateConversionProgress(self, value):
        loader = self.view_window.findChild(QObject, "avatarLoader")
        if loader and loader.item():
            loader.item().setProperty("conversionProgress", value / 100.0)

    def onConversionFinished(self):
        print("✅ Avatar conversion finished!")
        # 컨트롤 UI를 "변환 완료" 상태로 변경
        self.conversionFinishedForControl.emit()

    @pyqtSlot()
    def playConvertedVideo(self):
        print("🎬 변환된 비디오 재생 요청")
        video_path = "resource/output_character.mp4"
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            self.signalBridge.videoSelected.emit(video_path)
        else:
            print(f"❗ Error: Converted video file not found or is empty at {video_path}")
            self.goToMainMenu()

    @pyqtSlot()
    def goToMainMenu(self):
        print("🎬 메인 메뉴로 돌아갑니다.")
        QMetaObject.invokeMethod(self.view_window, "resetToInitialState", Qt.QueuedConnection)
        self.showMainMenu.emit()

    @pyqtSlot()
    def retryGame(self):
        print("🎮 게임을 다시 시작합니다.")
        if self.last_video_path:
            self.startSinglePlayer(self.last_video_path)
        else:
            print("❗ 마지막으로 플레이한 게임 정보가 없습니다.")

    @pyqtSlot(str)
    def startSinglePlayer(self, videoPath):
        if not videoPath:
            print("❗ 비디오가 선택되지 않았습니다.")
            return

        if self.game_window and self.game_window.isVisible():
            print("❗ 이미 게임이 실행 중입니다.")
            return

        print(f"🚀 싱글 플레이어 모드 시작: {videoPath}")
        self.last_video_path = videoPath  # 마지막 비디오 경로 저장
        self.gameStarted.emit() # 게임 시작 신호 전송

        # 배경을 비디오에서 이미지로 변경
        QMetaObject.invokeMethod(self.view_window, "showBackgroundImage", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.view_window, "stopForegroundVideo", Qt.QueuedConnection)

        # SinglePlayerApp에 필요한 인자(args) 생성
        json_path = videoPath.replace(".mp4", ".json")
        args = Namespace(
            ref=videoPath,
            json=json_path,
            imgsz=640,
            device=self.device,
            conf_thres=0.5,
        )

        # SinglePlayerApp 인스턴스 생성
        self.game_window = SinglePlayerApp(args, self.model, self.use_half)
        self.game_window.setAttribute(Qt.WA_DeleteOnClose) # 창이 닫힐 때 객체 자동 삭제
        
        # 게임 창이 닫힐 때 신호를 받기 위해 연결
        self.game_window.destroyed.connect(self.onGameFinished)

        # 메인 뷰와 동일한 화면에 전체 화면으로 표시
        screen_for_view = self.screens[0]
        screen_geometry = screen_for_view.geometry()
        self.game_window.move(screen_geometry.topLeft())
        self.game_window.showFullScreen()


    @pyqtSlot()
    def onGameFinished(self):
        print("🏁 게임 창이 닫혔습니다.")
        if self.game_window:
            score = self.game_window.final_score
            print(f"Final score from game window: {score}")
            self.showRank.emit(int(score))
        self.gameFinished.emit() # 게임 종료 신호 전송
        self.game_window = None


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    event_filter = AppEventFilter()
    app.installEventFilter(event_filter)
    
    print("🧠 YOLOv8 모델을 로드합니다...")
    model = YOLO("merge_test/yolov8l-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        model.fuse()
    except:
        pass
    use_half = (device == "cuda")
    if use_half:
        try:
            model.model.half()
        except:
            use_half = False
    print("✅ 모델 로드 완료.")
    model_data = {"model": model, "device": device, "use_half": use_half}

    screens = QGuiApplication.screens()
    print(f"총 {len(screens)}개의 모니터가 감지되었습니다.")
    for i, screen in enumerate(screens):
        print(f"🖥️ 모니터 {i}: name={screen.name()}, geometry={screen.geometry()}")

    screen_for_view = None
    screen_for_control = None

    if len(screens) > 1:
        for screen in screens:
            geo = screen.geometry()
            if geo.width() == 2560 and geo.height() == 1440:
                screen_for_view = screen
                print(f"✅ Main_view용 모니터 (2560x1440) 찾음: {screen.name()}")
            elif geo.width() == 1920 and geo.height() == 1080:
                screen_for_control = screen
                print(f"✅ Main_control용 모니터 (1920x1080) 찾음: {screen.name()}")

    if not screen_for_view or not screen_for_control:
        print("⚠️ 특정 해상도의 모니터를 찾지 못했습니다. 기본 설정(0, 1)을 사용합니다.")
        screen_for_view = screens[0]
        screen_for_control = screens[0] if len(screens) < 2 else screens[1]

    single_monitor_mode = len(screens) < 2

    view_engine = QQmlApplicationEngine()
    view_engine.rootContext().setContextProperty("targetScreen", screen_for_view)
    view_engine.load(QUrl("Main_view.qml"))

    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]
    view_window.setGeometry(screen_for_view.geometry())
    view_window.show()
    print(f"✅ Main_view.qml 로드 완료")

    main_engine = QQmlApplicationEngine()
    
    signalBridge = SignalBridge(view_window)
    controlBridge = ControlBridge(screens, signalBridge, model_data, view_window)
    
    controlBridge.showRank.connect(lambda score: view_window.setProperty('finalScore', score))

    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    
    main_engine.load(QUrl("Main_control.qml"))
    
    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]
    
    # 신호 연결 추가
    controlBridge.conversionStarted.connect(lambda: main_window.showConvertingScreen())
    controlBridge.conversionFinishedForControl.connect(lambda: main_window.showConvertedScreen())

    if single_monitor_mode:
        screen_geo = screen_for_control.geometry()
        width = 400
        height = screen_geo.height()
        main_window.setGeometry(screen_geo.width() - width, 0, width, height)
        main_window.show()
        print("✅ Main_control.qml을 창 모드로 띄움")
    else:
        main_window.setGeometry(screen_for_control.geometry())
        main_window.show()
        print(f"✅ Main_control.qml 모니터 {screens.index(screen_for_control)}에 전체화면으로 띄움")
    
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(main_window, "showAvatarScreen", Qt.QueuedConnection))
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(view_window, "showAvatarScreen", Qt.QueuedConnection))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
