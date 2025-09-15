import sys
import os
import subprocess
import atexit
import json

# PyQt5를 먼저 임포트하고 환경 변수를 설정합니다.
import PyQt5
if hasattr(PyQt5, 'QtCore'):
    pyqt_plugins_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugins_path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication, QKeyEvent, QImage
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, pyqtSlot, QVariant, Qt, QMetaObject, QEvent, QThread, QGenericArgument, Q_ARG

# 게임 관련 모듈 임포트
import torch
from ultralytics import YOLO
from argparse import Namespace

# --- 추가 임포트 ---
from avatar_qt import MannequinRenderer
import cv2
import numpy as np

import platform
import serial
# -----------------

# merge_test 폴더를 모듈 검색 경로에 추가
sys.path.insert(0, os.path.abspath('merge_test'))
sys.path.insert(0, os.path.abspath('merge_test/tools'))
from pages.Single_Player_app import SinglePlayerApp
from pages.Multi_Player_app import MultiPlayerApp
from video_to_json import create_json_from_video

def delete_output_files():
    """출력 비디오 파일을 삭제하는 함수"""
    print("Deleting output files...")
    files_to_delete = [
        "resource/output.mp4",
        "resource/output_character.mp4",
        "resource/output.json"
    ]
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except OSError as e:
                print(f"Error deleting file {f}: {e}")
        else:
            print(f"File not found, skipping: {f}")

# ⌨️ 전역 키 이벤트 필터: 'q' 키를 누르면 앱 종료
class AppEventFilter(QObject):
    def __init__(self, control_bridge, parent=None):
        super().__init__(parent)
        self.control_bridge = control_bridge

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Q:
            if self.control_bridge.game_window and self.control_bridge.game_window.isVisible():
                print("'q' key pressed. Closing game window.")
                self.control_bridge.game_window.close()
                return True
            else:
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

    def __init__(self, avatar_name, model, device, use_half, parent=None):
        super().__init__(parent)
        self.renderer = None
        self.avatar_name = avatar_name
        self.model = model
        self.device = device
        self.use_half = use_half

    @pyqtSlot()
    def run(self):
        """Long-running task for avatar conversion."""
        try:
            # Stage 1: Video to JSON
            self.totalProgress.emit(0)
            video_in = "resource/output.mp4"
            json_out = "resource/output.json"
            self.log.emit(f"Starting video to JSON conversion for {video_in}")
            create_json_from_video(
                video_path=video_in,
                model_path='merge_test/yolov8l-pose.pt', # Using the same model as main app
                output_json=json_out,
                imgsz=640,
                device=self.device,
                use_half=self.use_half,
                step=1 # Process every 3rd frame to match renderer stride
            )
            self.log.emit(f"Successfully created {json_out}.")
            self.totalProgress.emit(10)

            # Stage 2: Render frames (10% -> 60%)
            self.log.emit("Loading assets and rendering frames...")
            assets_dir = self.avatar_name
            self.renderer = MannequinRenderer(
                json_path=json_out,
                assets_dir=assets_dir,
                stride=1 # JSON already has a stride, so renderer uses 1
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
    showPostGameMenu = pyqtSignal(str)
    showRank = pyqtSignal(int)
    showMultiplayerResult = pyqtSignal(str)
    showMainMenu = pyqtSignal()
    showAvatarScreen = pyqtSignal()
    conversionStarted = pyqtSignal()
    conversionFinishedForControl = pyqtSignal()
    avatarNext = pyqtSignal()
    avatarPrevious = pyqtSignal()

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
        self.current_avatar_index = 0
        self.is_multi_player = False
        
        # OS별 포트 선택
        if platform.system() == "Windows":
            port = "COM7"
        elif platform.system() == "Linux":
            port = "/dev/ttyACM0"
        else:
            raise Exception("지원하지 않는 OS")
        
        self.ser = serial.Serial(port=port, baudrate=115200, timeout=1)

    @pyqtSlot(int)
    def onAvatarIndexChanged(self, index):
        print(f"Avatar index changed to: {index}")
        self.current_avatar_index = index

    @pyqtSlot(int)
    def choose(self, index):
        avatar_map = {0: "naruto_parts", 1: "dady_parts", 2: "ren_parts", 3: "rumi_parts"}
        avatar_name = avatar_map.get(index, "dady_parts") # Default to dady_parts if index is wrong
        self.startAvatarConversionWithName(avatar_name)

    @pyqtSlot(str)
    def selectVideo(self, videoPath):
        print(f"🎬 QML에서 영상 선택: {videoPath}")
        self.signalBridge.videoSelected.emit(videoPath)

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 버튼 클릭됨: Video Select 화면으로 전환 신호 전송 (1인 모드)")
        self.is_multi_player = False
        self.showVideoSelect.emit()

    @pyqtSlot()
    def openVideoSelectWindowForMultiplayer(self):
        print("🎬 버튼 클릭됨: Video Select 화면으로 전환 신호 전송 (2인 모드)")
        self.is_multi_player = True
        self.showVideoSelect.emit()
        
    @pyqtSlot()
    def avatarButtonClicked(self):
        print("🎬 아바타 버튼 클릭됨: 아바타 화면으로 전환 신호 전송")
        self.showAvatarScreen.emit()
        
    @pyqtSlot(str)
    def startAvatarConversionWithName(self, avatar_name):
        print(f"🔄 아바타 변환 시작 신호 수신: {avatar_name}")
        if self.conversion_thread and self.conversion_thread.isRunning():
            print("❗ Conversion is already in progress.")
            return

        # 컨트롤 UI를 "변환 중" 상태로 변경
        self.conversionStarted.emit()
        # 메인 뷰를 로딩 화면으로 변경
        QMetaObject.invokeMethod(self.view_window, "showAvatarLoading", Qt.QueuedConnection)

        self.conversion_thread = QThread()
        self.conversion_worker = ConversionWorker(avatar_name, self.model, self.device, self.use_half)
        self.conversion_worker.moveToThread(self.conversion_thread)

        self.conversion_thread.started.connect(self.conversion_worker.run)
        self.conversion_worker.finished.connect(self.conversion_thread.quit)
        self.conversion_worker.finished.connect(self.conversion_worker.deleteLater)
        self.conversion_worker.finished.connect(self.conversion_thread.deleteLater)
        self.conversion_thread.finished.connect(self.onConversionThreadFinished)
        
        self.conversion_worker.totalProgress.connect(self.updateConversionProgress)
        self.conversion_worker.finished.connect(self.onConversionFinished)
        self.conversion_worker.log.connect(lambda msg: print(f"[CONVERSION]: {msg}"))

        self.conversion_thread.start()

    @pyqtSlot()
    def startAvatarConversion(self):
        print(f"startAvatarConversion called from AvatarControl with index {self.current_avatar_index}")
        self.choose(self.current_avatar_index)

    def onConversionThreadFinished(self):
        print("Conversion thread finished, setting to None.")
        self.conversion_thread = None

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
            QMetaObject.invokeMethod(self.view_window, "playConvertedVideoInMain", Qt.QueuedConnection, Q_ARG(QVariant, video_path))
        else:
            print(f"❗ Error: Converted video file not found or is empty at {video_path}")
            self.goToMainMenu()

    @pyqtSlot()
    def onAvatarNext(self):
        self.avatarNext.emit()

    @pyqtSlot()
    def onAvatarPrevious(self):
        self.avatarPrevious.emit()

    @pyqtSlot()
    def goToMainMenu(self):
        print("🎬 메인 메뉴로 돌아갑니다.")
        delete_output_files()
        QMetaObject.invokeMethod(self.view_window, "resetToInitialState", Qt.QueuedConnection)
        self.showMainMenu.emit()

    @pyqtSlot()
    def retryGame(self):
        print("🎮 게임을 다시 시작합니다.")
        if self.last_video_path:
            self.startGame(self.last_video_path)
        else:
            print("❗ 마지막으로 플레이한 게임 정보가 없습니다.")

    @pyqtSlot()
    def showReplay(self):
        print("🎬 리플레이를 보여줍니다.")
        self.view_window.setProperty('multiplayerScores', {})

        video_path = "resource/output.mp4"
        if os.path.exists(video_path):
            QMetaObject.invokeMethod(self.view_window, "playVideo", Qt.QueuedConnection, Q_ARG(QVariant, video_path))
        else:
            print(f"❗ 리플레이 비디오 파일을 찾을 수 없습니다: {video_path}")

    @pyqtSlot(str)
    def startGame(self, videoPath):
        if self.is_multi_player:
            self._startMultiPlayer(videoPath)
        else:
            self._startSinglePlayer(videoPath)

    def _startSinglePlayer(self, videoPath):
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
        self.game_window = SinglePlayerApp(args, self.model, self.use_half, self.ser)
        move_mid = 'w'
        self.ser.write(move_mid.encode())
        self.game_window.setAttribute(Qt.WA_DeleteOnClose) # 창이 닫힐 때 객체 자동 삭제
        
        # 게임 창이 닫힐 때 신호를 받기 위해 연결
        self.game_window.destroyed.connect(self.onGameFinished)

        # 메인 뷰와 동일한 화면에 전체 화면으로 표시
        screen_for_view = self.screens[0]
        screen_geometry = screen_for_view.geometry()
        self.game_window.move(screen_geometry.topLeft())
        self.game_window.showFullScreen()

    def _startMultiPlayer(self, videoPath):
        if not videoPath:
            print("❗ 비디오가 선택되지 않았습니다.")
            return

        if self.game_window and self.game_window.isVisible():
            print("❗ 이미 게임이 실행 중입니다.")
            return

        print(f"🚀 멀티 플레이어 모드 시작: {videoPath}")
        self.last_video_path = videoPath
        self.gameStarted.emit()

        QMetaObject.invokeMethod(self.view_window, "showBackgroundImage", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.view_window, "stopForegroundVideo", Qt.QueuedConnection)

        json_path = videoPath.replace(".mp4", ".json")
        args = Namespace(
            ref=videoPath,
            json=json_path,
            imgsz=640,
            device=self.device,
            conf_thres=0.5,
        )

        self.game_window = MultiPlayerApp(args, self.model, self.use_half, self.ser)
        move_mid = 'w'
        self.ser.write(move_mid.encode())
        self.game_window.setAttribute(Qt.WA_DeleteOnClose)
        
        self.game_window.destroyed.connect(self.onGameFinished)

        screen_for_view = self.screens[0]
        screen_geometry = screen_for_view.geometry()
        self.game_window.move(screen_geometry.topLeft())
        self.game_window.showFullScreen()


    @pyqtSlot()
    def onGameFinished(self):
        print("🏁 게임 창이 닫혔습니다.")
        move_mid = 'w'
        self.ser.write(move_mid.encode())
        if self.game_window:
            if self.is_multi_player:
                scores = self.game_window.final_score
                print(f"Multiplayer scores from game window: {scores}")
                self.showMultiplayerResult.emit(json.dumps(scores))
            else:
                score = self.game_window.final_score
                print(f"Final score from game window: {score}")
                self.showRank.emit(int(score))
        
        if self.is_multi_player:
            self.showPostGameMenu.emit("PostGameMenu_Multi.qml")
        else:
            self.showPostGameMenu.emit("PostGameMenu.qml")
            
        self.game_window = None

def main():
    app = QApplication(sys.argv)
    atexit.register(delete_output_files)
    app.setQuitOnLastWindowClosed(False)

    
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
        # Try to find HDMI for control screen
        for screen in screens:
            if "HDMI" in screen.name():
                screen_for_control = screen
                print(f"✅ Control screen found by name: {screen.name()}")
                break
        
        # Assign the other screen to view
        if screen_for_control:
            for screen in screens:
                if screen != screen_for_control:
                    screen_for_view = screen
                    print(f"✅ View screen assigned: {screen.name()}")
                    break

    # Fallback logic if the above fails
    if not screen_for_view or not screen_for_control:
        print("⚠️ Could not find HDMI monitor or assign screens correctly. Using default order.")
        screen_for_view = screens[0]
        screen_for_control = screens[0] if len(screens) < 2 else screens[1]

    single_monitor_mode = len(screens) < 2

    view_engine = QQmlApplicationEngine()
    main_engine = QQmlApplicationEngine()

    # 브릿지 객체들을 먼저 생성합니다.
    # view_window는 아직 존재하지 않으므로 None으로 초기화하고 나중에 설정합니다.
    signalBridge = SignalBridge(None) 
    controlBridge = ControlBridge(screens, signalBridge, model_data, None)

    event_filter = AppEventFilter(controlBridge)
    app.installEventFilter(event_filter)

    # view_engine에 controlBridge를 설정합니다.
    view_engine.rootContext().setContextProperty("targetScreen", screen_for_view)
    view_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    view_engine.rootContext().setContextProperty("pyBridge", controlBridge) # pyBridge 추가
    view_engine.load(QUrl("Main_view.qml"))

    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]
    
    # 브릿지 객체에 view_window를 설정합니다.
    signalBridge.main_view_window = view_window
    controlBridge.view_window = view_window

    view_window.setGeometry(screen_for_view.geometry())
    view_window.show()
    print(f"✅ Main_view.qml 로드 완료")
    
    controlBridge.showRank.connect(lambda score: view_window.setProperty('finalScore', score))
    controlBridge.showMultiplayerResult.connect(lambda scores: view_window.setProperty('multiplayerScores', scores))

    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    main_engine.rootContext().setContextProperty("pyBridge", controlBridge) # pyBridge 추가
    
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
        main_window.showFullScreen()
        print(f"✅ Main_control.qml 모니터 {screens.index(screen_for_control)}에 전체화면으로 띄움")
    
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(main_window, "showAvatarScreen", Qt.QueuedConnection))
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(view_window, "showAvatarScreen", Qt.QueuedConnection))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()