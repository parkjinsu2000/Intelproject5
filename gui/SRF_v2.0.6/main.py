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
        "resource/output.json",
        "resource/output_with_audio.mp4",
        "resource/output_character_with_audio.mp4"
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

    def __init__(self, avatar_name, model, device, use_half, reference_video_path, parent=None):
        super().__init__(parent)
        self.renderer = None
        self.avatar_name = avatar_name
        self.model = model
        self.device = device
        self.use_half = use_half
        self.reference_video_path = reference_video_path

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
                
                # Stage 3: Writing video (60% -> 90% of total progress)
                video_progress = int((i + 1) * 100 / total_frames)
                total_progress = 60 + int(video_progress * 0.3)
                self.totalProgress.emit(total_progress)

            writer.release()
            self.log.emit("Finished writing video.")

            # Stage 4: Merge audio (90% -> 100%)
            self.log.emit("Merging audio to final video...")
            self._merge_audio_to_final_video()
            self.totalProgress.emit(100)

        except Exception as e:
            self.log.emit(f"Error writing video or merging audio: {e}")

    def _merge_audio_to_final_video(self):
        self.log.emit(f"Starting final audio merge...")
        character_video = "resource/output_character.mp4"
        output_video_with_audio = "resource/output_character_with_audio.mp4"

        if not self.reference_video_path:
            self.log.emit("No reference video path provided for audio merge, skipping.")
            # Copy silent video to the final name so playback doesn't fail
            import shutil
            shutil.copy(character_video, output_video_with_audio)
            return

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', character_video,
            '-i', self.reference_video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_video_with_audio
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.log.emit(f"Successfully merged audio to {output_video_with_audio}")
        except subprocess.CalledProcessError as e:
            self.log.emit(f"FFmpeg error during final merge: {e.stderr}")
        except FileNotFoundError:
            self.log.emit("'ffmpeg' not found. Cannot merge audio.")


# 🎮 컨트롤 브리지: 버튼 클릭 시 화면 전환 신호를 보냅니다.
class ControlBridge(QObject):
    showVideoSelect = pyqtSignal()
    gameStarted = pyqtSignal()
    showPostGameMenu = pyqtSignal(str)
    showRank = pyqtSignal(int)
    showMultiplayerResult = pyqtSignal(str)
    showMainMenu = pyqtSignal()
    showAvatarScreen = pyqtSignal()
    showCredits = pyqtSignal()
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

    @pyqtSlot()
    def onShowCredits(self):
        print("🎬 크레딧 버튼 클릭됨: 크레딧 신호 전송")
        self.showCredits.emit()
        
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
        self.conversion_worker = ConversionWorker(
            avatar_name, self.model, self.device, self.use_half, self.last_video_path
        )
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
        video_path = "resource/output_character_with_audio.mp4"
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

    def _merge_audio_to_output(self, reference_video_path):
        print(f"🔊 오디오 병합 시작: 'resource/output.mp4'와 '{reference_video_path}'의 오디오를 합칩니다.")
        recorded_video = "resource/output.mp4"
        output_video_with_audio = "resource/output_with_audio.mp4"

        if not os.path.exists(recorded_video):
            print(f"❗ 녹화된 비디오 파일이 없습니다: {recorded_video}")
            return

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', recorded_video,
            '-i', reference_video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_video_with_audio
        ]

        try:
            # ffmpeg 실행, 로그 출력을 위해 capture_output=True 사용
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ 오디오 병합 완료: {output_video_with_audio}")
            print(f"FFmpeg stdout: {result.stdout}")
            print(f"FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❗ FFmpeg 오류 발생:")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print("❗ 'ffmpeg'을 찾을 수 없습니다. 시스템에 설치되어 있는지 확인하세요.")

    @pyqtSlot()
    def onGameFinished(self):
        print("🏁 게임 창이 닫혔습니다.")
        move_mid = 'w'
        self.ser.write(move_mid.encode())

        # 오디오 병합 실행
        if self.last_video_path:
            self._merge_audio_to_output(self.last_video_path)
        else:
            print("❗ last_video_path가 설정되지 않아 오디오를 병합할 수 없습니다.")

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

    # --- HDMI 우선 검색: "HDMI", "DP", "HDMI-" 등 이름에 HDMI/DP 표기가 있는 스크린을 찾음 ---
    screen_for_view = None
    screen_for_control = None

    if len(screens) > 1:
        # 우선 HDMI 계열로 보이는 스크린 찾기
        for screen in screens:
            name = screen.name().upper()
            if "HDMI" in name or "DP" in name or "DISPLAY" in name:
                screen_for_view = screen
                print(f"✅ View(전체화면) 화면으로 선택: {screen.name()}")
                break

        # 만약 HDMI를 못 찾으면 첫 번째를 view로 지정
        if not screen_for_view:
            screen_for_view = screens[0]
            print("⚠️ HDMI/DP 표기가 있는 모니터를 찾지 못했습니다. 첫 번째 스크린을 view로 사용합니다.")

        # control은 view와 다른 화면을 쓰도록 시도
        for screen in screens:
            if screen != screen_for_view:
                screen_for_control = screen
                print(f"✅ Control 화면으로 선택: {screen.name()}")
                break

    # 단일 모니터 또는 control 못 찾았을 때의 폴백
    if not screen_for_control:
        if len(screens) > 1:
            # 두 번째 스크린를 control로 사용
            screen_for_control = screens[1] if screens[0] == screen_for_view else screens[0]
        else:
            # screen_for_control = screen_for_view
            screen_for_control = screens[0]
            screen_for_view = screens[0]

    single_monitor_mode = (len(screens) < 2)

    # ... (모델 로드 등 중간 생략) ...

    view_engine = QQmlApplicationEngine()
    main_engine = QQmlApplicationEngine()

    # 브릿지 등 설정 (기존 코드와 동일)
    signalBridge = SignalBridge(None) 
    controlBridge = ControlBridge(screens, signalBridge, model_data, None)

    event_filter = AppEventFilter(controlBridge)
    app.installEventFilter(event_filter)

    view_engine.rootContext().setContextProperty("targetScreen", screen_for_view)
    view_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    view_engine.rootContext().setContextProperty("pyBridge", controlBridge)
    view_engine.load(QUrl("Main_view.qml"))

    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]

    # 브릿지에 윈도우 설정
    signalBridge.main_view_window = view_window
    controlBridge.view_window = view_window

    # View 화면을 해당 스크린에 맞춰 전체화면으로 띄움
    try:
        # 화면 위치/크기 맞춤
        view_window.setGeometry(screen_for_view.geometry())
        # 최상단 수준 윈도우(예: QWindow/QQuickWindow/Window일 경우) 전체화면 호출
        view_window.showFullScreen()
        print(f"✅ Main_view.qml을 화면 '{screen_for_view.name()}'에서 전체화면으로 표시했습니다.")
    except Exception as e:
        print(f"⚠️ view_window 전체화면 설정 중 예외 발생: {e}")
        # 안전하게 창모드로 띄우기
        view_window.setGeometry(screen_for_view.geometry())
        view_window.show()
        print("✅ 대체로 창 모드로 view_window를 표시했습니다.")

    # control쪽 QML 로드
    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    main_engine.rootContext().setContextProperty("pyBridge", controlBridge)
    main_engine.load(QUrl("Main_control.qml"))

    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]

    # 기존의 변환 신호 연결
    controlBridge.conversionStarted.connect(lambda: main_window.showConvertingScreen())
    controlBridge.conversionFinishedForControl.connect(lambda: main_window.showConvertedScreen())

    # control 윈도우 보이기: 단일 모니터면 창 모드 (우측에 붙여서),
    # 멀티 모니면 control 스크린에서 전체화면으로 표시(주로 터치패널 등)
    if single_monitor_mode:
        screen_geo = screen_for_control.geometry()
        width = 400
        height = screen_geo.height()
        main_window.setGeometry(screen_geo.width() - width, 0, width, height)
        main_window.show()
        print("✅ Main_control.qml을 창 모드로 띄움 (단일 모니터 설정)")
    else:
        # control 화면을 전체화면으로 띄우기 (원하면 창 모드로 바꿔도 됨)
        main_window.setGeometry(screen_for_control.geometry())
        main_window.showFullScreen()
        print(f"✅ Main_control.qml 모니터 '{screen_for_control.name()}'에 전체화면으로 띄움")
    
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(main_window, "showAvatarScreen", Qt.QueuedConnection))
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(view_window, "showAvatarScreen", Qt.QueuedConnection))
    controlBridge.showCredits.connect(lambda: QMetaObject.invokeMethod(view_window, "showCreditVideo", Qt.QueuedConnection))
    controlBridge.showCredits.connect(lambda: QMetaObject.invokeMethod(main_window, "showCreditRoll", Qt.QueuedConnection))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()