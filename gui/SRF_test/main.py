import sys
import os
import atexit
import time

# PyQt5 먼저 + Qt 플러그인 경로
import PyQt5
if hasattr(PyQt5, 'QtCore'):
    pyqt_plugins_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugins_path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication, QImage
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import (
    QUrl, QObject, pyqtSignal, pyqtSlot, QVariant, Qt,
    QMetaObject, QEvent, QThread, Q_ARG
)

import torch
from ultralytics import YOLO
from argparse import Namespace

from avatar_qt import MannequinRenderer
import cv2
import numpy as np

# merge_test 경로 + 도구
sys.path.insert(0, os.path.abspath('merge_test'))
sys.path.insert(0, os.path.abspath('merge_test/tools'))
from pages.Single_Player_app import SinglePlayerApp
from video_to_json import create_json_from_video

# ------- 고정 경로 상수 --------
OUTPUT_INPUT      = "resource/output.mp4"
OUTPUT_JSON       = "resource/output.json"
OUTPUT_FINAL      = "resource/output_character.mp4"     # ✅ 항상 이 이름으로 덮어씀
OUTPUT_FINAL_TMP  = "resource/.output_character.tmp.mp4"  # 임시 파일
# --------------------------------

def cleanup_outputs(keep_input=True):
    """변환 산출물만 정리. output.mp4는 기본 유지."""
    print("Cleaning up old outputs...")
    # JSON만 지움 (최신 변환 시 다시 생성)
    if os.path.exists(OUTPUT_JSON):
        try:
            os.remove(OUTPUT_JSON)
            print(f"Deleted {OUTPUT_JSON}")
        except OSError as e:
            print(f"Error deleting {OUTPUT_JSON}: {e}")
    # 이전 임시 파일 있으면 정리
    if os.path.exists(OUTPUT_FINAL_TMP):
        try:
            os.remove(OUTPUT_FINAL_TMP)
        except OSError:
            pass
    # 입력 비디오는 유지
    if not keep_input and os.path.exists(OUTPUT_INPUT):
        try:
            os.remove(OUTPUT_INPUT)
            print(f"Deleted {OUTPUT_INPUT}")
        except OSError as e:
            print(f"Error deleting {OUTPUT_INPUT}: {e}")

class AppEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Q:
            print("'q' key pressed. Terminating application.")
            QGuiApplication.instance().quit()
            return True
        return super().eventFilter(obj, event)

class SignalBridge(QObject):
    videoSelected = pyqtSignal(str)
    def __init__(self, main_view_window, parent=None):
        super().__init__(parent)
        self.main_view_window = main_view_window
        self.videoSelected.connect(self.onVideoSelected)
    @pyqtSlot(str)
    def onVideoSelected(self, videoPath):
        print(f"🎬 시그널 수신 → 영상 변경: {videoPath}")
        if self.main_view_window:
            self.main_view_window.playVideo(videoPath)

# --- 아바타 변환 작업자 ---
class ConversionWorker(QObject):
    finished = pyqtSignal()
    totalProgress = pyqtSignal(int)
    log = pyqtSignal(str)
    videoReady = pyqtSignal(str)  # ✅ 완성 파일 경로

    def __init__(self, avatar_name, model, device, use_half, parent=None):
        super().__init__(parent)
        self.renderer = None
        self.avatar_name = avatar_name
        self.model = model
        self.device = device
        self.use_half = use_half

    @pyqtSlot()
    def run(self):
        try:
            # Stage 1: Video -> JSON
            self.totalProgress.emit(0)
            if not os.path.exists(OUTPUT_INPUT):
                raise FileNotFoundError(f"Input video not found: {OUTPUT_INPUT}")

            self.log.emit(f"Starting video->JSON: {OUTPUT_INPUT}")
            create_json_from_video(
                video_path=OUTPUT_INPUT,
                model_path='merge_test/yolov8l-pose.pt',
                output_json=OUTPUT_JSON,
                imgsz=640,
                device=self.device,
                use_half=self.use_half,
                step=3
            )
            self.log.emit(f"JSON created: {OUTPUT_JSON}")
            self.totalProgress.emit(10)

            # Stage 2: Render frames
            self.log.emit(f"Rendering frames with assets: {self.avatar_name}")
            self.renderer = MannequinRenderer(
                json_path=OUTPUT_JSON,
                assets_dir=self.avatar_name,
                stride=1
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
        self.totalProgress.emit(10 + int(value * 0.5))

    def write_video(self, qframes, fps):
        self.totalProgress.emit(60)
        if not qframes:
            self.log.emit("No frames to write.")
            return

        # ✅ 임시 파일에 먼저 쓴 뒤 원자적 교체
        tmp_path = OUTPUT_FINAL_TMP
        final_path = OUTPUT_FINAL

        try:
            # 혹시 이전 tmp 잔재가 있으면 삭제
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            first = qframes[0]
            h, w = first.height(), first.width()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open temp file")

            self.log.emit(f"Writing video (temp): {tmp_path}")
            total = len(qframes)
            for i, qframe in enumerate(qframes):
                img = qframe.convertToFormat(QImage.Format.Format_RGB888)
                ptr = img.constBits(); ptr.setsize(img.sizeInBytes())
                arr = np.array(ptr).reshape(h, w, 3)
                writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                self.totalProgress.emit(60 + int(((i + 1) * 100 / total) * 0.4))

            writer.release()
            # ✅ 완성되면 기존 파일을 원자적으로 교체 (읽는 쪽은 이전 inode 계속 사용)
            os.replace(tmp_path, final_path)
            self.log.emit(f"Final video ready: {final_path}")
            self.videoReady.emit(final_path)

        except Exception as e:
            self.log.emit(f"Error writing video: {e}")
            # 실패 시 tmp 제거 시도
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

class ControlBridge(QObject):
    showVideoSelect = pyqtSignal()
    gameStarted = pyqtSignal()
    gameFinished = pyqtSignal()
    showRank = pyqtSignal(int)
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
        self.last_converted_video = None
        self.conversion_thread = None
        self.conversion_worker = None
        self.currentAvatarIndex = 1  # 기본 dady_parts

    def avatar_name_from_index(self, index: int) -> str:
        avatar_map = {0: "naruto_parts", 1: "dady_parts", 2: "ren_parts", 3: "rumi_parts"}
        return avatar_map.get(int(index), "dady_parts")

    @pyqtSlot(int)
    def setCurrentAvatarIndex(self, index):
        self.currentAvatarIndex = int(index)
        print(f"[Bridge] currentAvatarIndex={self.currentAvatarIndex} ({self.avatar_name_from_index(self.currentAvatarIndex)})")

    @pyqtSlot(int)
    def choose(self, index):
        self.setCurrentAvatarIndex(index)
        self.startAvatarConversion()

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 Video Select 화면 요청")
        self.showVideoSelect.emit()

    @pyqtSlot()
    def avatarButtonClicked(self):
        print("🎬 아바타 화면 요청")
        self.showAvatarScreen.emit()

    @pyqtSlot()
    def startAvatarConversion(self):
        avatar_name = self.avatar_name_from_index(self.currentAvatarIndex)
        self.startAvatarConversionWithName(avatar_name)

    @pyqtSlot(str)
    def startAvatarConversionWithName(self, avatar_name):
        print(f"🔄 아바타 변환 시작: {avatar_name}")
        if self.conversion_thread and self.conversion_thread.isRunning():
            print("❗ Conversion is already in progress.")
            return

        # 현재 재생 중인 변환 결과를 정지하도록 뷰에 요청(있다면)
        if self.view_window:
            QMetaObject.invokeMethod(self.view_window, "showAvatarLoading", Qt.QueuedConnection)
            # 재생 중지용 메서드가 있다면 호출 (없으면 무시)
            try:
                QMetaObject.invokeMethod(self.view_window, "stopConvertedVideoInMain", Qt.QueuedConnection)
            except Exception:
                pass

        self.conversionStarted.emit()
        cleanup_outputs(keep_input=True)  # JSON/임시물만 정리, input 유지

        self.conversion_thread = QThread()
        self.conversion_worker = ConversionWorker(avatar_name, self.model, self.device, self.use_half)
        self.conversion_worker.moveToThread(self.conversion_thread)

        self.conversion_thread.started.connect(self.conversion_worker.run)
        self.conversion_worker.finished.connect(self.conversion_thread.quit)
        self.conversion_worker.finished.connect(self.conversion_worker.deleteLater)
        self.conversion_thread.finished.connect(self.conversion_thread.deleteLater)
        self.conversion_worker.totalProgress.connect(self.updateConversionProgress)
        self.conversion_worker.log.connect(lambda msg: print(f"[CONVERSION]: {msg}"))
        self.conversion_worker.videoReady.connect(self.onVideoReady)  # ✅ 결과 파일 즉시 재생

        self.conversion_thread.start()

    @pyqtSlot(int)
    def updateConversionProgress(self, value):
        loader = self.view_window.findChild(QObject, "avatarLoader") if self.view_window else None
        if loader and loader.item():
            loader.item().setProperty("conversionProgress", value / 100.0)

    @pyqtSlot(str)
    def onVideoReady(self, path):
        print(f"🎬 변환된 비디오 준비됨: {path}")
        self.last_converted_video = path
        # 컨트롤 UI에 "완료" 상태 표시
        self.conversionFinishedForControl.emit()
        # 메인 뷰에 재생 요청
        if self.view_window and os.path.exists(path) and os.path.getsize(path) > 0:
            QMetaObject.invokeMethod(self.view_window, "playConvertedVideoInMain", Qt.QueuedConnection, Q_ARG(QVariant, path))
        else:
            print(f"❗ Error: Converted video not found or empty at {path}")
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
        cleanup_outputs(keep_input=True)
        if self.view_window:
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
        self.last_video_path = videoPath
        self.gameStarted.emit()

        if self.view_window:
            QMetaObject.invokeMethod(self.view_window, "showBackgroundImage", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self.view_window, "stopForegroundVideo", Qt.QueuedConnection)

        json_path = videoPath.replace(".mp4", ".json")
        args = Namespace(ref=videoPath, json=json_path, imgsz=640, device=self.device, conf_thres=0.5)
        self.game_window = SinglePlayerApp(args, self.model, self.use_half)
        self.game_window.setAttribute(Qt.WA_DeleteOnClose)
        self.game_window.destroyed.connect(self.onGameFinished)

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
        self.gameFinished.emit()
        self.game_window = None


def main():
    app = QApplication(sys.argv)
    atexit.register(lambda: cleanup_outputs(keep_input=True))
    app.setQuitOnLastWindowClosed(False)

    event_filter = AppEventFilter()
    app.installEventFilter(event_filter)

    print("🧠 YOLOv8 모델을 로드합니다...")
    model = YOLO("merge_test/yolov8l-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass
    use_half = (device == "cuda")
    if use_half:
        try:
            model.model.half()
        except Exception:
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
    main_engine = QQmlApplicationEngine()

    signalBridge = SignalBridge(None)
    controlBridge = ControlBridge(screens, signalBridge, model_data, None)

    view_engine.rootContext().setContextProperty("targetScreen", screen_for_view)
    view_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    view_engine.rootContext().setContextProperty("pyBridge", controlBridge)
    view_engine.load(QUrl("Main_view.qml"))
    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]
    signalBridge.main_view_window = view_window
    controlBridge.view_window = view_window
    view_window.setGeometry(screen_for_view.geometry())
    view_window.show()
    print(f"✅ Main_view.qml 로드 완료")

    controlBridge.showRank.connect(lambda score: view_window.setProperty('finalScore', score))

    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    main_engine.rootContext().setContextProperty("pyBridge", controlBridge)
    main_engine.load(QUrl("Main_control.qml"))
    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]
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