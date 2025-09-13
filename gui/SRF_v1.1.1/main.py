import sys
import os
import subprocess

# PyQt5를 먼저 임포트하고 환경 변수를 설정합니다.
import PyQt5
if hasattr(PyQt5, 'QtCore'):
    pyqt_plugins_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugins_path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication, QKeyEvent
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, pyqtSlot, QVariant, Qt, QMetaObject, QEvent

# 게임 관련 모듈 임포트
import torch
from ultralytics import YOLO
from argparse import Namespace

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


# 🎮 컨트롤 브리지: 버튼 클릭 시 화면 전환 신호를 보냅니다.
class ControlBridge(QObject):
    showVideoSelect = pyqtSignal()
    gameStarted = pyqtSignal()
    gameFinished = pyqtSignal()
    showRank = pyqtSignal(int)
    showMainMenu = pyqtSignal()
    showAvatarScreen = pyqtSignal()

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
        QMetaObject.invokeMethod(self.view_window, "showAvatarLoading", Qt.QueuedConnection)

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
        # 배경 비디오를 다시 시작하지 않음
        self.game_window = None


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False) # 마지막 창이 닫혀도 앱이 종료되지 않도록 설정

    # 전역 이벤트 필터 설치
    event_filter = AppEventFilter()
    app.installEventFilter(event_filter)
    
    # --- YOLO 모델 로드 (애플리케이션 시작 시 한 번만) ---
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
    # ----------------------------------------------------

    screens = QGuiApplication.screens()
    print(f"총 {len(screens)}개의 모니터가 감지되었습니다.")
    for i, screen in enumerate(screens):
        print(f"🖥️ 모니터 {i}: name={screen.name()}, geometry={screen.geometry()}")

    # 해상도로 특정 모니터 찾기
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

    # 특정 모니터를 찾지 못했을 경우의 대비책
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
    # view_window를 ControlBridge에 전달
    controlBridge = ControlBridge(screens, signalBridge, model_data, view_window)
    
    # 점수 신호를 QML 속성에 연결
    controlBridge.showRank.connect(lambda score: view_window.setProperty('finalScore', score))

    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    
    main_engine.load(QUrl("Main_control.qml"))
    
    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]
    
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
    
    # 아바타 화면 전환 신호 연결
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(main_window, "showAvatarScreen", Qt.QueuedConnection))
    controlBridge.showAvatarScreen.connect(lambda: QMetaObject.invokeMethod(view_window, "showAvatarScreen", Qt.QueuedConnection))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()