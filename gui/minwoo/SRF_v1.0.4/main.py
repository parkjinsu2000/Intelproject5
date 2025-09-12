import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication, QKeyEvent
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, pyqtSlot, QVariant, Qt, QMetaObject, QEvent

# 🔔 시그널 브리지: QML에서 Python으로 데이터를 전달하고, 다시 다른 QML로 명령을 보냅니다.
class SignalBridge(QObject):
    videoSelected = pyqtSignal(str)

    def __init__(self, main_view_window, parent=None):
        super().__init__(parent)
        self.main_view_window = main_view_window
        # videoSelected 시그널이 발생하면 onVideoSelected 슬롯을 호출합니다.
        self.videoSelected.connect(self.onVideoSelected)

    @pyqtSlot(str)
    def onVideoSelected(self, videoPath):
        print(f"🎬 시그널 수신 → 영상 변경: {videoPath}")
        
        # 🟢 PyQt를 통해 QML의 함수를 호출
        # Main_view.qml에 있는 playVideo 함수를 호출하여 영상 경로를 전달합니다.
        self.main_view_window.playVideo(videoPath)

# 🎮 컨트롤 브리지: 버튼 클릭 시 화면 전환 신호를 보냅니다.
class ControlBridge(QObject):
    # 🎬 화면 전환을 위한 시그널 정의
    showVideoSelect = pyqtSignal()

    def __init__(self, screens, signalBridge, parent=None):
        super().__init__(parent)
        self.screens = screens
        self.signalBridge = signalBridge
        self.single_monitor_mode = len(self.screens) < 2
        self.control_screen = self.screens[1] if not self.single_monitor_mode else self.screens[0]

    @pyqtSlot(str)
    def selectVideo(self, videoPath):
        print(f"🎬 QML에서 영상 선택: {videoPath}")
        self.signalBridge.videoSelected.emit(videoPath)

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 버튼 클릭됨: Video Select 화면으로 전환 신호 전송")
        # 시그널을 발생시켜 QML에서 화면을 전환하도록 합니다.
        self.showVideoSelect.emit()


def main():
    app = QApplication(sys.argv)
    
    # 📝 모든 키보드 이벤트를 가로채는 이벤트 필터
    def handle_global_keypress(obj, event):
        if event.type() == QEvent.KeyPress:
            key_event = QKeyEvent(event)
            if key_event.key() == Qt.Key_Q:
                print("⌨️ 'Q' 키 입력 감지. 애플리케이션을 종료합니다.")
                QApplication.quit()
                return True # 이벤트를 처리했으므로 True 반환
        return False # 다른 이벤트는 무시하고 계속 진행

    app.installEventFilter(app)
    app.eventFilter = handle_global_keypress # 이벤트 필터 함수 설정

    screens = QGuiApplication.screens()
    print(f"총 {len(screens)}개의 모니터가 감지되었습니다.")
    for i, screen in enumerate(screens):
        print(f"🖥️ 모니터 {i}: name={screen.name()}, geometry={screen.geometry()}")

    single_monitor_mode = len(screens) < 2
    
    screen_for_view = screens[0]
    screen_for_control = screens[1] if not single_monitor_mode else screens[0]


    # 🔹 Main_view.qml → 모니터 0
    view_engine = QQmlApplicationEngine()
    view_engine.rootContext().setContextProperty("targetScreen", screen_for_view)
    view_engine.load(QUrl("Main_view.qml"))

    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]
    view_window.setScreen(screen_for_view)
    view_window.showFullScreen()
    print(f"✅ Main_view.qml 모니터 {screens.index(screen_for_view)}에 전체화면으로 띄움")

    # 🔹 메인 애플리케이션 창 (Main_control.qml) → 모니터 1
    main_engine = QQmlApplicationEngine()
    
    # 🛠️ controlBridge 객체와 targetScreen 객체를 QML 파일 로드 전에 먼저 생성
    signalBridge = SignalBridge(view_window)
    controlBridge = ControlBridge(screens, signalBridge)
    main_engine.rootContext().setContextProperty("targetScreen", screen_for_control)
    # 🛠️ 수정: controlBridge를 컨텍스트 속성으로 설정합니다.
    main_engine.rootContext().setContextProperty("controlBridge", controlBridge)
    
    main_engine.load(QUrl("Main_control.qml"))
    
    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]
    # 🛠️ 수정: 창 객체에 controlBridge 속성을 직접 할당하는 코드는 컨텍스트 속성 설정으로 대체되었으므로 제거합니다.
    # main_window.setProperty("controlBridge", controlBridge)

    main_window.setScreen(screen_for_control) 
    
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
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
