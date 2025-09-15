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
        QMetaObject.invokeMethod(
            self.main_view_window,
            "playVideo",
            Qt.DirectConnection,
            QVariant(videoPath)
        )

# 🎮 컨트롤 브리지: 버튼 클릭 시 Video_select_control.qml 창을 띄우는 역할
class ControlBridge(QObject):
    def __init__(self, screens, signalBridge, parent=None):
        super().__init__(parent)
        self.screens = screens
        self.signalBridge = signalBridge
        self.video_select_engine = None
        self.video_select_window = None

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 버튼 클릭됨: Video_select_control.qml 창 띄우기 (모니터 1)")
        
        # 이미 창이 열려 있으면 다시 열지 않습니다.
        if self.video_select_window and self.video_select_window.isVisible():
            print("❗ Video_select_control.qml 창이 이미 열려 있습니다.")
            return

        # 새 창을 위한 QML 엔진 생성
        self.video_select_engine = QQmlApplicationEngine()
        
        # 컨텍스트 속성 설정 (시그널 브리지와 타겟 스크린)
        self.video_select_engine.rootContext().setContextProperty("targetScreen", self.screens[1])
        self.video_select_engine.rootContext().setContextProperty("signalEmitter", self.signalBridge)

        # QML 파일 로드
        self.video_select_engine.load(QUrl("Video_select_control.qml"))

        if not self.video_select_engine.rootObjects():
            print("❗ Video_select_control.qml 로드 실패")
            return

        self.video_select_window = self.video_select_engine.rootObjects()[0]
        # ✅ 수정: QML에 있는 targetScreen 속성에 스크린 객체를 전달하여 QML이 스스로 위치를 설정하도록 합니다.
        self.video_select_window.setProperty("targetScreen", self.screens[1])
        self.video_select_window.showFullScreen()
        print("✅ Video_select_control.qml 모니터 1에 전체화면으로 띄움")

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

    if len(screens) < 2:
        print("❗ 2개 이상의 모니터가 필요합니다.")
        sys.exit(-1)

    # 🔹 Main_view.qml → 모니터 0
    view_engine = QQmlApplicationEngine()
    view_engine.rootContext().setContextProperty("targetScreen", screens[0])
    view_engine.load(QUrl("Main_view.qml"))

    if not view_engine.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)

    view_window = view_engine.rootObjects()[0]
    view_window.setScreen(screens[0])
    view_window.showFullScreen()
    print("✅ Main_view.qml 모니터 0에 전체화면으로 띄움")

    # 🔹 메인 애플리케이션 창 (Main_control.qml) → 모니터 1
    main_engine = QQmlApplicationEngine()
    
    # 🛠️ controlBridge 객체와 targetScreen 객체를 QML 파일 로드 전에 먼저 생성
    signalBridge = SignalBridge(view_window)
    controlBridge = ControlBridge(screens, signalBridge)
    main_engine.rootContext().setContextProperty("targetScreen", screens[1]) 
    
    main_engine.load(QUrl("Main_control.qml"))
    
    if not main_engine.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)

    main_window = main_engine.rootObjects()[0]
    # 🛠️ 수정: 창 객체에 controlBridge 속성을 직접 할당합니다.
    main_window.setProperty("controlBridge", controlBridge)

    main_window.setScreen(screens[1]) 
    main_window.setGeometry(screens[1].geometry())
    main_window.showFullScreen()
    print("✅ Main_control.qml 모니터 1에 전체화면으로 띄움")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
