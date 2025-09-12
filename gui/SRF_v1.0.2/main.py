import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QObject, pyqtSlot

class ControlBridge(QObject):
    def __init__(self, screens):
        super().__init__()
        self.screens = screens

    @pyqtSlot()
    def openVideoSelectWindow(self):
        print("🎬 버튼 클릭됨: Video_select_control.qml 창 띄우기 (모니터 2)")
        engine_video = QQmlApplicationEngine()
        engine_video.rootContext().setContextProperty("targetScreen", self.screens[1])  # 모니터 2
        engine_video.load(QUrl("Video_select_control.qml"))

        if not engine_video.rootObjects():
            print("❗ Video_select_control.qml 로드 실패")
            return

        video_window = engine_video.rootObjects()[0]
        video_window.setScreen(self.screens[1])
        video_window.showFullScreen()
        print("✅ Video_select_control.qml 모니터 2에 전체화면으로 띄움")

def main():
    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    print(f"총 {len(screens)}개의 모니터가 감지되었습니다.")
    for i, screen in enumerate(screens):
        print(f"🖥️ 모니터 {i}: name={screen.name()}, geometry={screen.geometry()}")

    if len(screens) < 2:
        print("❗ 2개 이상의 모니터가 필요합니다.")
        sys.exit(-1)

    # 🔹 Main_view.qml → 모니터 1
    engine_view = QQmlApplicationEngine()
    engine_view.load(QUrl("Main_view.qml"))
    if not engine_view.rootObjects():
        print("❗ Main_view.qml 로드 실패")
        sys.exit(-1)
    main_view_window = engine_view.rootObjects()[0]
    main_view_window.setScreen(screens[0])
    main_view_window.showFullScreen()

    # 🔹 Main_control.qml → 모니터 2
    bridge = ControlBridge(screens)
    engine_control = QQmlApplicationEngine()
    engine_control.rootContext().setContextProperty("targetScreen", screens[1])
    engine_control.rootContext().setContextProperty("controlBridge", bridge)
    engine_control.load(QUrl("Main_control.qml"))
    if not engine_control.rootObjects():
        print("❗ Main_control.qml 로드 실패")
        sys.exit(-1)
    main_control_window = engine_control.rootObjects()[0]
    main_control_window.setScreen(screens[1])
    main_control_window.showFullScreen()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
