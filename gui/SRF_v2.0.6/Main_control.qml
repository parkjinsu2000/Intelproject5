import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.15
import QtQuick.Layouts 1.15

import "common" // 새로 만든 컴포넌트들을 불러옵니다.

ApplicationWindow {
    id: appWindow
    property var targetScreen

    color: "black"

    // ✅ Python에서 전달된 targetScreen이 설정되면 위치와 해상도 설정
    Connections {
        target: appWindow
        function onTargetScreenChanged() {
            if (targetScreen !== undefined && targetScreen !== null) {
                appWindow.screen = targetScreen
                appWindow.x = targetScreen.geometry.x
                appWindow.y = targetScreen.geometry.y
                appWindow.width = targetScreen.geometry.width
                appWindow.height = targetScreen.geometry.height
                console.log("✅ main.qml targetScreen 전달됨:", targetScreen.name)
                console.log("🧭 QML에서 적용된 screen 이름:", screen.name)
                console.log("📍 위치:", x, y, "크기:", width, height)
            } else {
                console.log("❗ targetScreen이 전달되지 않았습니다.")
            }
        }
    }

    MediaPlayer {
      id: sharedPlayer
      source: "resource/background_control.mp4"
      autoPlay: true
      loops: MediaPlayer.Infinite
    }

    VideoOutput {
      anchors.fill: parent
      source: sharedPlayer
    }

    // 화면 전환을 위한 Loader
    Loader {
        id: pageLoader
        anchors.fill: parent
        source: "MainMenu.qml" // 초기 화면
    }

    Loader {
        id: creditLoader
        z: 1
    }

    // Python의 ControlBridge에서 오는 신호를 처리
    Connections {
        target: controlBridge // 전역 컨텍스트 속성
        function onShowVideoSelect() {
            console.log("🔄 화면 전환 신호 수신: Video_select_control.qml 로드")
            pageLoader.source = "Video_select_control.qml"
        }

        function onGameStarted() {
            console.log("🎮 게임 시작 신호 수신: 컨트롤 UI 숨김")
            pageLoader.visible = false
        }

        function onShowPostGameMenu(qmlFile) {
            console.log("🏁 게임 종료 신호 수신: 컨트롤 UI 표시")
            pageLoader.source = qmlFile // 게임 후 메뉴로 복귀
            pageLoader.visible = true
        }
        function onShowMainMenu() {
            console.log("🔄 메인 메뉴로 돌아가기")
            pageLoader.source = "MainMenu.qml"
        }
    }

    function showAvatarScreen() {
        console.log("🔄 화면 전환: AvatarControl.qml 로드")
        pageLoader.source = "AvatarControl.qml"
    }

    function showConvertingScreen() {
        console.log("🔄 화면 전환: AvatarConverting.qml 로드")
        pageLoader.source = "AvatarConverting.qml"
    }

    function showConvertedScreen() {
        console.log("🔄 화면 전환: AvatarConverted.qml 로드")
        pageLoader.source = "AvatarConverted.qml"
    }

    function showCreditRoll() {
        creditLoader.source = "Credit.qml"
        creditLoader.item.start()
    }
}
