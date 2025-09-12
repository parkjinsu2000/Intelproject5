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

    visible: true
    visibility: "FullScreen"
    width: 1920
    height: 1080
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

    // 배경 이미지
    Image {
        source: "resource/background_control.png"
        anchors.fill: parent
        fillMode: Image.PreserveAspectCrop
    }

    // 화면 전환을 위한 Loader
    Loader {
        id: pageLoader
        anchors.fill: parent
        source: "MainMenu.qml" // 초기 화면
    }

    // Python의 ControlBridge에서 오는 신호를 처리
    Connections {
        target: controlBridge // 전역 컨텍스트 속성
        function onShowVideoSelect() {
            console.log("🔄 화면 전환 신호 수신: Video_select_control.qml 로드")
            pageLoader.source = "Video_select_control.qml"
        }
    }
}