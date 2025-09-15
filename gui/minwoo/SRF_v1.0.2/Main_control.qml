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
property var controlBridge

visible: true
visibility: "FullScreen"
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

// 배경 동영상 컴포넌트 사용
BackgroundVideo {
    anchors.fill: parent
}

// 🔹 버튼이 포함된 UI
ColumnLayout {
    anchors.centerIn: parent
    spacing: 40

    // 🔸 1인 모드 버튼
    Button {
        width: 320
        height: 90
        background: Rectangle { color: "#DD000000"; radius: 12 }
        contentItem: Text {
            text: "🕹️ 1인 모드"
            font.pixelSize: 32
            color: "white"
            anchors.centerIn: parent
            layer.enabled: true
            layer.effect: DropShadow {
                color: "black"
                radius: 8
                samples: 16
                verticalOffset: 2
            }
        }
        onClicked: {
            console.log("🕹️ 1인 모드 버튼 클릭")
            // Python의 controlBridge를 통해 Video_select_control.qml 창을 띄웁니다.
            if (controlBridge) {
                controlBridge.openVideoSelectWindow()
            } else {
                console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
            }
        }
    }

    // 🔸 2인 모드 버튼
    Button {
        width: 320
        height: 90
        background: Rectangle { color: "#DD000000"; radius: 12 }
        contentItem: Text {
            text: "👥 2인 모드"
            font.pixelSize: 32
            color: "white"
            anchors.centerIn: parent
            layer.enabled: true
            layer.effect: DropShadow {
                color: "black"
                radius: 8
                samples: 16
                verticalOffset: 2
            }
        }
        onClicked: {
            console.log("👥 2인 모드 버튼 클릭")
            // 2인 모드 창을 띄우는 로직 추가
            if (controlBridge) {
                 controlBridge.openVideoSelectWindow() // 임시로 같은 창 호출
            } else {
                console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
            }
        }
    }
}

}