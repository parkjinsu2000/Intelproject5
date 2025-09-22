import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    FontLoader {
      id: neodgm
      source: "resource/fonts/neodgm.ttf"
    }

    // 🔹 버튼이 포함된 UI
    RowLayout {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 80
        spacing: 100

        // 🔸 메뉴 1 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "아바타"
                font.family: neodgm.name
                font.pixelSize: 32
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }
            onClicked: {
                controlBridge.avatarButtonClicked()
            }
        }

        // 🔸 메뉴 2 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "메인으로"
                font.family: neodgm.name
                font.pixelSize: 32
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }
            onClicked: {
                console.log("메인으로 버튼 클릭")
                controlBridge.goToMainMenu()
            }
        }

        // 🔸 메뉴 3 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "다시하기"
                font.family: neodgm.name
                font.pixelSize: 32
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }
            onClicked: {
                console.log("다시하기 버튼 클릭")
                controlBridge.retryGame()
            }
        }
    }
}
