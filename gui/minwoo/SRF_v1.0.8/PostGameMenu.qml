import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    id: postGameMenu
    signal avatarButtonClicked

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
            background: Rectangle { color: "#DD000000"; radius: 12 }
            contentItem: Text {
                text: "아바타"
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
                controlBridge.avatarButtonClicked()
            }
        }

        // 🔸 메뉴 2 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#DD000000"; radius: 12 }
            contentItem: Text {
                text: "메인으로"
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
                console.log("메인으로 버튼 클릭")
            }
        }

        // 🔸 메뉴 3 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#DD000000"; radius: 12 }
            contentItem: Text {
                text: "다시하기"
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
                console.log("다시하기 버튼 클릭")
            }
        }
    }
}
