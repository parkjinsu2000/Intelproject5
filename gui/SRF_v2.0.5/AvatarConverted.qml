import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    FontLoader {
      id: neodgm
      source: "resource/fonts/neodgm.ttf"
    }

    Rectangle {
        anchors.fill: parent
        color: "transparent"
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20

        Text {
            text: "변환 완료!"
            font.pixelSize: 48
            color: "white"
            Layout.alignment: Qt.AlignHCenter
            font.family: neodgm.name
            layer.enabled: true
            layer.effect: Glow {
                color: "white"
                radius: 3
            }
        }

        Button {
            id: playButton
            text: "변환된 영상 보기"
            font.pixelSize: 48
            Layout.preferredWidth: 400
            Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter

            background: Rectangle {
                color: "#27AE60"
                radius: 10
            }

            contentItem: Text {
                text: parent.text
                color: "white"
                anchors.centerIn: parent
                font.family: neodgm.name
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }

            onClicked: {
                controlBridge.playConvertedVideo()
                playButton.visible = false
                mainMenuButton.visible = true
            }
        }

        Button {
            id: mainMenuButton
            text: "메인으로 돌아가기"
            font.pixelSize: 32
            Layout.preferredWidth: 400
            Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter
            visible: false

            background: Rectangle {
                color: "#3498DB"
                radius: 10
            }

            contentItem: Text {
                text: parent.text
                color: "white"
                anchors.centerIn: parent
                font.family: neodgm.name
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }

            onClicked: {
                controlBridge.goToMainMenu()
            }
        }
    }
}
