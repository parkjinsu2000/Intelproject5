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
                radius: 15
                border.width: 2
                border.color: "white"
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#5DADE2" }
                    GradientStop { position: 1.0; color: "#2E86C1" }
                }
                transform: Scale {
                    origin.x: parent.width / 2
                    origin.y: parent.height / 2
                    xScale: playButton.pressed ? 0.95 : 1.0
                    yScale: playButton.pressed ? 0.95 : 1.0
                }
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
            font.pixelSize: 48
            Layout.preferredWidth: 400
            Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter
            visible: false

            background: Rectangle {
                radius: 15
                border.width: 2
                border.color: "white"
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#58D68D" }
                    GradientStop { position: 1.0; color: "#28B463" }
                }
                transform: Scale {
                    origin.x: parent.width / 2
                    origin.y: parent.height / 2
                    xScale: mainMenuButton.pressed ? 0.95 : 1.0
                    yScale: mainMenuButton.pressed ? 0.95 : 1.0
                }
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
