
import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

Item {
    id: avatarLoading

    FontLoader {
      id: neodgm
      source: "resource/fonts/neodgm.ttf"
    }

    property real conversionProgress: 0 // Value from 0.0 to 1.0

    anchors.fill: parent

    // 배경 비디오
    MediaPlayer {
        id: loadingPlayer
        source: "resource/output.mp4"
        autoPlay: true
        loops: MediaPlayer.Infinite
        volume: 0.3 // 배경음이므로 소리를 약간 줄입니다.
    }

    VideoOutput {
        anchors.fill: parent
        source: loadingPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }

    // 진행 상태를 표시하기 위한 반투명 오버레이와 프로그레스 바
    Rectangle {
        anchors.fill: parent
        color: "#80000000" // 50% black
    }

    Column {
        width: parent.width * 0.7
        spacing: 20
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: parent.height * 0.2 // 하단에서 20% 위에 위치

        Text {
            text: "캐릭터 변환 중..."
            color: "white"
            font.pixelSize: 48
            anchors.horizontalCenter: parent.horizontalCenter
            font.family: neodgm.name
            layer.enabled: true
            layer.effect: Glow {
                color: "white"
                radius: 3
            }
        }

        ProgressBar {
            id: progressBar
            width: parent.width
            value: avatarLoading.conversionProgress

            background: Rectangle {
                color: "#4A4A4A"
                radius: height / 2
                implicitHeight: 20
            }

            contentItem: Item {
                Rectangle {
                    width: progressBar.visualPosition * progressBar.width
                    height: progressBar.height
                    color: "#5DADE2"
                    radius: height / 2
                }
            }
        }
    }
}
