import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15

Window {
    id: appWindow
    screen: targetScreen
    x: targetScreen.geometry.x
    y: targetScreen.geometry.y
    width: targetScreen.geometry.width
    height: targetScreen.geometry.height
    visible: true
    visibility: "FullScreen"
    flags: Qt.FramelessWindowHint
    color: "black"

    // 🎬 배경 영상
    VideoOutput {
        id: videoOutput
        anchors.fill: parent
        source: mediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }

    MediaPlayer {
        id: mediaPlayer
        volume: 0.0
        loops: MediaPlayer.Infinite
        source: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_pade.mp4"

        Component.onCompleted: mediaPlayer.play()

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                mediaPlayer.play()
            }
        }
    }

    // 🔹 StackView: 이 윈도우의 모든 화면 전환을 관리합니다.
    StackView {
        id: screenStackView
        anchors.fill: parent
        initialItem: initialScreen

        Component {
            id: initialScreen

            Item {
                anchors.fill: parent

                Column {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.top: parent.top
                    anchors.topMargin: 700
                    spacing: 40

                    // 🔸 1인 모드 버튼
                    Button {
                        width: 320
                        height: 90
                        background: Rectangle {
                            color: "#DD000000"
                            radius: 12
                        }
                        contentItem: Item {
                            anchors.fill: parent
                            Text {
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
                        }
                        onClicked: {
                            screenStackView.push("Video_select_control.qml") // 기존 기능 유지
                            controlBridge.openVideoSelectWindow() // 새 창 띄우기
                        }
                    }

                    // 🔸 2인 모드 버튼
                    Button {
                        width: 320
                        height: 90
                        background: Rectangle {
                            color: "#DD000000"
                            radius: 12
                        }
                        contentItem: Item {
                            anchors.fill: parent
                            Text {
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
                        }
                        onClicked: {
                            screenStackView.push("Video_select_control.qml") // 기존 기능 유지
                            controlBridge.openVideoSelectWindow() // 새 창 띄우기
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        console.log("🧭 QML에서 적용된 screen 이름:", screen.name)
        console.log("📍 위치:", x, y, "크기:", width, height)
    }
}
