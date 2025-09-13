import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Layouts 1.15

Item {
    id: avatarSelection

    Rectangle {
        anchors.fill: parent
        color: "#80000000" // Semi-transparent background
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20

        Text {
            text: "아바타 선택"
            font.pixelSize: 48
            color: "white"
            Layout.alignment: Qt.AlignHCenter
        }

        RowLayout {
            spacing: 20
            
            // Frog Avatar
            ColumnLayout {
                spacing: 10
                Rectangle {
                    width: 320
                    height: 180
                    color: "black"
                    radius: 10
                    
                    VideoOutput {
                        anchors.fill: parent
                        source: frogPlayer
                    }

                    MediaPlayer {
                        id: frogPlayer
                        source: "resource/videos/frog.mp4"
                        autoPlay: true
                        loops: MediaPlayer.Infinite
                    }
                }
                Text {
                    text: "개구리"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // Jump Avatar
            ColumnLayout {
                spacing: 10
                Rectangle {
                    width: 320
                    height: 180
                    color: "black"
                    radius: 10

                    VideoOutput {
                        anchors.fill: parent
                        source: jumpPlayer
                    }

                    MediaPlayer {
                        id: jumpPlayer
                        source: "resource/videos/jump.mp4"
                        autoPlay: true
                        loops: MediaPlayer.Infinite
                    }
                }
                Text {
                    text: "점프"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // Squat Avatar
            ColumnLayout {
                spacing: 10
                Rectangle {
                    width: 320
                    height: 180
                    color: "black"
                    radius: 10

                    VideoOutput {
                        anchors.fill: parent
                        source: squatPlayer
                    }

                    MediaPlayer {
                        id: squatPlayer
                        source: "resource/videos/squart.mp4"
                        autoPlay: true
                        loops: MediaPlayer.Infinite
                    }
                }
                Text {
                    text: "스쿼트"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }
    }
}
