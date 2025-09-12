import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15

Window {
    id: videoWindow
    screen: targetScreen
    x: targetScreen.geometry.x
    y: targetScreen.geometry.y
    width: targetScreen.geometry.width
    height: targetScreen.geometry.height
    visible: true
    visibility: "FullScreen"
    color: "black"

    // 🎬 배경 영상 출력
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
                console.log("재생이 멈췄습니다 → 다시 시작")
                mediaPlayer.play()
            }
        }
    }

    // 🔹 콘텐츠 오버레이
    Item {
        anchors.fill: parent

        ColumnLayout {
            anchors.centerIn: parent
            spacing: 60

            GridLayout {
                id: grid
                columns: 3
                rowSpacing: 40
                columnSpacing: 40

                Repeater {
                    model: videoModel

                    delegate: ColumnLayout {
                        spacing: 10

                        Button {
                            width: 300
                            height: 240
                            background: Image {
                                source: model.thumbnail
                                fillMode: Image.PreserveAspectCrop
                                anchors.fill: parent
                            }
                            onClicked: {
                                console.log("썸네일 클릭:", model.videoPath)
                            }
                        }

                        Button {
                            width: 300
                            height: 60
                            text: model.name
                            font.pixelSize: 18
                            background: Rectangle {
                                color: "#6680B0E0"
                                radius: 8
                            }
                            onClicked: {
                                console.log("이름 클릭:", model.videoPath)
                            }
                        }
                    }
                }
            }

            Button {
                width: 300
                height: 70
                text: "시작하기"
                font.pixelSize: 20
                background: Rectangle {
                    color: "royalblue"
                    radius: 10
                }
                onClicked: {
                    console.log("게임 시작!")
                }
            }
        }

        Text {
            text: "✅ 화면 표시 테스트"
            color: "white"
            font.pixelSize: 40
            anchors.centerIn: parent
        }
    }

    ListModel {
        id: videoModel

        ListElement {
            name: "썸네일 1"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/biggibiggi.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/biggibiggi.png"
        }
        ListElement {
            name: "썸네일 2"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/frog.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/frog.png"
        }
        ListElement {
            name: "썸네일 3"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/jump.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/jump.png"
        }
        ListElement {
            name: "썸네일 4"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/naruto.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/naruto.png"
        }
        ListElement {
            name: "썸네일 5"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/sodapop.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/sodapop.png"
        }
        ListElement {
            name: "썸네일 6"
            videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/tokatoka.mp4"
            thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/tokatoka.png"
        }
    }

    Component.onCompleted: {
        console.log("🧭 screen 이름:", screen.name)
        console.log("📍 위치:", x, y, "크기:", width, height)
    }
}
