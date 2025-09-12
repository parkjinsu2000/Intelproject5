import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Rectangle {
    id: videoSelectScreen
    color: "black"

    // 🔹 콘텐츠 전체
    Item {
        anchors.fill: parent

        // 🔹 썸네일 그리드 영역
        ColumnLayout {
            id: contentLayout
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.topMargin: 80
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

                        // 썸네일 이미지 버튼
                        Item {
                            width: 300
                            height: 480
                            Image {
                                source: model.thumbnail
                                fillMode: Image.PreserveAspectCrop
                                anchors.fill: parent
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    console.log("썸네일 클릭:", model.videoPath)
                                    controlBridge.selectVideo(model.videoPath)
                                }
                            }
                        }

                        // 텍스트 버튼
                        Item {
                            width: 300
                            height: 60
                            Rectangle { anchors.fill: parent; color: "#6680B0E0"; radius: 8 }
                            Text {
                                text: model.name
                                anchors.centerIn: parent
                                font.pixelSize: 18
                                color: "white"
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    console.log("이름 클릭:", model.videoPath)
                                    controlBridge.selectVideo(model.videoPath)
                                }
                            }
                        }
                    }
                }
            }
        }

        // 🔹 하단 중앙의 시작 버튼
        Item {
            id: startButton
            width: 300
            height: 70
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 60    // 아래 여백 조절

            Rectangle { anchors.fill: parent; color: "royalblue"; radius: 10 }
            Text {
                text: "시작하기"
                anchors.centerIn: parent
                font.pixelSize: 20
                color: "white"
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    console.log("게임 시작!")
                    // signalEmitter.startGame()
                }
            }
        }
    }

    // 🔹 비디오 리스트 모델
    ListModel {
        id: videoModel
        ListElement { name: "썸네일 1"; videoPath: "resource/videos/biggibiggi.mp4"; thumbnail: "resource/videos/biggibiggi.png" }
        ListElement { name: "썸네일 2"; videoPath: "resource/videos/frog.mp4"; thumbnail: "resource/videos/frog.png" }
        ListElement { name: "썸네일 3"; videoPath: "resource/videos/jump.mp4"; thumbnail: "resource/videos/jump.png" }
        ListElement { name: "썸네일 4"; videoPath: "resource/videos/naruto.mp4"; thumbnail: "resource/videos/naruto.png" }
        ListElement { name: "썸네일 5"; videoPath: "resource/videos/sodapop.mp4"; thumbnail: "resource/videos/sodapop.png" }
        ListElement { name: "썸네일 6"; videoPath: "resource/videos/tokatoka.mp4"; thumbnail: "resource/videos/tokatoka.png" }
    }
}
