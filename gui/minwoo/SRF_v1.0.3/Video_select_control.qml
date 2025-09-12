import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

// import "common" // BackgroundVideo를 불러옵니다.

Rectangle {
    id: videoSelectScreen
    color: "black"

    Image {
        source: "resource/background_control.png"
        anchors.fill: parent
        fillMode: Image.PreserveAspectCrop
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

                        // 썸네일 이미지 버튼
                        Item {
                            width: 300
                            height: 240
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

            // 시작 버튼
            Item {
                width: 300
                height: 70
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
    }

    // ListModel은 이 화면에 고유한 로직이므로 여기에 유지합니다.
    ListModel {
        id: videoModel
        // ... (기존 ListElement 데이터 유지) ...
        ListElement { name: "썸네일 1"; videoPath: "resource/videos/biggibiggi.mp4"; thumbnail: "resource/videos/biggibiggi.png" }
        ListElement { name: "썸네일 2"; videoPath: "resource/videos/frog.mp4"; thumbnail: "resource/videos/frog.png" }
        ListElement { name: "썸네일 3"; videoPath: "resource/videos/jump.mp4"; thumbnail: "resource/videos/jump.png" }
        ListElement { name: "썸네일 4"; videoPath: "resource/videos/naruto.mp4"; thumbnail: "resource/videos/naruto.png" }
        ListElement { name: "썸네일 5"; videoPath: "resource/videos/sodapop.mp4"; thumbnail: "resource/videos/sodapop.png" }
        ListElement { name: "썸네일 6"; videoPath: "resource/videos/tokatoka.mp4"; thumbnail: "resource/videos/tokatoka.png" }
    }
}