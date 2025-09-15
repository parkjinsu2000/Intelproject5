import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Rectangle {
    id: videoSelectScreen
    color: "black"

    Image{
        source: "resource/background_control.png"
        anchors.fill: parent
        fillMode: Image.PreserveAspectCrop
    }

    Item {
        anchors.fill: parent

        // 🔹 썸네일 가로 리스트
        ListView {
            id: thumbList
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.topMargin: 80
            height: 780                // (이미지 480 + 텍스트 60 + 간격 약간)
            orientation: ListView.Horizontal
            spacing: 40
            model: videoModel
            clip: true                  // 뷰 영역 밖은 잘라내기
            cacheBuffer: 800            // 스크롤 부드럽게

            // 휠로 좌우 스크롤(마우스 휠이 세로 기본이라 좌우로 매핑)
            WheelHandler {
                target: thumbList
                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                onWheel: (event)=> {
                    thumbList.contentX += event.angleDelta.y !== 0
                                          ? -event.angleDelta.y
                                          : -event.angleDelta.x
                }
            }

            ScrollBar.horizontal: ScrollBar {
                policy: ScrollBar.AlwaysOff
                visible: false
            }

            delegate: Item {
                width: 400
                height: 550

                Column {
                    anchors.fill: parent
                    spacing: 10

                    // 썸네일 이미지 버튼
                    Item {
                        width: 400
                        height: 700
                        Image {
                            anchors.fill: parent
                            source: model.thumbnail
                            fillMode: Image.PreserveAspectCrop
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
                        width: 400
                        height: 60
                        Rectangle { anchors.fill: parent; color: "#6680B0E0"; radius: 8 }
                        Text {
                            text: model.name
                            anchors.centerIn: parent
                            font.pixelSize: 18
                            color: "white"
                            elide: Text.ElideRight
                            horizontalAlignment: Text.AlignHCenter
                            width: parent.width - 20
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

        // 🔹 하단 중앙의 시작 버튼
        Item {
            id: startButton
            width: 300
            height: 70
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 60

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
        ListElement { name: "빼빼빼기"; videoPath: "resource/videos/biggibiggi.mp4"; thumbnail: "resource/videos/biggibiggi.png" }
        ListElement { name: "개구리"; videoPath: "resource/videos/frog.mp4"; thumbnail: "resource/videos/frog.png" }
        ListElement { name: "뛰어"; videoPath: "resource/videos/jump.mp4"; thumbnail: "resource/videos/jump.png" }
        ListElement { name: "나루토"; videoPath: "resource/videos/naruto.mp4"; thumbnail: "resource/videos/naruto.png" }
        ListElement { name: "소다팝"; videoPath: "resource/videos/sodapop.mp4"; thumbnail: "resource/videos/sodapop.png" }
        ListElement { name: "토카토카"; videoPath: "resource/videos/tokatoka.mp4"; thumbnail: "resource/videos/tokatoka.png" }
    }
}
