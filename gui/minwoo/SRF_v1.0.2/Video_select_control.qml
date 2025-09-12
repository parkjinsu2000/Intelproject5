import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15 // Import this to use ApplicationWindow

import "common" // BackgroundVideo를 불러옵니다.

ApplicationWindow {
id: videoSelectScreen
// 🔔 Python에서 연결되는 시그널 송신기
property var signalEmitter
// ✅ 수정: Python에서 전달받을 targetScreen 속성을 추가합니다.
property var targetScreen

visible: true
visibility: "FullScreen"
color: "black"

// ✅ 수정: targetScreen이 설정되면 위치와 해상도 설정
Connections {
    target: videoSelectScreen
    function onTargetScreenChanged() {
        if (targetScreen !== undefined && targetScreen !== null) {
            videoSelectScreen.screen = targetScreen
            videoSelectScreen.x = targetScreen.geometry.x
            videoSelectScreen.y = targetScreen.geometry.y
            videoSelectScreen.width = targetScreen.geometry.width
            videoSelectScreen.height = targetScreen.geometry.height
            console.log("✅ Video_select_control.qml targetScreen 전달됨:", targetScreen.name)
            console.log("🧭 QML에서 적용된 screen 이름:", screen.name)
            console.log("📍 위치:", x, y, "크기:", width, height)
        } else {
            console.log("❗ targetScreen이 전달되지 않았습니다.")
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
                                if (signalEmitter) {
                                    signalEmitter.videoSelected(model.videoPath)
                                }
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
                                if (signalEmitter) {
                                    signalEmitter.videoSelected(model.videoPath)
                                }
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
    ListElement { name: "썸네일 1"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/biggibiggi.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/biggibiggi.png" }
    ListElement { name: "썸네일 2"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/frog.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/frog.png" }
    ListElement { name: "썸네일 3"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/jump.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/jump.png" }
    ListElement { name: "썸네일 4"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/naruto.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/naruto.png" }
    ListElement { name: "썸네일 5"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/sodapop.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/sodapop.png" }
    ListElement { name: "썸네일 6"; videoPath: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/tokatoka.mp4"; thumbnail: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/videos/tokatoka.png" }
}

}