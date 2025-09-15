import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

Item {
    id: avatarSelection

    property int currentIndex: 0
    property var avatars: [narutoAvatar, dadyAvatar, renAvatar, rumiAvatar]
    property var players: [narutoPlayer, dadyPlayer, renPlayer, rumiPlayer]
    property alias selectedIndex: avatarSelection.currentIndex

    onCurrentIndexChanged: {
        pyBridge.onAvatarIndexChanged(currentIndex)
    }

    function updateSelection() {
        for (var i = 0; i < avatars.length; i++) {
            var avatar = avatars[i];
            var player = players[i];

            if (i === currentIndex) {
                avatar.opacity = 1.0;
                avatar.scale = 1.1;
                if (player) player.play();
            } else {
                avatar.opacity = 0.5;
                avatar.scale = 1.0;
                if (player) player.pause();
            }
        }
    }

    function selectNext() {
        currentIndex = (currentIndex + 1) % avatars.length;
        updateSelection();
    }

    function selectPrevious() {
        currentIndex = (currentIndex - 1 + avatars.length) % avatars.length;
        updateSelection();
    }

    Component.onCompleted: {
        updateSelection();
        pyBridge.onAvatarIndexChanged(currentIndex) // 초기 인덱스 전달
    }

    Rectangle {
        anchors.fill: parent
        color: "#80000000" // Semi-transparent background
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 100

        Text {
            text: "아바타 선택"
            font.pixelSize: 48
            color: "white"
            Layout.alignment: Qt.AlignHCenter
        }

        Button {
            text: "변환하기"
            onClicked: {
                // currentIndex는 0,1,2,3
                pyBridge.choose(avatarSelection.currentIndex)
            }
        }

        RowLayout {
            spacing: 50
            
            // 나루토
            ColumnLayout {
                id: narutoAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556
                    height: 720
                    color: "black"
                    radius: 10
                    
                    VideoOutput {
                        anchors.fill: parent
                        source: narutoPlayer
                    }

                    MediaPlayer {
                        id: narutoPlayer
                        source: "resource/naruto_select.mp4"
                        loops: MediaPlayer.Infinite
                        autoPlay: true
                    }
                }
                Text {
                    text: "나루토"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // 신형만
            ColumnLayout {
                id: dadyAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556
                    height: 720
                    color: "black"
                    radius: 10

                    VideoOutput {
                        anchors.fill: parent
                        source: dadyPlayer
                    }

                    MediaPlayer {
                        id: dadyPlayer
                        source: "resource/dady_select.mp4"
                        loops: MediaPlayer.Infinite
                        autoPlay: false
                    }
                }
                Text {
                    text: "신형만"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // 렌고쿠
            ColumnLayout {
                id: renAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556
                    height: 720
                    color: "black"
                    radius: 10

                    VideoOutput {
                        anchors.fill: parent
                        source: renPlayer
                    }

                    MediaPlayer {
                        id: renPlayer
                        source: "resource/ren_select.mp4"
                        loops: MediaPlayer.Infinite
                        autoPlay: false
                    }
                }
                Text {
                    text: "렌고쿠"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // 루미
            ColumnLayout {
                id: rumiAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556
                    height: 720
                    color: "black"
                    radius: 10

                    VideoOutput {
                        anchors.fill: parent
                        source: rumiPlayer
                    }

                    MediaPlayer {
                        id: rumiPlayer
                        source: "resource/rumi_select.mp4"
                        loops: MediaPlayer.Infinite
                        autoPlay: false
                    }
                }
                Text {
                    text: "루미"
                    font.pixelSize: 24
                    color: "white"
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }
    }
}