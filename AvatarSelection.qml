import QtQuick 2.15
import QtMultimedia 5.15 // MediaPlayer를 안쓰지만 혹시 모르니 남겨둡니다.
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

Item {
    id: avatarSelection

    property int currentIndex: 0
    property var avatars: [narutoAvatar, dadyAvatar, renAvatar, rumiAvatar]
    // MediaPlayer를 사용하지 않으므로 players 속성은 제거합니다.
    property alias selectedIndex: avatarSelection.currentIndex

    onCurrentIndexChanged: {
        pyBridge.onAvatarIndexChanged(currentIndex)
    }

    // AnimatedImage를 제어하도록 수정한 함수
    function updateSelection() {
        // 각 캐릭터의 AnimatedImage id를 배열로 만듭니다.
        var animations = [narutoAnimation, dadyAnimation, renAnimation, rumiAnimation];

        for (var i = 0; i < avatars.length; i++) {
            var avatar = avatars[i];
            var animation = animations[i];

            if (i === currentIndex) {
                avatar.opacity = 1.0;
                avatar.scale = 1.1;
                if (animation) animation.playing = true; // 선택되면 재생
            } else {
                avatar.opacity = 0.5;
                avatar.scale = 1.0;
                if (animation) animation.playing = false; // 선택 안되면 멈춤
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
        pyBridge.onAvatarIndexChanged(currentIndex)
    }

    Rectangle {
        anchors.fill: parent
        color: "#80000000" // 반투명 배경
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
                    width: 556; height: 720; color: "transparent"
                    
                    AnimatedImage {
                        id: narutoAnimation
                        anchors.fill: parent
                        source: "resource/naruto.gif" // GIF 파일 경로
                        playing: false // 처음에는 멈춤
                    }
                }
                Text { text: "나루토"; font.pixelSize: 24; color: "white"; Layout.alignment: Qt.AlignHCenter }
            }

            // 신형만
            ColumnLayout {
                id: dadyAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556; height: 720; color: "transparent"
                    AnimatedImage {
                        id: dadyAnimation
                        anchors.fill: parent
                        source: "resource/daddy.gif"
                        playing: false
                    }
                }
                Text { text: "신형만"; font.pixelSize: 24; color: "white"; Layout.alignment: Qt.AlignHCenter }
            }

            // 렌고쿠
            ColumnLayout {
                id: renAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556; height: 720; color: "transparent"
                    AnimatedImage {
                        id: renAnimation
                        anchors.fill: parent
                        source: "resource/ren.gif"
                        playing: false
                    }
                }
                Text { text: "렌고쿠"; font.pixelSize: 24; color: "white"; Layout.alignment: Qt.AlignHCenter }
            }

            // 루미
            ColumnLayout {
                id: rumiAvatar
                spacing: 10
                Behavior on opacity { NumberAnimation { duration: 200 } }
                Behavior on scale { ScaleAnimator { duration: 200 } }

                Rectangle {
                    width: 556; height: 720; color: "transparent"
                    AnimatedImage {
                        id: rumiAnimation
                        anchors.fill: parent
                        source: "resource/rumi.gif"
                        playing: false
                    }
                }
                Text { text: "루미"; font.pixelSize: 24; color: "white"; Layout.alignment: Qt.AlignHCenter }
            }
        }
    }
}