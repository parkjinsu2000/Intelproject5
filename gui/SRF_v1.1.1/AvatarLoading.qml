import QtQuick 2.15
import QtMultimedia 5.15

Item {
    id: avatarLoading

    anchors.fill: parent

    MediaPlayer {
        id: loadingPlayer
        source: "resource/output.mp4"
        autoPlay: true
        loops: MediaPlayer.Infinite
    }

    VideoOutput {
        anchors.fill: parent
        source: loadingPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }
}