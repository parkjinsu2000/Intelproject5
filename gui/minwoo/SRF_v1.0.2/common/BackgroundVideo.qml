import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Controls 2.15

Item {
    id: root

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
        source: "file:///home/ubuntu/Qt/SRF_v1.0.2/resource/background_video_pade.mp4"

        Component.onCompleted: mediaPlayer.play()

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                console.log("재생이 멈췄습니다 → 다시 시작")
                mediaPlayer.play()
            }
        }
    }
}