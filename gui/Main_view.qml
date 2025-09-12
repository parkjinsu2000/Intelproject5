import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15

ApplicationWindow {
    visible: true
    visibility: "FullScreen"
    width: 1920
    height: 1080
    color: "black"

    // 🎬 배경 영상 출력
    VideoOutput {
        id: videoOutput
        anchors.fill: parent
        source: mediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }

    // 🎬 안정적인 영상 재생기
    MediaPlayer {
        id: mediaPlayer
        volume: 0.0
        loops: MediaPlayer.Infinite
        source: "file:///home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_large.mp4"

        Component.onCompleted: mediaPlayer.play()

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                console.log("재생이 멈췄습니다 → 다시 시작")
                mediaPlayer.play()
            }
        }
    }

  
}
