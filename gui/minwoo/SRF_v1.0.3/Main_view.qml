import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15
import QtQml 2.15

ApplicationWindow {
    id: rootWindow
    property var targetScreen

    visible: true
    visibility: "FullScreen"
    color: "black"

    // ✅ targetScreen이 전달되면 위치와 해상도 설정
    Connections {
        target: rootWindow
        function onTargetScreenChanged() {
            if (targetScreen !== undefined && targetScreen !== null) {
                rootWindow.screen = targetScreen
                rootWindow.x = targetScreen.geometry.x
                rootWindow.y = targetScreen.geometry.y
                rootWindow.width = targetScreen.geometry.width
                rootWindow.height = targetScreen.geometry.height
                console.log("✅ targetScreen 전달됨:", targetScreen.name)
                console.log("🧭 QML에서 적용된 screen 이름:", screen.name)
                console.log("📍 위치:", x, y, "크기:", width, height)
            } else {
                console.log("❗ targetScreen이 전달되지 않았습니다.")
            }
        }
    }

    // 🎬 배경 영상 출력
    VideoOutput {
        id: videoOutput
        anchors.fill: parent
        source: mediaPlayer
        fillMode: VideoOutput.PreserveAspectFit
    }

    // 🎬 안정적인 영상 재생기
    MediaPlayer {
        id: mediaPlayer
        volume: 0.5
        loops: MediaPlayer.Infinite
        autoPlay: true
        source: "file:///home/ubuntu04/Intelproject5/gui/minwoo/SRF_v1.0.3/resource/openning_sound.mp4"  // ✅ 초기 영상 경로 지정

        onStatusChanged: {
            if (status === MediaPlayer.Loaded) {
                console.log("✅ 초기 영상 로딩 완료 → 재생 시작")
                mediaPlayer.play()
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 초기 영상 경로:", mediaPlayer.source)
            }
        }

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                console.log("⏹️ 멈춤 상태 → 다시 재생 시도")
                mediaPlayer.play()
            }
        }

        onSourceChanged: {
            console.log("🔄 source 변경됨:", mediaPlayer.source)
        }
    }

    // 🔧 외부에서 호출 가능한 함수
    function playVideo(videoPath) {
        console.log("📺 영상 경로 변경 요청:", videoPath)
        mediaPlayer.stop()
        mediaPlayer.source = ""
        mediaPlayer.source = videoPath
    }
}
