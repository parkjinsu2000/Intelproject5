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
        id: backgroundVideoOutput
        anchors.fill: parent
        source: backgroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }

    // 🎬 안정적인 배경 영상 재생기
    MediaPlayer {
        id: backgroundMediaPlayer
        volume: 0.5
        loops: MediaPlayer.Infinite
        autoPlay: true
        source: "file:///home/ubuntu04/Intelproject5/gui/minwoo/SRF_v1.0.3/resource/openning_sound.mp4"

        onStatusChanged: {
            if (status === MediaPlayer.Loaded) {
                console.log("✅ 배경 영상 로딩 완료 → 재생 시작")
                backgroundMediaPlayer.play()
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 배경 영상 경로:", backgroundMediaPlayer.source)
            }
        }

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                console.log("⏹️ 배경 영상 멈춤 상태 → 다시 재생 시도")
                backgroundMediaPlayer.play()
            }
        }
    }

    // 🎬 전경 영상 출력 (선택된 비디오)
    VideoOutput {
        id: foregroundVideoOutput
        anchors.fill: parent
        source: foregroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectFit
        visible: false
    }

    // 🎬 전경 영상 재생기
    MediaPlayer {
        id: foregroundMediaPlayer
        volume: 1.0 // 배경음보다 크게
        autoPlay: false
        loops: 0 // 한 번만 재생

        onStatusChanged: {
            if (status === MediaPlayer.EndOfMedia) {
                console.log("⏹️ 전경 영상 재생 완료");
                foregroundVideoOutput.visible = false;
                foregroundMediaPlayer.source = ""; // 비디오 언로드
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 전경 영상 경로:", foregroundMediaPlayer.source)
                foregroundVideoOutput.visible = false;
            }
        }
    }


    // 🔧 외부에서 호출 가능한 함수
    function playVideo(videoPath) {
        console.log("📺 영상 경로 변경 요청:", videoPath)
        foregroundMediaPlayer.stop()
        foregroundMediaPlayer.source = videoPath
        foregroundVideoOutput.visible = true
        foregroundMediaPlayer.play()
    }
}