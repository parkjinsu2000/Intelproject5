import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15
import QtQml 2.15

ApplicationWindow {
    id: rootWindow
    property var targetScreen
    property int finalScore: -1
    property bool videoEnabled: true // 비디오 배경 활성화 여부

    visible: true
    // visibility: "FullScreen"
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

    onFinalScoreChanged: {
        console.log("🏆 finalScore changed to:", finalScore)
        if (finalScore >= 0) {
            hideRankTimer.start()
        }
    }

    // 배경 이미지 (게임 중 표시)
    Image {
        id: gameBackgroundImage
        source: "resource/background_control.png"
        anchors.fill: parent
        fillMode: Image.PreserveAspectCrop
        visible: false // 초기에는 숨김
        z: 0
    }

    // 🎬 배경 영상 출력
    VideoOutput {
        id: backgroundVideoOutput
        anchors.fill: parent
        source: backgroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
        z: 0
    }

    // 🎬 안정적인 배경 영상 재생기
    MediaPlayer {
        id: backgroundMediaPlayer
        volume: 0.5
        loops: MediaPlayer.Infinite
        autoPlay: true
        source: "resource/openning_sound.mp4"

        onStatusChanged: {
            if (status === MediaPlayer.Loaded) {
                console.log("✅ 배경 영상 로딩 완료 → 재생 시작")
                if (rootWindow.videoEnabled) {
                    backgroundMediaPlayer.play()
                }
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 배경 영상 경로:", backgroundMediaPlayer.source)
            }
        }

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                if (rootWindow.videoEnabled) {
                    console.log("⏹️ 배경 영상 멈춤 상태 → 다시 재생 시도")
                    backgroundMediaPlayer.play()
                }
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
        z: 1
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
                backgroundMediaPlayer.volume = 0.5; // 배경음 원래대로
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 전경 영상 경로:", foregroundMediaPlayer.source)
                foregroundVideoOutput.visible = false;
                backgroundMediaPlayer.volume = 0.5; // 배경음 원래대로
            }
        }
    }

    // 🏆 랭크 이미지 표시
    Image {
        id: rankImage
        anchors.centerIn: parent
        width: parent.width / 2.5
        height: parent.height / 2.5
        source: {
            if (finalScore >= 80) {
                return "resource/score_S.png";
            } else if (finalScore >= 60) {
                return "resource/score_A.png";
            } else {
                return "resource/score_B.png";
            }
        }
        visible: finalScore >= 0
        fillMode: Image.PreserveAspectFit
        z: 10

        Behavior on opacity { NumberAnimation { duration: 300 } }
    }

    // ⏱️ 랭크 이미지를 숨기는 타이머
    Timer {
        id: hideRankTimer
        interval: 4000 // 4초
        repeat: false
        onTriggered: {
            rootWindow.finalScore = -1
        }
    }

    Loader {
        id: avatarLoader
        anchors.fill: parent
        z: 20
    }


    // 🔧 외부에서 호출 가능한 함수
    function showBackgroundImage() {
        console.log("🖼️ 배경을 이미지로 변경")
        rootWindow.videoEnabled = false
        backgroundMediaPlayer.stop()
        backgroundVideoOutput.visible = false
        gameBackgroundImage.visible = true
    }

    function playVideo(videoPath) {
        console.log("📺 영상 경로 변경 요청:", videoPath)
        backgroundMediaPlayer.volume = 0.1 // 배경음 줄이기
        foregroundMediaPlayer.stop()
        foregroundMediaPlayer.source = videoPath
        foregroundVideoOutput.visible = true
        foregroundMediaPlayer.play()
    }

    function muteBackground(mute) {
        backgroundMediaPlayer.muted = mute;
        console.log("🔊 배경 영상 음소거:", mute)
    }

    function stopForegroundVideo() {
        foregroundMediaPlayer.stop();
        foregroundVideoOutput.visible = false;
        console.log("⏹️ 전경 영상 중지됨");
    }

    function resumeBackgroundVideo() {
        console.log("▶️ 배경 영상 다시 재생 시도")
        if (rootWindow.videoEnabled && backgroundMediaPlayer.playbackState !== MediaPlayer.PlayingState) {
            backgroundMediaPlayer.play();
        }
    }

    function resetToInitialState() {
        console.log("🔄 Main_view를 초기 상태로 리셋합니다.")
        rootWindow.finalScore = -1
        rootWindow.videoEnabled = true
        gameBackgroundImage.visible = false
        backgroundVideoOutput.visible = true
        if (backgroundMediaPlayer.source !== "resource/openning_sound.mp4") {
            backgroundMediaPlayer.source = "resource/openning_sound.mp4"
        }
        backgroundMediaPlayer.volume = 0.5 // 볼륨을 원래대로 복원
        backgroundMediaPlayer.play()
    }

    function showAvatarScreen() {
        console.log("🔄 화면 전환: AvatarSelection.qml 로드")
        avatarLoader.source = "AvatarSelection.qml"
    }
<<<<<<< HEAD
=======

    function showAvatarLoading() {
        console.log("🔄 화면 전환: AvatarLoading.qml 로드")
        avatarLoader.source = "AvatarLoading.qml"
    }

    function clearAvatarLoader() {
        console.log("🧹 아바타 로더 초기화")
        avatarLoader.source = ""
    }
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
}
