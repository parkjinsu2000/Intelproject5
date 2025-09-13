import QtQuick 2.15
import QtMultimedia 5.15
import "./common"

FullWindow {
    id: root

    Video {
        id: video
        anchors.fill: parent
        source: "resource/output.mp4"
        loops: Video.Infinite
        autoPlay: true
        fillMode: Video.PreserveAspectCrop
    }
}
