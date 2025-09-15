import QtQuick 2.15
import QtQuick.Window 2.15


Window {
    screen: targetScreen
    x: targetScreen.geometry.x
    y: targetScreen.geometry.y
    width: targetScreen.geometry.width
    height: targetScreen.geometry.height
    visible: true
    visibility: "FullScreen"
    flags: Qt.FramelessWindowHint
}