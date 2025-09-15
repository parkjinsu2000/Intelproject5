import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    Rectangle {
        anchors.fill: parent
        color: "transparent" // Or another background color if you prefer
    }

    Text {
        anchors.centerIn: parent
        text: "캐릭터 변환 중..."
        color: "white"
        font.pixelSize: 48
    }
}
