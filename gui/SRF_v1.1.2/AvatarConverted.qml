import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    Rectangle {
        anchors.fill: parent
        color: "transparent"
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20

        Text {
            text: "변환 완료!"
            font.pixelSize: 48
            color: "white"
            Layout.alignment: Qt.AlignHCenter
        }

        Button {
            text: "변환된 영상 보기"
            font.pixelSize: 32
            Layout.preferredWidth: 400
            Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter

            background: Rectangle {
                color: "#27AE60" // A nice green color
                radius: 10
            }

            contentItem: Text {
                text: parent.text
                color: "white"
                font: parent.font
                anchors.centerIn: parent
            }

            onClicked: {
                controlBridge.playConvertedVideo()
            }
        }
    }
}
