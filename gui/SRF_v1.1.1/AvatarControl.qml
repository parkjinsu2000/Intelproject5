import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: avatarControl

    RowLayout {
        anchors.centerIn: parent
        spacing: 50 // 두 버튼 그룹 사이의 간격

        // 왼쪽 컨트롤 그룹
        ColumnLayout {
            spacing: 10 // 위아래 버튼 사이의 간격

            Button {
                text: "<"
                font.pixelSize: 20
                width: 80
                height: 50
                Layout.alignment: Qt.AlignHCenter // 큰 버튼 위에 수평으로 중앙 정렬

                background: Rectangle {
                    color: "#5DADE2" // 파란색 배경
                    radius: 8
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("왼쪽 버튼 클릭")
                }
            }

            Button {
                text: "변환하기"
                font.pixelSize: 28
                width: 200
                height: 70
                background: Rectangle {
                    color: "#5DADE2" // 파란색 배경
                    radius: 10
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("변환하기 버튼 클릭")
                }
            }
        }

        // 오른쪽 컨트롤 그룹
        ColumnLayout {
            spacing: 10 // 위아래 버튼 사이의 간격

            Button {
                text: ">"
                font.pixelSize: 20
                width: 80
                height: 50
                Layout.alignment: Qt.AlignHCenter // 큰 버튼 위에 수평으로 중앙 정렬

                background: Rectangle {
                    color: "#5DADE2" // 파란색 배경
                    radius: 8
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("오른쪽 버튼 클릭")
                }
            }

            Button {
                text: "메인으로"
                font.pixelSize: 28
                width: 200
                height: 70
                background: Rectangle {
                    color: "#5DADE2" // 파란색 배경
                    radius: 10
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("메인으로 버튼 클릭")
                }
            }
        }
    }
}