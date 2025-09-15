import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    // 🔹 버튼이 포함된 UI
    ColumnLayout {
        anchors.centerIn: parent
        spacing: 40

        // 🔸 1인 모드 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#DD000000"; radius: 12 }
            contentItem: Text {
                text: "🕹️ 1인 모드"
                font.pixelSize: 32
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: DropShadow {
                    color: "black"
                    radius: 8
                    samples: 16
                    verticalOffset: 2
                }
            }
            onClicked: {
                console.log("🕹️ 1인 모드 버튼 클릭")
                // Python의 controlBridge를 통해 화면 전환 신호를 보냅니다.
                if (controlBridge) {
                    controlBridge.openVideoSelectWindow()
                } else {
                    console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
                }
            }
        }

        // 🔸 2인 모드 버튼
        Button {
            width: 320
            height: 90
            background: Rectangle { color: "#DD000000"; radius: 12 }
            contentItem: Text {
                text: "👥 2인 모드"
                font.pixelSize: 32
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: DropShadow {
                    color: "black"
                    radius: 8
                    samples: 16
                    verticalOffset: 2
                }
            }
            onClicked: {
                console.log("👥 2인 모드 버튼 클릭")
                // 2인 모드 창을 띄우는 로직 추가
                if (controlBridge) {
                     controlBridge.openVideoSelectWindowForMultiplayer() // 임시로 같은 창 호출
                } else {
                    console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
                }
            }
        }
    }
}
