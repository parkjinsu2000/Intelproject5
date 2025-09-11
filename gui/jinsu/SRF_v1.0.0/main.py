# main.py 파일
from PyQt5.QtWidgets import QApplication, QMainWindow  # QMainWindow를 불러옵니다.

def main():
    app = QApplication([])
    window = QMainWindow()  # MainWindow를 QMainWindow로 수정합니다.
    window.show()
    app.exec_() # PyQt5는 exec_()를 사용합니다.

if __name__ == "__main__":
    main()