import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

def main():
    # Qt 앱 실행
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
