#!/usr/bin/python3
import sys
from PySide2.QtWidgets import QApplication

from dex import DexWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dex = DexWindow()
    dex.show()
    sys.exit(app.exec_())