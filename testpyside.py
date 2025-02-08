import sys
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Test Window")
window.resize(400, 300)
window.show()
sys.exit(app.exec())
