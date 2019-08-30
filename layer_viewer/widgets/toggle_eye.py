import pyqtgraph as pg
import os
from pyqtgraph.Qt import QtCore, QtGui


class ToggleEye(QtGui.QLabel):
    activeChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super(ToggleEye, self).__init__(parent=parent)
        self._active = True

        path = os.path.dirname(os.path.abspath(__file__))

        self._eye_open = QtGui.QPixmap(os.path.join(path, "icons/stock-eye-20.png"))
        self._eye_closed = QtGui.QPixmap(
            os.path.join(path, "icons/stock-eye-20-gray.png")
        )
        self.setPixmap(self._eye_open)

    def active(self):
        return self._active

    def setActive(self, b):
        if b == self._active:
            return
        self._active = b
        if b:
            self.setPixmap(self._eye_open)
        else:
            self.setPixmap(self._eye_closed)

    def toggle(self):
        if self.active():
            self.setActive(False)
        else:
            self.setActive(True)

    def mousePressEvent(self, ev):
        self.toggle()
        self.activeChanged.emit(self._active)


class TripleToggleEye(QtGui.QLabel):
    stateChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(TripleToggleEye, self).__init__(parent=parent)

        self._state = 1
        path = os.path.dirname(os.path.abspath(__file__))

        self._eye_open = QtGui.QPixmap(os.path.join(path, "icons/stock-eye-20.png"))
        self._eye_closed = QtGui.QPixmap(
            os.path.join(path, "icons/stock-eye-20-gray.png")
        )
        self._eye_open_exclusive = QtGui.QPixmap(
            os.path.join(path, "icons/stock-eye-green.png")
        )

        self._state_to_pixmap = {
            0: self._eye_closed,
            1: self._eye_open,
            2: self._eye_open_exclusive,
        }
        self.setPixmap(self._eye_open)
        self.last = 1

    def mousePressEvent(self, event):
        self.last = 1

    def mouseDoubleClickEvent(self, event):
        self.last = 2

    def setState(self, state):
        if state != self._state:
            self._state = state
            self.setPixmap(self._state_to_pixmap[self._state])

    def setActive(self, binary_state):
        self.setState(bool(binary_state))

    def state(self):
        return self._state

    def mouseReleaseEvent(self, event):

        if self.last == 1:
            if self._state == 0:
                self._state = 1

            elif self._state == 1 or self._state == 2:
                self._state = 0
        else:
            self._state = 2

        self.setPixmap(self._state_to_pixmap[self._state])
        # print("emit change ",self._state)
        self.stateChanged.emit(self._state)
