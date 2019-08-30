from .layer_base import LayerBase
from ..widgets import TripleToggleEye, ToggleEye, FractionSelectionBar
from ..pixel_path import *
from .layer_controller import *
import pyqtgraph as pg
import os
from pyqtgraph.Qt import QtCore, QtGui
import numpy

###############################################################################
from builtins import range

# from past.utils import old_div
import warnings
from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
    QEvent,
    QRect,
    QSize,
    QTimer,
    QPoint,
    QItemSelectionModel,
)
from PyQt5.QtGui import QPainter, QFontMetrics, QFont, QPalette, QMouseEvent, QPixmap
from PyQt5.QtWidgets import (
    QStyledItemDelegate,
    QWidget,
    QListView,
    QStyle,
    QLabel,
    QGridLayout,
    QSpinBox,
    QApplication,
)


class BinaryObjectLayer(LayerBase):
    def __init__(self, name, data=None, color=None):
        super(BinaryObjectLayer, self).__init__(name=name)

        # if color is None:
        #     s4 = (lut_size + with_background) * 4
        #     color = numpy.random.randint(low=0, high=255, size=3)

        lut = numpy.zeros([2, 4])
        lut[:, 0:3] = color
        lut[:, 3] = 255
        lut[0, 3] = 0

        self.lut = lut.astype("int64")

        self.m_data = data

        self.m_image_item = pg.ImageItem()
        if self.m_data is not None:

            self.m_image_item.setImage(self._apply_lut(self.m_data), autoLevels=False)

        self.m_ctrl_widget = LayerItemWidget(name=self.name, add_gradient_widgtet=False)
        # self.m_ctrl_widget.setLut(lut)
        self.viewer = None

        w = self.m_ctrl_widget
        self.m_ctrl_widget.toggleEye.setActive(True)

        def toogleEyeChanged(state):
            if self.viewer.m_exlusive_layer is not None:
                self.viewer.m_exlusive_layer.setVisible(True)
                self.viewer.m_exlusive_layer = None
            if state == 2:
                self.viewer.showAndHideOthers(self.name)
            else:
                self.setVisible(bool(state))

        self.m_ctrl_widget.toggleEye.stateChanged.connect(toogleEyeChanged)

        self.m_ctrl_widget.bar.fractionChanged.connect(self.setOpacity)

        self.m_ctrl_widget.layer = self

    def _apply_lut(self, image):
        image = image.astype("int64")
        img = numpy.take(self.lut, image, axis=0, mode="clip").astype("uint8")
        return img

    def ctrl_widget(self):
        return self.m_ctrl_widget

    def get_image_item(self):
        return self.m_image_item

    def setOpacity(self, opacity):
        self.m_ctrl_widget.setFraction(opacity)
        self.m_image_item_group.setOpacity(opacity)

    def setVisible(self, visible):
        self.m_ctrl_widget.toggleEye.setState(visible)
        self.m_image_item.setVisible(visible)

    def setZValue(self, z):
        self.m_image_item.setZValue(z)

    def updateData(self, image):
        self.m_image_item.updateImage(self._apply_lut(image))
        self.m_data = image

    def setData(self, image):
        self.m_image_item.setImage(self._apply_lut(image), autoLevels=False)
        self.m_data = image
