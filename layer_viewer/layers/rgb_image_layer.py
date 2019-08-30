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


class RGBImageLayer(LayerBase):
    def __init__(
        self, name, data=None, autoLevels=True, levels=None, autoHistogramRange=False
    ):
        super(RGBImageLayer, self).__init__(name=name)

        self.m_data = data
        self.m_autoLevels = autoLevels
        self.m_levels = levels
        self.m_autoHistogramRange = autoHistogramRange
        self.m_image_item = pg.ImageItem()
        if self.m_data is not None:
            self.m_image_item.setImage(
                self.m_data, autoLevels=self.m_autoLevels, levels=self.m_levels
            )

        self.m_ctrl_widget = LayerItemWidget(name=self.name, add_gradient_widgtet=False)
        self.viewer = None

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

    def ctrl_widget(self):
        return self.m_ctrl_widget

    def get_image_item(self):
        return self.m_image_item

    def setOpacity(self, opacity):
        self.m_ctrl_widget.setFraction(opacity)
        # self.m_ctrl_widget.bar.update()
        self.m_image_item.setOpacity(opacity)

    def setVisible(self, visible):
        self.m_ctrl_widget.toggleEye.setState(visible)
        self.m_image_item.setVisible(visible)

    def setZValue(self, z):
        self.m_image_item.setZValue(z)

    def updateData(self, image):
        self.m_image_item.updateImage(image)
        self.m_data = image

    def setData(self, image):
        self.m_image_item.setImage(
            image,
            autoLevels=self.m_autoLevels,
            levels=self.m_levels,
            autoHistogramRange=self.m_autoHistogramRange,
        )
        self.m_data = image
