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


class MultiChannelImageLayer(LayerBase):
    class CtrlWidget(LayerItemWidget):
        def __init__(self, name):
            super().__init__(
                name=name, add_gradient_widgtet=True, channel_selector=True
            )

            self.asRgb = QtGui.QCheckBox()
            self.asRgb.setToolTip("Show As RGB")
            self._layout.addWidget(self.asRgb, 3, 0)

    def __init__(
        self,
        name,
        data=None,
        autoLevels=True,
        levels=None,
        autoHistogramRange=False,
        cmap=None,
    ):
        super().__init__(name=name)

        self.current_channel = 0
        self.as_rgb = False
        self.m_data = data
        self.m_autoLevels = autoLevels
        self.m_levels = levels
        self.m_autoHistogramRange = autoHistogramRange
        self.m_image_item = pg.ImageItem()
        if self.m_data is not None:

            self.m_image_item.setImage(
                self.m_data[..., self.current_channel],
                autoLevels=self.m_autoLevels,
                levels=self.m_levels,
            )

        self.m_ctrl_widget = MultiChannelImageLayer.CtrlWidget(name=self.name)
        self.viewer = None

        self.cmap = cmap
        if self.cmap is not None:
            self.m_ctrl_widget.gradientWidget.loadPreset(self.cmap)

        # setup ctrl widget
        w = self.m_ctrl_widget
        if self.m_data is not None:
            w.channelSelector.setRange(0, self.m_data.shape[2] - 1)
        w.toggleEye.setActive(True)

        def toogleEyeChanged(state):
            if self.viewer.m_exlusive_layer is not None:
                self.viewer.m_exlusive_layer.setVisible(True)
                self.viewer.m_exlusive_layer = None
            if state == 2:
                self.viewer.showAndHideOthers(self.name)
            else:
                self.setVisible(bool(state))

        w.toggleEye.stateChanged.connect(toogleEyeChanged)

        def channelChanged(channel):
            if not self.as_rgb:
                self.current_channel = channel
                if self.m_data is not None:
                    self.m_image_item.setImage(
                        self.m_data[..., self.current_channel],
                        autoLevels=self.m_autoLevels,
                        levels=self.m_levels,
                        autoHistogramRange=self.m_autoHistogramRange,
                    )

        w.channelSelector.valueChanged.connect(channelChanged)

        def asRgbChanged(as_rgb):
            self.as_rgb = bool(as_rgb)
            self.m_ctrl_widget.channelSelector.setEnabled(not self.as_rgb)
            self.m_ctrl_widget.gradientWidget.setEnabled(not self.as_rgb)
            if self.as_rgb:
                self.m_image_item.setLookupTable(None)
                self.m_image_item.setImage(
                    self.m_data,
                    autoLevels=self.m_autoLevels,
                    levels=self.m_levels,
                    autoHistogramRange=self.m_autoHistogramRange,
                )
            else:
                lut = w.gradientWidget.getLookupTable(512)
                self.m_image_item.setLookupTable(lut)
                self.m_image_item.setImage(
                    self.m_data[..., self.current_channel],
                    autoLevels=self.m_autoLevels,
                    levels=self.m_levels,
                    autoHistogramRange=self.m_autoHistogramRange,
                )

        w.asRgb.stateChanged.connect(asRgbChanged)
        w.bar.fractionChanged.connect(self.setOpacity)

        def update():
            lut = w.gradientWidget.getLookupTable(512)
            self.m_image_item.setLookupTable(lut)

        w.gradientWidget.sigGradientChanged.connect(update)
        w.layer = self

    def ctrl_widget(self):
        return self.m_ctrl_widget

    def get_image_item(self):
        return self.m_image_item

    def updateData(self, image):
        self.m_data = image
        self.m_ctrl_widget.channelSelector.setRange(0, self.m_data.shape[2] - 1)
        self.m_image_item.updateImage(image[..., self.current_channel])
        self.update_rgb_enabled()

    def setData(self, image):
        self.m_data = image

        if image is None:
            self.m_image_item.clear()
            # self.m_image_item.setImage(None, autoLevels=self.m_autoLevels,
            #         levels=self.m_levels, autoHistogramRange=self.m_autoHistogramRange)
        else:
            self.m_ctrl_widget.channelSelector.setRange(0, self.m_data.shape[2] - 1)

            if self.as_rgb:
                self.m_image_item.setImage(
                    image,
                    autoLevels=self.m_autoLevels,
                    levels=self.m_levels,
                    autoHistogramRange=self.m_autoHistogramRange,
                )
            else:
                self.m_image_item.setImage(
                    image[..., self.current_channel],
                    autoLevels=self.m_autoLevels,
                    levels=self.m_levels,
                    autoHistogramRange=self.m_autoHistogramRange,
                )
        self.update_rgb_enabled()

    def update_rgb_enabled(self):
        if len(self.m_data.shape) == 3 and self.m_data.shape[2] > 4:
            self.m_ctrl_widget.asRgb.setEnabled(False)
        else:
            self.m_ctrl_widget.asRgb.setEnabled(True)
