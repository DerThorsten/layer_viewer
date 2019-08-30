from .layer_base import LayerBase
from ..distinct_colors import *
from ..pixel_path import *
from .layer_controller import *
from ..widgets import TripleToggleEye, ToggleEye, FractionSelectionBar, GradientWidget
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


class LabelLayer(LayerBase):
    labelsChangedSignal = pyqtSignal(object)

    # the gui for the user in the layer stack
    class CtrlWidget(QWidget):
        def __init__(self, name=None, parent=None, current_label=1, disc_rad=1):
            super(LabelLayer.CtrlWidget, self).__init__(parent=parent)
            # self._layer = None

            self._font = QFont(QFont().defaultFamily(), 9)
            self._fm = QFontMetrics(self._font)
            self.bar = FractionSelectionBar(initial_fraction=1.0)
            self.bar.setFixedHeight(10)
            self.nameLabel = QLabel(parent=self)
            self.nameLabel.setFont(self._font)
            self.nameLabel.setText(str(name))
            self.opacityLabel = QLabel(parent=self)
            self.opacityLabel.setAlignment(Qt.AlignRight)
            self.opacityLabel.setFont(self._font)
            self.opacityLabel.setText(
                "\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction()))
            )

            self.toggleEye = TripleToggleEye(parent=self)
            self.toggleEye.setActive(True)
            self.toggleEye.setFixedWidth(35)
            self.toggleEye.setToolTip("Visibility")

            self.channelSelector = QSpinBox(parent=self)
            self.channelSelector.setFrame(False)
            self.channelSelector.setFont(self._font)
            self.channelSelector.setMaximumWidth(35)
            self.channelSelector.setAlignment(Qt.AlignRight)
            self.channelSelector.setToolTip("Channel")
            self.channelSelector.setVisible(False)

            self._layout = QtGui.QGridLayout(self)
            self._layout.addWidget(self.toggleEye, 0, 0)
            self._layout.addWidget(self.nameLabel, 0, 1)
            self._layout.addWidget(self.opacityLabel, 0, 2)
            self._layout.addWidget(self.channelSelector, 1, 0)

            self._layout.addWidget(self.bar, 1, 1, 1, 2)

            self.labelLabel = QLabel(parent=self)
            self.labelLabel.setAlignment(Qt.AlignRight)
            self.labelLabel.setFont(self._font)
            self.labelLabel.setText("Label:")
            self._layout.addWidget(self.labelLabel, 3, 0, 1, 1)

            self.labelSelector = QSpinBox(parent=self)
            self.labelSelector.setFrame(False)
            self.labelSelector.setFont(self._font)
            self.labelSelector.setMaximumWidth(35)
            self.labelSelector.setAlignment(Qt.AlignRight)
            self.labelSelector.setToolTip("Label")
            self.labelSelector.setVisible(True)
            self.labelSelector.setMinimum(0)
            self.labelSelector.setMaximum(255)
            self.labelSelector.setValue(current_label)
            self._layout.addWidget(self.labelSelector, 3, 1, 1, 1)

            self.brushLabel = QLabel(parent=self)
            self.brushLabel.setAlignment(Qt.AlignRight)
            self.brushLabel.setFont(self._font)
            self.brushLabel.setText("Brush:")
            self._layout.addWidget(self.brushLabel, 3, 2, 1, 1)

            self.brushSelector = QSpinBox(parent=self)
            self.brushSelector.setFrame(False)
            self.brushSelector.setFont(self._font)
            self.brushSelector.setMaximumWidth(35)
            self.brushSelector.setAlignment(Qt.AlignRight)
            self.brushSelector.setToolTip("Brush")
            self.brushSelector.setVisible(True)
            self.brushSelector.setMinimum(0)
            self.brushSelector.setMaximum(255)
            self.brushSelector.setValue(disc_rad)
            self._layout.addWidget(self.brushSelector, 3, 3, 1, 1)

            self._layout.setColumnMinimumWidth(2, 35)
            self._layout.setSpacing(0)
            self.setLayout(self._layout)

            def f(frac):
                self.opacityLabel.setText(
                    "\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction()))
                )

            self.bar.fractionChanged.connect(f)

        def setFraction(self, opacity):
            self.bar.setFraction(opacity)
            self.opacityLabel.setText(
                "\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction()))
            )

        def getCurrentLabel(self):
            return self.labelSelector.value

        def setNumClasses(self, setNumClasses):
            self.labelSelector.setRange(0, setNumClasses)

        def setName(self, name):
            self.nameLabel.setText(str(name))

        def mousePressEvent(self, ev):
            super(LabelLayer.CtrlWidget, self).mousePressEvent(ev)

    # we need a special image item to handle mouse interaction
    class LabelLayerImageItem(pg.ImageItem):
        def __init__(self, label_layer):
            super(LabelLayer.LabelLayerImageItem, self).__init__()
            self.label_layer = label_layer
            self.m_label_data = label_layer.m_label_data
            self._pixel_path = label_layer._pixel_path

            # otherwise we do not get key events?
            self.setFlag(self.ItemIsFocusable, True)

        def mouseClickEvent(self, ev):
            pass

        def mouseDragEvent(self, ev):
            modifiers = QtGui.QApplication.keyboardModifiers()
            if modifiers != QtCore.Qt.ShiftModifier:

                if ev.isStart():

                    self._pixel_path.clear()
                    self._pixel_path.add(ev.pos())

                elif ev.isFinish():
                    # add the labels
                    self._pixel_path.insert_to_image(
                        label_image=self.m_label_data,
                        label=self.label_layer.current_label,
                        rad=self.label_layer.disk_rad
                        * [1, 3][self.label_layer.current_label == 0],
                    )
                    # update image
                    self.setImage(self.m_label_data, autoLevels=False)
                    self.label_layer.m_temp_path.setPath(QtGui.QPainterPath())

                    # send signal
                    self.label_layer.labelsChangedSignal.emit(self.label_layer)

                else:
                    self._pixel_path.add(ev.pos())
                    self.label_layer.m_temp_path.setPath(self._pixel_path.qpath)

                ev.accept()

        def keyPressEvent(self, event):
            modifiers = QtGui.QApplication.keyboardModifiers()

            for label in range(10):
                if event.key() == getattr(QtCore.Qt, f"Key_{label}"):
                    if modifiers == QtCore.Qt.ControlModifier:
                        self.label_layer.setCurrentDiskRadius(label)
                    else:
                        self.label_layer.setCurrentLabel(label)
                    event.accept()
                    break
                if event.key() == QtCore.Qt.Key_E:
                    if modifiers != QtCore.Qt.ControlModifier:
                        self.label_layer.setCurrentLabel(0)
                        event.accept()

    # we need a custom graphics item group to handle key interaction
    class MyQGraphicsItemGroup(QtGui.QGraphicsItemGroup):
        keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

        def __init__(self, image_item, temp_path):
            super(LabelLayer.MyQGraphicsItemGroup, self).__init__()
            self.image_item = image_item
            self.temp_path = temp_path
            self.addToGroup(self.image_item)
            self.addToGroup(self.temp_path)

        def keyPressEvent(self, event):
            self.image_item.keyPressEvent(event)

    def __init__(self, name, data=None, lut=None):
        super(LabelLayer, self).__init__(name=name)

        self.lut = get_label_lut()

        self.m_label_data = data

        self.current_label = 1
        self.disk_rad = 1

        # self.m_image_item_group.setHandlesChildEvents

        self.m_ctrl_widget = LabelLayer.CtrlWidget(
            name=self.name, current_label=self.current_label, disc_rad=self.disk_rad
        )

        self._pixel_path = PixelPath()
        self.m_image_item = LabelLayer.LabelLayerImageItem(label_layer=self)
        self.m_image_item.setLookupTable(self.lut)
        self.m_temp_path = QtGui.QGraphicsPathItem()

        self._set_pen()

        self.m_image_item_group = LabelLayer.MyQGraphicsItemGroup(
            self.m_image_item, self.m_temp_path
        )

        if self.m_label_data is not None:
            self.m_image_item.setImage(self.m_label_data, autoLevels=False)

        self.m_image_item_group.addToGroup(self.m_image_item)
        self.m_image_item_group.addToGroup(self.m_temp_path)
        self.disk = disk(self.disk_rad)

        # self.m_ctrl_widget.setLut(lut)
        self.viewer = None

        w = self.m_ctrl_widget

        # toggle eye
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

        # opacity
        self.m_ctrl_widget.bar.fractionChanged.connect(self.setOpacity)

        # current label
        def onLabelChange(label):
            self.current_label = label
            self._set_pen()

        self.m_ctrl_widget.labelSelector.valueChanged.connect(onLabelChange)

        # current label
        def onDiskSizeChange(disk_rad):
            # print('disk_rad', disk_rad)
            self.disk_rad = disk_rad
            self._set_pen()

        self.m_ctrl_widget.brushSelector.valueChanged.connect(onDiskSizeChange)

        self.m_ctrl_widget.layer = self

    def _set_pen(self):
        fac = [1, 3][self.current_label == 0]
        if self.current_label == 0:
            color = (0, 0, 0, 255.0 * 0.7)
        else:
            color = self.lut[self.current_label, :]
        p = pg.mkPen(color=color, width=fac * self.disk_rad * 2 + 1, cosmetic=False)
        p.setCapStyle(QtCore.Qt.RoundCap)
        self.m_temp_path.setPen(p)

    def setNumClasses(self, num_classes):
        self.m_ctrl_widget.setNumClasses(num_classes)

    def ctrl_widget(self):
        return self.m_ctrl_widget

    def get_image_item(self):
        return self.m_image_item_group

    def setCurrentLabel(self, label):
        self.current_label = label
        self._set_pen()
        self.m_ctrl_widget.labelSelector.setValue(label)

    def setCurrentDiskRadius(self, rad):
        self.disc_rad = rad
        self._set_pen()
        self.m_ctrl_widget.brushSelector.setValue(rad)

    def updateData(self, image):
        self.m_label_data = image
        self.m_image_item.m_label_data = self.m_label_data
        self.m_image_item.updateImage((image))

    def setData(self, image):
        self.m_label_data = image
        self.m_image_item.m_label_data = self.m_label_data
        self.m_image_item.setImage((image), autoLevels=False)
