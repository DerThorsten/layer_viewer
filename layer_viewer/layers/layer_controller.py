from ..widgets import TripleToggleEye, ToggleEye, FractionSelectionBar, GradientWidget
import pyqtgraph as pg
import os
from pyqtgraph.Qt import QtCore, QtGui
import numpy

###############################################################################
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


class LayerItemWidget(QWidget):
    def __init__(
        self,
        name=None,
        parent=None,
        add_gradient_widgtet=False,
        channel_selector=False,
        add_as_rgb_button=False,
    ):
        super(LayerItemWidget, self).__init__(parent=parent)
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
        self.opacityLabel.setText(u"\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction())))

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
        self.channelSelector.setVisible(channel_selector)

        self._layout = QtGui.QGridLayout(self)
        self._layout.addWidget(self.toggleEye, 0, 0)
        self._layout.addWidget(self.nameLabel, 0, 1)
        self._layout.addWidget(self.opacityLabel, 0, 2)
        self._layout.addWidget(self.channelSelector, 1, 0)

        self._layout.addWidget(self.bar, 1, 1, 1, 2)

        if add_gradient_widgtet:
            self.gradientWidget = GradientWidget(orientation="top")
            self.gradientWidget.loadPreset("grey")
            self._layout.addWidget(self.gradientWidget, 3, 1, 1, 2)

        # if add_as_rgb_button:
        #     self.asRgb = QCheckBox(  )
        #     #self.asRgb.setFrame( False )
        #     #self.asRgb.setFont( self._font )
        #     self.asRgb.setMaximumWidth( 35 )
        #     self.asRgb.setAlignment(Qt.AlignRight)
        #     self.asRgb.setToolTip("Show As RGB")
        #     #self.asRgb.setVisible(channel_selector)
        #     self._layout.addWidget( self.gradientWidget, 3,0,1,2)

        self._layout.setColumnMinimumWidth(2, 35)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        def f(frac):
            self.opacityLabel.setText(
                u"\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction()))
            )

        self.bar.fractionChanged.connect(f)

    def setFraction(self, opacity):
        self.bar.setFraction(opacity)
        self.opacityLabel.setText(u"\u03B1=%0.1f%%" % (100.0 * (self.bar.fraction())))

    def setName(self, name):
        self.nameLabel.setText(str(name))

    def mousePressEvent(self, ev):
        super(LayerItemWidget, self).mousePressEvent(ev)
