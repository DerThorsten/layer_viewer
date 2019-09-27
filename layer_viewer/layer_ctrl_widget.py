import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import os

from . widgets.drag_and_drop_list_widget import DrangAndDropListWidget


class LayerCtrlDrangAndDropListWidget(DrangAndDropListWidget):
    def __init__(self, parent=None):
        DrangAndDropListWidget.__init__(self, parent)


    def dropEvent(self, e):
        super(LayerCtrlDrangAndDropListWidget, self).dropEvent(e)
        print("fobu")
        n = self.count()
        for i in range(self.count()):
            item = self.item(i)
            w = self.itemWidget(item)
            layer = w.layer
            # layer.setZValue(n - 1 - i)
            layer.setZValue(i)


class LayerCtrlWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.widget_layout = QtGui.QVBoxLayout()

        self.list_widget = LayerCtrlDrangAndDropListWidget(parent=self)
        # self.list_widget.horizontalScrollBar().setDisabled(True);
        self.widget_layout.addWidget(self.list_widget)
        self.setLayout(self.widget_layout)
        self.n_layers = 0

    def removeLayer(self, layer):
        _list_widget_item = layer._list_widget_item
        qIndex = self.list_widget.indexFromItem(_list_widget_item)
        self.list_widget.model().removeRow(qIndex.row())

    def add_layer(self, layer):
        self.addLayer(layer=layer)
        
    def addLayer(self, layer):

        w = layer.ctrl_widget()
        w.toggleEye.setActive(True)
        myQListWidgetItem = QtGui.QListWidgetItem()
        myQListWidgetItem.setSizeHint(w.sizeHint())
        layer._list_widget_item = myQListWidgetItem
        self.list_widget.addItem(myQListWidgetItem)
        self.list_widget.setItemWidget(myQListWidgetItem, w)

        # w.toggleEye.activeChanged.connect(layer.setVisible)
        # w.bar.fractionChanged.connect(layer.setOpacity)

        self.n_layers += 1

    def dragEnterEvent(self, item):
        pass


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    widget = LayerCtrlWidget()
    widget.show()

    sys.exit(app.exec_())
