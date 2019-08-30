import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import os


class DrangAndDropListWidget(QtGui.QListWidget):
    def __init__(self, parent=None):
        QtGui.QListWidget.__init__(self, parent)

        # Enable drag & drop ordering of items.
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)

    def on_rowsInserted(self, parent_index, start, end):
        pass

    def dragEnterEvent(self, e):
        pass
        super(DrangAndDropListWidget, self).dragEnterEvent(e)

    def dropEvent(self, e):
        pass
        super(DrangAndDropListWidget, self).dropEvent(e)

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

        self.list_widget = DrangAndDropListWidget(parent=self)
        # self.list_widget.horizontalScrollBar().setDisabled(True);
        self.widget_layout.addWidget(self.list_widget)
        self.setLayout(self.widget_layout)
        self.n_layers = 0

    def removeLayer(self, layer):
        _list_widget_item = layer._list_widget_item
        qIndex = self.list_widget.indexFromItem(_list_widget_item)
        self.list_widget.model().removeRow(qIndex.row())

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
