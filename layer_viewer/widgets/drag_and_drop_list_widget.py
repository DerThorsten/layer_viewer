from pyqtgraph.Qt import QtCore, QtGui

class DrangAndDropListWidget(QtGui.QListWidget):
    def __init__(self, parent=None):
        QtGui.QListWidget.__init__(self, parent)

        # Enable drag & drop ordering of items.
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)



    #def dragEnterEvent(self, e):
    #    super(DrangAndDropListWidget, self).dragEnterEvent(e)

    #def dropEvent(self, e):
    #    super(DrangAndDropListWidget, self).dropEvent(e)
