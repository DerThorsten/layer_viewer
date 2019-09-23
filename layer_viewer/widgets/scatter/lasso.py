import numpy
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class Lasso(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.point_list = []
        self.path = None

    def add(self, point):
        qpoint = QtCore.QPointF(*point)
        if self.path is None:
            self.path = QtGui.QPainterPath(qpoint)
            self.path.setFillRule(QtCore.Qt.WindingFill)
        else:
            self.path.lineTo(qpoint)
        self.point_list.append(point)
    def close_path(self):
        self.path.closeSubpath()

    def __bool__(self):
        return bool(self.point_list)

    def __contains__(self, point):
        if not self.point_list:
            return False
        x, y = point[0],point[1]
        qpoint = QtCore.QPointF(float(x),float(y))
        return self.path.contains(qpoint)

    def contains(self, points):
        assert isinstance(points, numpy.ndarray)

        bb = self.path.boundingRect()
        tl = bb.topLeft()
        tl = numpy.array([tl.x(), tl.y()])
        br = bb.bottomRight()
        br = numpy.array([br.x(), br.y()])

        inidx = numpy.where(numpy.all(numpy.logical_and(tl <= points, points <= br), axis=1))[0]
        #print("inidx",inidx.shape)
        #print("points",points.shape)
        contained = []
        for indice in inidx:
            #print("indice",indice)
            p = points[indice, :]
            #print("p",p)
            if self.__contains__(p):
                contained.append(indice)
        return numpy.array(contained)

    def maybe_closed_array(self):
        if len(self.point_list) >=2:
            return numpy.array(self.point_list + self.point_list[0:1])
        else:
            return numpy.array(self.point_list)
    def __array__(self):
        return numpy.array(self.point_list)

    def __len__(self):
        return len(self.point_list)



class Lasso2(object):

    def __init__(self, kd_tree):
        self.reset()
        self.kd_tree = kd_tree

    def reset(self):
        self.point_list = []
        self.path = None

    def add(self, point):
        qpoint = QtCore.QPointF(*point)
        if self.path is None:
            self.path = QtGui.QPainterPath(qpoint)
            self.path.setFillRule(QtCore.Qt.WindingFill)
        else:
            self.path.lineTo(qpoint)
        self.point_list.append(point)
    def close_path(self):
        self.path.closeSubpath()

    def __bool__(self):
        return bool(self.point_list)

    def __contains__(self, point):
        if not self.point_list:
            return False
        x, y = point[0],point[1]
        qpoint = QtCore.QPointF(float(x),float(y))
        return self.path.contains(qpoint)

    def contains(self, points):
        assert isinstance(points, numpy.ndarray)
        return self.kd_tree.query_closed_qpath(self.path)

    def maybe_closed_array(self):
        if len(self.point_list) >=2:
            return numpy.array(self.point_list + self.point_list[0:1])
        else:
            return numpy.array(self.point_list)
    def __array__(self):
        return numpy.array(self.point_list)

    def __len__(self):
        return len(self.point_list)