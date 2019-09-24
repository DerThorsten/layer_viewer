import numpy
from scipy.spatial import cKDTree
from pyqtgraph.Qt import QtCore, QtGui


def bounding_ball(bounding_box):
    min_coords = bounding_box[0]
    max_coords = bounding_box[1]
    rads = [ma - mi for ma,mi in zip(min_coords, max_coords)]
    pos = [mi + ((ma - mi)/2.0) for ma,mi in zip(min_coords, max_coords)]
    r = max(rads[0], rads[1])
    return pos,r


def in_bounding_box(points, bounding_box):
    tl = bounding_box[0]
    tl = numpy.require(tl)
    br = bounding_box[1]
    tl = numpy.require(br)

    indices = numpy.where(numpy.all(numpy.logical_and(tl <= points, points <= br), axis=1))[0]
    return indices

class KdTree(object):
    def __init__(self, points):
        self.points = points
        self.kdtree  = cKDTree(data=self.points)

    def query_nearest(self, pos, k):
        return self.query_k_nearest(pos, k=1)

    def query_k_nearest(self, pos, k):
        return self.kdtree.query(x=pos, k=k)

    def query_ball(self, pos, r):
        return self.kdtree.query_ball_point(x=pos, r=r, return_sorted=False, return_length=False)

    def query_bounding_box(self, bounding_box):
        
        pos,r = bounding_ball(bounding_box=bounding_box)
        indices = self.query_ball(pos=pos, r=r)
        sub_indices = in_bounding_box(self.points[indices], bounding_box=bounding_box)
        return indices[sub_indices]
        
    def query_closed_qpath(self, qpath):
        br = qpath.boundingRect()
        tl = br.topLeft()
        br = br.bottomRight()
        bounding_box = ((tl.x(), tl.y()), (br.x(), br.y()))
        indices = self.query_bounding_box(bounding_box)
        contained = []
        for indice in indices:
            p = self.points[indice, :]
            p = QtCore.QPointF(float(p[0]),float(p[1]))
            if  qpath.contains(p):
                contained.append(indice)
        return numpy.array(contained)