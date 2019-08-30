import numpy

# from bresenham import bresenham
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.draw import line as skimage_line

from pyqtgraph.Qt import QtCore, QtGui


def bresenham_line(pstart, pstop):
    x0, y0 = pstart
    x0 = int(x0)
    y0 = int(y0)

    x1, y1 = pstop
    x1 = int(x1)
    y1 = int(y1)
    rr, cc = skimage_line(x0, y0, x1, y1)
    return [(r, c) for r, c in zip(rr, cc)]


class PixelPath(object):
    def __init__(self):

        self._path = []
        self._pix_path = []
        self.draw_kernels = dict()

        self.qpath = QtGui.QPainterPath()

    def clear(self):
        self._path = []
        self._pix_path = []
        # there is no clear
        self.qpath = QtGui.QPainterPath()

    def add(self, pos):
        _pix_path = self._pix_path

        # it was > 1, but I think it should be > 0
        if len(_pix_path) > 0:
            self._pix_path.extend(bresenham_line(self._pix_path[-1], pos))
            self.qpath.lineTo(float(pos[0]), float(pos[1]))
        else:
            self._pix_path.append(pos)
            self.qpath.moveTo(float(pos[0]), float(pos[1]))

    def insert_to_image(self, label_image, label, rad):
        path = numpy.array(self._pix_path)
        int_path = path.astype("int")

        if rad == 0:
            wx = int_path[:, 0]
            wy = int_path[:, 1]
        else:

            # bouning box
            bb_min = numpy.min(int_path, axis=0)
            bb_max = numpy.max(int_path, axis=0)
            bb_size = bb_max - bb_min + 1 + 2 * rad

            # remove min
            int_path -= bb_min
            int_path += rad

            # label patch
            label_patch = numpy.zeros(bb_size, dtype="uint8")

            # write to label patch
            label_patch[int_path[:, 0], int_path[:, 1]] = 2

            # dilate
            if rad not in self.draw_kernels:
                self.draw_kernels[rad] = disk(rad)
            label_patch = dilation(label_patch, self.draw_kernels[rad])

            # as coordinates
            wx, wy = numpy.where(label_patch != 0)

            wx += bb_min[0] - rad
            wy += bb_min[1] - rad

        wx = numpy.clip(wx, 0, label_image.shape[0] - 1)
        wy = numpy.clip(wy, 0, label_image.shape[1] - 1)
        label_image[wx, wy] = label
        return label_image
