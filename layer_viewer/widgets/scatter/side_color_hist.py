from pyqtgraph.graphicsItems.PlotDataItem import *
import numpy
import scipy.ndimage

class SideColorHist(PlotDataItem):
    def __init__(self, values, bins, color, root, sigma):
        super().__init__()
        if values is not None:
            h,_ = numpy.histogram(values, bins=bins, range=root.levels, density=True)
            h = scipy.ndimage.gaussian_filter(h, sigma=sigma)
            r = numpy.linspace(root.levels[0], root.levels[1], bins).squeeze()
            self.setData(r,h)

        self.rotate(90)
        self.setFillLevel(0.0)
        self.setPen([255.0 * c for c in color])
        self.setFillBrush((100,100,100, 255.0/2.0))