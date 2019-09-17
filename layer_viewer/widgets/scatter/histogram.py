import sys
from vispy import scene, app
import numpy as np
import vispy
from vispy.scene import visuals
import vispy.scene as scene
from colored_hist import colored_hist 
import numpy
class CustomHistogramVisual(vispy.visuals.MeshVisual):
    """Visual that calculates and displays a histogram of data

    Parameters
    ----------
    data : array-like
        Data to histogram. Currently only 1D data is supported.
    bins : int | array-like
        Number of bins, or bin edges.
    color : instance of Color
        Color of the histogram.
    orientation : {'h', 'v'}
        Orientation of the histogram.
    """
    def __init__(self, data, cm, values, range, value_range, bins=4, color='w', orientation='h'):
        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        data_in = np.asarray(data)
        if data_in.ndim != 1:
            raise ValueError('Only 1D data currently supported')

        X, Y = (0, 1) if orientation == 'h' else (1, 0)

        # do the histogramming
        values = numpy.clip(values, *value_range)
        data, bin_edges, cvalues = colored_hist(data=data_in, values=values, range=list(range), n_bins=bins)
        data, bin_edges = np.histogram(data_in, bins, range=range)
        #print(data.min(), data.max())

        #cvalues = numpy.nan_to_num(cvalues)
        normalized_values =numpy.clip(cvalues, value_range[0], value_range[1])
        normalized_values -= value_range[0]
        normalized_values /= (value_range[1] - value_range[0])
        val_colors = cm.mapToFloat(normalized_values)
        val_colors = np.repeat(val_colors, 2, axis=0)
        #val_colors = numpy.concatenate([val_colors, val_colors], axis=0)
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, X] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, Y] = data
        rr[2::3, Y] = data
        
        bin_edges.astype(np.float32)
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1,
                                dtype=np.uint32)[:, np.newaxis]

        print(rr.shape,tris.shape)

        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets
        print(rr,tris)
        vispy.visuals.MeshVisual.__init__(self, rr, tris,face_colors=val_colors)

    def set_data(self, data, cm, values, range, value_range, bins=4, color='w', orientation='h'):
       #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        data_in = np.asarray(data)
        if data_in.ndim != 1:
            raise ValueError('Only 1D data currently supported')

        X, Y = (0, 1) if orientation == 'h' else (1, 0)

        # do the histogramming
        values = numpy.clip(values, *value_range)
        data, bin_edges, cvalues = colored_hist(data=data_in, values=values, range=list(range), n_bins=bins)
        data, bin_edges = np.histogram(data_in, bins, range=range)
        #print(data.min(), data.max())

        #cvalues = numpy.nan_to_num(cvalues)
        normalized_values =numpy.clip(cvalues, value_range[0], value_range[1])
        normalized_values -= value_range[0]
        normalized_values /= (value_range[1] - value_range[0])
        val_colors = cm.mapToFloat(normalized_values)
        val_colors = np.repeat(val_colors, 2, axis=0)
        #val_colors = numpy.concatenate([val_colors, val_colors], axis=0)
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, X] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, Y] = data
        rr[2::3, Y] = data
        
        bin_edges.astype(np.float32)
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1,
                                dtype=np.uint32)[:, np.newaxis]

        print(rr.shape,tris.shape)

        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets
        self.set_data(vertices=rr, faces=tris, face_colors=val_colors)

CustomHistogram = scene.visuals.create_visual_node(CustomHistogramVisual)


def custom_hist(self, data, cm, values,  range, value_range, bins=30, color='w', orientation='h'):
    self._configure_2d()
    hist = CustomHistogram(data=data, bins=bins, 
        cm=cm, values=values,
        color=color, orientation=orientation, range=range,
        value_range=value_range)
    self.view.add(hist)
    self.view.camera.set_range()
    return hist
