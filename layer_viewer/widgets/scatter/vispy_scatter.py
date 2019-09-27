import numpy as np
import vispy.plot as vp
from vispy.color import get_colormap
import numpy
import math
from PyQt5.QtWidgets import *
import vispy.app
import vispy
import sys
from vispy import scene, app
from vispy.color import ColorArray
from vispy.visuals.filters import Clipper, Alpha, ColorFilter

import numpy as np
import vispy.plot as vp
from vispy.scene import visuals

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..toggle_eye import ToggleEye
from pyqtgraph.widgets.HistogramLUTWidget import HistogramLUTWidget
from ...widgets import  ToggleEye, FractionSelectionBar, GradientWidget
import vispy.scene as scene



from .patch_gradients import patch_gradients
from .side_color_hist import SideColorHist
from .lasso import Lasso
from . spatial import *
from . misc import block_signals, clip_norm
from . marker_proxy import MarkerProxy
from . hist_proxy import HistProxy

from sklearn.preprocessing import MinMaxScaler

import functools
import logging




import scipy.ndimage


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

patch_gradients()


class GridLayoutWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QtGui.QGridLayout())


class Selector(object):
    def __init__(self, root):
        self.root = root

    def widget(self):
        raise NotImplementedError()

    def disable(self):
        raise NotImplementedError()

    def enable(self):
        raise NotImplementedError()



class BallSelector(Selector):
    name = "Ball"

    class CtrlWidget(GridLayoutWidget):
        def __init__(self):
            super().__init__()
            l = self.layout()
            self.rad_spinner = QSpinBox()
            self.rad_spinner.setMinimum(0)
            l.addWidget(self.rad_spinner, 0,0)

    def __init__(self, root):
        super().__init__(root=root)
        self._widget = BallSelector.CtrlWidget()

    def widget(self):
        return self._widget

class KnnSelector(Selector):
    name = "KNN"

    class CtrlWidget(GridLayoutWidget):
        def __init__(self):
            super().__init__()
            l = self.layout()
            self.knn_spinner = QSpinBox()
            self.knn_spinner.setMinimum(0)
            l.addWidget(self.knn_spinner, 0,0)

    def __init__(self, root):
        super().__init__(root=root)
        self._widget = KnnSelector.CtrlWidget()

    def widget(self):
        return self._widget

class LassoSelector(Selector):
    name = "Lasso"
    def __init__(self, root):
        super().__init__(root=root)
        self._widget = QWidget()

    def widget(self):
        return self._widget


SELECTORS_CLS  = [
    LassoSelector,
    KnnSelector,
    BallSelector
]




class CtrlWidget(QWidget):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.setLayout(QtGui.QGridLayout())
        l =  self.layout()

        for i,w in enumerate(self.root.selectors):
            l.addWidget(QLabel(w.name), 0, i*2)
            l.addWidget(w.widget(), 0, i*2 + 1)    

if False:
    class CtrlWidget(QWidget):
        def __init__(self, root):
            super().__init__()
            self.root = root
        
            self.combo_box = QtGui.QComboBox()
            self._selection_types = ["ball","lasso", "ball"]
            self.selection_type = self._selection_types[0]
            self.combo_box.addItems(self._selection_types)
            self.spinner_rad = QSpinBox()
            self.spinner_knn = QSpinBox()

            self._init_ui()
            self._connect_signals()

        def _init_ui(self):

            self.spinner_rad.setMinimum(0)
            self.spinner_knn.setMinimum(0)

            self.spinner_rad.setValue(5)
            self.spinner_knn.setValue(100)

            self.spinner_knn.setEnabled(False)
            self.spinner_rad.setEnabled(True)
            

            self.setLayout(QtGui.QGridLayout())
            l = self.layout()
            l.addWidget(QtGui.QLabel("SelectionType"), 0, 1)
            l.addWidget(self.combo_box, 0, 2)
            l.addWidget(QtGui.QLabel("ball radius"), 0, 3)
            l.addWidget(self.spinner_rad, 0, 4)
            l.addWidget(QtGui.QLabel("k"), 0, 5)
            l.addWidget(self.spinner_knn, 0, 6)

        def _connect_signals(self):
            def f(i):
                m = self._selection_types[i]
                if m == "lasso":
                    self.spinner_knn.setEnabled(False)
                    self.spinner_rad.setEnabled(False)
                    self.root.sel_ellips.visible = False

                elif m == "ball":
                    self.spinner_knn.setEnabled(False)
                    self.spinner_rad.setEnabled(True)
                    self.root.sel_ellips.visible = True

                elif m == "knn":
                    self.spinner_knn.setEnabled(True)
                    self.spinner_rad.setEnabled(False)
                    self.root.sel_ellips.visible = False


            self.combo_box.currentIndexChanged.connect(f)

            def f(r):
                self.root.sel_ellips.radius = (r,r)
            # todo remove me
            #self.root._init_crosshair()
            self.spinner_rad.valueChanged.connect(f)


class ScatterDataset(object):
    def __init__(self, name, pos, color,values, z_index, root):
        self.name = name
        self.pos = pos
        self.is_sel = self.name == "sel"

        # we never need a kd tree for datasets where pos is None
        # and the only ds which can have a pos == None is the
        # selection dataset itself, therefore we can just
        # ignore the kdtree attribute (not have it) when we do not
        # have a pos
        if self.pos is not None:
            logger.debug("build kdtree")
            self.kdtree = KdTree(pos)
            logger.debug("build kdtree done")
        self.color = color
        self.values = values
        self.z_index = z_index
        self.root = root
        self.clipnormed_values = None
        self.colored_values = None
        self.alpha = Alpha(1.0) 
        self.color_filter = ColorFilter((1,1,1,1))
        filters = ([self.color_filter],[])[self.is_sel]
        self.scatter = MarkerProxy(plot=root.plot_scatter,z_index=z_index,
             filters=filters + [self.alpha], symbol='o', edge_width=0.1)
        bins = self.root.config['n_bins_histxy']
        sigma = self.root.config['sigma_histxy']
        self.histx = HistProxy(plot=root.plot_hist_x, filters=filters, bins=bins, sigma=sigma, orientation='h')
        self.histy = HistProxy(plot=root.plot_hist_y, filters=filters, bins=bins, sigma=sigma, orientation='v')
        self.histv = None
        
        self.is_visible = True

    def build_plots(self):
        
        # pos can be none in case of the "selection dataset"
        # which only has self.pos!=None when there is an 
        # actual selection
        if self.pos is not None:
            # where to clip the 
            clip_range = self.root.levels

            # clip and normalize values st. they are in [0,1]
            self.clipnormed_values = clip_norm(self.values, clip_range[0], clip_range[1])
            self.colored_values = self.root.apply_cm(self.clipnormed_values)
            self.scatter.set_data(pos=self.pos, face_color=self.colored_values)
            
            # pos range is the min/max of the x/y coordinates
            # for all datasets
            pos_range = self.root.data_list.pos_range
            cm = self.root.get_cm()
            self.histx.set_data(data=self.pos[:,0], range=pos_range[0], cm=cm, values=self.values, value_range=clip_range)
            self.histy.set_data(data=self.pos[:,1], range=pos_range[1], cm=cm, values=self.values, value_range=clip_range)

            self.scatter.visible = self.is_visible
            self.histx.visible = self.is_visible
            self.histy.visible = self.is_visible

    def on_colormap_changed(self):
        if self.pos is not None :
            assert self.clipnormed_values is not None
            self.colored_values = self.root.apply_cm(self.clipnormed_values)
            self.scatter.set_data(self.pos, face_color=self.colored_values)
            self.histx.on_colormap_changed(self.root.get_cm())
            self.histy.on_colormap_changed(self.root.get_cm())
          
    def on_levels_changed(self):
        self.build_plots()

    def setVisible(self, v):
        self.is_visible = v
        self.scatter.visible = v
        self.histx.visible = v
        self.histy.visible = v

    def setAlpha(self, alpha):
    
        self.alpha.alpha = alpha
        self.scatter.update()

    def setData(self, pos, values):
        self.pos = pos 
        self.values = values
    
    def unsetData(self):
        self.pos = None 
        self.values = None

class ScatterDatasetList(list):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.value_range = None
        self.pos_range = None

    def build_from_list(self, data_list):

        # create then data set from the dict
        for i, data in enumerate(data_list):
            ds = ScatterDataset(
                name=data['name'],
                pos=data['pos'],
                values=data['values'],
                color=data['color'],
                z_index = -i,
                root=self.root,
            )
            self.append(ds)


        # find levels aka min max of values
        scaler = MinMaxScaler()
        for data in self:
            scaler.partial_fit(data.values[...,None])
        value_range = scaler.data_min_, scaler.data_max_
        self.value_range = tuple([float(f) for f in value_range])


        # find min max of positions
        scaler = MinMaxScaler()
        for data in self:
            scaler.partial_fit(data.pos)
        self.pos_range = (scaler.data_min_[0], scaler.data_max_[0]),(scaler.data_min_[1], scaler.data_max_[1])

        ds = ScatterDataset(
            name='sel',
            pos=None,
            values=None,
            color=(0,0,1),
            z_index = len(data_list),
            root=self.root,
        )
        self.append(ds)


    def build_plots(self):
        for data in self:
            data.build_plots()

    def on_colormap_changed(self):
        for data in self:
            data.on_colormap_changed()

    def on_levels_changed(self):
        for data in self:
            data.on_levels_changed()



class DatasetCtrlWidget(QWidget):
    def __init__(self, root):
        super().__init__()
        self.setLayout(QtGui.QGridLayout())
        g = self.layout()
        
        def on_eye(self, index, eye):
            d = root.data_list[index].setVisible(eye.active())
            root.plots[index].setVisible(eye.active())

        def on_bar(self, index, bar):
            d = root.data_list[index].setAlpha(bar.fraction())
            root.plots[index].setOpacity(bar.fraction())

        for i,data in enumerate(root.data_list):
            bar = FractionSelectionBar()
            bar.setFixedHeight(20)
            eye = ToggleEye()


            eye.activeChanged.connect(functools.partial(on_eye, index=i, eye=eye))
            bar.fractionChanged.connect(functools.partial(on_bar, index=i, bar=bar))

            g.addWidget(QtGui.QLabel(data.name),i,0)
            g.addWidget(eye,i,1)
            g.addWidget(bar,i,2,1,20)



class VisPyScatter(QWidget):

    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()

        self.config = {
            "trigger_cm_when_finished" : False,
            "trigger_level_when_finished" : True,
            "show_edges": True,
            "n_bins_histv":100,
            "n_bins_histxy":255,
            "sigma_histxy": 2.0,
            "sigma_histv" : 2.0,
            
        }
        self.config.update(kwargs)
        self.selectors = [cls(root=self) for cls in SELECTORS_CLS]
        # the data and the global levels
        self.data_list = ScatterDatasetList(root=self)
        self.levels = None

        # the vispy figure / widgets
        self.vispy_fig = None
        self.plot_hist_x = None
        self.ploty = None
        self._init_vispy_fig()

        self.sel_ellips = None
        self.sel_ellips_alpha = Alpha(0.2)

        # qt widgets
        self.histlut = HistogramLUTWidget(fillHistogram=True)
        self.histlut.item.gradient.restoreState(
            pg.graphicsItems.GradientEditorItem.Gradients['thermal'])
        self.data_ctrl_widget = None
        self.ctrl_widget = CtrlWidget(root=self)

        # plots for side value hists
        self.plots = []

        # init the ui
        self._init_ui()

        # connect signals
        self._connect_signals()

        # TODO
        self.bars = []

        # selection
        self.selection = None

    @property
    def selection_type(self):
        return self.ctrl_widget.selection_type
    

    def interactive_ds(self):
        return  self.data_list[0]

    def sel_ds(self):
        return self.data_list[-1]

    def on_selection_changed(self, selection):
        self.selectionChanged.emit()
        if selection is None or len(selection) == 0:
            logger.debug("on_deselect")
            self.on_deselect()
        else:
            logger.debug("on_new_select")
            self.on_new_select(selection)
        self.selection = selection




    def on_new_select(self, selection):
        ds_sel = self.data_list[-1]
        ds = self.interactive_ds() 
        sel_pos = ds.pos[selection, :]
        sel_values = ds.values[selection]
        ds_sel.setData(pos=sel_pos, values=sel_values)
        ds_sel.build_plots()
        ds_sel.setVisible(True)
        self.synch_axis(with_sel=True)

        for ds in self.data_list[:-1]:
            ds.color_filter.filter = (0.5,0.5,0.5,0.5)
        self.sel_ds().setVisible(True)

    def on_deselect(self):
        for ds in self.data_list[:-1]:
            ds.color_filter.filter = (1,1,1,1)

        self.sel_ds().unsetData()
        self.sel_ds().setVisible(False)
        self.synch_axis()


    def _init_ui(self):

        # set main layut
        self.setLayout(QGridLayout())

        # the layout
        layout = self.layout()

        # add the vispy fig 
        layout.addWidget(self.vispy_fig.native, 0, 0)

        # add the histogram lut
        layout.addWidget(self.histlut, 0, 1)


        # make sure the scatter widget takes most of the space when window is maximized
        self.vispy_fig.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.histlut.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.ctrl_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

            
        self.layout().addWidget(self.ctrl_widget,1,0,1,2)

    def _init_vispy_fig(self):

        def conf_plot(plt):
            plt._configure_2d()
            plt.set_gl_state(depth_test=False)


        self.vispy_fig = vp.Fig(show=False)
        k = 4
        self.plot_scatter = self.vispy_fig[0:k, 0:k]
        conf_plot(self.plot_scatter)
        self.plot_hist_x = self.vispy_fig[k, 0:k]
        conf_plot(self.plot_hist_x)
        self.plot_hist_y = self.vispy_fig[0:k, k]
        conf_plot(self.plot_hist_y)


    def _init_crosshair(self):
        if self.sel_ellips is None:
            self.sel_ellips = visuals.Ellipse(center=(0.0,0.0), radius=(3, 3),
                border_color=(0, 0, 1, 1), color=(1,0,1,0.1),
                parent=self.plot_scatter.view.scene)

        self.sel_ellips.order = -1
        self.sel_ellips.visible = True
        self.sel_ellips.attach(self.sel_ellips_alpha)

    def _connect_signals(self):

        # handle changing gradients
        if self.config['trigger_cm_when_finished']:
            self.histlut.gradient.sigGradientChangeFinished.connect(self.on_colormap_changed)
        else:
            self.histlut.gradient.sigGradientChanged.connect(self.on_colormap_changed)

        # handle changing level
        if self.config['trigger_level_when_finished']:
            self.histlut.sigLevelChangeFinished.connect(self.on_levels_changed)
        else:
            self.histlut.sigLevelsChanged.connect(self.on_levels_changed)

        # to handle cross-hair
        self.vispy_fig.events.mouse_move.connect(self.on_mouse_move)
        self.vispy_fig.events.mouse_press.connect(self.on_mouse_press)
        self.vispy_fig.events.mouse_release.connect(self.on_mouse_release)


        self.vispy_fig.events.key_release.connect(self.on_key_release)
        self.vispy_fig.events.key_press.connect(self.on_key_press)

        # to link scatter axis with histograms
        self.plot_scatter.view.scene.transform.changed.connect(self.on_transform_changed)
        self.vispy_fig.events.resize.connect(self.on_transform_changed)


            



    def on_transform_changed(self, _=None):
        self.synch_axis()

    def synch_axis(self, with_sel=False):


        if with_sel:
            max_bin_x = max([d.histx.max_bin_count for d in self.data_list])
            max_bin_y = max([d.histy.max_bin_count for d in self.data_list])
        else:
            max_bin_x = max([d.histx.max_bin_count for d in self.data_list[:-1]])
            max_bin_y = max([d.histy.max_bin_count for d in self.data_list[:-1]])


        xaxis = self.plot_scatter.xaxis
        yaxis = self.plot_scatter.yaxis

        cam_rec = self.plot_hist_x.view.camera.rect
        new = vispy.geometry.Rect()
        new.left = xaxis.axis.domain[0]
        new.right = xaxis.axis.domain[1]
        new.top = max_bin_x
        new.bottom = 0
        self.plot_hist_x.view.camera.rect = new

        cam_rec = self.plot_hist_y.view.camera.rect
        new = vispy.geometry.Rect()
        new.top = yaxis.axis.domain[1]
        new.bottom = yaxis.axis.domain[0]
        new.left = 0
        new.right = max_bin_y
        self.plot_hist_y.view.camera.rect = new


    def map_coords(self, pos):
        # this is hacky!
        s = self.data_list[0].scatter.item
        tr = self.vispy_fig.scene.node_transform(s)
        return tr.map(pos)[0:2]

    def on_key_press(self, event):
        if self.selection_type == "ball":
            print(event.key, type(event.key))
            logger.debug("KEY %s %s", str(event.key), str(QtCore.Qt.Key_Shift))
            if event.key == vispy.util.keys.SHIFT:
                self.sel_ellips_alpha.alpha = 1.0
                self.sel_ellips.update()

    def on_key_release(self, event):
        modifiers = QtGui.QApplication.keyboardModifiers()
        if self.selection_type == "ball":
            if event.key ==vispy.util.keys.SHIFT:
                self.sel_ellips_alpha.alpha = 0.2
                self.sel_ellips.update()

    def on_mouse_press(self, event):
        if self.selection_type == "ball":
            pos =  self.map_coords(event.pos)
            self.select_from_ball(pos=pos)
            self.sel_ellips_alpha.alpha = 1.0
            self.sel_ellips.update()

    def on_mouse_release(self, event):
        if self.selection_type == "ball":
            self.sel_ellips_alpha.alpha = 0.2
            self.sel_ellips.update()
        else:
            self.sel_ellips.visible = False

    def on_mouse_move(self, event):
        
        modifiers = QtGui.QApplication.keyboardModifiers()
       
        if self.selection_type == "ball":
            pos =  self.map_coords(event.pos)
            self.sel_ellips.center = pos
            if modifiers == QtCore.Qt.ShiftModifier:
                self.sel_ellips_alpha.alpha = 1.0
                self.select_from_ball(pos=pos)
            else:
                self.sel_ellips_alpha.alpha = 0.2
   
    def select_from_ball(self, pos):
        self.sel_ellips_alpha.alpha = 1.0
        self.sel_ellips.center = pos
        radius = self.sel_ellips.radius
        if not isinstance(radius, (int,float)):
            assert math.isclose(radius[0], radius[1])
            radius = radius[0]

        ds = self.data_list[0]
        kd = ds.kdtree
        logger.debug("radius %s", radius)
        logger.debug("pos %s", pos)
        selection = kd.query_ball(pos=pos, r=self.sel_ellips.radius[0])
        
        # check if selection is nonempty
        if selection is not None and  len(selection) > 0:
            self.on_selection_changed(selection=selection)
        else:
            # the new selection is empty
            # check if we had an old selection
            if self.selection is not None:
                self.on_selection_changed(selection=None)

    def delect_selection(self):
        self.on_selection_changed(None)
      
        
    # start vispy app via run
    def show(self):
        super().show()
        self.vispy_fig.app.run()

    def get_cm(self):
        return self.histlut.item.gradient.colorMap()

    def apply_cm(self, values):
        cm = self.get_cm()
        color = cm.mapToFloat(values)
        #color[:,3] *= 0.5
        return color

    def set_data(self, data_list, levels=None):

        with block_signals(self, self.histlut.item):
        
            # clear data
            self.data_list.clear()
            

            # find levels aka min max
            self.data_list.build_from_list(data_list)
            
            # if no level is specified,
            # we use the range of the data 
            self.levels = levels
            if self.levels is None:
                self.levels = self.data_list.value_range
            self.levels = tuple([float(f) for f in self.levels])

            # build the plot
            self.data_list.build_plots()

            # set the range
            self.plot_scatter.view.camera.set_range()
            self.plot_hist_x.view.camera.set_range()
            self.plot_hist_y.view.camera.set_range()
            self._init_crosshair()


            self.histlut.item.region.setRegion(self.levels)
            self.histlut.setHistogramRange(self.data_list.value_range[0], self.data_list.value_range[1])

            logger.debug(f"value range {self.data_list.value_range}")
            logger.debug(f"levels      {self.levels}")




            for p in self.plots:
                self.histlut.vb.removeItem(p)
            self.plots.clear()



            # sideSideColorHist (TODO: re-factor me)
            n_bins = self.config['n_bins_histv']
            for i,data in enumerate(self.data_list[:]):
                p = SideColorHist(values=data.values, bins=n_bins, color=data.color, root=self, sigma=self.config['sigma_histv'])
                self.histlut.vb.addItem(p)
                self.plots.append(p)


            # aspect ratio
            self.plot_scatter.camera.aspect = 1.0



            if self.data_ctrl_widget is not None:
                self.layout().removeWidget(self.data_ctrl_widget)

            self.data_ctrl_widget = DatasetCtrlWidget(self)

            #self.vispy_fig.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
            self.data_ctrl_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)


            self.layout().addWidget(self.data_ctrl_widget,2,0,1,2)
            #self.layout().removeWidget(w)

            logger.debug("set data end ")
        
        self.synch_axis()


    def on_colormap_changed(self):
        self.data_list.on_colormap_changed()

    def on_levels_changed(self):
        self.levels = self.histlut.getLevels()
        self.data_list.on_levels_changed()






















