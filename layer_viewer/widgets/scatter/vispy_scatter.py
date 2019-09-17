import numpy as np
import vispy.plot as vp
from vispy.color import get_colormap
import numpy

from PyQt5.QtWidgets import *
import vispy.app
import vispy
import sys
from vispy import scene, app

import numpy as np
import vispy.plot as vp
from vispy.scene import visuals

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..toggle_eye import ToggleEye
from pyqtgraph.widgets.HistogramLUTWidget import HistogramLUTWidget
from ...widgets import  ToggleEye, FractionSelectionBar, GradientWidget
import vispy.scene as scene




from .histogram import custom_hist
from .patch_gradients import patch_gradients
from .lasso import Lasso


class VisPyScatterPlot(QWidget):

    selectionChanged = QtCore.pyqtSignal(numpy.ndarray)


    def __init__(self, with_background_scatter=False):
        super().__init__()

        self.with_background_scatter = with_background_scatter
        self.selection =  None

        # create figure with plot
        fig = vp.Fig(show=False)#, bgcolor=(0,0,0)  )


        self.setLayout(QGridLayout())
        self.layout().addWidget(fig.native,0,0,5,1)
        #self.layout().addWidget(self.ctrl_widget,5,0,1,1)

        self.fig = fig
        #self.fig._bg_color = (0,0,0)
        self.k = 4
        plt = fig[0:self.k, 0:self.k]
        plt.set_gl_state(depth_test=False)



        self.histcolor = (0.3, 0.5, 0.8)
        self.n_bins = 100

        self.histx = None
        self.histy = None


        plt._configure_2d()
        selected = None

        self.plt = plt
        #self.plt.custom_hist = custom_hist

        
        self.data = None
        self.color = None
        self.scatter =  None
        self.scatter_sub = None


        self.data_bg = None
        self.color_bg = None
        self.scatter_bg = None

        ldata =None
        self.lasso = Lasso()
        self.lasso_line = vp.Line(pos=None,parent=plt.view.scene, width=50, method='agg',color=(1,0,0))#, marker_size=0.0)
        self.lasso_line.visible = False
        self.lasso_line.order = 20

        self.sub_data = None

        # connect events
        self.fig.events.mouse_press.connect(self.on_mouse_press)
        self.fig.events.mouse_release.connect(self.on_mouse_release)
        self.fig.events.mouse_move.connect(self.on_mouse_move)

        self.cm = None

    def get_cm(self):
        return self.root.get_cm()

    def set_data(self, data, color, values, data_bg=None, color_bg=None, values_bg=None):

        self.data = data
        self.data_bg = data_bg
        self.values = values
        self.values_bg = values_bg

        if self.with_background_scatter:
            assert data_bg is not None
            assert color_bg is not None
        else:
            assert data_bg is None
            assert color_bg is None


        # scatter fg
        if self.scatter is None:
            self.scatter = vp.Markers(pos=data,symbol='o', face_color=color, edge_width=0.1, edge_width_rel=None,parent=self.plt.view.scene)
            self.scatter.order =-5

        else:
            self.scatter.set_data(pos=data,face_color=color)

        # scatter bg
        if self.with_background_scatter:
            if self.scatter_bg is None:
                self.scatter_bg = vp.Markers(pos=data_bg,symbol='o', face_color=color_bg, edge_width=0.1, edge_width_rel=None,parent=self.plt.view.scene)
                self.scatter_bg.order =-3
            else:
                self.scatter_bg.set_data(pos=data_bg,face_color=color_bg)
        self.plt.view.camera.set_range()

        # histogram
        plotx = self.fig[self.k, 0:self.k]
        ploty = self.fig[0:self.k, self.k]
        cm = self.get_cm()
        value_range = self.root.levels
        if not self.with_background_scatter:
           
            minvals = numpy.min(data, axis=0)
            maxvals = numpy.max(data, axis=0)

            for d in range(2):
                plot = [ploty,plotx][d==0]
                orientation = ['v','h'][d==0]
                r = (minvals[d], maxvals[d])
                custom_hist(plot, data=data[:,d],     range=r, color=(1,0,0), orientation=orientation, cm=cm, values=self.values, value_range=value_range)
        else:

            all_data = numpy.concatenate([data,data_bg], axis=0)
            all_values = numpy.concatenate([self.values, self.values_bg])
            minvals = numpy.min(all_data, axis=0)
            maxvals = numpy.max(all_data, axis=0)
            
            
            print("data",data.min(), data.max())
            for d in range(2):
                plot = [ploty,plotx][d==0]
                orientation = ['v','h'][d==0]
                r = (minvals[d], maxvals[d])
                custom_hist(plot, data=data[:,d],     range=r, color=(1,0,0), orientation=orientation, cm=cm, values=self.values, value_range=value_range)
                custom_hist(plot, data=data_bg[:,d],  range=r, color=(0,1,0), orientation=orientation, cm=cm, values=self.values_bg, value_range=value_range)
                custom_hist(plot, data=all_data[:,d], range=r, color=(0,0,1), orientation=orientation, cm=cm, values=all_values, value_range=value_range)
            
    def map_to_dataspace(self, pos):
        tr = self.fig.scene.node_transform(self.scatter)
        return tr.map(pos)[0:2]

  
    def on_mouse_press(self, event):
        pass

    def on_mouse_release(self, event):
        if event.handled :
            return

        if hasattr(event.last_event,'is_custom_drag') and event.last_event.is_custom_drag :
            self.on_drag_stop(event)


    def on_mouse_move(self, event):

        if event.is_dragging:
            modifiers = QtGui.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ControlModifier:
                event.is_custom_drag = True
                if not event.last_event.is_dragging:
                   self.on_drag_start(event.last_event)

                if self.lasso:
                    self.on_drag(event)


    def on_drag_start(self, event):
        #print("on_drag_start")
        self.lasso.reset()
        pos = self.map_to_dataspace(event.pos)
        self.lasso.add(pos)
        self.lasso_line.visible = True

    def on_drag_stop(self, event):
        self._build_sub()
        self.lasso_line.visible = False

    def on_drag(self, event):
        #print("on_drag")
        pos = self.map_to_dataspace(event.pos)
        self.lasso.add(pos)
        self.lasso_line.set_data(pos=self.lasso.maybe_closed_array(), width=5)


    def _build_sub(self):
        if len(self.lasso) > 2:
            self.lasso.close_path()
            self.sub_data = None
            
            self.selection = self.lasso.contains(self.data)
            print("self.selection: {}".format(len(self.selection)))
            if len(self.selection) > 0:
                self.sub_data = self.data[self.selection,:]
                if self.scatter_sub is None:
                    self.scatter_sub = vp.Markers(pos=self.sub_data, face_color=(0.1, 0.9, 0.8),symbol='o', parent=self.plt.view.scene)
                    self.scatter_sub.order =-10
                else:

                    self.scatter_sub.set_data(self.sub_data, face_color=(0.1, 0.9, 0.8),symbol='o')

                self.scatter_sub.interactive = True
                self.scatter_sub.visible = True
            self.path = None
            self.selectionChanged.emit(self.selection)
    def show(self):
        super().show()
        self.fig.app.run()





class VisPyScatter(QWidget):
    def __init__(self, with_background_scatter=False):
        super().__init__()
        self.with_background_scatter =with_background_scatter
        self.setLayout(QGridLayout())


        self.plot = VisPyScatterPlot(with_background_scatter=with_background_scatter)
        self.plot.root = self
        self.histlut = HistogramLUTWidget()
        self.histlut.item.gradient.restoreState(
            pg.graphicsItems.GradientEditorItem.Gradients['thermal'])

        self.levels = None
        self.data = None
        self.values = None
        self.values_bg = None
        self.data_bg = None

        self.init_ui()

    def init_ui(self):

        self.plot.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.histlut.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


        self.layout().addWidget(self.plot, 0,0, 1,1)
        self.layout().addWidget(self.histlut, 0,1, 1,1)

        # levels changed
        def f():
            self.on_levels_change(self.histlut.getLevels())
        self.histlut.sigLevelChangeFinished.connect(f)

        def f():
            self.on_colormap_changed(self.get_cm())

        # cm changed
        self.histlut.item.gradient.sigGradientChanged.connect(self.on_colormap_changed)

    def set_levels(self, minv, maxv):
        self.levels = (minv, maxv)
        self.histlut.item.setLevels(self.levels[0], self.levels[1])
        self.histlut.setHistogramRange(self.levels[0], self.levels[1])

    def get_cm(self):
        return self.histlut.item.gradient.colorMap()


    def normalize_values(self, values):
        normalized_values = numpy.clip(values, self.levels[0], self.levels[1])
        normalized_values -=  self.levels[0]
        normalized_values /= (self.levels[1] - self.levels[0])
        return normalized_values

    def apply_cm(self, values):
        return self.get_cm().mapToFloat(values)

    # this is ONLY triggered by changes from the gui?
    def on_levels_change(self, levels):
        self.levels = levels

        if self.values is not None:
            print("Acc")
            # color the data    
            self.normalized_values = self.normalize_values(self.values)
            color = self.apply_cm(self.normalized_values)
            color_bg = None
            if self.with_background_scatter:
                self.normalized_values_bg = self.normalize_values(self.values_bg)
                color_bg = self.apply_cm(self.normalized_values_bg)
            self.plot.set_data(data=self.data, color=color, 
                values=self.values, values_bg=self.values_bg,
                data_bg=self.data_bg, color_bg=color_bg)


    def on_colormap_changed(self, cm):

        if self.normalized_values is not None:
            color = self.get_cm().mapToFloat(self.normalized_values)
            if self.with_background_scatter:
                color_bg = self.get_cm().mapToFloat(self.normalized_values_bg)
            else:
                color_bg = None


        self.plot.set_data(
            data=self.data, 
            color=color, 
            values=self.values,
            data_bg=self.data_bg,
            color_bg=color_bg,
            values_bg=self.values_bg
            )



    def set_data(self, data, values, data_bg=None, values_bg=None):


        self.values = values
        self.values_bg  = values_bg
        self.data = data
        self.data_bg = data_bg
        realmin = values.min()
        realmax = values.max()
        hist,b= numpy.histogram(values,bins=100, range=self.levels)
        r =numpy.linspace(realmin,realmax, 100)
        self.histlut.item.plot.setData(r,hist)

        # color the data 
        self.normalized_values = self.normalize_values(self.values)
        color = self.apply_cm(self.normalized_values)

        if self.with_background_scatter:
            self.normalized_values_bg = self.normalize_values(self.values_bg)
            color_bg = self.apply_cm(self.normalized_values_bg)
        else:
            self.data_bg = None
            self.values_bg = None
            color_bg = None

        self.plot.set_data(data=self.data, color=color, values=self.values,
            data_bg=self.data_bg, color_bg=color_bg, values_bg=self.values_bg)



    def show(self):
        super().show()
        self.plot.fig.app.run()
