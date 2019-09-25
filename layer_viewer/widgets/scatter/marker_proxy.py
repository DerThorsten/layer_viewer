import vispy.plot
from . proxy import Proxy

# the original Markersobject cannot 
# be constructed properly without data
class MarkerProxy(Proxy):
    def __init__(self,plot, filters, **kwargs):
        super().__init__()
        #super().__init__(*args, **kwargs)

        self.parent = plot.view.scene
        self.kwargs = kwargs
        self.filters = filters
        self.item = None

    def set_data(self, pos, face_color):
        if self.item is None:
            self.item = vispy.plot.Markers(pos=pos, face_color=face_color,
                parent=self.parent, **self.kwargs) 
            for filt in self.filters:
                self.item.attach(filt)
            self.item.set_gl_state(u_antialias=False)
            self.item.set_gl_state(depth_test=False)
        else:
            self.item.set_data(pos=pos, face_color=face_color,
                **self.kwargs)

