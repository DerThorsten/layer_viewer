
from .histogram import custom_hist
from . proxy import Proxy

# the original Hist cannot 
# be constructed properly without data
class HistProxy(Proxy):
    def __init__(self,plot, filters, **kwargs):
        super().__init__()
        #super().__init__(*args, **kwargs)

        self.plot = plot
        self.kwargs = kwargs
        self.filters = filters
        self.item = None
        self._visible = True
        self._cm = None
    def set_data(self, **kwargs):

        kwargs  = {**self.kwargs, **kwargs}
        if self.item is None:
            self.item = custom_hist(self.plot, **kwargs)
            for filt in self.filters:
                self.item.attach(filt)
            self.item.set_gl_state(u_antialias=False)
            self.item.set_gl_state(depth_test=False)
            self.item.visible = self._visible
            if self._cm is not None:
                self.item.on_colormap_changed(cm)

        else:
            self.item.set_data(**kwargs)
        self._cm = None


    @property
    def max_bin_count(self):
        if self.item is None:
            return 0
        else:
            return self.item.max_bin_count

    def on_colormap_changed(self, cm):
        self._cm = cm
        if self.item is not None:
            self.item.on_colormap_changed(cm)
