class Proxy(object):
    
    def update(self):
        if self.item is not None:
            self.item.update()

    @property
    def visible(self):
        return self._visible
        
    @visible.setter 
    def visible(self, value): 
        self._visible = value
        if self.item is not None:
            self.item.visible = value
