# volume object module

class Volume(object):
    def __init__(self, img, labels):
        # These values are created
        # when the class is instantiated.
        self.img = img
        self.labels = labels
        self.data = layer_list(self.labels)
        
