import skimage.data

from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *


def test_layer_viewer_widget(qtbot):
    # test data
    image = skimage.data.astronaut().swapaxes(0, 1)

    viewer = LayerViewerWidget()
    # viewer.show()
    viewer.setWindowTitle('LayerViewer')

    layer = MultiChannelImageLayer(name='img', data=image[...])
    viewer.addLayer(layer=layer)

    assert viewer.hasLayer('img')
    assert viewer.hasLayer('labels') is False

    labels = numpy.zeros(image.shape[0:2], dtype='uint8')
    label_layer = LabelLayer(name='labels', data=None)
    viewer.addLayer(layer=label_layer)
    viewer.setData('labels', image=labels)

    assert viewer.hasLayer('img')
    assert viewer.hasLayer('labels')
