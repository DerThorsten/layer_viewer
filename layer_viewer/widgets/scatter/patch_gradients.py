

import pyqtgraph as pg


def patch_gradients():


    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    to_rm = []
    for k,v in Gradients.items():
        if v['mode'] == 'hsv':
            to_rm.append(k)

    for k in to_rm:
        del Gradients[k]


