from contextlib import contextmanager
import numpy

def clip_norm(data, minv, maxv):
    c  = numpy.clip(data, minv, maxv)
    c -= minv
    c  /=(maxv -  minv)
    return c


@contextmanager
def block_signals(*args, **kwds):
    for arg in args:
        arg.blockSignals(True)
    for k,v in kwds.items():
        v.blockSignals(True)
    yield
    for arg in args:
        arg.blockSignals(False)
    for k,v in kwds.items():
        v.blockSignals(False)
