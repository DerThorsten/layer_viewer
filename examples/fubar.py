from PyQt5.QtWidgets import *
import vispy.app
import vispy
import sys
from vispy import scene, app
import numpy as np
import vispy.plot as vp
from vispy.scene import visuals

np.random.seed(2324)
n = 10000
data = np.empty((n, 2))
lasti = 0
for i in range(1, 20):
    nexti = lasti + (n - lasti) // 2
    scale = np.abs(np.random.randn(2)) + 0.1
    scale[1] = scale.mean()
    data[lasti:nexti] = np.random.normal(size=(nexti-lasti, 2),
                                         loc=np.random.randn(2),
                                         scale=scale / i)
    lasti = nexti
color = (0.3, 0.5, 0.8)
n_bins = 100









color = (0.3, 0.5, 0.8)
n_bins = 100

fig = vp.Fig(show=False)
line = fig[0:4, 0:4].plot(data, symbol='o', width=0,
                          face_color=color + (1,), edge_color=None,
                          marker_size=4)
print(line,type(line))
line.set_gl_state(depth_test=False)
a = fig[4, 0:4].histogram(data[:, 0], bins=n_bins, color=color, orientation='h')
fig[0:4, 4].histogram(data[:, 1], bins=n_bins, color=color, orientation='v')










w = QMainWindow()
widget = QWidget()
w.setCentralWidget(widget)
widget.setLayout(QVBoxLayout())
widget.layout().addWidget(fig.native)
widget.layout().addWidget(QPushButton())
w.show()
vispy.app.run()