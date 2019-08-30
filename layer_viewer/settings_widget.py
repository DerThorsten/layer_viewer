import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import (
    Parameter,
    ParameterTree,
    ParameterItem,
    registerParameterType,
)


class SettingsWidget(ParameterTree):
    def __init__(self, *args, **kwargs):
        super(SettingsWidget, self).__init__(*args, **kwargs)

        self.qtSettings = QtCore.QSettings("layer_viewer", "settings")

        useOpenGl = self.qtSettings.value("Use OpenGL", True)
        useAA = self.qtSettings.value("Use Anti-Aliasing", False)
        pattern = self.qtSettings.value("bg-type", "LinearGradientPattern")

        params = [
            {
                "name": "Global Options",
                "type": "group",
                "children": [
                    {
                        "name": "Use OpenGL",
                        "type": "bool",
                        "value": useOpenGl,
                        "tip": "can lead to speedups if enabled",
                    },
                    {"name": "Use Anti-Aliasing", "type": "bool", "value": useAA},
                ],
            },
            {
                "name": "ViewBox Options",
                "type": "group",
                "children": [
                    {
                        "name": "ViewBox Background",
                        "type": "group",
                        "children": [
                            {
                                "name": "bg-type",
                                "type": "list",
                                "values": [
                                    "SolidPattern",
                                    "LinearGradientPattern",
                                    "Dense1Pattern",
                                    "Dense2Pattern",
                                    "Dense3Pattern",
                                    "Dense4Pattern",
                                    "Dense5Pattern",
                                    "Dense6Pattern",
                                    "Dense7Pattern",
                                    "NoBrush",
                                    "HorPattern",
                                    "VerPattern",
                                    "CrossPattern",
                                    "BDiagPattern",
                                    "FDiagPattern",
                                    "DiagCrossPattern",
                                ],
                                "value": pattern,
                            },
                            {
                                "name": "bg-color 1",
                                "type": "color",
                                "value": (180,) * 3,
                                "tip": "background color 1",
                            },
                            {
                                "name": "bg-color 2",
                                "type": "color",
                                "value": (60,) * 3,
                                "tip": "background color 2",
                            },
                        ],
                    },
                    {"name": "Show Axis", "type": "bool", "value": False},
                ],
            },
            # {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
            #     {'name': 'Save State', 'type': 'action'},
            #     {'name': 'Restore State', 'type': 'action', 'children': [
            #         {'name': 'Add missing items', 'type': 'bool', 'value': True},
            #         {'name': 'Remove extra items', 'type': 'bool', 'value': True},
            #     ]},
            # ]}
        ]

        self.p = Parameter.create(name="params", type="group", children=params)

        self.setParameters(self.p, showTop=False)
        self.setWindowTitle("Layer Viewer Settings")
        # TODO: the parameters are not saved beyond the exit of the program, do that

    def __getitem__(self, key):
        return self.p[key]
