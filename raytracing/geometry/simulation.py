from ..plotting import Plotable
from .base import bounding_box


class Simulation(Plotable):
    def __init__(self, scene, TX, *RXS):
        super().__init__()

        self.scene = scene
        self.TX = TX
        self.RXS = RXS
        # self.domain = bounding_box([scene.domain, TX.domain, *[RX.domain for RX in RXS]])

    def plot(self):
        self.scene.on(self.ax).plot()
        self.TX.on(self.ax).plot(color="b")

        for RX in self.RXS:
            RX.on(self.ax).plot(color="r")

        return self.ax
