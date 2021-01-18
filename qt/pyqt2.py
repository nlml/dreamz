import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        app = pg.mkQApp()
        self.win = pg.GraphicsWindow(title="Live webcam")
        self.win.resize(100, 100)

        box = self.win.addViewBox(lockAspect=True)
        box.invertY()
        self.vis = pg.ImageItem(border=None)
        box.addItem(self.vis)

        self.label = QtGui.QLabel()
        # self.win.addWidget(self.label)

        #### Set Data  #####################
        self.counter = 0
        self.fps = 0.0
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):

        self.data = np.random.rand(100, 100, 3)
        self.vis.setImage(self.data)

        now = time.time()
        dt = now - self.lastupdate
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.99 + fps2 * 0.01
        tx = "Mean Frame Rate:  {fps:.3f} FPS".format(fps=self.fps)
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
