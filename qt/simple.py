import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


class App:
    def __init__(self, parent=None):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(600, 600)
        self.img = pg.ImageItem()
        self.plot = self.win.addPlot()
        self.plot.addItem(self.img)
        self.win.setWindowTitle('My Title')
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_for_new_data_and_replot)
        self.timer.start(100)
        self.win.show()
        ####  Set Data  
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        self._update()

    def _update(self):

        self.data = np.random.rand(100, 100, 3)
        self.img.setImage(self.data)

        now = time.time()
        dt = (now - self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.99 + fps2 * 0.01
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
