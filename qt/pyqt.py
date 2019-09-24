# import sys
# from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PyQt5.QtCore import Qt
# import torch


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.label = QtWidgets.QLabel()
#         canvas = QtGui.QPixmap(400, 300)
#         self.label.setPixmap(canvas)
#         self.setCentralWidget(self.label)
#         self.draw_something()

#     def draw_something(self):
#         painter = QtGui.QPainter(self.label.pixmap())
#         painter.drawLine(10, 10, 300, 200)
#         painter.end()


# m = torch.jit.load('/home/liam/cppn_trace_nice_cool_big2.pth')
# m = m.cuda()
# import pdb; pdb.set_trace()
# x, o = get_xy_mesh([200, 200]), torch.FloatTensor([0.3, 0.4])


# app = QtWidgets.QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec_()

import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0, 0, 1000, 1000))

        #  image plot
        self.img = pg.ImageItem(border=None)
        self.view.addItem(self.img)

        # Set Data

        self.x = np.linspace(0, 50., num=100)
        self.X, self.Y = np.meshgrid(self.x, self.x)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Start
        self._update()

    def _update(self):

        self.data = np.random.rand(1000, 1000, 3)
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
