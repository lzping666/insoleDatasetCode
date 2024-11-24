# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets

class Terrain(object):
    def __init__(self, vid_output_path):
        """
        Initialize the graphics window for visualizing contact forces
        """

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('3D Contact Forces')
        self.window.setGeometry(1000, 0, 1920, 1080)
        self.window.setCameraPosition(distance=1.5, elevation=1.8)
        self.window.setBackgroundColor(0.8)
        self.window.show()

        zgrid = gl.GLGridItem(color=(255, 255, 255, 226))
        self.window.addItem(zgrid)

        data = np.load(vid_output_path)
        grf = data['pred_grf'][0]  # Assume shape is (frames, points, 3)
        num_points = grf.shape[0]
        print("GRF shape:", grf.shape)
        contact = np.zeros_like(grf)
        vis_contact_scale = 200
        contact_end = contact + grf / vis_contact_scale
        self.contact_force = np.stack([contact, contact_end], axis=2)

        self.line_contact = {}
        for i in range(num_points):
            self.line_contact[str(i)] = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                                          color=pg.glColor((0, 255, 0)),
                                                          width=3,
                                                          antialias=True)
            self.window.addItem(self.line_contact[str(i)])

        self.frame_idx = 0

    def update(self):
        print('Visualizing contact forces for frame:', self.frame_idx)
        for i in range(self.contact_force.shape[1]):
            self.line_contact[str(i)].setData(pos=self.contact_force[self.frame_idx, i])

        self.frame_idx = (self.frame_idx + 1) % self.contact_force.shape[0]

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def animation(self, frametime=100):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_output_path', type=str, required=True, help='Path to the inferred motion and forces data')
    args = parser.parse_args()

    t = Terrain(args.vid_output_path)
    t.animation()