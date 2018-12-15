from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d
import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
from sympy import Symbol, sympify, pi
from my_fuzzy import fx, f

class MyMplCanvas(FigureCanvas):
    """FigureCanvas的最終的父類別是QWidget"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        # 配置中文顯示
        #plt.rcParams['font.family'] = ['SimHei']    # 用来正常顯示中文
        #plt.rcParams['axes.unicode_minus'] = False  # 用来正常顯示正負號
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)    # 新建一個figure
        #self.axes = self.fig.add_subplot(111)                  # 建立一個子圖，如果要建立複合圖，可以在這裡修改

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        '''定義FigureCanvas的尺寸策略，這部分的意思是設置FigureCanvas，使之盡可能向外填充空間。'''
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    

    def plot(self, u1, u2, fg=None):
        pts = 100
        a = np.linspace(u1[0], u1[1], pts)
        b= np.linspace(u2[0], u2[1], pts)

        x1 = Symbol('x1')
        x2 = Symbol('x2')
        
        label = fg
        gx = np.zeros([pts, pts])
        x1_ = np.zeros([pts, pts])
        x2_ = np.zeros([pts, pts])
 
        for i in range(pts):
            for j in range(pts):
                x1_[i,j] = a[i]
                x2_[i,j] = b[j]

                if fg:
                    gx[i,j] = f(fg, a[i], b[j])
                else:
                    gx[i,j] = fx(a[i], b[j])
                    label = "0.5*x1**2 + 0.2*x2**2 + 0.7*x2 - 0.5*x1*x2"
        #self.fig.clf()   # clear the entire current figure

        self.fig.gca().cla()         # 清空當前的axes
        self.fig.clear()             # 清空colorbar
        self.ax = self.fig.gca(projection='3d')

        wire = self.ax.plot_wireframe(x1_, x2_, gx, label=str(label), rstride=5, cstride=5)

        # Retrive data from internal storage of plot_wireframe, then delete it
        nx, ny, _ = np.shape(wire._segments3d)
        wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
        wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
        wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
        wire.remove()

        # create data for a LineCollection
        wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
        wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
        wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
        to_delete = np.arange(0, nx*ny, ny)
        wire_x1 = np.delete(wire_x1, to_delete, axis=1)
        wire_y1 = np.delete(wire_y1, to_delete, axis=1)
        wire_z1 = np.delete(wire_z1, to_delete, axis=1)
        scalars = np.delete(wire_z, to_delete)

        segs = [list(zip(xl, yl, zl)) for xl, yl, zl in zip(wire_x1.T, wire_y1.T, wire_z1.T)]

        # Plots the wireframe by a  a line3DCollection
        my_wire = art3d.Line3DCollection(segs, cmap="hsv")
        my_wire.set_array(scalars)
        self.ax.add_collection(my_wire)

        self.fig.colorbar(my_wire)
  
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('g(x1,x2)')
        #self.ax.legend()
        self.fig.canvas.draw()

        

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=10, height=9, dpi=100)
        self.mpl_ntb = NavigationToolbar(self.mpl, self)  # 添加完整的 toolbar

        self.layout.addWidget(self.mpl)
        self.layout.addWidget(self.mpl_ntb)