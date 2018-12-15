# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication,  QMainWindow
from PyQt5.QtGui import QIcon
import PyQt5.QtGui as QtGui
from PyQt5 import QtCore

from mpl_toolkits.mplot3d import art3d

from fuzzy_HW2 import Ui_MainWindow           # 繼承Qt Designer
from my_fuzzy import *
import math
from sympy import Symbol, cos, sin, diff, limit, sympify, pi   # 計算fuzzynum


class MymainWin(QMainWindow, Ui_MainWindow):
    def __init__(self,  parent=None):
        super(MymainWin,  self).__init__(parent)          # 繼承Qt5物件的方法
        # Qt Designer內的物件
        self.setupUi(self)
        self.setWindowTitle("HW2 fuzzy modeling")
        self.matplotlibwidget_result.setVisible(False)

        # 設定字體和大小
        self.edit_e.setFont(QtGui.QFont('SansSerif', 12))
        self.edit_equation.setFont(QtGui.QFont('SansSerif',12))
 
        # initial
        self.initial()
        self.fg = None

        # plot answer
        self.plot_answer()

        # signal & emmit
        self.setup()


    def initial(self):
        # 準確度設定
        self.e = float(self.edit_e.text())
        
        # 前件部範圍
        self.u1a = [int(self.edit_x1a_0.text()), int(self.edit_x1a_1.text())]    # HW2_a
        self.u2a = [int(self.edit_x2a_0.text()), int(self.edit_x2a_1.text())]
        self.u1b = [int(self.edit_x1b_0.text()), int(self.edit_x1b_1.text())]    # HW2_b
        self.u2b = [int(self.edit_x2b_0.text()), int(self.edit_x2b_1.text())]

        self.xa1_range = np.around(np.arange(self.u1a[0], self.u1a[1]+0.001, 0.001), decimals=3)
        self.xa2_range = np.around(np.arange(self.u2a[0], self.u2a[1]+0.001, 0.001), decimals=3)
        self.xb1_range = np.around(np.arange(self.u1b[0], self.u1b[1]+0.001, 0.001), decimals=3)
        self.xb2_range = np.around(np.arange(self.u2b[0], self.u2b[1]+0.001, 0.001), decimals=3)

        # 所需要的fuzzyset數量
        self.xn1a = int(self.premise_xn1a.text())
        self.xn2a = int(self.premise_xn2a.text())
        self.consequent_gna.setText(str(self.xn1a*self.xn2a))

        self.xn1b = int(self.premise_xn1b.text())
        self.xn2b = int(self.premise_xn2b.text())
        self.consequent_gnb.setText(str(self.xn1b*self.xn2b))

        # fuzzyset
        self.Ax1a = get_fuzzyset(self.xa1_range, self.xn1a)
        self.Ax2a = get_fuzzyset(self.xa2_range, self.xn2a)

        self.Ax1b = get_fuzzyset(self.xb1_range, self.xn1b)
        self.Ax2b = get_fuzzyset(self.xb2_range, self.xn2b)

        # 模糊集合觸發點
        self.fire_Ax1a = np.around(np.linspace(self.u1a[0], self.u1a[1], self.xn1a), decimals=3)
        self.fire_Ax2a = np.around(np.linspace(self.u2a[0], self.u2a[1], self.xn2a), decimals=3)
        self.fire_Ax1b = np.around(np.linspace(self.u1b[0], self.u1b[1], self.xn1b), decimals=3)
        self.fire_Ax2b = np.around(np.linspace(self.u2b[0], self.u2b[1], self.xn2b), decimals=3)


    def setup(self):
        # change equation
        self.pushButton_go.clicked.connect(self.Compute_Universal_Approximation_Theorem)

        # plot fuzzyset
        self.pushButton_plot_a.clicked.connect(self.plot_fuzzyset_a)
        self.pushButton_plot_b.clicked.connect(self.plot_fuzzyset_b)

        # plot result
        self.pushButton_result_a.clicked.connect(self.plot_result_a)
        self.pushButton_result_b.clicked.connect(self.plot_result_b)

        # fuzzyset change
        self.premise_xn1a.textChanged.connect(self.initial)
        self.premise_xn2a.textChanged.connect(self.initial)
        self.premise_xn1b.textChanged.connect(self.initial)
        self.premise_xn2b.textChanged.connect(self.initial)

        # arange change
        self.edit_x1a_0.textChanged.connect(self.initial)
        self.edit_x1a_1.textChanged.connect(self.initial)
        self.edit_x2a_0.textChanged.connect(self.initial)
        self.edit_x2a_1.textChanged.connect(self.initial)

        self.edit_x1b_0.textChanged.connect(self.initial)
        self.edit_x1b_1.textChanged.connect(self.initial)
        self.edit_x2b_0.textChanged.connect(self.initial)
        self.edit_x2b_1.textChanged.connect(self.initial)

        # 準確度設定
        self.edit_e.textChanged.connect(self.initial)

    # 計算模糊集合數量
    def Compute_Universal_Approximation_Theorem(self):
        e = self.e
        pts = 50

        self.fg = sympify(self.edit_equation.text())    # str -> equation
        self.plot_answer(self.fg)

        x1 = Symbol('x1')
        x2 = Symbol('x2')

        u1 = [self.u1a[0], self.u1a[1]]
        u2 = [self.u2a[0], self.u2a[1]]
        
        
        a = np.linspace(u1[0], u1[1], pts)
        b = np.linspace(u2[0], u2[1], pts)


        #print("---------------------------(a)--------------------------------")
        # 定理12.2 (a)
        fg_x1 = diff(self.fg, x1)
        fg_x2 = diff(self.fg, x2)

        max_fg_x1 = 0
        max_fg_x2 = 0

        # 計算最大值
        for i in range(pts):
            for j in range(pts):
                if max_fg_x1 < abs(fg_x1.evalf(subs={x1: a[i], x2: b[j]})):
                    max_fg_x1 = abs(fg_x1.evalf(subs={x1: a[i], x2: b[j]}))
                if max_fg_x2 < abs(fg_x2.evalf(subs={x1: a[i], x2: b[j]})):
                    max_fg_x2 = abs(fg_x2.evalf(subs={x1: a[i], x2: b[j]}))


        #print("max_fg_x1=",round(max_fg_x1, 5))
        #print("max_fg_x2=",round(max_fg_x2,5))

        _translate = QtCore.QCoreApplication.translate
        self.label_fx1.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">max_f</span><span style=\" font-size:12pt; vertical-align:sub;\">x1</span><span style=\" font-size:12pt;\"> = "+str(round(max_fg_x1, 5))+"< /span > </p > </body > </html >"))
        self.label_fx2.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">max_f</span><span style=\" font-size:12pt; vertical-align:sub;\">x2</span><span style=\" font-size:12pt;\"> = "+str(round(max_fg_x2, 5)) +"< /span > </p > </body > </html >"))


        ha = round(e/(max_fg_x1+max_fg_x2+1e-5), 5)
        xn1 = math.ceil((u1[1]-u1[0])/ha + 1)
        xn2 = math.ceil((u2[1]-u2[0])/ha + 1)

        self.premise_xn1a.setText(str(xn1))
        self.premise_xn2a.setText(str(xn2))

        #print("ha=", ha)
        #print("error=", round(max_fg_x1*ha+max_fg_x2*ha, 2))

        self.label_ha.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">ha = "+str(ha)+"</span></p></body></html>"))
        self.label_ea.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">error = "+str(round(max_fg_x1*ha+max_fg_x2*ha, 2))+"< /span > </p > </body > </html >"))
        
        
        #print("---------------------------(b)--------------------------------")
        # 定理12.3 (b)
        fg_x1_x1 = diff(fg_x1, x1)
        fg_x2_x2 = diff(fg_x2, x2)

        max_fg_x1_x1 = 0
        max_fg_x2_x2 = 0

        # 計算最大值
        for i in range(pts):
            for j in range(pts):
                if max_fg_x1_x1 < abs(fg_x1_x1.evalf(subs={x1: a[i], x2: b[j]})):
                    max_fg_x1_x1 = abs(fg_x1_x1.evalf(subs={x1: a[i], x2: b[j]}))
                if max_fg_x2_x2 < abs(fg_x2_x2.evalf(subs={x1: a[i], x2: b[j]})):
                    max_fg_x2_x2 = abs(fg_x2_x2.evalf(subs={x1: a[i], x2: b[j]}))

        #print("max_fg_x1_x1=", round(max_fg_x1_x1,5))
        #print("max_fg_x2_x2=",round(max_fg_x2_x2,5))
        self.label_fx1x1.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">max_f</span><span style=\" font-size:12pt; vertical-align:sub;\">x1x1</span><span style=\" font-size:12pt;\"> = "+str(round(max_fg_x1_x1,5))+"</span></p></body></html>"))
        self.label_fx2x2.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">max_f</span><span style=\" font-size:12pt; vertical-align:sub;\">x2x2</span><span style=\" font-size:12pt;\"> = "+str(round(max_fg_x2_x2, 5))+"</span > </p > </body > </html >"))



        hb = round(((8*e)/(max_fg_x1_x1+max_fg_x2_x2+1e-5))**0.5, 5)
        xn1 = math.ceil((u1[1]-u1[0])/hb + 1)
        xn2 = math.ceil((u2[1]-u2[0])/hb + 1)

        self.premise_xn1b.setText(str(xn1))
        self.premise_xn2b.setText(str(xn2))

        #print("hb=", hb)
        #print("error=", round((max_fg_x1_x1*hb**2+max_fg_x2_x2*hb**2)/8, 3))
        self.label_hb.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">hb = "+str(hb)+"</span></p></body></html>"))
        self.label_eb.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">error = "+str(round((max_fg_x1_x1*hb**2+max_fg_x2_x2*hb**2)/8, 3))+"</span></p></body></html>"))

        


    def plot_answer(self, fg=None):
        self.matplotlibwidget_result.setVisible(True)
        self.matplotlibwidget_result.mpl.plot(self.u1a, self.u2a, fg)


    def arg_plot_fuzzyset_a(self):
        if self.comboBox_a.currentIndex()==0:
            return self.xa1_range, self.Ax1a
        elif self.comboBox_a.currentIndex()==1:
            return self.xa2_range, self.Ax2a

    def arg_plot_fuzzyset_b(self):
        if self.comboBox_b.currentIndex() == 0:
            return self.xb1_range, self.Ax1b
        elif self.comboBox_b.currentIndex() == 1:
            return self.xb2_range, self.Ax2b
        

    def plot_fuzzyset_a(self):
        arange, Ax = self.arg_plot_fuzzyset_a()
        plot_fuzzyset(arange, Ax)
    
    def plot_fuzzyset_b(self):
        arange, Ax = self.arg_plot_fuzzyset_b()
        plot_fuzzyset(arange, Ax)


    def plot_result_a(self):
        x1_ = np.zeros([self.xn1a, self.xn2a])
        x2_ = np.zeros([self.xn1a, self.xn2a])
        fuzzy_fx = np.zeros([self.xn1a, self.xn2a])

        x1 = Symbol('x1')
        x2 = Symbol('x2')

        for i in range(self.xn1a):
            for j in range(self.xn2a):
                # x
                x1_[i,j] = self.fire_Ax1a[i]
                x2_[i,j] = self.fire_Ax2a[j]

                # membership
                tmp_x1 = int(np.where(self.xa1_range == self.fire_Ax1a[i])[0])
                tmp_x2 = int(np.where(self.xa2_range == self.fire_Ax2a[j])[0])
                Ai1 = self.Ax1a[i][tmp_x1]
                Aj2 = self.Ax2a[j][tmp_x2]

                # f(x)
                # 假設f(x)=g(x)
                if self.fg:
                    fuzzy_fx[i,j] = (f(self.fg, self.fire_Ax1a[i], self.fire_Ax2a[j]) * Ai1 * Aj2) / (Ai1*Aj2)
                else:
                    fuzzy_fx[i,j]=(fx(self.fire_Ax1a[i],self.fire_Ax2a[j])*Ai1*Aj2) / (Ai1*Aj2)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        wire = ax.plot_wireframe(x1_, x2_, fuzzy_fx, rstride=5, cstride=5)

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

        segs = [list(zip(xl, yl, zl)) for xl, yl, zl in
                zip(wire_x1.T, wire_y1.T, wire_z1.T)]

        # Plots the wireframe by a  a line3DCollection
        my_wire = art3d.Line3DCollection(segs, cmap="hsv")
        my_wire.set_array(scalars)
        ax.add_collection(my_wire)

        plt.colorbar(my_wire)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('g(x1,x2)')
        #ax.legend()
        plt.show()

    def plot_result_b(self):
        x1_ = np.zeros([self.xn1b, self.xn2b])
        x2_ = np.zeros([self.xn1b, self.xn2b])
        fuzzy_fx = np.zeros([self.xn1b, self.xn2b])

        x1 = Symbol('x1')
        x2 = Symbol('x2')

        for i in range(self.xn1b):
            for j in range(self.xn2b):
                # x
                x1_[i,j] = self.fire_Ax1b[i]
                x2_[i,j] = self.fire_Ax2b[j]

                # membership
                tmp_x1 = int(np.where(self.xb1_range == self.fire_Ax1b[i])[0])
                tmp_x2 = int(np.where(self.xb2_range == self.fire_Ax2b[j])[0])
                Ai1 = self.Ax1b[i][tmp_x1]
                Aj2 = self.Ax2b[j][tmp_x2]
    
                # f(x)
                if self.fg:
                    fuzzy_fx[i,j] = (f(self.fg, self.fire_Ax1b[i], self.fire_Ax2b[j]) * Ai1 * Aj2) / (Ai1*Aj2)
                else:
                    fuzzy_fx[i,j] = (fx(self.fire_Ax1b[i],self.fire_Ax2b[j])*Ai1*Aj2) / (Ai1*Aj2)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        wire = ax.plot_wireframe(x1_, x2_, fuzzy_fx, rstride=5, cstride=5)

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

        segs = [list(zip(xl, yl, zl)) for xl, yl, zl in
                zip(wire_x1.T, wire_y1.T, wire_z1.T)]

        # Plots the wireframe by a  a line3DCollection
        my_wire = art3d.Line3DCollection(segs, cmap="hsv")
        my_wire.set_array(scalars)
        ax.add_collection(my_wire)

        plt.colorbar(my_wire)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('g(x1,x2)')
        #ax.legend()
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ICIP_lab.png"))
    win = MymainWin()

    win.show()
    sys.exit(app.exec_())
    #app.exec_()
