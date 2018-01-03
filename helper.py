#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:29:19 2017

@author: alex
"""

import matplotlib.pyplot as plt

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
from PyQt5.Qt import QMainWindow, QApplication
import numpy as np
from cloud import CloudModel, settings
from colorsys import hsv_to_rgb


class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        widget = GLWidget(self)
        self.setCentralWidget(widget)



arrow_ang = 15*np.pi/180

Vmax = 10

def show_distr(model, z = 0):
    
    if z > settings["z_max"] or z < settings["z_min"]: 
        print("inapropriate Z")
        return
    
    num = int(z/model.delta.z)
    
    if z >0 : num = num + int(settings["size_z"]/2)
    if z <0 : num = int(settings["size_z"]/2) - num
    
    U = model.U[:,:,num]
    V = model.V[:,:,num]
    
    full_V = np.sqrt(np.power(U,2) + np.power(V,2))
    angle = np.arctan2(V,U)
    
    print("full-V-shape:", full_V.shape)
    print("angle-shape:", angle.shape)
    
    
    class GLWidget(QGLWidget):
        def __init__(self,parent):
            QGLWidget.__init__(self,parent)
            self.setMinimumSize(500,500)
    
        def timerEvent(self, QTimerEvent):
            model.new_state()
            self.paintGL()
            self.swapBuffers()
        
        def paintGL(self):

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            delta_x = 2/settings["size_x"]
            delta_y = 2/settings["size_y"]
            for i in range(0, settings["size_x"]):
                
                glColor3d(0,1,0)
                glBegin(GL_LINES)
                glVertex2d(i*delta_x -1,-1)
                glVertex2d(i*delta_x -1, 1)

                glEnd()
                
            for i in range(0, settings["size_y"]):
                glBegin(GL_LINES)
                glVertex2d(-1, i*delta_y - 1)
                glVertex2d(1, i*delta_y - 1)

                glEnd()             
                      
            for i in range(0, settings["size_x"]):
                for j in range(0, settings["size_y"]):
                    center_pos = (i*delta_x +delta_x/2 -1, j*delta_y +delta_x/2 -1)
                    ang = angle[i,j]
                    l = np.minimum(delta_x, delta_y)/2
                    
                    x0 = l*np.cos(ang) + center_pos[0]
                    y0 = l*np.sin(ang) + center_pos[1]
                    
                    x1 = center_pos[0] - l*np.cos(ang)
                    y1 = center_pos[1] - l*np.sin(ang)
                    
                    y2 = y0 - l/3*np.sin(ang - arrow_ang)
                    x2 = x0 - l/3*np.cos(ang - arrow_ang)
                    
                    y3 = y0 - l/3*np.cos(np.pi/2 - ang - arrow_ang)
                    x3 = x0 - l/3*np.sin(np.pi/2 - ang - arrow_ang)
                    
                    
                    hue = full_V[i,j]/Vmax
                    if hue >1: 
                        hue = 360
                    else: hue = hue*360
                    
                    color = hsv_to_rgb(hue,1,1)
                    
                    glColor3d(color[0],color[1],color[2])
                    glBegin(GL_LINES)
                    glVertex2d(x0,y0)
                    glVertex2d(x1,y1)
                    
                    glVertex2d(x0,y0)
                    glVertex2d(x2,y2)
                    
                    glVertex2d(x0,y0)
                    glVertex2d(x3,y3)
                    
                    glEnd()

            glFlush()

        def resizeGL(self, w, h):
        
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(40.0, 1.0, 1.0, 30.0)
            
        def initializeGL(self):
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClearDepth(1.0)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(40.0, 1.0, 1.0, 30.0)
            self.startTimer(2000)
    
    
    class Window(QMainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.setMinimumSize(500, 500)
            widget = GLWidget(self)
            self.setCentralWidget(widget)
    
    app = QApplication(['Spiral Widget Demo'])
    window = Window()

    window.show()
    app.exec_()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    