
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
from PyQt5.Qt import QMainWindow, QApplication
import numpy as np
from cloud import CloudModel, settings, ValueError
from colorsys import hsv_to_rgb


model = CloudModel()

arrow_ang = 15*np.pi/180

Vmax = 20

class GLWidget(QGLWidget):
    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)

        self.U = model.U[:, :, 3]
        self.V = model.V[:, :, 3]

        self.full_V = np.sqrt(np.power(self.U, 2) + np.power(self.V, 2))
        self.angle = np.arctan2(self.V, self.U)

    def timerEvent(self, QTimerEvent):
        try:
            self.U,self.V, A,S = model.new_state()
        except ValueError as e:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            delta_x = 2 / settings["size_x"]
            delta_y = 2 / settings["size_y"]
            for i in range(0, settings["size_x"]):
                glColor3d(0, 1, 0)
                glBegin(GL_LINES)
                glVertex2d(i * delta_x - 1, -1)
                glVertex2d(i * delta_x - 1, 1)
                
                glEnd()

            for i in range(0, settings["size_y"]):
                glBegin(GL_LINES)
                glVertex2d(-1, i * delta_y - 1)
                glVertex2d(1, i * delta_y - 1)
                
                glEnd()

            i,j,k = e.val
            print(i,j,k)
            glColor3d(1, 0, 0)
            glBegin(GL_QUADS)
            glVertex2d(i * delta_x -1, j * delta_y -1)
            glVertex2d((i+1) * delta_x -1, j * delta_y -1)
                    
            glVertex2d((i+1) * delta_x -1, (j+1) * delta_y -1)
            glVertex2d(i * delta_x -1, (j+1) * delta_y -1)
            glEnd()
        
            glFlush()
            self.swapBuffers()
            self.killTimer(self.timerId)
        self.U = self.U[:,:,3]
        self.V = self.V[:,:,3]
        self.full_V = np.sqrt(np.power(self.U, 2) + np.power(self.V, 2))
        self.angle = np.arctan2(self.V, self.U)

        self.paintGL()
        self.swapBuffers()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        delta_x = 2 / settings["size_x"]
        delta_y = 2 / settings["size_y"]
        #for i in range(0, settings["size_x"]):
            #glColor3d(0, 1, 0)
            #glBegin(GL_LINES)
            #glVertex2d(i * delta_x - 1, -1)
            #glVertex2d(i * delta_x - 1, 1)

            #glEnd()

        #for i in range(0, settings["size_y"]):
         #   glBegin(GL_LINES)
          #  glVertex2d(-1, i * delta_y - 1)
           # glVertex2d(1, i * delta_y - 1)

            #glEnd()

        for i in range(0, settings["size_x"]):
            for j in range(0, settings["size_y"]):
                center_pos = (i * delta_x + delta_x / 2 - 1, j * delta_y + delta_x / 2 - 1)
                ang = self.angle[i, j]
                l = np.minimum(delta_x, delta_y) / 2

                x0 = l * np.cos(ang) + center_pos[0]
                y0 = l * np.sin(ang) + center_pos[1]

                x1 = center_pos[0] - l * np.cos(ang)
                y1 = center_pos[1] - l * np.sin(ang)

                y2 = y0 - l / 3 * np.sin(ang - arrow_ang)
                x2 = x0 - l / 3 * np.cos(ang - arrow_ang)

                y3 = y0 - l / 3 * np.cos(np.pi / 2 - ang - arrow_ang)
                x3 = x0 - l / 3 * np.sin(np.pi / 2 - ang - arrow_ang)

                hue = self.full_V[i, j] / Vmax
                
                #print(self.full_V[i, j])
                if hue > 1:
                    hue = (hue -int(hue))*0.5
                else:
                    hue = hue*0.5
                    
                
                color = hsv_to_rgb(hue, 1, 1)

                glColor3d(color[0], color[1], color[2])
                glLineWidth(2)
                glBegin(GL_LINES)
                glVertex2d(x0, y0)
                glVertex2d(x1, y1)

                glVertex2d(x0, y0)
                glVertex2d(x2, y2)

                glVertex2d(x0, y0)
                glVertex2d(x3, y3)

                glEnd()
        glFlush()

    def resizeGL(self, w, h):

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, 1.0, 1.0, 30.0)

    def initializeGL(self):
        glClearColor(0.2, 0, 1.0, 0.5)
        glClearDepth(1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, 1.0, 1.0, 30.0)
        self.timerId = self.startTimer(100)


class GLWidget_RO(QGLWidget):
    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)
        #Dro = np.gradient(model.Ro, model.delta.x,model.delta.y,model.delta.z)
        
        #self.U = Dro[0][:, :, 0]
        #self.V = Dro[1][:, :, 0]
        self.Ro = model.Ro[:,:,3]
        self.max = 3


    def timerEvent(self, QTimerEvent):

        a,s,d,self.Ro = model.new_state()
        self.Ro = self.Ro[:,:,3]
        self.paintGL()
        self.swapBuffers()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        delta_x = 2 / settings["size_x"]
        delta_y = 2 / settings["size_y"]
        for i in range(0, settings["size_x"]):
            glColor3d(0, 1, 0)
            glBegin(GL_LINES)
            glVertex2d(i * delta_x - 1, -1)
            glVertex2d(i * delta_x - 1, 1)

            glEnd()

        for i in range(0, settings["size_y"]):
            glBegin(GL_LINES)
            glVertex2d(-1, i * delta_y - 1)
            glVertex2d(1, i * delta_y - 1)

            glEnd()

        for i in range(0, settings["size_x"]):
            for j in range(0, settings["size_y"]):
                hue = self.Ro[i,j]/self.max
                
                #print(self.full_V[i, j])
                if hue > 1:
                    hue = (hue -int(hue))*0.5
                else:
                    hue = hue*0.5
                    
                
                color = hsv_to_rgb(hue, 1, 1)

                glColor3d(color[0], color[1], color[2])
                glBegin(GL_QUADS)
                glVertex2d(i * delta_x -1, j * delta_y -1)

                glVertex2d((i+1) * delta_x -1, j * delta_y -1)
                
                glVertex2d((i+1) * delta_x -1, (j+1) * delta_y -1)
                glVertex2d(i * delta_x -1, (j+1) * delta_y -1)
                glEnd()
        glFlush()

    def resizeGL(self, w, h):

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, 1.0, 1.0, 30.0)

    def initializeGL(self):
        glClearColor(1.0, 1.0, 0, 1.0)
        glClearDepth(1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, 1.0, 1.0, 30.0)
        self.startTimer(100)




class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        #self.setMinimumSize(500, 500)
        widget = GLWidget_RO(self)
        self.setCentralWidget(widget)


app = QApplication(['Навье-Стокс для красавчиков:))'])
window = Window()

window.show()
app.exec_()