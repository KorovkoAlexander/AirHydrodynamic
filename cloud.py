import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from akima import interpolate
from enum import Enum

settings = {"size_x": 50,
            "size_y": 50,
            "size_z": 5,
            "x_min": -500,
            "x_max": 500,
            "y_min": -500,
            "y_max": 500,
            "z_min": 0,
            "z_max": 500,
            "time_tick": 0.1} # 1second

glob_vars = {"Cp" : 1006,
             "gamma":0.009741,
             "l":1.8961e-5,
             "l1":1.327e-5,
             "k": 50,
             "kz": 15,
             "w":1/(24*60*60),
             "g": 9.81}

class Delta():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0


class CloudModel():
    def __init__(self):
        delta = Delta()
        delta.x = (settings["x_max"] - settings["x_min"])/settings["size_x"]
        delta.y = (settings["y_max"] - settings["y_min"])/settings["size_y"]
        delta.z = (settings["z_max"] - settings["z_min"])/settings["size_z"]
        self.delta = delta
        
        self.scp1 = Karetka(settings["size_y"], settings["size_z"], 30, 50)
        self.scp2 = Karetka(settings["size_y"], settings["size_z"], 30, 50)
        
        size = (settings["size_x"],settings["size_y"],settings["size_z"])
        self.U = np.zeros(size)
        self.V = np.zeros(size)
        self.W = np.zeros(size)
        self.Ro = np.random.uniform(3, 4, size)
        
        a,b,c = self.scp1.get()
        
        self.U[0,:,:] = a
        self.V[0,:,:] = b
        self.W[0,:,:] = c
        
        a,b,c = self.scp2.get()
        
        self.U[-1,:,:] = a
        self.V[-1,:,:] = b
        self.W[-1,:,:] = c
        
        self.Ro[0,:,:] = np.random.uniform(3, 4, (settings["size_y"], settings["size_z"]))
        self.Ro[-1,:,:] = np.random.uniform(3, 4, (settings["size_y"], settings["size_z"]))         
        self.Ro[:,0,:] = np.random.uniform(3, 4, (settings["size_x"], settings["size_z"]))         
        self.Ro[:,-1,:] = np.random.uniform(3, 4, (settings["size_x"], settings["size_z"]))         

    def new_state(self):
        
        a,b,c = self.scp1.get()
        
        self.U[0,:,:] = a
        self.V[0,:,:] = b
        self.W[0,:,:] = c
        
        a,b,c = self.scp2.get()
        
        self.U[-1,:,:] = a
        self.V[-1,:,:] = b
        self.W[-1,:,:] = c
        
        self.U[:,0,:] = self.U[:,1,:]
        self.V[:,0,:] = self.V[:,1,:]
        self.W[:,0,:] = self.W[:,1,:]
        
        self.U[:,-1,:] = self.U[:,self.U.shape[1]-2,:]
        self.V[:,-1,:] = self.V[:,self.V.shape[1]-2,:]
        self.W[:,-1,:] = self.W[:,self.W.shape[1]-2,:]
        
        self.Ro[0,:,:] = np.random.uniform(3, 4, (settings["size_y"], settings["size_z"]))
        self.Ro[-1,:,:] = np.random.uniform(3, 4, (settings["size_y"], settings["size_z"]))         
        self.Ro[:,0,:] = np.random.uniform(3, 4, (settings["size_x"], settings["size_z"]))         
        self.Ro[:,-1,:] = np.random.uniform(3, 4, (settings["size_x"], settings["size_z"]))         
        
        
        #size = (settings["size_x"],settings["size_y"],settings["size_z"])
        newU = self.U
        newV = self.V
        newW = self.W
        newRo = self.Ro
        
        #Dp = np.gradient(8.31*self.Ro*320, self.delta.x,self.delta.y,self.delta.z)
        #a = 1 - glob_vars["gamma"]*8.31/glob_vars["g"]

        
        #print(np.sum(np.sum(np.sum(self.Ro, axis = 0), axis =0), axis = 0))
        for i in range(1,settings["size_x"] -1):
            for j in range(1,settings["size_y"] -1):
                for k in range(1,settings["size_z"]-1):
                    Fx = glob_vars["k"]*(\
                                  (self.U[i+1,j,k] - 2*self.U[i,j,k] + self.U[i-1,j,k])/(self.delta.x**2) +\
                                  (self.U[i,j+1,k] - 2*self.U[i,j,k] + self.U[i,j-1,k])/(self.delta.y**2) +\
                                  (self.U[i,j,k+1] - 2*self.U[i,j,k] + self.U[i,j,k-1])/(self.delta.x**2))
                    Fy = glob_vars["k"]*(\
                                  (self.V[i+1,j,k] - 2*self.V[i,j,k] + self.V[i-1,j,k])/(self.delta.x**2) +\
                                  (self.V[i,j+1,k] - 2*self.V[i,j,k] + self.V[i,j-1,k])/(self.delta.y**2) +\
                                  (self.V[i,j,k+1] - 2*self.V[i,j,k] + self.V[i,j,k-1])/(self.delta.x**2))
                    Fz = glob_vars["kz"]*(\
                                  (self.W[i+1,j,k] - 2*self.W[i,j,k] + self.W[i-1,j,k])/(self.delta.x**2) +\
                                  (self.W[i,j+1,k] - 2*self.W[i,j,k] + self.W[i,j-1,k])/(self.delta.y**2) +\
                                  (self.W[i,j,k+1] - 2*self.W[i,j,k] + self.W[i,j,k-1])/(self.delta.x**2))
                    
                    
                    newRo[i,j,k] = self.Ro[i,j,k] - settings["time_tick"]*( \
                         (self.U[i,j,k]*(self.Ro[i+1,j,k] - self.Ro[i-1,j,k])/(2*self.delta.x)) +\
                         (self.Ro[i,j,k]*(self.U[i+1,j,k] -  self.U[i-1,j,k])/(2*self.delta.x)) +\
                         (self.V[i,j,k]*(self.Ro[i,j+1,k] - self.Ro[i,j-1,k])/(2*self.delta.y)) +\
                         (self.Ro[i,j,k]*(self.V[i,j+1,k] -  self.V[i,j-1,k])/(2*self.delta.y)) +\
                         (self.W[i,j,k]*(self.Ro[i,j,k+1] - self.Ro[i,j,k-1])/(2*self.delta.z)) +\
                         (self.Ro[i,j,k]*(self.W[i,j,k+1] -  self.W[i,j,k-1])/(2*self.delta.z)))
                    
                     
                    if(newRo[i,j,k] <=0):
                        print("WARNING")
                        newRo[i,j,k] = 0.001
                    
                    ux = (self.U[i+1,j,k] - self.U[i-1,j,k])/(2*self.delta.x)
                    uy = (self.U[i,j+1,k] - self.U[i,j-1,k])/(2*self.delta.y)
                    uz = (self.U[i,j,k+1] - self.U[i,j,k-1])/(2*self.delta.z)
                    
                    vx = (self.V[i+1,j,k] - self.V[i-1,j,k])/(2*self.delta.x)
                    vy = (self.V[i,j+1,k] - self.V[i,j-1,k])/(2*self.delta.y)
                    vz = (self.V[i,j,k+1] - self.V[i,j,k-1])/(2*self.delta.z)
                    
                    wx = (self.W[i+1,j,k] - self.W[i-1,j,k])/(2*self.delta.x)
                    wy = (self.W[i,j+1,k] - self.W[i,j-1,k])/(2*self.delta.y)
                    wz = (self.W[i,j,k+1] - self.W[i,j,k-1])/(2*self.delta.z)
                    
                    px = 8.31*320*(self.Ro[i+1,j,k] - self.Ro[i-1,j,k])/(2*self.delta.x)
                    py = 8.31*320*(self.Ro[i,j+1,k] - self.Ro[i,j-1,k])/(2*self.delta.y)
                    pz = 8.31*320*(self.Ro[i,j,k+1] - self.Ro[i,j,k-1])/(2*self.delta.z)
                    
                    
                    VgradU = np.sum(self.U[i,j,k]*ux + self.V[i,j,k]*uy + self.W[i,j,k]*uz)
                    VgradV = np.sum(self.U[i,j,k]*vx + self.V[i,j,k]*vy + self.W[i,j,k]*vz)
                    VgradW = np.sum(self.U[i,j,k]*wx + self.V[i,j,k]*wy + self.W[i,j,k]*wz)
                    
                    #print(VgradU, VgradV, VgradW)
                    
                    newU[i,j,k] =self.U[i,j,k] + settings["time_tick"]*(-VgradU  - px/self.Ro[i,j,k] + Fx)
                        #glob_vars["l"]*self.V[i,j,k] - glob_vars["l1"]*self.W[i,j,k] + Fx)
                    #if newU[i,j,k] < 1e-10: newU[i,j,k] = 0
                    
                    newV[i,j,k] =self.V[i,j,k] + settings["time_tick"]*(-VgradV - py/self.Ro[i,j,k] + Fy)
                        #glob_vars["l"]*self.U[i,j,k] + Fy)
                    
                    #if newV[i,j,k] < 1e-10: newV[i,j,k] = 0
                    
                    newW[i,j,k] =self.W[i,j,k] + settings["time_tick"]*(-VgradW - pz/self.Ro[i,j,k] + Fz) 
                        #glob_vars["l1"]*self.U[i,j,k] + Fz) #- glob_vars["g"])
    
        
                    
        #print(np.max(newRo[:,3,3]))
        self.U = newU
        self.V = newV
        self.W = newW
        self.Ro = newRo
        return (self.U, self.V, self.W, self.Ro)
        

class ValueError(Exception):
    def __init__(self, val):
        self.val = val
        
    def __str__(self):
        return repr("R<0")
        
class Sides(Enum):
    left = 1,
    right = 2,
    top = 3,
    bottom = 4        
        
class StartCondProvider_2():
    def __init__(self):
        self.tau = 60
        self.width = 10
        
        self.y = np.zeros((2,self.tau))
        self.y[0] = self.y[1] = np.linspace(self.width/2+5, settings["size_y"] - self.width/2-5, self.tau)
        
        self.x = np.zeros((2,self.tau))
        self.x[0] = self.x[1] = np.linspace(self.width/2+5, settings["size_x"] - self.width/2-5, self.tau)

        
        mod_y0 = np.concatenate((self.y[0], np.random.uniform(self.width/2, settings["size_y"] - self.width/2, 1)))
        mod_y1 = np.concatenate((self.y[1], np.random.uniform(self.width/2, settings["size_y"] - self.width/2, 1)))
        
        mod_x0 = np.concatenate((self.x[0], np.random.uniform(self.width/2, settings["size_x"] - self.width/2, 1)))
        mod_x1 = np.concatenate((self.x[1], np.random.uniform(self.width/2, settings["size_x"] - self.width/2, 1)))
        
        x = np.concatenate((np.arange(self.tau), np.array([2*self.tau])))
        
        new_y0 = interpolate(x, mod_y0, np.arange(2*self.tau))
        new_y1 = interpolate(x, mod_y1, np.arange(2*self.tau))
        new_x0 = interpolate(x, mod_x0, np.arange(2*self.tau))
        new_x1 = interpolate(x, mod_x1, np.arange(2*self.tau))
        
        self.y = np.vstack((new_y0, new_y1))
        self.x = np.vstack((new_x0, new_x1))
        self.pointer = 0
    
    def get(self):
        if self.pointer >=self.tau:
            self.y = self.y[:,self.tau-1:-1]
            self.x = self.x[:,self.tau-1:-1]
        
            mod_y0 = np.concatenate((self.y[0], np.random.uniform(self.width/2+5, settings["size_y"] -self.width/2-5, 1)))
            mod_y1 = np.concatenate((self.y[1], np.random.uniform(self.width/2+5, settings["size_y"] -self.width/2-5, 1)))
        
            mod_x0 = np.concatenate((self.x[0], np.random.uniform(self.width/2+5, settings["size_x"] - self.width/2-5, 1)))
            mod_x1 = np.concatenate((self.x[1], np.random.uniform(self.width/2+5, settings["size_x"] - self.width/2-5, 1)))
        
            x = np.concatenate((np.arange(self.tau), np.array([2*self.tau])))
            
            new_y0 = interpolate(x, mod_y0, np.arange(2*self.tau))
            new_y1 = interpolate(x, mod_y1, np.arange(2*self.tau))
            new_x0 = interpolate(x, mod_x0, np.arange(2*self.tau))
            new_x1 = interpolate(x, mod_x1, np.arange(2*self.tau))
            self.y = np.vstack((new_y0, new_y1))
            self.x = np.vstack((new_x0, new_x1))
            
            print("hi")
            #self.y = self.y[self.tau-1:-1]
            #self.x = self.x[self.tau-1:-1]
            self.pointer = 0
            
        #left,right, top, bottom
        
        
        Ul = Ur = np.random.uniform(-10, -5, (settings["size_y"],settings["size_z"]))
        Vl = Wl = Vr = Wr = np.random.uniform(-3, 3, (settings["size_y"],settings["size_z"]))
        
        #Ur = np.random.uniform(-15, -10, (settings["size_y"],settings["size_z"]))
        #Vr = Wr = np.random.uniform(-3, 3, (settings["size_y"],settings["size_z"]))
        
        Vt = Vb = np.random.uniform(-10, -5, (settings["size_x"],settings["size_z"]))
        Ut = Wt = Ub = Wb = np.random.uniform(-3, 3, (settings["size_x"],settings["size_z"]))
        
        #left
        a= self.y[0, self.pointer]
        
        print(a)
        w = int(a + self.width/2) - int(a - self.width/2)
        Ul[int(a - self.width/2):int(a + self.width/2), :] = np.full((w, settings["size_z"]), 15)
        Vl[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        Wl[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        
        #right
        a= self.y[1, self.pointer]
        w = int(a + self.width/2) - int(a - self.width/2)
        Ur[int(a - self.width/2):int(a + self.width/2), :] = np.full((w, settings["size_z"]), 15)
        Vr[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        Wr[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        
        #top
        a= self.x[0, self.pointer]
        w = int(a + self.width/2) - int(a - self.width/2)
        Ut[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        Vt[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(10,15,(w, settings["size_z"]))
        Wt[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        
        #bottom
        a= self.x[1, self.pointer]
        w = int(a + self.width/2) - int(a - self.width/2)
        Ub[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        Vb[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(10,15,(w, settings["size_z"]))
        Wb[int(a - self.width/2):int(a + self.width/2), :] = np.random.uniform(-3,3,(w, settings["size_z"]))
        
        self.pointer = self.pointer + 1
        
        return ((Ul,Vl,Wl),(Ur,Vr,Wr),(Ut,Vt,Wt),(Ub,Vb,Wb))
        
        
        


class Karetka():
    def __init__(self, size_x,size_y, a_max, sigma):
        self.tau = 60
        self.size_x = size_x
        self.size_y = size_y
        self.a_max = a_max
        self.sigma = sigma
        
        
        self.y = self.a_max*np.sin(np.linspace(-np.pi/2,np.pi/2,self.tau))
        self.x = np.linspace(0, self.size_x, self.tau)

        mod_y = np.concatenate((self.y, np.random.uniform(self.a_max/2, self.a_max, 1)))               
        mod_x = np.concatenate((self.x, np.random.uniform(self.a_max/2, self.size_x, 1)))
            
        x = np.concatenate((np.arange(self.tau), np.array([2*self.tau])))
        
        self.y = interpolate(x, mod_y, np.arange(2*self.tau))
        self.x = interpolate(x, mod_x, np.arange(2*self.tau))
        self.pointer = 0
    
    def get(self):
        if self.pointer >=self.tau:
            self.y = self.y[self.tau-1:-1]
            self.x = self.x[self.tau-1:-1]
        
            mod_y = np.concatenate((self.y, np.random.uniform(self.a_max/2, self.a_max, 1)))               
            mod_x = np.concatenate((self.x, np.random.uniform(self.a_max/2, self.size_x, 1)))
            
            x = np.concatenate((np.arange(self.tau), np.array([2*self.tau])))
        
            self.y = interpolate(x, mod_y, np.arange(2*self.tau))
            self.x = interpolate(x, mod_x, np.arange(2*self.tau))
            
            self.pointer = 0
            
        #left,right, top, bottom
        
        a = self.y[self.pointer]
        xk = self.x[self.pointer]
        
        
        Ul = t = self.a_max*(np.exp(-((np.arange(self.size_x) - xk)**2)/(2*self.sigma)) -0.1)
        for i in range(self.size_y-1): Ul = np.vstack((Ul, t))
        Vl = Wl = t = np.random.uniform(-1,1,(self.size_x, self.size_y))
        
        self.pointer = self.pointer + 1
        
        return (Ul.transpose(),Vl,Wl)
        
        
        
        
        
        
        