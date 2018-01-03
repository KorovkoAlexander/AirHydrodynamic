import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from akima import interpolate
from enum import Enum

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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

        mod = SourceModule("""

            __constant__ float k = 50;
            __constant__ float kz = 15;
            __constant__ float delta_x = {delta_x};
            __constant__ float delta_y = {delta_y};
            __constant__ float delta_z = {delta_z};
            __constant__ int size_x = {size_x};
            __constant__ int size_y = {size_y};
            __constant__ int size_z = {size_z};
            __constant__ int stride_x = {stride_x};
            __constant__ int stride_y = {stride_y};
            __constant__ int stride_z = {stride_z};
            __constant__ float time_tick = {tt};


            __global__ void compute(float* out_U,float* out_V,float* out_W, float* out_Ro,
                                    float* in_U,float* in_V,float* in_W,float* in_Ro)
                {
                    const int tid =stride_x + stride_y + stride_z +  threadIdx.x + blockDim.x * blockIdx.x +
                                    threadIdx.y + blockDim.y * blockIdx.y +
                                    threadIdx.z + blockDim.z * blockIdx.z;

                    if (tid < (size_x-2)*(size_y-2)*(size_z-2))
                    {
                        int id = tid;

                        int left_id = tid - stride_x/sizeof(float);
                        int right_id = tid + stride_x/sizeof(float);

                        int front_id = tid + stride_y/sizeof(float);
                        int back_id = tid - stride_y/sizeof(float);

                        int up_id = tid + stride_z/sizeof(float);
                        int down_id = tid - stride_z/sizeof(float);

                        float Fx = k*(
                                (in_U[right_id] - 2*in_U[id] + in_U[left_id])/(delta_x*delta_x) +
                                (in_U[front_id] - 2*in_U[id] + in_U[back_id])/(delta_y*delta_y) +
                                (in_U[up_id] - 2*in_U[id] + in_U[down_id])/(delta_z*delta_z)
                        );

                        float Fy = k*(
                                (in_V[right_id] - 2*in_V[id] + in_V[left_id])/(delta_x*delta_x) +
                                (in_V[front_id] - 2*in_V[id] + in_V[back_id])/(delta_y*delta_y) +
                                (in_V[up_id] - 2*in_V[id] + in_V[down_id])/(delta_z*delta_z)
                        );

                        float Fz = kz*(
                                (in_W[right_id] - 2*in_W[id] + in_W[left_id])/(delta_x*delta_x) +
                                (in_W[front_id] - 2*in_W[id] + in_W[back_id])/(delta_y*delta_y) +
                                (in_W[up_id] - 2*in_W[id] + in_W[down_id])/(delta_z*delta_z)
                        );


                        out_Ro[id] = in_Ro[id] - time_tick*(
                                in_U[id]*(in_Ro[right_id] - in_Ro[left_id])/(2*delta_x) +
                                in_Ro[id]*(in_U[right_id] - in_U[left_id])/(2*delta_x) +

                                in_V[id]*(in_Ro[front_id] - in_Ro[back_id])/(2*delta_y) +
                                in_Ro[id]*(in_V[front_id] - in_V[back_id])/(2*delta_y) +

                                in_W[id]*(in_Ro[up_id] - in_Ro[down_id])/(2*delta_z) +
                                in_Ro[id]*(in_W[up_id] - in_W[down_id])/(2*delta_z) +
                        );

                        if(out_Ro[id] <=0)
                        {
                            out_Ro[id] = 0.001;
                        }

                        float ux = (in_U[right_id] - in_U[left_id])/(2*delta_x);
                        float uy = (in_U[front_id] - in_U[back_id])/(2*delta_y);
                        float uz = (in_U[up_id] - in_U[down_id])/(2*delta_z);

                        float vx = (in_V[right_id] - in_V[left_id])/(2*delta_x);
                        float vy = (in_V[front_id] - in_V[back_id])/(2*delta_y);
                        float vz = (in_V[up_id] - in_V[down_id])/(2*delta_z);

                        float wx = (in_W[right_id] - in_W[left_id])/(2*delta_x);
                        float wy = (in_W[front_id] - in_W[back_id])/(2*delta_y);
                        float wz = (in_W[up_id] - in_W[down_id])/(2*delta_z);

                        float px = 8.31*320*(in_Ro[right_id] - in_U[left_id])/(2*delta_x);
                        float py = 8.31*320*(in_Ro[front_id] - in_U[back_id])/(2*delta_y);
                        float pz = 8.31*320*(in_Ro[up_id] - in_U[down_id])/(2*delta_z);


                        float VgradU = in_U[id]*ux + in_V[id]*uy + in_W[id]*uz;
                        float VgradV = in_U[id]*vx + in_V[id]*vy + in_W[id]*vz;
                        float VgradW = in_U[id]*wx + in_V[id]*wy + in_W[id]*wz;

                        out_U[id] = in_U[id] + time_tick*(-VgradU - px/in_Ro[id] + Fx);
                        out_V[id] = in_V[id] + time_tick*(-VgradV - py/in_Ro[id] + Fy);
                        out_W[id] = in_W[id] + time_tick*(-VgradW - pz/in_Ro[id] + Fz);

                    }

                }
            """.format(delta_x = self.delta.x, 
                delta_y = self.delta.y,
                delta_z = self.delta.z,
                size_x = settings["size_x"],
                size_y = settings["size_y"],
                size_z = settings["size_z"],
                stride_x  = self.U.strides[0], 
                stride_y  = self.U.strides[1], 
                stride_z  = self.U.strides[2],
                tt = settings["time_tick"]))        

    def cuda_next():
        
        # border conditions------Start--
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

        # border conditions------End------


        U = self.U.astype(np.float32)
        V = self.V.astype(np.float32)
        W = self.W.astype(np.float32)
        Ro = self.Ro.astype(np.float32)

        size_x, size_y, size_z = U.shape

        U_gpu = cuda.mem_alloc(U.nbytes)
        V_gpu = cuda.mem_alloc(V.nbytes)
        W_gpu = cuda.mem_alloc(W.nbytes)
        Ro_gpu = cuda.mem_alloc(Ro.nbytes)

        U_gpu_out = cuda.mem_alloc(U.nbytes)
        V_gpu_out = cuda.mem_alloc(V.nbytes)
        W_gpu_out = cuda.mem_alloc(W.nbytes)
        Ro_gpu_out = cuda.mem_alloc(Ro.nbytes)

        cuda.memcpy_htod(U_gpu, U)
        cuda.memcpy_htod(V_gpu, V)
        cuda.memcpy_htod(W_gpu, W)
        cuda.memcpy_htod(Ro_gpu, Ro)
        
        func = mod.get_function("compute")
        func(U_gpu_out,
             V_gpu_out,
             W_gpu_out,
             U_gpu,
             V_gpu,
             W_gpu,
             block=(settings["size_x"],settings["size_y"],settings["size_z"]))

        cuda.memcpy_dtoh(U, U_gpu_out)
        cuda.memcpy_dtoh(V, V_gpu_out)
        cuda.memcpy_dtoh(W, W_gpu_out)
        cuda.memcpy_dtoh(Ro, Ro_gpu_out)

        self.U = U
        self.V = V
        self.W = W
        self.Ro = Ro
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
        
        
        
        
        
        
        