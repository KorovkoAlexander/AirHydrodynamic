AirHydro
==================
>AirHydro is a project, allowing you to simulate the behavior of the gas flow with custom border conditions.
Requirements
----------------
 >- python 3.6
 >- numpy (1.13.1)
 >- pyopengl (3.1.0)
 >- pyqt5

Images below show the test example. I tried to show the movement of the sky cloud. You can test it yourself, running the test script:
 > python test.py


Density  | Velocity
-------- | ---
![ cloud density](https://user-images.githubusercontent.com/35064209/34520862-c9b84568-f09b-11e7-897a-6d8289c6eabc.gif)| ![cloud velocity](https://user-images.githubusercontent.com/35064209/34520871-de94fc2e-f09b-11e7-8dbd-0f98a84cc1d2.gif)
Green areas are areas with high density.Blue areas are areas with low density.Red areas are areas with medium density.    | Here we have the distribution of point velocities in two coordinates (X,Y). Green arrows mark high velocity, red - low velocity.

Project Organization
--------------------------

AirHydro has 2 main classes: CloudModel and ConditionProvider defined in cloud.py.
All the work does CloudModel. It provides to the user 4 3-dimentional tensors with velocities(Vx named U in code, Vy named V, Vz named W) and density(Ro).   

All you need to do to make your own model is to define custom Border Condition Provider by inheriting from ConditionProvider or simply by defining the same methods.

Under the hood
---------------------
Here we have the Navier-Stokes equesions in their most common view:

![equation](http://latex.codecogs.com/gif.latex?\frac{dV}{dt}=\frac{-1}{\rho}\nabla{p}-2wV+g)

![equation](http://latex.codecogs.com/gif.latex?\frac{dT}{dt}-\frac{\gamma}{g\rho}\frac{dp}{dt}=\frac{1}{C_{p}}\varepsilon)

![equation](http://latex.codecogs.com/gif.latex?\frac{d\rho}{dt}+div{\rho{V}}=0)

![equation](http://latex.codecogs.com/gif.latex?p=R\rho{T})

We would use it in rectangular coord system, so we rewrite it:

![equation](http://latex.codecogs.com/gif.latex?\frac{du}{dt}=\frac{-1}{\rho}\frac{dp}{dx}-l_{1}w+F_{x};)

![equation](http://latex.codecogs.com/gif.latex?\frac{dv}{dt}=\frac{-1}{\rho}\frac{dp}{dy}-lu+F_{y};)


![equation](http://latex.codecogs.com/gif.latex?\frac{dw}{dt}=\frac{-1}{\rho}\frac{dp}{dz}-l_{1}u-g+F_{z};)

Where  ![equation](http://latex.codecogs.com/gif.latex?u,v,z) are the vilocities along X, Y, Z.
![equation](http://latex.codecogs.com/gif.latex?l_{1}) and   ![equation](http://latex.codecogs.com/gif.latex?l) are constants. 

![equation](http://latex.codecogs.com/gif.latex?F_{x}=k\(\frac{\partial^{2}u}{\partial{x^2}}+\frac{\partial^{2}u}{\partial{y^2}}+\frac{\partial^{2}u}{\partial{z^2}}\))

![equation](http://latex.codecogs.com/gif.latex?F_{x}=k\(\frac{\partial^{2}v}{\partial{x^2}}+\frac{\partial^{2}v}{\partial{y^2}}+\frac{\partial^{2}v}{\partial{z^2}}\))

![equation](http://latex.codecogs.com/gif.latex?F_{x}=k\(\frac{\partial^{2}w}{\partial{x^2}}+\frac{\partial^{2}w}{\partial{y^2}}+\frac{\partial^{2}w}{\partial{z^2}}\))

Also we have equations for temperature, dencity and preasure:

![equation](http://latex.codecogs.com/gif.latex?\frac{dT}{dt}-\frac{\gamma}{g\rho}\frac{dp}{dt}=\frac{1}{C_{p}}\varepsilon)

![equation](http://latex.codecogs.com/gif.latex?\frac{d\rho}{dt}+\frac{d\rho{u}}{dx}+\frac{d\rho{v}}{dy}+\frac{d\rho{w}}{dz}=0.)

Now the problem is that we cant compute all variables in continous time and space.So we woud use the discrete analog of them. 

We would use the discrete rectangular coord system, defined like:

![equation](http://latex.codecogs.com/gif.latex?i=\frac{x}{\Delta{x}};j=\frac{y}{\Delta{y}};k=\frac{z}{\Delta{z}};s=\frac{t}{\Delta{t}})

where ![equation](http://latex.codecogs.com/gif.latex?\Delta{x},\Delta{y},\Delta{z},\Delta{t}) are steps through space and time.

In this specific coord system our equations would look like:

![equation](http://latex.codecogs.com/gif.latex?\frac{u_{s+1}-u_{s-1}}{2\Delta{t}}+u_{i,j,k,s}\frac{u_{i+1}-u_{i-1}}{2\Delta{x}}+v_{i,j,k,s}\frac{u_{j+1}-u_{j-1}}{2\Delta{y}}+w_{i,j,k,s}\frac{u_{k+1}-u_{k-1}}{2\Delta{z}}=)

![equation](http://latex.codecogs.com/gif.latex?=\frac{-1}{\rho_{i,j,k,s}}\frac{p_{i+1}-p_{i-1}}{2\Delta{x}}+lv_{i,j,k,s}-l_{1}w_{i,j,k,s}+)

![equation](http://latex.codecogs.com/gif.latex?+\frac{k}{{\Delta{x}}^2}{[u_{i+1}-2u_{i}+u_{i+1}+u_{j+1}-2u_{j+1}+u_{j-1}+u_{k+1}-2u_{k}+u_{k-1}}])

----------------------------------------------------------------------

![equation](http://latex.codecogs.com/gif.latex?\frac{v_{s+1}-v_{s-1}}{2\Delta{t}}+u_{i,j,k,s}\frac{v_{i+1}-v_{i-1}}{2\Delta{x}}+v_{i,j,k,s}\frac{v_{j+1}-v_{j-1}}{2\Delta{y}}+w_{i,j,k,s}\frac{v_{k+1}-v_{k-1}}{2\Delta{z}}=)

![equation](http://latex.codecogs.com/gif.latex?=\frac{-1}{\rho_{i,j,k,s}}\frac{p_{i+1}-p_{i-1}}{2\Delta{x}}+lu_{i,j,k,s}+)

![equation](http://latex.codecogs.com/gif.latex?+\frac{k}{{\Delta{x}}^2}{[v_{i+1}-2v_{i}+v_{i+1}+v_{j+1}-2v_{j+1}+v_{j-1}+v_{k+1}-2v_{k}+v_{k-1}}])

-----------------------------------------------------------------------

![equation](http://latex.codecogs.com/gif.latex?\frac{w_{s+1}-w_{s-1}}{2\Delta{t}}+u_{i,j,k,s}\frac{w_{i+1}-w_{i-1}}{2\Delta{x}}+u_{i,j,k,s}\frac{w_{j+1}-w_{j-1}}{2\Delta{y}}+w_{i,j,k,s}\frac{w_{k+1}-w_{k-1}}{2\Delta{z}}=)

![equation](http://latex.codecogs.com/gif.latex?=\frac{-1}{\rho_{i,j,k,s}}\frac{p_{i+1}-p_{i-1}}{2\Delta{x}}+lu_{i,j,k,s}+)

![equation](http://latex.codecogs.com/gif.latex?+\frac{k}{{\Delta{x}}^2}{[w_{i+1}-2w_{i}+w_{i+1}+w_{j+1}-2w_{j+1}+w_{j-1}+w_{k+1}-2w_{k}+w_{k-1}}])

------------------------------------------------------------------------------

![equation](http://latex.codecogs.com/gif.latex?\frac{\rho_{s+1}-\rho_{s-1}}{2\Delta{t}}+u_{i,j,k,s}\frac{\rho_{i+1}-\rho_{i-1}}{2\Delta{x}}+v_{i,j,k,s}\frac{\rho_{j+1}-\rho_{j-1}}{2\Delta{y}}+w_{i,j,k,s}\frac{\rho_{k+1}-\rho_{k-1}}{2\Delta{z}}=0)


------------------------------------------------------------------------------
These are 4 equations of 5. Let us suggest that the temperature is constant in high scale mode, so the fifth equesion is not nessesary now.

Now we only need to set start conditions and let the code do its deal.

Start Conditions
---------------------

As start conditions we set  velocity and dencity in every point in space in first moment of computations. Its not nessesary to reset it later.

Boundary Conditions
----------------------------

At every time step we should set new Boundary Conditions, by using the  ConditionProvider class. It implements one method 
>get()

which returns the values of velocities U, V, W and dencity Ro on every side of computing space. I recomend you to look at the example it test.py. Its much more informative.

Some more nice images:
Density  | Velocity
----------  | ------------
![enter image description here](https://user-images.githubusercontent.com/35064209/34519054-60cb4e36-f093-11e7-976b-86b8b8373fb8.png)| ![enter image description here](https://user-images.githubusercontent.com/35064209/34519065-6cdb10c6-f093-11e7-8d2b-fc1ed4bc102f.png)
![enter image description here](https://user-images.githubusercontent.com/35064209/34519057-6466a3d8-f093-11e7-9154-f2aa3e3f1ed1.png) | ![enter image description here](https://user-images.githubusercontent.com/35064209/34519063-699fc14a-f093-11e7-8924-a67a7bc1e8f5.png) 
==|===========================



