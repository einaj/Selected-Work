"""
Program for plotting waves produced by wave2D
"""

import numpy as np
from wave2D import wave2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

"""
Plot plug wave
"""

#define equation
q = lambda x,y: c**2
f = lambda x,y,t: 0
b = 0

#set initial conditions
V = lambda x,y: 0
def I(x,y):
    try:
        I = np.zeros(len(x[0,:]))
        for i in range(len(x[0,:])):
            I[i] = 0 if abs(x[0,i]-Lx/2.0) > 0.1 else 1
        return I
    except:
        return 0 if abs(x-Lx/2.0) > 0.1 else 1

#set other parameters
Lx = Ly = 1. ; Nx = Ny = 40; c = 0.5
dt = (Lx/Nx)/c; T = 4

"""
x-direction
"""

test = wave2D(f,q,b)
test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
sol = test.solution
x,y = np.meshgrid(test.x,test.y)
fig = plt.figure()


def update_plot(frame_number, sol, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, sol[:,:,frame_number], cmap="Blues")

ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0,1)
plot = [ax.plot_surface(x, y, sol[:,:,0],cmap="Blues")]
plug_wave_animation = animation.FuncAnimation(fig, update_plot, fargs=(sol, plot), frames=int(T/dt)+1,interval=50)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Plug wave x-diretion')
#plug_wave_animation.save('../figs/plug_wave_x.mp4')
plt.show()

"""
y-direction
"""
V = lambda x,y: 0
def I(x,y):
    try:
        I = np.zeros((len(x[0,:]),len(y[0,:])))
        for i in range(len(y[:,0])):
            I[i,:] = 0 if abs(y[i,0]-Ly/2.0) > 0.1 else 1
        return I
    except:
        return 0 if abs(y-Ly/2.0) > 0.1 else 1

test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
sol = test.solution
x,y = np.meshgrid(test.x,test.y)
fig = plt.figure()


def update_plot(frame_number, sol, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, sol[:,:,frame_number], cmap="Blues")

ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0,1)
plot = [ax.plot_surface(x, y, sol[:,:,0],cmap="Blues")]
plug_wave_animation = animation.FuncAnimation(fig, update_plot, fargs=(sol, plot), frames=int(T/dt)+1,interval=50)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Plug wave y-diretion')
#plug_wave_animation.save('../figs/plug_wave_y.mp4')
plt.show()

"""
Plot standing undamped waves
"""
#define equation
q = lambda x,y: c**2
f = lambda x,y,t: 0
b = 0

#set initial conditions
def I(x,y):
    return np.cos(kx*x)*np.cos(ky*y)

V = lambda x,y: 0

#set other parameters
Lx = Ly = 1.; Nx = 50; Ny = 50; c = 0.5
dt = 0.01 ; T = 4


#choose variables for exact solution
A = 1 ; mx = my =2 ; kx = mx*np.pi/Lx ; ky = my*np.pi/Ly
omega = 2*np.pi

test = wave2D(f,q,b)
test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
sol = test.solution

x,y = np.meshgrid(test.x,test.y)
fig = plt.figure()

def update_plot(frame_number, sol, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, sol[:,:,frame_number], cmap="Blues")

ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1,1)
plot = [ax.plot_surface(x, y, sol[:,:,0],cmap="Blues")]
standing_wave_animation = animation.FuncAnimation(fig, update_plot, fargs=(sol, plot), frames=int(T/dt),interval=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Standing undamped wave')
#standing_wave_animation.save('../figs/standing_wave.mp4')
plt.show()
