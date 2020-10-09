from wave2D import wave2D
import nose.tools as nt
import numpy as np
from numpy import cos,sin,exp,pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

"""
Script for simulating tsunamis over a subseahill using the wave2D class
"""

#define equations
def B_gaussian(x,y):
    return B0 + Ba*exp(-((x-Bmx)/Bs)**2 - ((y-Bmy)/(b*Bs))**2)

def B_cosine(x,y):
    return B0  + Ba*cos(pi*(x-Bmx)/(2*Bs))*cos(pi*(y-Bmy)/(2*Bs))



q = lambda x,y: (np.ones(x.shape)*I0 - B_cosine(x,y))*g
f = lambda x,y,t : 0
b_ = 0 #b for the PDE

#set initial values:
V = lambda x,y : 0
def I(x,y):
    return I0 + Ia*exp(-((x-Im)/Is)**2)


#set other parameters
Lx = Ly = 1; Nx = Ny = 50
dt = 0.00005 ; T = 0.8

I0 = 5; Ia = 1; Im = 0; Is = 0.2

B0 = 0; Ba = 4.5; Bmx = Lx/2; Bmy=Ly/2; Bs = 0.5; b=1
g = 9.81

tsunami = wave2D(f,q,b_)
tsunami.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
sol = tsunami.solution


x = tsunami.x ; y = tsunami.y
x,y = np.meshgrid(x,y)

#dont plot every frame
reduct_factor = 110
sol_reduced = sol[:,:,::reduct_factor]

def update_plot(frame_number, sol, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, sol_reduced[:,:,frame_number], cmap="Blues")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(fr"Tsunami over centered Gaussian bottom b={b}")
#ax.set_zlim(2,9)
plot = [ax.plot_surface(x, y, sol[:,:,0],cmap="Blues"),ax.plot_surface(x,y,B_cosine(x,y),cmap="Greys")]
ani = animation.FuncAnimation(fig, update_plot, fargs=(sol, plot), frames=int(T/dt/reduct_factor),interval=50)
#ani.save('../figs/tsunami_gaussian_b=1.mp4')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
