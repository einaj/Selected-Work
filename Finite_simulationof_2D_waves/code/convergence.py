from wave2D import wave2D
import numpy as np
from numpy import cos, pi, sin, exp
import matplotlib.pyplot as plt

"""
Compute convergence rate of a standing wave with exact Solution

u_e(x,y,t) = A*cos(k_x*x)*cos(k_y*y)*cos(omega*t)

using the wave2D class
"""

#define equation
f = lambda x,y,t: 0
q = lambda x,y: c**2
b = 0

#set initial values
def I(x,y):
    return cos(kx*x)*cos(ky*y)
V = lambda x,y: 0

#set other parameters
Lx = Ly = 1.
c = 0.5
T = 10

#choose variables for exact solution
A = 1 ; mx = my = 2; kx = mx*pi/Lx ; ky = my*pi/Ly
omega = 2*pi

def u_e(x,y,t):
    return A*cos(kx*x)*cos(ky*y)*cos(omega*t)

Nx = Ny = 20

#make list for error norm and discretization parameter
err = []
hs = []


for dt in [0.4,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]:
    approx = wave2D(f,q,b)
    approx.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
    u = approx.solution
    x,y= np.meshgrid(approx.x,approx.y)
    #choose point to check here (10)
    error = np.absolute(u[:,:,10] - u_e(x,y,approx.t[10]))
    e_norm = np.amax(error)
    err.append(e_norm)
    hs.append(approx.h)

rates = []
hplot = []
for i in range(1,len(err)):
    r = np.log(err[i-1]/err[i])/np.log(hs[i-1]/hs[i])
    hplot.append((hs[i-1]+hs[i])/2)
    rates.append(r)
print("convergence rates:")
print(rates)

plt.plot(np.log(hplot),rates)
plt.title(r"""convergence rate for $u$ with exact solution
$u_e(x,y,z) =Acos(k_xx)cos(k_yy)cos(\omega t)$""",fontsize=16)
plt.xlabel(r'discretization parameter $\log(h)$   $h =\Delta x\Delta y \Delta t$',fontsize=16)
plt.ylabel(r'convergence rate $r$',fontsize=16)
plt.plot(np.log(hplot),np.ones(len(hplot))*2,'k--')
#plt.savefig('../figs/convergence_standing_wave.pdf')
plt.show()

"""
Compute convergence rate of a standing wave with exact Solution

u_e(x,y,t) = (A*cos(w*t) + B*sin(w*t))*exp(-c*t)*cos(kx*x)*cos(ky*y)
"""

#define exact solution
def u_e(x,y,t):
    return (A*cos(w*t) + B*sin(w*t))*exp(-c*t)*cos(kx*x)*cos(ky*y)

#define equation
q = lambda x,y : 1
b = 0.5

#use sympy to find source term f V and I ---> (source_term.py)
#f found with w = 1 and q(x,y) = 1 and b = 0.5 for simplicity
def f(x,y,t):
    return  (-0.5*A*sin(t) - A*cos(t) - B*sin(t) + 0.5*B*cos(t)\
     + c**2*(A*cos(t) + B*sin(t)) + 2*c*(A*sin(t) - B*cos(t)) \
     - 0.5*c*(A*cos(t) + B*sin(t)) + kx**2*(A*cos(t) + B*sin(t))\
      + ky**2*(A*cos(t) + B*sin(t)))*exp(-c*t)*cos(kx*x)*cos(ky*y)

#set initial condtitions
def I(x,y):
    return A*cos(kx*x)*cos(ky*y)

def V(x,y):
    return (-A*c + B*w)*cos(kx*x)*cos(ky*y)

#set other parameters
Lx = Ly = 1.; Nx = Ny = 20;
dt = (Lx/Nx) ; T = 5

#choose variables for exact solution
A = 1 ; mx = my =2 ; kx = mx*np.pi/Lx ; ky = my*np.pi/Ly
w = 1 ; A = 0.5; B = 0.7; c = 2

err = []
hs = []

approx = wave2D(f,q,b)
for dt in [0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]:
    approx.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
    u = approx.solution
    x,y= np.meshgrid(approx.x,approx.y)
    #choose point to check here (10)
    error = np.absolute(u[:,:,10] - u_e(x,y,approx.t[10]))
    e_norm = np.amax(error)
    err.append(e_norm)
    hs.append(approx.h)

rates = []
hplot = []
for i in range(1,len(err)):
    r = np.log(err[i-1]/err[i])/np.log(hs[i-1]/hs[i])
    hplot.append((hs[i-1]+hs[i])/2)
    rates.append(r)

print("convergence rates:")
print(rates)

plt.plot(np.log(hplot),rates)
plt.title(r"""convergence rate for $u$ with exact solution
$u_e(x,y,z) = (Acos(\omega t) + Bsin(\omega t))e^{-ct}cos(k_xx)cos(k_yy)$""",fontsize=16)
plt.xlabel(r'discretization parameter $\log(h)$   $h =\Delta x\Delta y \Delta t$',fontsize=16)
plt.ylabel(r'convergence rate $r$',fontsize=18)
plt.plot(np.log(hplot),np.ones(len(hplot))*2,'k--')
#plt.savefig('../figs/convergence_manufactured_wave.pdf')
plt.show()
