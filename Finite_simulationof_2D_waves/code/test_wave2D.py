from wave2D import wave2D
import nose.tools as nt
import numpy as np
from numpy import cos,sin,exp,pi

"""
Testing various aspects of the wave2D class with nosetests

to run either do:
1) >>> python3 test_wave2D.py
or
2) >>> nosetests -v test_wave2D.py

"""

def test_constant_solution():
    """
    Test that solution u(x,y,t) = c is constant for both scalar and vectorized
    """
    #define equation
    q = lambda x,y: 0.5
    f = lambda x,y,t: 0
    b = 0

    #set initial conditions
    I = lambda x,y: c
    V = lambda x,y: 0

    #set other parameters
    c = 3 ; Lx = Ly = 1 ; Nx = Ny = 10
    dt = 0.1; T = 10

    #define exact solution
    exact = np.ones((Nx+1,Ny+1))*c


    test = wave2D(f,q,b)
    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='scalar')
    sol = test.solution

    for n in range(sol.shape[2]):
        for j in range(sol.shape[1]):
            for i in range(sol.shape[0]):
                assert(sol[i,j,n] == exact[i,j])

    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
    sol = test.solution

    for n in range(sol.shape[2]):
        for j in range(sol.shape[1]):
            for i in range(sol.shape[0]):
                assert(sol[i,j,n] == exact[i,j])

@nt.raises(AttributeError)
def test_error_if_solve_method_has_not_been_called():
    """
    Test that error raised if attributes dont exist
    """
    #define equation
    b = 0
    q = lambda x,y: 1
    f = lambda x,y,t: x+y+t
    test = wave2D(f,q,b)
    sol = test.solution


def test_scalar_and_vectorized_is_equal():
    """
    Test that the scalar and the vectorized method give the same solution
    """

    #define equation
    q = lambda x,y: 2*x+2*y
    f = lambda x,y,t: 0
    b = 0

    #set initial conditions
    I = lambda x,y: c
    V = lambda x,y: 0.5

    #set other parameters
    c = 3 ; Lx = Ly = 1 ; Nx = Ny = 10
    dt = 0.01; T = 0.5

    test = wave2D(f,q,b)
    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
    sol1 = test.solution
    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='scalar')
    sol2 = test.solution

    for n in range(sol1.shape[2]):
        for j in range(sol1.shape[1]):
            for i in range(sol1.shape[0]):
                assert(abs(sol1[i,j,n] - sol2[i,j,n])<1e-10)

def test_plug_wave():
    """
    Test plug wave is equal to initial after one period for both scalar and vectorized in x and y-direction
    """
    #define equation
    q = lambda x,y: c**2
    f = lambda x,y,t: 0
    b = 0

    #set initial conditions
    V = lambda x,y: 0
    #test x-direction
    def I(x,y):
        try:
            I = np.zeros(len(x[0,:]))
            for i in range(len(x[0,:])):
                I[i] = 0 if abs(x[0,i]-Lx/2.0) > 0.1 else 1
            return I
        except:
            return 0 if abs(x-Lx/2.0) > 0.1 else 1

    #set other parameters
    Lx = Ly = 1. ; Nx = Ny = 10; c = 0.5
    dt = (Lx/Nx)/c; T = 4

    test = wave2D(f,q,b)
    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
    sol = test.solution
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            assert(sol[i,j,0] == sol[i,j,-1])

    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='scalar')
    sol = test.solution
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            assert(sol[i,j,0] == sol[i,j,-1])

    #test y-direction
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
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            assert(sol[i,j,0] == sol[i,j,-1])

    test.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='scalar')
    sol = test.solution
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            assert(sol[i,j,0] == sol[i,j,-1])


def test_convergence_manufactured_solution():
    """
    Test converge of rate manufactured solution of a standing, damped wave (expected r=2)
    """
    #define exact solution
    def u_e(x,y,t):
        return (A*cos(w*t) + B*sin(w*t))*exp(-c*t)*cos(kx*x)*cos(ky*y)

    #define equation
    q = lambda x,y : 1
    b = 0.5

    #use sympy to find source term f V and I ---> (source_term.py)
    #f found with w = 1 and q(x,y) = 1 and b = 0 for simplicity
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
    Lx = Ly = 1.; Nx = Ny = 30;
    dt = (Lx/Nx) ; T = 5

    #choose variables for exact solution
    A = 1 ; mx = my =2 ; kx = mx*np.pi/Lx ; ky = my*np.pi/Ly
    w = 1 ; A = 0.5; B = 0.7; c = 2

    err = []
    hs = []

    approx = wave2D(f,q,b)

    for dt in [0.0001,0.00005]:
        approx.solve(I,V,Lx,Ly,Nx,Ny,dt,T,version='vectorized')
        u = approx.solution
        x,y= np.meshgrid(approx.x,approx.y)
        #choose point to check here (10)
        error = np.absolute(u[:,:,10] - u_e(x,y,approx.t[10]))
        e_norm = np.amax(error)
        err.append(e_norm)
        hs.append(approx.h)

    r = np.log(err[0]/err[1])/np.log(hs[0]/hs[1])
    expected = 2.
    assert(abs(r-expected)<1e-3)



if __name__ == "__main__":
    import nose
    nose.run()
