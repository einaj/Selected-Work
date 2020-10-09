#!/usr/bin/env python
import numpy as np
class wave2D():
    """
    Class for solving a 2D wave equation with variable coefficients
    DtDtu + bDtu = DxqDxu + DyqDyu + f
    """

    def __init__(self,f,q,b):
        """
        Establish the equation : DtDtu + bDtu = DxqDxU + DxqDxy + f
        The class should be initialized the following arguments

        Arguments:   f -- function of x,y and t
                     q -- function of x and y
                     b -- scalar value, dampening factor
        """
        self.f = f
        self.q = q
        self.b = b
        self._u = None

    def __call__(self,i,j,n):
        """
        Returns the value of the wave at mesh point (x[i],y[j]) at time t[n].

        Arguments:   i -- array index of desired x point
                     j -- array index of desired y point
                     n -- array index of desired time point
        """
        try:
            call = self._u[i,j,n]
            return call
        except:
            raise Exception("Call solution function solve() before __call__")

    def solve(self,I,V,Lx,Ly,Nx,Ny,dt,T,version='scalar'):
        """
        Solves the 2D wave equation with ghost cells in x and y directions
        Stores properties solution,x,y,t,h for user to extract after calling
        this method

        Arguments:
                    Functions:
                    I       -- initial function of x and y
                    V       -- initial velocity function of x and y
                    Scalars:
                    Lx      -- length of grid in x - direction
                    Ly      -- length of grid in y - direction
                    Nx      -- number of meshes in grid in x - direction
                    Ny      -- number of meshes in grid in y - direction
                    dt      -- time step
                    T       -- end time
                    version -- string, either "scalar" or "vectorized", sets
                                choice of solving method.
        """
        #Store variables
        self.I = I
        self.V = V
        self._Lx = Lx
        self._Ly = Ly
        self._Nx = Nx
        self._Ny = Ny
        self._dt = dt
        self._T = T
        self._version = version


        #check version
        if version == 'scalar':
            advance = self._advance_scalar
        elif version == 'vectorized':
            advance = self._advance_vectorized
        else:
            raise Exception('Incorrection version, choose "scalar" or "vectorized"')

        #create mesh points
        dx = Lx/(Nx)
        dy = Ly/(Ny)
        x = np.linspace(-dx, Lx+dx, Nx+3)  # mesh points in x dir
        y = np.linspace(-dy, Ly+dy, Ny+3)  # mesh points in y dir
        #we fix the value of the end grid points to make ghost cells
        x[0] = x[2] ; x[-1] = x[-3]
        y[0] = y[2] ; y[-1] = y[-3]

        #find max wave velocity
        max_c = np.sqrt(np.amax(self.q(x[:],y[:])))

        #check stability
        safety_factor = 0.9
        stability_limit = safety_factor*np.sqrt(dx**2 + dy**2)/(max_c)
        if dt >= stability_limit:
            print(f"Warning: dt = {dt} exceed stability limit ({stability_limit:.5f})")

        #save discretization parameter
        self._h = dx*dy*dt

        Nt = int(round(T/float(dt)))
        t = np.linspace(0, Nt*dt, Nt+1)

        u = np.zeros((Nx+3,Ny+3,Nt+1))

        #store variables
        self._x = x; self._y = y; self._dx = dx; self._dy = dy; self._Nt = Nt
        self.t = t; self._u = u

        #Set u[:,:,0] at initial condition
        self._set_initial_condtition()
        self._initial_step()

        #advance for all n up to Nt
        if self._version == 'scalar':
            self._advance_scalar()
        elif self._version == 'vectorized':
            self._advance_vectorized()


    def _initial_step(self):
        """
        Advance first step with either scalar method or vectorized
        """
        #get values
        x = self._x; y = self._y; u = self._u; t = self.t; b=self.b
        t=self.t; dt = self._dt; f = self.f; V = self.V

        #help variables
        beta = (1-b*dt/2)/(1+b*dt/2)

        if self._version == 'scalar':
            #advance first step special case
            for i in range(1,len(x)-1):
                for j in range(1,len(y)-1):
                    u[i,j,1] = (self._DxqDxu([i,j,0]) + self._DyqDyu([i,j,0])\
                                + f(x[i],y[j],0)) * dt**2/(1+b*dt/2)/(1+beta) \
                                + 2*u[i,j,0]/(1+b*dt/2)/(1+beta) + beta/(1+beta)*2*dt*V(x[i],y[i])


                #fix end points ghost cells
                for i in range(1,len(x)-1):
                    u[i,0,1] = u[i,2,1] ; u[i,-1,1] = u[i,-3,1]
                for j in range(1,len(y)-1):
                    u[0,j,1] = u[2,j,1] ; u[-1,j,1] = u[-3,j,1]

                u[0,0,1] = u[2,2,1] ; u[0,-1,1] = u[2,-3,1]
                u[-1,0,1] = u[-3,2,1] ; u[-1,-1,1] = u[-3,-3,1]


        elif self._version == 'vectorized':
            x,y = np.meshgrid(x[1:-1],y[1:-1])
            u[1:-1,1:-1,1] = (self._DxqDxu([None,None,0]) + self._DyqDyu([None,None,0])\
                        + f(x,y,t[0])) * dt**2/(1+b*dt/2)/(1+beta) \
                        + 2*u[1:-1,1:-1,0]/(1+b*dt/2)/(1+beta) + beta/(1+beta)*2*dt*V(x,y)

            #fix end points ghost cells
            u[1:-1,0,1] = u[1:-1,2,1] ; u[1:-1,-1,1] = u[1:-1,-3,1]
            u[0,1:-1,1] = u[2,1:-1,1] ; u[-1,1:-1,1] = u[-3,1:-1,1]

            u[0,0,1] = u[2,2,1] ; u[0,-1,1] = u[2,-3,1]
            u[-1,0,1] = u[-3,2,1] ; u[-1,-1,1] = u[-3,-3,1]

        self._u = u



    def _advance_scalar(self):
        """
        Adavance time steps from 1 to Nt with scalar method
        """
        #get values
        x = self._x; y = self._y; u = self._u; t = self.t; b=self.b
        t=self.t; dt = self._dt; f = self.f; V = self.V

        #help variables
        beta = (1-b*dt/2)/(1+b*dt/2)

        for n in range(1,len(t)-1):
            for i in range(1,len(x)-1):
                for j in range(1,len(y)-1):
                    u[i,j,n+1] = (self._DxqDxu([i,j,n]) + self._DyqDyu([i,j,n])\
                                + f(x[i],y[j],t[n])) * dt**2/(1+b*dt/2) \
                                + 2*u[i,j,n]/(1+b*dt/2) - beta*u[i,j,n-1]

            #fix end points ghost cells
            for i in range(1,len(x)-1):
                u[i,0,n+1] = u[i,2,n+1] ; u[i,-1,n+1] = u[i,-3,n+1]
            for j in range(1,len(y)-1):
                u[0,j,n+1] = u[2,j,n+1] ; u[-1,j,n+1] = u[-3,j,n+1]

            u[0,0,n+1] = u[2,2,n+1] ; u[0,-1,n+1] = u[2,-3,n+1]
            u[-1,0,n+1] = u[-3,2,n+1] ; u[-1,-1,n+1] = u[-3,-3,n+1]

        self._u = u

    def _advance_vectorized(self):
        """
        Adavance time steps from 1 to Nt with vectorized method
        """
        #get values
        x = self._x; y = self._y; u = self._u; t = self.t; b=self.b
        t=self.t; dt = self._dt; f = self.f; V = self.V

        #help variables
        beta = (1-b*dt/2)/(1+b*dt/2)
        x,y = np.meshgrid(x[1:-1],y[1:-1])

        for n in range(1,len(t)-1):
            u[1:-1,1:-1,n+1] = (self._DxqDxu([None,None,n]) + self._DyqDyu([None,None,n])\
                        + f(x,y,t[n])) * dt**2/(1+b*dt/2) \
                        + 2*u[1:-1,1:-1,n]/(1+b*dt/2) - beta*u[1:-1,1:-1,n-1]

            #fix end points ghost cells
            u[1:-1,0,n+1] = u[1:-1,2,n+1] ; u[1:-1,-1,n+1] = u[1:-1,-3,n+1]
            u[0,1:-1,n+1] = u[2,1:-1,n+1] ; u[-1,1:-1,n+1] = u[-3,1:-1,n+1]

            u[0,0,n+1] = u[2,2,n+1] ; u[0,-1,n+1] = u[2,-3,n+1]
            u[-1,0,n+1] = u[-3,2,n+1] ; u[-1,-1,n+1] = u[-3,-3,n+1]

    def _set_initial_condtition(self):
        """
        Sets the initial condtidtion u[:,:,0] = I[:,:,0] using ghost cells
        """
        u = self._u
        x = self._x
        y = self._y
        I = self.I

        if self._version == 'scalar':
            for i in range(len(x)):
                for j in range(len(y)):
                    u[i,j,0]=I(x[i],y[j])

        elif self._version == 'vectorized':
            x,y = np.meshgrid(x,y)
            u[:,:,0] = I(x,y)

        self._u = u


    def _DxqDxu(self,index):
        """
        Help function for computing the x-derivative
        """
        [i,j,n] = index

        #get variables
        q = self.q; x = self._x; y = self._y; dx = self._dx; u = self._u

        if self._version == 'scalar':

            #help variables
            q_plus_half = 0.5*(q(x[i],y[j]) + q(x[i+1],y[j]))
            q_minus_half = 0.5*(q(x[i],y[j]) + q(x[i-1],y[j]))

            return (q_plus_half*(u[i+1,j,n]-u[i,j,n]) \
                    - q_minus_half*(u[i,j,n]-u[i-1,j,n]))/dx**2

        elif self._version == 'vectorized':
            #help variables
            x,y = np.meshgrid(x[1:-1],y[1:-1])
            q_plus_half = 0.5*(q(x,y) + q(x+dx,y))
            q_minus_half = 0.5*(q(x,y) + q(x-dx,y))

            return (q_plus_half*(u[2:,1:-1,n]-u[1:-1,1:-1,n]) \
                    - q_minus_half*(u[1:-1,1:-1,n]-u[0:-2,1:-1,n]))/dx**2


    def _DyqDyu(self,index):
        """
        Help function for computing the y-derivative
        """
        [i,j,n] = index
        #get variables
        q = self.q; x = self._x; y = self._y; dy = self._dy; u = self._u

        if self._version == 'scalar':

            #help variables
            q_plus_half = 0.5*(q(x[i],y[j]) + q(x[i],y[j+1]))
            q_minus_half = 0.5*(q(x[i],y[j]) + q(x[i],y[j-1]))

            return (q_plus_half*(u[i,j+1,n]-u[i,j,n]) \
                    - q_minus_half*(u[i,j,n]-u[i,j-1,n]))/dy**2

        elif self._version =='vectorized':
            #help variables
            x,y = np.meshgrid(x[1:-1],y[1:-1])
            q_plus_half = 0.5*(q(x,y) + q(x,y+dy))
            q_minus_half = 0.5*(q(x,y) + q(x,y-dy))

            return (q_plus_half*(u[1:-1,2:,n]-u[1:-1,1:-1,n]) \
                    - q_minus_half*(u[1:-1,1:-1,n]-u[1:-1,0:-2,n]))/dy**2



    @property
    def solution(self):
        """
        Return solution without ghost cells
        """
        if self._u is not None:
            return self._u[1:-1,1:-1,:]
        else:
            raise AttributeError("Solution does not exist, use self.solve()")

    @property
    def x(self):
        """
        Return x without ghost cells
        """
        try:
            return self._x[1:-1]
        except AttributeError:
            raise AttributeError("x does not exist, use self.solve()")

    @property
    def y(self):
        """
        Return y without ghost cells
        """
        try:
            return self._y[1:-1]
        except AttributeError:
            raise AttributeError("y does not exist, use self.solve()")

    @property
    def time(self):
        """
        Return t for all time steps
        """
        try:
            return self.t
        except AttributeError:
            raise AttributeError("t does not exist, use self.solve()")

    @property
    def h(self):
        """
        Return discretization parameter h
        """
        try:
            return self._h
        except AttributeError:
            raise AttributeError("h does not exist, use self.solve()")



if __name__ == '__main__':
    pass
