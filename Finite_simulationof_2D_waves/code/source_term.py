import sympy as sym
import numpy as np

"""
Help script to find source term f and initial condtions with sympy
"""

def pde_source_term(u , q):
    """
    find source term (f) for PDEs with form
    DtDtu + bDtu = DxqDxu + DyqDyu + f
    """
    return sym.diff(u(x,y,t),t,2) + b*sym.diff(u(x,y,t),t) \
    - sym.diff(q(x,y)*sym.diff(u(x,y,t),x),x) - sym.diff(q(x,y)*sym.diff(u(x,y,t),y),y)

def initial_value(u):
    """
    calculate initial function I(x,y)
    """
    return u(x,y,t).subs(t,0)

def initial_velocity(u):
    """
    calculate initial velocity V(x,y)
    """
    return sym.diff(u(x,y,t),t).subs(t,0)

#define case specific functions and variables
u, x, y, t, q, w, A ,B, c, dt, dx, dy, kx, ky, b = sym.symbols("u x y t q w A B c dt dx dy kx ky b")
f = None # global variable for the source term in the ODE

def u_e(x,y,t):
    return (A*sym.cos(w*t) + B*sym.sin(w*t))*sym.exp(-c*t)*sym.cos(kx*x)*sym.cos(ky*y)

def q(x,y):
    return 1


def write_source_term(u,q):
    print("----------------------")
    print(f'=== finding source term for exact solution: u(x,y,t) = {u(x,y,t)} ===')
    print("----------------------")
    #use w=1 and b=0.5
    print(f"f(x,y,t) = {sym.simplify(pde_source_term(u,q).subs(w,1).subs(b,0.5))}")
    print("----------------------")
    print(f"I(x,y) = {sym.simplify(initial_value(u))}     V(x,y) = {sym.simplify(initial_velocity(u))}"   )
    print("----------------------")



if __name__ == "__main__":
    write_source_term(u_e,q)
