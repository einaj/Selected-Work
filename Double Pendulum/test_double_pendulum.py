import nose.tools as nt                     # Using nosetests
import numpy as np
from double_pendulum import DoublePendulum

def test_special_method_call_in_Double_Pendulum_class_keeps_a_pendulum_at_rest():
    """Test that call method keeps pendulum at rest for all angles = 0."""
    # Initial conditions for pendulum in equiliblrium state
    theta10  = 0; theta20 = 0
    omega10  = 0; omega20 = 0
    analytic = [0, 0, 0, 0]
    eps      = 10**(-7)

    D_pendel = DoublePendulum()
    computed = D_pendel(0,[theta10, theta20, omega10, omega20])

    for i in range(4):
        assert(abs(computed[i] - analytic[i]) < eps)

@nt.raises(AttributeError)
def test_error_if_solve_method_has_not_been_called():
    """
    Test that the solve method has been called. Error raised if attributes dont exist.
    """
    D_pendel = DoublePendulum()
    theta1   = D_pendel.theta1
    omega1   = D_pendel.omega1
    theta2   = D_pendel.theta2
    omega2   = D_pendel.omega2
    time     = D_pendel.t

def test_only_the_latest_solution_is_stored():
    """Tests that latest solution overwrites previous ones."""
    y0_1 = [0, 0, 0, 0]
    T_1  = 5
    dt_1 = 0.1

    y0_2 = [2, 3, 4, 5]
    T_2  = 15
    dt_2 = 0.01

    y0_3 = [1, 4, 0.1, 1.34]
    T_3  = 10
    dt_3 = 0.05

    D_pendel = DoublePendulum()
    D_pendel.solve(y0_1, T_1, dt_1)
    len_1 = len(D_pendel.t) #store previous length
    D_pendel.solve(y0_2, T_2, dt_2)
    len_2 = len(D_pendel.t) #store previous length
    D_pendel.solve(y0_3, T_3, dt_3)

    #Check length of t
    assert(len(D_pendel.t) != len_1)
    assert(len(D_pendel.t) != len_2)

    D_pendel2 = DoublePendulum()
    D_pendel2.solve(y0_3, T_3, dt_3)
    # Solve D_pendel2 for case #3 only
    # Check so that D_pendel is the latest solution
    for i in range(len(D_pendel.x1)):
        assert(D_pendel.x1[i] == D_pendel2.x1[i])
        assert(D_pendel.y1[i] == D_pendel2.y1[i])
        assert(D_pendel.x2[i] == D_pendel2.x2[i])
        assert(D_pendel.y2[i] == D_pendel2.y2[i])

def test_solve_method_in_Double_Pendulum_class_theta_omega_zero_arrays():
    """
    Test solve method keeps pendulum at for inital angles and velocity is zero, while t = i*dt.
    """
    y0 = [0, 0, 0, 0]
    T  = 5
    dt = 0.1

    pendel = DoublePendulum()
    pendel.solve(y0, T, dt)

    for i in range(len(pendel.t)):
        assert(pendel.t[i]      == i*pendel.dt)
        assert(pendel.theta1[i] == 0)
        assert(pendel.theta2[i] == 0)
        assert(pendel.omega1[i] == 0)
        assert(pendel.omega2[i] == 0)


if __name__ == "__main__":
    import nose
    nose.run()
