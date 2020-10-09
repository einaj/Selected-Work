import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html for documentation.


class DoublePendulum():
    """Class for modeling a double pendulum."""
    def __init__(self, L1=1, L2=1, M1=1, M2=1, g=9.81):
        """
        The class should be called with the following arguments:

        Paramenters: M1 -- Mass of inner pendulum (closest to center) [kg]
                     M2 -- Mass of outer pendulum (furthest from center) [kg]
                     L1 -- Length of massless rod between center and M1 [m]
                     L2 -- Length of massless rod between M1 and M2 [m]
                     g -- Gravitational acceleration [m/s^2]
        """
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2

        self.g = g

    def __call__(self, t, y):
        """
        __call__ special method for caluculating the derivative of the angles
        theta1 and theta2 [angle(unitless)] and the angular velocities omega1
        and omega2 [1/s] of the dobble pendulum.

        Paramenters: t -- Time [s]
                     y -- Tuple consisting of initial values for theta1 [],
                          theta2 [], omega1 [1/s] and omega2 [1/s]. Should be
                          given as a tuple (theta1, theta2, omega1, omega2).

        Returns: Derivative of angles and angular velocities.
        """
        L1 = self.L1
        L2 = self.L2
        M1 = self.M1
        M2 = self.M2
        g  = self.g

        theta1, theta2, omega1, omega2 = y # y[0], y[1], y[2], y[3]
        diff_theta = theta2 - theta1       # Difference in angles

        topp1 = (M2*L1*omega1**2*np.sin(diff_theta)*np.cos(diff_theta)
                 + M2*g*np.sin(theta2)*np.cos(diff_theta)
                 + M2*L2*omega2**2*np.sin(diff_theta)
                 - (M1 + M2)*g*np.sin(theta1))

        bunn1 = (M1 + M2)*L1 - M2*L1*np.cos(diff_theta)**2

        topp2 = (-M2*L2*omega2**2*np.sin(diff_theta)*np.cos(diff_theta)
                 + (M1 + M2)*g*np.sin(theta1)*np.cos(diff_theta)
                 - (M1 + M2)*L1*omega1**2*np.sin(diff_theta)
                 - (M1 + M2)*g*np.sin(theta2))

        bunn2 = (M1 + M2)*L2 - M2*L2*np.cos(diff_theta)**2

        d_theta1__dt = omega1      # Angular velocity omega
        d_theta2__dt = omega2      # Angular velocity omega
        d_omega1__dt = topp1/bunn1 # Angular acceleration
        d_omega2__dt = topp2/bunn2 # Angular acceleration

        return d_theta1__dt, d_theta2__dt, d_omega1__dt, d_omega2__dt

    def solve(self, y0, T, dt, angles="rad"):
        """
        Solves the ODE using scipy.integrate solve_ivp function.

        Parameters: y0 -- An initial values for the function y we want to find
                          at a time t=0. Should be given as a list
                          [theta1, theta2, omega1, omega2].
                    T  -- End time for the experiment
                    dt -- Timestep we want to use in our calculation.

                    (NB! Both T and dt are given as scalar values. That is,
                    solve method should be called as
                    object.solve([theta1, theta2, omega1, omega2], T, dt), where
                    theta1, theta2, omega1, omega2, T and dt are scalar values.)

        Kwargs: angles -- specifying if the given values of theta and omega
                          are given in degrees or radians. Should be specified
                          as "deg" or "rad". If angles="deg", it will be changed
                          to radians and the result will allways be given in
                          radians.

        Returns: Nothing. Sets the solution as a private variable _solution.
        """
        self._dt = dt
        self._T = T

        if angles == "deg":
            y0 = np.array(y0)
            y0 = y0*(np.pi/180) # Changing from degrees to radians.

        sol = solve_ivp(self, [0, T], y0, t_eval=np.arange(0, T, dt),method="Radau")
        #store solutions
        self._solution = sol

    def create_animation(self,axis=[-3,3,-3,3]):
        """
        Creates animation of the pendulums, and traces the movement of the
        lower pendulum.

        Parameters: No parameters

        Kwargs:     axis -- A list containing minimum and maximum x and y values
                            for the coordinate system.

        Returns: Nothing. Creates an animation that can be:
                                - saved with save_animation()
                                - showed with show_animation()
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #make it pretty
        plt.axis("equal")
        plt.axis("off")
        plt.axis(axis)

        # Make an 'empty' plot object to be updated throughout the animation
        self.pendulums, = ax.plot([],[],'ro-',lw=2)

        #trace movement of lower pendulum
        self.trace, = ax.plot([],[],'b-')
        self.foox = [] ; self.fooy = []

        #timer
        self.time_template = 'time = %.1f s'
        self.time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)

        #animate
        self.animation = animation.FuncAnimation(fig,self._next_frame,
                                                 frames=range(0,len(self.t),2),
                                                 repeat=None,
                                                 interval=1000*self.dt,
                                                 blit=True)

    def _next_frame(self,i):
        """
        Method used by create_animation() to update the plot for each frame.

        Parametes:  i -- the current iteration/frame, automatically updated
                         by the FuncAnimation() method.

        Returns: the updated plots and the new time text.
        """
        self.pendulums.set_data((0,self.x1[i],self.x2[i]),
                                (0,self.y1[i],self.y2[i]))

        self.time_text.set_text(self.time_template % (i*dt))
        self.foox.append(self.x2[i]) ; self.fooy.append(self.y2[i])
        self.trace.set_data(self.foox,self.fooy)

        return self.pendulums,self.time_text,self.trace


    def show_animation(self):
        """Shows the animation"""
        plt.show()

    def save_animation(self,filename="pendel.mp4",res=100):
        """
        Saves the animation as desired filename

        Parameters: (None)

        Kwargs:    filename -- String containing desired filename: "example.mp4"
                               Set to "pendel.mp4" as default.
                   res      -- Set default to dpi = 100, changes resolution.

        Returns: Nothing. Saves the file as specified in filename.
        """
        if os.path.isfile(filename):
            os.system("rm "+filename)
        self.animation.save(filename,fps=60,dpi=res)

    @property
    def solution(self):
        """Returns the latest stored solution."""
        try:
            return self._solution
        except AttributeError:
            raise AttributeError("Solutions for the double pendulum does not "
                                 "exist. You need to run the solve method to "
                                 "compute how the pendulum moves. Run solve "
                                 "method first.")

    @property
    def dt(self):
        """Returns dt as given in solve."""
        return self._dt

    @property
    def T(self):
        """Returns T as given in solve."""
        return self._T

    @property
    def t(self):
        """
        Returns the time calculated in solve method. Raises error if it doesn't
        exit.
        """
        try:
            return self.solution.t
        except AttributeError:
            raise AttributeError("Pendulum time does not exist. You need to "
                                  "run the solve method to compute a time. Run "
                                  "solve method first.")

    @property
    def theta1(self):
        """Returns theta of upper pendulum, raises error if it doesn't exist."""
        try:
            return self.solution.y[0]
        except AttributeError:
            raise AttributeError("Theta1 values for pendulum does not exist "
                                  "yet. You need to run the solve method to "
                                  "compute the theta1 values. Run solve method "
                                  "first.")

    @property
    def theta2(self):
        """Returns theta of lower pendulum, raises error if it doesn't exist."""
        try:
            return self.solution.y[1]
        except AttributeError:
            raise AttributeError("Theta2 values for pendulum does not exist "
                                 "yet. You need to run the solve method to "
                                 "compute the theta2 values. Run solve method "
                                 "first.")

    @property
    def omega1(self):
        """Returns omega of upper pendulum, raises error if it doesn't exist."""
        try:
            return self.solution.y[2]
        except AttributeError:
            raise AttributeError("Omega1 values for pendulum does not exist "
                                 "yet. You need to run the solve method to "
                                 "compute the omega1 values. Run solve method "
                                 "first.")

    @property
    def omega2(self):
        """Returns omega of lower pendulum, raises error if it doesn't exist."""
        try:
            return self.solution.y[3]
        except AttributeError:
            raise AttributeError("Omega1 values for pendulum does not exist "
                                 "yet. You need to run the solve method to "
                                 "compute the omega2 values. Run solve method "
                                 "first.")

    @property
    def x1(self):
        """Returns x position of upper pendulum"""
        theta1 = self.theta1
        x1_pos = self.L1*np.sin(theta1)
        return x1_pos

    @property
    def x2(self):
        """Returns x position of lower pendulum"""
        theta2 = self.theta2
        x2_pos = self.x1 + self.L2*np.sin(theta2)
        return x2_pos

    @property
    def y1(self):
        """Returns y position of upper pendulum"""
        theta1 = self.theta1
        y1_pos = -self.L1*np.cos(theta1)
        return y1_pos

    @property
    def y2(self):
        """Returns y position of lower pendulum"""
        theta2 = self.theta2
        y2_pos = self.y1 - self.L2*np.cos(theta2)
        return y2_pos

    @property
    def vx1(self):
        """Retuns speed in x direction for upper pendulum"""
        #gradient documentation:
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
        return np.gradient(self.x1,self.t)

    @property
    def vx2(self):
        """Retuns speed in x direction for lower pendulum"""
        return np.gradient(self.x2,self.t)

    @property
    def vy1(self):
        """Retuns speed in y direction for upper pendulum"""
        return np.gradient(self.y1,self.t)

    @property
    def vy2(self):
        """Retuns speed in y direction for lower pendulum"""
        return np.gradient(self.y2,self.t)

    @property
    def potential(self):
        """Returns calculated total potential energy """
        P1 = self.M1*self.g*(self.y1 + self.L1)
        P2 = self.M2*self.g*(self.y2 + self.L1 + self.L2)
        P = P1 + P2
        return P

    @property
    def kinetic(self):
        """Returns calculated total kinetic energy """
        K1 = 0.5*self.M1*(self.vx1**2 + self.vy1**2)
        K2 = 0.5*self.M2*(self.vx2**2 + self.vy2**2)
        K = K1 + K2
        return K


if __name__ == "__main__":
    #Plotting double penulum, movement and energy levels
    y0 = [-np.pi/2,-np.pi,-0.6, 0]
    T  = 10
    dt = float(1/120)

    pendel = DoublePendulum()
    pendel.solve(y0,T,dt)

    time = pendel.t
    Ke   = pendel.kinetic
    Pe   = pendel.potential
    Te   = Ke + Pe

    #plot Energy and time
    plt.plot(time,Ke)
    plt.plot(time,Pe)
    plt.plot(time,Te)
    plt.title("Double pendulum")
    plt.legend(["Kinetic energy","Potential energy","Total energy"])
    plt.xlabel("time [s]") ; plt.ylabel("Energy [J]")
    plt.show()

    #create, save and show animation
    pendel.create_animation()
    #pendel.save_animation("pendel.mp4",res=200)
    pendel.show_animation()
