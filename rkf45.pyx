cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef heun_euler_2(y,t,func,h):
    """simplest adaptive RK method, combining heun and euler's method to get an error estimate for controlling an adaptive stepsize"""
    #given the initial conditions and the function f in dy/dt = f(t,y), solve for y values in the range [t,t_final] using an adaptive step size, where an original step size is given, the error is estimated by comparing the next y value calculated for methods of two different orders, and the error estimate is used to decide whether to accept or reject the current iteration's solution, and to choose the next step size (which is based on the order of the solutions, the relative and absolute error tolerance, and scaling factors set to allow step sizes to increase or decrease no more than a certain amount in the hope of maintaining stability and preventing unnecessarily long time to converge to a solution)
    cdef float b1 = 1
    cdef float k1 = func(t,y)
    cdef float y_next_int = y + h * b1 * k1
    cdef float y_next = y + h/2 * (k1 + func(t+h,y_next_int))
    cdef float error = y_next - y_next_int
    cdef float tol = 1e-7
    cdef float En = error/tol
    cdef float h_next = h * (1/En) #need the previous step size used

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef heun_euler(y,t,func,h_prev):
    """simplest adaptive RK method, combining heun and euler's method to get an error estimate for controlling an adaptive stepsize"""
    cdef float error = heun(y,t,func) - euler(y,t,func)
    cdef float tol = 1e-7
    cdef float En = error/tol
    cdef float h = h_prev * (1/En) #need the previous step size used


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef heun(y, t, func):
    """Heun's method. 2nd order method with two stages. First writing for a first order ODE, then a separate version for a second order ODE split into two first order ODEs. this gives the next value of y"""
    # cdef float c2 = 1
    # cdef float a21 = 1
    # cdef float b1 = 0.5
    # cdef float b2 = 0.5
    cdef float h = 0.001
    # cdef float k1 = func(t,y)
    # cdef float k2 = func(t + c2*h, y + h*(a21 * k1))
    cdef float y_next_int = y + h * func(t,y)
    cdef float y_next = y + h/2 * (func(t,y) + func(t+h,y_next_int))
    return y_next

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rk2(y, t, func):
    """RK method. 2nd order method. First writing for a first order ODE, then a separate version for a second order ODE split into two first order ODEs. this gives the next value of y"""
    cdef float c2 = 1
    cdef float a21 = 1
    cdef float b1 = 0.5
    cdef float b2 = 0.5
    cdef float h = 0.001
    cdef float k1 = func(t,y)
    cdef float k2 = func(t + c2*h, y + h*(a21 * k1))
    cdef float y_next = y + h * (b1 * k1 + b2 * k2) 
    return y_next

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef euler(y, t, func):
    """Euler's method. 1st order method. First writing for a first order ODE, then a separate version for a second order ODE split into two first order ODEs."""
    cdef float h = 0.001
    cdef float b1 = 1
    cdef float k1 = func(t,y)
    cdef float y_next = y + h * b1 * k1
    return y_next

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rkf45(double[:,::1] x0,double[:,::1] v0,double[:,::1] a,double[:,::1] x1,double[:,::1] v1):
    """Runge-Kutta-Fehlberg method. Calculates 4th order and 5th order RK method results to estimate local truncation error to allow for adaptive step size."""
    cdef int i

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_positions(double[:,::1] x0,double[:,::1] v0,double[:,::1] a,double[:,::1] x1,double[:,::1] v1,double dt,double[:] m,double[:,::1] spring_force,double[:,::1] volume_correction_force, double drag,double[:,::1] bc_forces, long[:] fixed_nodes):
    """taking into account boundary conditions, drag, velocity, volume correction and spring forces, calculate the particle accelerations and update the particle positions and velocities"""
    cdef int i
    cdef int j
    with nogil:#calculate the acceleration on each node due to the forces acting on the nodes and update the velocites and positions
        for i in range(x0.shape[0]):
            for j in range(3):
                a[i,j] = (spring_force[i,j] + volume_correction_force[i,j] + bc_forces[i,j] - drag * v0[i,j])/m[i]
                v1[i,j] = a[i,j] * dt + v0[i,j]
                x1[i,j] = a[i,j] * dt * dt + v0[i,j] * dt + x0[i,j]
        for i in range(fixed_nodes.shape[0]):#after the fact, can place nodes back into position if they are supposed to be held fixed
            for j in range(3):
                x1[fixed_nodes[i],j] = x0[fixed_nodes[i],j]
                a[fixed_nodes[i],j] = 0