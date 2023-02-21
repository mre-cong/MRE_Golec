cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_positions(double[:,::1] x0,double[:,::1] v0,double[:,::1] a,double[:,::1] x1,double[:,::1] v1,double dt,double[:] m,double[:,::1] spring_force,double[:,::1] volume_correction_force, double drag,double[:,::1] bc_forces, double[:] fixed_nodes):
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
                x1[i,j] = x0[i,j]
                #because of the scaling difference with simulation size between the number of surface nodes that could be held fixed and the total number of nodes, the overhead from any kind of conditional check to prevent motion is probably greater than the cost of updating and then reverting the positions of nodes that are held fixed in place at a relatively small cutoff threshold (in terms of simulation volume by number of elements)