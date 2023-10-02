cimport cython
from libc.math cimport sqrt
from libc.math cimport pow

#calculate the spring force for each spring
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void get_spring_forces(const double[:,::1] node_posns, double[:,::1] springs, double[:,::1] spring_force):
    cdef int i
    cdef int j
    cdef int node_i
    cdef int node_j
    cdef double[3] single_spring_force
    with nogil:
        for i in range(spring_force.shape[0]):
            for j in range(3):
                spring_force[i][j] = 0.0
        for i in range(springs.shape[0]):
            node_i = int(springs[i][0])
            node_j = int(springs[i][1])
            get_spring_force(i,node_i,node_j,node_posns,springs,single_spring_force)
            for j in range(3):
                spring_force[node_i][j] += single_spring_force[j]
                spring_force[node_j][j] += -1 * single_spring_force[j]

#calculate the spring force for a spring. want to try two variations, one that memoryviews into the entire node_posns array, and another that copies the relevant node positions for the spring. that second version is necessary for dealing with a multiprocessing situation, since multiple processes could be trying to access the same data (node position) at the same time and ersult in an error. i guess it is less of an issue since it should be constant
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_spring_force(const int row, const int node_i, const int node_j, const double[:,::1] node_posns, double[:,::1] springs, double[3] spring_force) nogil:
    cdef int i
    cdef double[3] rij
    cdef double k = springs[row][2]
    cdef double eq_length = springs[row][3]
    for i in range(3):
        rij[i] = node_posns[node_i,i] - node_posns[node_j,i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    cdef double spring_mag = -1 * k * (rij_mag - eq_length)
    for i in range(3):
        spring_force[i] = spring_mag * rij[i] / rij_mag

#calculate the spring force for each spring
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void get_spring_forces_WCA(const double[:,::1] node_posns, double[:,::1] springs, double[:,::1] spring_force):
    cdef int i
    cdef int j
    cdef int node_i
    cdef int node_j
    cdef double[3] single_spring_force
    with nogil:
        for i in range(spring_force.shape[0]):
            for j in range(3):
                spring_force[i][j] = 0.0
        for i in range(springs.shape[0]):
            node_i = int(springs[i][0])
            node_j = int(springs[i][1])
            get_spring_force_WCA(i,node_i,node_j,node_posns,springs,single_spring_force)
            for j in range(3):
                spring_force[node_i][j] += single_spring_force[j]
                spring_force[node_j][j] += -1 * single_spring_force[j]

#calculate the spring force for a spring. want to try two variations, one that memoryviews into the entire node_posns array, and another that copies the relevant node positions for the spring. that second version is necessary for dealing with a multiprocessing situation, since multiple processes could be trying to access the same data (node position) at the same time and ersult in an error. i guess it is less of an issue since it should be constant
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_spring_force_WCA(const int row, const int node_i, const int node_j, const double[:,::1] node_posns, double[:,::1] springs, double[3] spring_force) nogil:
    cdef int i
    cdef double[3] rij
    cdef double k = springs[row][2]
    cdef double eq_length = springs[row][3]
    for i in range(3):
        rij[i] = node_posns[node_i,i] - node_posns[node_j,i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    cdef double spring_mag = -1 * k * (rij_mag - eq_length)
    for i in range(3):
        spring_force[i] = spring_mag * rij[i] / rij_mag
    cdef double wca_mag
    cdef double sigma = 0.45*eq_length
    cdef double cutoff_length = pow(2,(1/6))*sigma
    if rij_mag <= cutoff_length:#if the spring has shrunk to 2^(1/6)*10% or less of it's equilibrium length, we want to introduce an additional repulsive force to prevent volume collapse/inversion of the volume elements
        wca_mag = get_wca_force(rij_mag,sigma)
        for i in range(3):
            spring_force[i] += wca_mag * rij[i] / rij_mag
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_wca_force(double r, double sigma) nogil:
    cdef double eps_constant = 1
    cdef double sigma_over_separation = sigma/r
    # potential = 4*eps_constant*(pow(sigma_over_separation,12) - pow(sigma_over_separation,6))
    cdef double force_mag = 4*eps_constant*(12*pow(sigma_over_separation,13)/sigma - 6* pow(sigma_over_separation,7)/sigma)
    return force_mag

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot_prod(double[3] vec1, double[3] vec2) nogil:
    cdef double result = 0 
    cdef int i
    for i in range(3):
        result += vec1[i]*vec2[i]
    return result