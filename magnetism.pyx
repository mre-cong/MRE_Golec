cimport cython
from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport fabs
cimport libc.math
cimport numpy as np
import numpy as np

cdef float mu0_32bit = 4*np.pi*1e-7
cdef double mu0 = 4*np.pi*1e-7

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot_prod(double[:] vec1, double[:] vec2) nogil:
    cdef double result = 0 
    cdef int i
    for i in range(3):
        result += vec1[i]*vec2[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] get_dip_dip_force(double[:] m_i, double[:] m_j, double[:] r_i, double[:] r_j):
    """Get the force on dipole i due to dipole j"""
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    # cdef double[3] rij_hat
    # for i in range(3):
    #     rij_hat[i] = rij[i]/rij_mag
    cdef np.ndarray[np.float64_t, ndim=1] force = np.empty((3,),dtype=np.float64)
    cdef double mi_dot_r = dot_prod(m_i,rij)
    cdef double mj_dot_r = dot_prod(m_j,rij)
    cdef double m_dot_m = dot_prod(m_i,m_j)
    cdef double prefactor = 3*mu0/(4*np.pi*pow(rij_mag,5))
    for i in range(3):
        force[i] = prefactor*(mj_dot_r*m_i[i] + mi_dot_r*m_j[i] + m_dot_m*rij[i] - 5*rij[i]*mi_dot_r*mj_dot_r/pow(rij_mag,2))
    return force

@cython.boundscheck(False)
@cython.wraparound(False)
#carbonyl iron parameters Ms = 1990 kA/m, chi_initial = 131
cdef np.ndarray[np.float64_t, ndim=1] get_magnetization(double[:] H, double chi, double Ms):
    cdef double H_mag = sqrt(dot_prod(H,H))
    if H_mag == 0:
        return np.zeros((3,),dtype=np.float64)
    cdef double M_mag = Ms*chi*H_mag/(Ms + chi*H_mag)
    cdef np.ndarray[np.float64_t, ndim=1] M = np.empty((3,),dtype=np.float64)
    cdef int i
    cdef double[3] H_hat
    for i in range(3):
        H_hat[i] = H[i]/H_mag
        M[i] = M_mag*H_hat[i]
    return M

#magnetic permeability of free space mu0 = 4*pi*1e-7 H/m
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] get_dipole_field(double[:] r_i, double[:] r_j,  double[:] m):
    """Get the B-Field at a point i due to a dipole at point j"""
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    cdef double[3] rij_hat
    for i in range(3):
        rij_hat[i] = rij[i]/rij_mag
    cdef double m_dot_r_hat = dot_prod(m,rij_hat)
    cdef np.ndarray[np.float64_t, ndim=1] B = np.empty((3,),dtype=np.float64)
    cdef double prefactor = mu0/(4*np.pi*pow(rij_mag,3))
    for i in range(3):
        B[i] = prefactor*(3*m_dot_r_hat*rij_hat[i] - m[i])
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] get_dipole_field_normalized(double[:] r_i, double[:] r_j,  double[:] m, double l_e):
    """Get the B-Field at a point i due to a dipole at point j"""
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    cdef double[3] rij_hat
    for i in range(3):
        rij_hat[i] = rij[i]/rij_mag
    cdef double m_dot_r_hat = dot_prod(m,rij_hat)
    cdef np.ndarray[np.float64_t, ndim=1] B = np.empty((3,),dtype=np.float64)
    cdef double prefactor = mu0/(4*np.pi*pow(rij_mag,3)*pow(l_e,3))
    for i in range(3):
        B[i] = prefactor*(3*m_dot_r_hat*rij_hat[i] - m[i])
    return B

#TODO function for getting the magnetization correct via iteration. can start with a hard coded number of iterations, but a smarter implementation would include a convergence criteria. magnetization finding should happen simultaneously. first the magnetization due to the external field is found, then all the fields at the points representing the other particles are evaluated, then the total field at each point is used to update the magnetizations, and the process repeats (find fields, find magnetization). When the magnetization magnitude or direction change is small enough (as a whole, and where the largest change for any given particle is small enough) the system is considered to have converged and the loop can be exited. definining good exit criteria may be tricky. magnitude changes are easy enough to get, by keeping variables tracking new and old magnetization. directional changes would come from doing dot products of old and new magnetization to get cosine of the angle between them. the angle itself isn't that important... the cosine of the angle is a good enough value to compare against. if the direction isn't changing the value is 1, if it is changing it is less than one, -1 would be a complete inversion of the direction of the magnetization.
#because of the iterative nature of the method, it is best to calculate the necessary and unchanging vectors and vector magnitudes at the start of the computation, so that there is no unnecessary recomputation of these constant values
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] get_magnetization_iterative(double[:] Hext, double[:,::1] particle_posns, double particle_radius, double chi, double Ms):
    """Get the magnetization of the particles based on the total effective field at the center of each particle. Particle_radius is the radius in meters"""
    cdef int i
    cdef int j
    cdef int count
    cdef int max_iters = 5
    cdef np.ndarray[np.float64_t, ndim=2] current_M = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] next_M = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef double[3] init_M = np.empty((3,),dtype=np.float64)
    #for each particle, get the initial magnetization due to the external field
    init_M = get_magnetization(Hext,chi,Ms)
    for i in range(particle_posns.shape[0]):
        current_M[i,0] = init_M[0]
        current_M[i,1] = init_M[1]
        current_M[i,2] = init_M[2]
    #for each particle, get the magnetization due to each other particle, keeping track of current and next values of magnetization
    #starts with getting the dipolar field due to each particle at each other particle's position
    cdef np.ndarray[np.float64_t, ndim=2] H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef double[3] B_dip = np.zeros((3,),dtype=np.float64)
    #TODO do things better, so you don't recalculate rij vectors unnecessarily. requires making adjustments to the function that calculates dipole fields to take rij vector as an argument
    cdef double[3] r_i = np.empty((3,),dtype=np.float64)
    cdef double[3] r_j = np.empty((3,),dtype=np.float64)
    cdef double[3] m_j = np.empty((3,),dtype=np.float64)
    cdef double particle_V = (4/3)*np.pi*pow(particle_radius,3)
    cdef double[3] H_tot = np.empty((3,),dtype=np.float64)
    for count in range(max_iters):
        for i in range(particle_posns.shape[0]):
            #get particle i position and particle j position, don't calculate field for itself
            for j in range(particle_posns.shape[0]):
                if i == j:
                    pass
                else:
                    r_i[0] = particle_posns[i,0]
                    r_i[1] = particle_posns[i,1]
                    r_i[2] = particle_posns[i,2]
                    r_j[0] = particle_posns[j,0]
                    r_j[1] = particle_posns[j,1]
                    r_j[2] = particle_posns[j,2]
                    m_j[:] = particle_V*current_M[j,:]
                    B_dip = get_dipole_field(r_i,r_j,m_j)
                    H_dip[i,0] += B_dip[0]/mu0
                    H_dip[i,1] += B_dip[1]/mu0
                    H_dip[i,2] += B_dip[2]/mu0
            H_tot[:] = Hext[:] + H_dip[i,:]
            next_M[i,:] = get_magnetization(H_tot,chi,Ms)
        H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
        current_M = next_M
    return next_M

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] get_magnetization_iterative_normalized(double[:] Hext, double[:,::1] particle_posns, double particle_radius, double chi, double Ms, double l_e):
    """Get the magnetization of the particles based on the total effective field at the center of each particle. Particle_radius is the radius in meters"""
    cdef int i
    cdef int j
    cdef int count
    cdef int max_iters = 5
    cdef np.ndarray[np.float64_t, ndim=2] current_M = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] next_M = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef double[3] init_M = np.empty((3,),dtype=np.float64)
    #for each particle, get the initial magnetization due to the external field
    init_M = get_magnetization(Hext,chi,Ms)
    for i in range(particle_posns.shape[0]):
        current_M[i,0] = init_M[0]
        current_M[i,1] = init_M[1]
        current_M[i,2] = init_M[2]
    #for each particle, get the magnetization due to each other particle, keeping track of current and next values of magnetization
    #starts with getting the dipolar field due to each particle at each other particle's position
    cdef np.ndarray[np.float64_t, ndim=2] H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
    cdef double[3] B_dip = np.zeros((3,),dtype=np.float64)
    #TODO do things better, so you don't recalculate rij vectors unnecessarily. requires making adjustments to the function that calculates dipole fields to take rij vector as an argument
    cdef double[3] r_i = np.empty((3,),dtype=np.float64)
    cdef double[3] r_j = np.empty((3,),dtype=np.float64)
    cdef double[3] m_j = np.empty((3,),dtype=np.float64)
    cdef double particle_V = (4/3)*np.pi*pow(particle_radius,3)
    cdef double[3] H_tot = np.empty((3,),dtype=np.float64)
    for count in range(max_iters):
        for i in range(particle_posns.shape[0]):
            #get particle i position and particle j position, don't calculate field for itself
            for j in range(particle_posns.shape[0]):
                if i == j:
                    pass
                else:
                    r_i[0] = particle_posns[i,0]
                    r_i[1] = particle_posns[i,1]
                    r_i[2] = particle_posns[i,2]
                    r_j[0] = particle_posns[j,0]
                    r_j[1] = particle_posns[j,1]
                    r_j[2] = particle_posns[j,2]
                    m_j[:] = particle_V*current_M[j,:]
                    B_dip = get_dipole_field_normalized(r_i,r_j,m_j,l_e)
                    H_dip[i,0] += B_dip[0]/mu0
                    H_dip[i,1] += B_dip[1]/mu0
                    H_dip[i,2] += B_dip[2]/mu0
            H_tot[:] = Hext[:] + H_dip[i,:]
            next_M[i,:] = get_magnetization(H_tot,chi,Ms)
        H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float64)
        current_M = next_M
    return next_M

#TODO function for returning the vector rij. additional for getting rij_hat, rij_magnitude? to store vectors for reuse in the function to get magnetizations for all the particles in the simulation


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] get_dip_dip_forces(double[:,::1] M, double[:,::1] particle_posns, double particle_radius):
    """Get the dipole-dipole interaction forces for each particle pair, returning the vector sum of the dipole forces acting on each dipole"""
    cdef int N_particles = particle_posns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] forces = np.zeros((N_particles,3),dtype=np.float64)
    cdef int i
    cdef int j
    cdef double particle_V = (4/3)*np.pi*pow(particle_radius,3)
    cdef np.ndarray[np.float64_t, ndim=2] moments = np.empty((N_particles,3),dtype=np.float64)
    cdef double[3] r_i = np.empty((3,),dtype=np.float64)
    cdef double[3] r_j = np.empty((3,),dtype=np.float64)
    cdef double[3] force = np.empty((3,),dtype=np.float64)
    cdef double[3] wca_force = np.empty((3,),dtype=np.float64)#additional repulsive force to keep the particles from collapsing on one another
    for i in range(N_particles):
        moments[i,0] = M[i,0]*particle_V
        moments[i,1] = M[i,1]*particle_V
        moments[i,2] = M[i,2]*particle_V
    for i in range(N_particles):
        for j in range(i+1,N_particles):
            r_i[0] = particle_posns[i,0]
            r_i[1] = particle_posns[i,1]
            r_i[2] = particle_posns[i,2]
            r_j[0] = particle_posns[j,0]
            r_j[1] = particle_posns[j,1]
            r_j[2] = particle_posns[j,2]
            force = get_dip_dip_force(moments[i,:],moments[j,:],r_i,r_j)
            wca_force = get_particle_wca_force(r_i,r_j,particle_radius)
            forces[i,0] += force[0]
            forces[i,1] += force[1]
            forces[i,2] += force[2]
            forces[j,0] -= force[0]
            forces[j,1] -= force[1]
            forces[j,2] -= force[2]
            forces[i,0] += wca_force[0]
            forces[i,1] += wca_force[1]
            forces[i,2] += wca_force[2]
            forces[j,0] -= wca_force[0]
            forces[j,1] -= wca_force[1]
            forces[j,2] -= wca_force[2]
    return forces

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] get_dip_dip_forces_normalized(double[:,::1] M, double[:,::1] particle_posns, double particle_radius, double l_e):
    """Get the dipole-dipole interaction forces for each particle pair, returning the vector sum of the dipole forces acting on each dipole"""
    cdef int N_particles = particle_posns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] forces = np.zeros((N_particles,3),dtype=np.float64)
    cdef int i
    cdef int j
    cdef double particle_V = (4/3)*np.pi*pow(particle_radius,3)
    cdef np.ndarray[np.float64_t, ndim=2] moments = np.empty((N_particles,3),dtype=np.float64)
    cdef double[3] r_i = np.empty((3,),dtype=np.float64)
    cdef double[3] r_j = np.empty((3,),dtype=np.float64)
    cdef double[3] force = np.empty((3,),dtype=np.float64)
    cdef double[3] wca_force = np.empty((3,),dtype=np.float64)#additional repulsive force to keep the particles from collapsing on one another
    for i in range(N_particles):
        moments[i,0] = M[i,0]*particle_V
        moments[i,1] = M[i,1]*particle_V
        moments[i,2] = M[i,2]*particle_V
    for i in range(N_particles):
        for j in range(i+1,N_particles):
            r_i[0] = particle_posns[i,0]
            r_i[1] = particle_posns[i,1]
            r_i[2] = particle_posns[i,2]
            r_j[0] = particle_posns[j,0]
            r_j[1] = particle_posns[j,1]
            r_j[2] = particle_posns[j,2]
            force = get_dip_dip_force(moments[i,:],moments[j,:],r_i,r_j)
            wca_force = get_particle_wca_force_normalized(r_i,r_j,particle_radius,l_e)
            forces[i,0] += force[0]
            forces[i,1] += force[1]
            forces[i,2] += force[2]
            forces[j,0] -= force[0]
            forces[j,1] -= force[1]
            forces[j,2] -= force[2]
            forces[i,0] += wca_force[0]
            forces[i,1] += wca_force[1]
            forces[i,2] += wca_force[2]
            forces[j,0] -= wca_force[0]
            forces[j,1] -= wca_force[1]
            forces[j,2] -= wca_force[2]
    return forces

#should this be for a single particle pair, should i wrap it in a function that goes over all particle pairs? Should this be wrapped into the higher level dipole-dipole force calculation for all particle pairs?
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] get_particle_wca_force(double[:] r_i, double[:] r_j, double particle_radius):
    cdef double wca_mag
    cdef double sigma = 1#2*particle_radius/(3e-6)
    cdef double cutoff_length = pow(2,(1/6))*sigma
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag =  sqrt(dot_prod(rij,rij))
    cdef np.ndarray[np.float64_t, ndim=1] force = np.zeros((3,),dtype=np.float64)
    cdef eps_constant = (1e-7)*4*pow(np.pi,2)*pow(1.9e6,2)*pow(1.5e-6,3)/72#mu0*4*(pi**2)*(Ms**2)*(R**3)/72
    if rij_mag <= cutoff_length:#if the spring has shrunk to 2^(1/6)*10% or less of it's equilibrium length, we want to introduce an additional repulsive force to prevent volume collapse/inversion of the volume elements
        wca_mag = get_wca_force(eps_constant,rij_mag,sigma)
        for i in range(3):
            force[i] += wca_mag * rij[i] / rij_mag
    return force
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] get_particle_wca_force_normalized(double[:] r_i, double[:] r_j, double particle_radius,double l_e):
    """Get a repuslive force between the particles that is supposed to prevent volume overlap. particle_radius is the radius of the particle in meters"""
    cdef double wca_mag
    cdef double sigma = (2*particle_radius+1e-6)/l_e#want there to be some volume/space between the particles even if they are strongly attracted
    cdef double cutoff_length = pow(2,(1/6))*sigma
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag =  sqrt(dot_prod(rij,rij))
    cdef np.ndarray[np.float64_t, ndim=1] force = np.zeros((3,),dtype=np.float64)
    cdef double eps_constant = (1e-7)*4*pow(np.pi,2)*pow(1.9e6,2)*pow(1.5e-6,3)/72#mu0*pi*(Ms**2)*(R**3)/72
    if rij_mag <= cutoff_length:#if the distance between particles has shrunk to 2^(1/6)*10% or less of their combined radii, we want to introduce an additional repulsive force to prevent volume collapse of the particles on top of one another
        wca_mag = get_wca_force(eps_constant,rij_mag,sigma)
        for i in range(3):
            force[i] += wca_mag * rij[i] / rij_mag
    return force

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_wca_force(double eps_constant, double r, double sigma) nogil:
    cdef double sigma_over_separation = sigma/r
    # potential = 4*eps_constant*(pow(sigma_over_separation,12) - pow(sigma_over_separation,6))
    cdef double force_mag = 4*eps_constant*(12*pow(sigma_over_separation,13)/sigma - 6* pow(sigma_over_separation,7)/sigma)
    return force_mag

#### 32 bit versions of normalized(scaled) functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float dot_prod_32bit(float[:] vec1, float[:] vec2) nogil:
    cdef float result = 0 
    cdef int i
    for i in range(3):
        result += vec1[i]*vec2[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
#carbonyl iron parameters Ms = 1990 kA/m, chi_initial = 131
cdef np.ndarray[np.float32_t, ndim=1] get_magnetization_32bit(float[:] H, float chi, float Ms):
    cdef float H_mag = sqrt(dot_prod_32bit(H,H))
    if H_mag == 0:
        return np.zeros((3,),dtype=np.float32)
    cdef float M_mag = Ms*chi*H_mag/(Ms + chi*H_mag)
    cdef np.ndarray[np.float32_t, ndim=1] M = np.empty((3,),dtype=np.float32)
    cdef int i
    cdef float[3] H_hat
    for i in range(3):
        H_hat[i] = H[i]/H_mag
        M[i] = M_mag*H_hat[i]
    return M

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=2] get_magnetization_iterative_normalized_32bit(float[:] Hext, float[:,::1] particle_posns, float particle_radius, float chi, float Ms, float l_e):
    """Get the magnetization of the particles based on the total effective field at the center of each particle. Particle_radius is the radius in meters"""
    cdef int i
    cdef int j
    cdef int count
    cdef int max_iters = 5
    cdef np.ndarray[np.float32_t, ndim=2] current_M = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] next_M = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
    cdef float[3] init_M = np.empty((3,),dtype=np.float32)
    #for each particle, get the initial magnetization due to the external field
    init_M = get_magnetization_32bit(Hext,chi,Ms)
    for i in range(particle_posns.shape[0]):
        current_M[i,0] = init_M[0]
        current_M[i,1] = init_M[1]
        current_M[i,2] = init_M[2]
    #for each particle, get the magnetization due to each other particle, keeping track of current and next values of magnetization
    #starts with getting the dipolar field due to each particle at each other particle's position
    cdef np.ndarray[np.float32_t, ndim=2] H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
    cdef float[3] B_dip = np.zeros((3,),dtype=np.float32)
    #TODO do things better, so you don't recalculate rij vectors unnecessarily. requires making adjustments to the function that calculates dipole fields to take rij vector as an argument
    cdef float[3] r_i = np.empty((3,),dtype=np.float32)
    cdef float[3] r_j = np.empty((3,),dtype=np.float32)
    cdef float[3] m_j = np.empty((3,),dtype=np.float32)
    cdef float particle_V = (4/3)*np.pi*pow(particle_radius,3)
    cdef float[3] H_tot = np.empty((3,),dtype=np.float32)
    for count in range(max_iters):
        for i in range(particle_posns.shape[0]):
            #get particle i position and particle j position, don't calculate field for itself
            for j in range(particle_posns.shape[0]):
                if i == j:
                    pass
                else:
                    r_i[0] = particle_posns[i,0]
                    r_i[1] = particle_posns[i,1]
                    r_i[2] = particle_posns[i,2]
                    r_j[0] = particle_posns[j,0]
                    r_j[1] = particle_posns[j,1]
                    r_j[2] = particle_posns[j,2]
                    m_j[:] = particle_V*current_M[j,:]
                    B_dip = get_dipole_field_normalized_32bit(r_i,r_j,m_j,l_e)
                    H_dip[i,0] += B_dip[0]/mu0
                    H_dip[i,1] += B_dip[1]/mu0
                    H_dip[i,2] += B_dip[2]/mu0
            H_tot[:] = Hext[:] + H_dip[i,:]
            next_M[i,:] = get_magnetization_32bit(H_tot,chi,Ms)
        H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
        current_M = next_M
    return next_M

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] get_dipole_field_normalized_32bit(float[:] r_i, float[:] r_j,  float[:] m, float l_e):
    """Get the B-Field at a point i due to a dipole at point j"""
    cdef float[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef float rij_mag = sqrt(dot_prod_32bit(rij,rij))
    cdef float[3] rij_hat
    for i in range(3):
        rij_hat[i] = rij[i]/rij_mag
    cdef float m_dot_r_hat = dot_prod_32bit(m,rij_hat)
    cdef np.ndarray[np.float32_t, ndim=1] B = np.empty((3,),dtype=np.float32)
    cdef float prefactor = mu0_32bit/(4*np.pi*pow(rij_mag,3)*pow(l_e,3))
    for i in range(3):
        B[i] = prefactor*(3*m_dot_r_hat*rij_hat[i] - m[i])
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=1] get_dip_dip_force_32bit(float[:] m_i, float[:] m_j, float[:] r_i, float[:] r_j):
    """Get the force on dipole i due to dipole j"""
    cdef float[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef float rij_mag = sqrt(dot_prod_32bit(rij,rij))
    # cdef float[3] rij_hat
    # for i in range(3):
    #     rij_hat[i] = rij[i]/rij_mag
    cdef np.ndarray[np.float32_t, ndim=1] force = np.empty((3,),dtype=np.float32)
    cdef float mi_dot_r = dot_prod_32bit(m_i,rij)
    cdef float mj_dot_r = dot_prod_32bit(m_j,rij)
    cdef float m_dot_m = dot_prod_32bit(m_i,m_j)
    cdef float prefactor = 3*mu0_32bit/(4*np.pi*pow(rij_mag,5))
    # print('cpu results')
    # print(f'prefactor*rij_mag = {prefactor*rij_mag}')
    # print(f'mi_dot_r_hat *1e6 = {mi_dot_r/rij_mag*1e6}')
    # print(f'mj_dot_r_hat*1e6 = {mj_dot_r/rij_mag*1e6}')
    # print(f'm_dot_m*1e6 = {m_dot_m*1e6}')
    for i in range(3):
        force[i] = prefactor*(mj_dot_r*m_i[i] + mi_dot_r*m_j[i] + m_dot_m*rij[i] - 5*rij[i]*mi_dot_r*mj_dot_r/pow(rij_mag,2))
        # print(f'force component {i} = {force[i]}')
    return force

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=2] get_dip_dip_forces_normalized_32bit(float[:,::1] M, float[:,::1] particle_posns, float particle_radius, float l_e):
    """Get the dipole-dipole interaction forces for each particle pair, returning the vector sum of the dipole forces acting on each dipole"""
    cdef int N_particles = particle_posns.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] forces = np.zeros((N_particles,3),dtype=np.float32)
    cdef int i
    cdef int j
    cdef float particle_V = (4/3)*np.pi*pow(particle_radius,3)
    # cdef float MAX_FORCE_NORM = (3*mu0_32bit)/(2*np.pi*pow(2*particle_radius,4))*pow(particle_V,2)*pow(Ms,2)
    # cdef float[3] rij = np.empty((3,),dtype=np.float32)
    # cdef float rij_mag = 0
    # cdef float force_norm = 0
    # cdef float force_norm_ratio = 1
    cdef np.ndarray[np.float32_t, ndim=2] moments = np.empty((N_particles,3),dtype=np.float32)
    cdef float[3] r_i = np.empty((3,),dtype=np.float32)
    cdef float[3] r_j = np.empty((3,),dtype=np.float32)
    cdef float[3] force = np.empty((3,),dtype=np.float32)
    cdef float[3] wca_force = np.empty((3,),dtype=np.float32)#additional repulsive force to keep the particles from collapsing on one another
    for i in range(N_particles):
        moments[i,0] = M[i,0]*particle_V
        moments[i,1] = M[i,1]*particle_V
        moments[i,2] = M[i,2]*particle_V
    for i in range(N_particles):
        for j in range(i+1,N_particles):
            r_i[0] = particle_posns[i,0]
            r_i[1] = particle_posns[i,1]
            r_i[2] = particle_posns[i,2]
            r_j[0] = particle_posns[j,0]
            r_j[1] = particle_posns[j,1]
            r_j[2] = particle_posns[j,2]
            # for my_counter in range(3):
            #     rij[my_counter] = r_i[my_counter] - r_j[my_counter]
            # rij_mag = sqrt(dot_prod_32bit(rij,rij))
            force = get_dip_dip_force_32bit(moments[i,:],moments[j,:],r_i,r_j)
            # force_norm = sqrt(dot_prod_32bit(force,force))
            # if rij_mag < 4.5e-6:
            #     print(f'force: {force}')
            #     print(f'force_norm_si: {force_norm}')
            wca_force = get_particle_wca_force_normalized_32bit(r_i,r_j,particle_radius,l_e)
            # if force_norm > MAX_FORCE_NORM:
            #     force_norm_ratio = MAX_FORCE_NORM/force_norm
            #     print(f'force norm ratio: {force_norm_ratio}\n force norm: {force_norm}')
            #     force[0] *= force_norm_ratio
            #     force[1] *= force_norm_ratio
            #     force[2] *= force_norm_ratio
            #     print(f'new force: {force}')
            #     print(f'wca_force: {wca_force}')
            forces[i,0] += force[0]
            forces[i,1] += force[1]
            forces[i,2] += force[2]
            forces[j,0] -= force[0]
            forces[j,1] -= force[1]
            forces[j,2] -= force[2]
            forces[i,0] += wca_force[0]
            forces[i,1] += wca_force[1]
            forces[i,2] += wca_force[2]
            forces[j,0] -= wca_force[0]
            forces[j,1] -= wca_force[1]
            forces[j,2] -= wca_force[2]
    return forces

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=2] get_dip_dip_forces_normalized_32bit_v2(float[:,::1] moments, float[:,::1] particle_posns, float particle_radius, float l_e):
    """Get the dipole-dipole interaction forces for each particle pair, returning the vector sum of the dipole forces acting on each dipole"""
    cdef int N_particles = particle_posns.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] forces = np.zeros((N_particles,3),dtype=np.float32)
    cdef int i
    cdef int j
    cdef float[3] r_i = np.empty((3,),dtype=np.float32)
    cdef float[3] r_j = np.empty((3,),dtype=np.float32)
    cdef float[3] force = np.empty((3,),dtype=np.float32)
    cdef float[3] wca_force = np.empty((3,),dtype=np.float32)#additional repulsive force to keep the particles from collapsing on one another
    for i in range(N_particles):
        for j in range(i+1,N_particles):
            r_i[0] = particle_posns[i,0]
            r_i[1] = particle_posns[i,1]
            r_i[2] = particle_posns[i,2]
            r_j[0] = particle_posns[j,0]
            r_j[1] = particle_posns[j,1]
            r_j[2] = particle_posns[j,2]
            force = get_dip_dip_force_32bit(moments[i,:],moments[j,:],r_i,r_j)
            wca_force = get_particle_wca_force_normalized_32bit(r_i,r_j,particle_radius,l_e)
            forces[i,0] += force[0]
            forces[i,1] += force[1]
            forces[i,2] += force[2]
            forces[j,0] -= force[0]
            forces[j,1] -= force[1]
            forces[j,2] -= force[2]
            # print(f'i={i},j={j}')
            forces[i,0] += wca_force[0]
            forces[i,1] += wca_force[1]
            forces[i,2] += wca_force[2]
            forces[j,0] -= wca_force[0]
            forces[j,1] -= wca_force[1]
            forces[j,2] -= wca_force[2]
    return forces

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=1] get_particle_wca_force_normalized_32bit(float[:] r_i, float[:] r_j, float particle_radius,float l_e):
    """Get a repulsive force between the particles that is supposed to prevent volume overlap. particle_radius is the radius of the particle in meters"""
    cdef float wca_mag
    cdef float SURFACE_TO_SURFACE_SPACING = 1e-7
    cdef float sigma = (2*particle_radius+SURFACE_TO_SURFACE_SPACING)#/l_e#want there to be some volume/space between the particles even if they are strongly attracted
    cdef float cutoff_length = pow(2,(1/6))*sigma
    cdef float[3] rij
    cdef int i
    for i in range(3):
        rij[i] = (r_i[i] - r_j[i])#/l_e
    cdef float rij_mag =  sqrt(dot_prod_32bit(rij,rij))
    cdef np.ndarray[np.float32_t, ndim=1] force = np.zeros((3,),dtype=np.float32)
    cdef float eps_constant = (1e-7)*4*pow(np.pi,2)*pow(1.9e6,2)*pow(1.5e-6,3)/72#mu0*pi*(Ms**2)*(R**3)/72
    if rij_mag <= cutoff_length:#if the distance between particles has shrunk to 2^(1/6)*10% or less of their combined radii, we want to introduce an additional repulsive force to prevent volume collapse of the particles on top of one another
        wca_mag = get_wca_force_32bit(eps_constant,rij_mag,sigma)
        for i in range(3):
            force[i] += wca_mag * rij[i] / rij_mag
        # print(f'wca_force={force}')
    return force

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float get_wca_force_32bit(float eps_constant, float r, float sigma) nogil:
    cdef float sigma_over_separation = sigma/r
    # potential = 4*eps_constant*(pow(sigma_over_separation,12) - pow(sigma_over_separation,6))
    cdef float force_mag = 4*eps_constant*(12*pow(sigma_over_separation,13)/sigma - 6* pow(sigma_over_separation,7)/sigma)
    return force_mag