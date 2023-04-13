cimport cython
from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport fabs
cimport libc.math
cimport numpy as np
import numpy as np

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
cpdef np.ndarray[np.float64_t, ndim=1] get_dip_dip_force(double[:] m_i, double[:] m_j, double[:] r_i, double[:] r_j):
    """Get the force on dipole i due to dipole j"""
    cdef double[3] rij
    cdef int i
    for i in range(3):
        rij[i] = r_i[i] - r_j[i]
    cdef double rij_mag = sqrt(dot_prod(rij,rij))
    cdef double[3] rij_hat
    for i in range(3):
        rij_hat[i] = rij[i]/rij_mag
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
    cdef double M_mag = Ms*chi*H_mag/(Ms + chi*H_mag)
    cdef np.ndarray[np.float64_t, ndim=1] M = np.empty((3,),dtype=np.float64)
    cdef int i
    cdef double[3] H_hat
    for i in range(3):
        H_hat[i] = H[i]/H_mag
        M[i] = M_mag*H_hat[i]
    return M

#magnetic permeability of free space mu0 = 3*pi*1e-7 H/m
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] get_dipole_field(double[:] r_i, double[:] r_j,  double[:] m):
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
    cdef double prefactor = mu0/(4*np.pi*rij_mag*rij_mag*rij_mag)
    for i in range(3):
        B[i] = prefactor*(3*m_dot_r_hat*rij_hat[i] - m[i])
    return B