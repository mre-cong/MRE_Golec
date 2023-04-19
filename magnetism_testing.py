import numpy as np
import magnetism

#TODO test the calculation of magnetization, dipole fields, and dipole dipole forces. compare results to by hand calculations
#1) magnetization response to 0 field, saturating field, subsaturating fields (resulting magnetization 0.5*Ms), and fields along and between two cardinal directions
#2) dipole fields, directly above, directly to the right/left, directly below, and at an abritrary angle relative to the magnetization vector direction. for unit magnetization, unit distance. increment distance via doubling/halving to ensure it behaves according to the expected power law dependence
#3) dipole-dipole forces: parallel orientations, anti-parallel, perpendicular orientations. side by side, stacked, one with zero magnetization, etc. aim for unit force results to begin with, and test power law dependence on separation matches expected power law

#cpdef np.ndarray[np.float64_t, ndim=1] get_magnetization(double[:] H, double chi, double Ms):

#cpdef np.ndarray[np.float64_t, ndim=1] get_dipole_field(double[:] r_i, double[:] r_j,  double[:] m):

#np.ndarray[np.float64_t, ndim=1] get_dip_dip_force(double[:] m_i, double[:] m_j, double[:] r_i, double[:] r_j):

#magnetization testing
# Hext = np.zeros((3,),dtype=np.float64)
# chi = 1
# Ms = 1
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)

# assert np.allclose(magnetization,np.zeros((3,),dtype=np.float64)),'incorrect magnetization'

# Hext[0] = 1
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)
# assert np.allclose(magnetization,np.array([0.5,0,0])),'incorrect magnetization'

# Hext = np.zeros((3,),dtype=np.float64)
# Hext[1] = 1
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)
# assert np.allclose(magnetization,np.array([0,0.5,0])),'incorrect magnetization'

# Hext = np.zeros((3,),dtype=np.float64)
# Hext[2] = 1
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)
# assert np.allclose(magnetization,np.array([0,0,0.5])),'incorrect magnetization'

# Hext[0] = np.sqrt(2)
# Hext[2] = np.sqrt(2)
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)
# correct_answer = 2/3*np.array([1/np.sqrt(2),0,1/np.sqrt(2)])
# assert np.allclose(magnetization,correct_answer),'incorrect magnetization when field at angle off axis'

# Hext = np.zeros((3,),dtype=np.float64)
# Hext[0] = 100
# magnetization = magnetism.get_magnetization(Hext,chi,Ms)
# assert np.allclose(magnetization,np.array([100/101,0,0])),'incorrect magnetization at saturating field'

#dipole field testing
# m = np.zeros((3,),dtype=np.float64)
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.zeros((3,),dtype=np.float64)), 'dipole field incorrect when magnetic moment is zero'

# m[0] = 1
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.array([2e-7,0,0]))

# #double distance, should be 1/(2**3) the value
# r_i[0] = 2
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.array([(2/8)*1e-7,0,0]))

# #to the side
# r_i = np.zeros((3,),dtype=np.float64)
# r_i[2] = 1
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.array([-1e-7,0,0]))

# #below
# r_i = np.zeros((3,),dtype=np.float64)
# r_i[0] = -1
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.array([2e-7,0,0]))

# #does adjusting r_j (the position of the dipole) work properly?
# r_j[0] = -1
# r_i[0] = 1
# #distance now doubled
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)
# assert np.allclose(dipole_field,np.array([(2/8)*1e-7,0,0]))

# #does adjusting r_j (the position of the dipole) work properly?
# r_j[0] = 1
# r_j[2] = 2
# r_i[0] = 0
# dipole_field = magnetism.get_dipole_field(r_i,r_j, m)

#dipole-dipole force testing
#unit distance separation, no magnetic moments
m_i = np.zeros((3,),dtype=np.float64)
r_i = np.zeros((3,),dtype=np.float64)
m_j = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
assert np.allclose(mag_force,np.array([0,0,0]))
#unit distance separation, unit magnetic moments, attractive parallel alignment
m_i[0] = 1
m_j[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
assert np.allclose(mag_force,np.array([-6e-7,0,0]))

#unit distance separation, unit magnetic moments, repulsive parallel alignment
r_i[0] = 0
r_i[1] = 1
m_i[0] = 1
m_j[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
assert np.allclose(mag_force,np.array([0,3e-7,0]))

#unit distance separation, unit magnetic moments, anti-parallel alignment
m_i[0] = -1
m_j[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
assert np.allclose(mag_force,np.array([0,-3e-7,0]))

#unit distance separation, unit magnetic moment and double magnetic moment, parallel alignment
m_i[0] = 2
m_j[0] = 1
r_i = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
assert np.allclose(mag_force,np.array([-12e-7,0,0]))

#twice unit distance separation, unit magnetic moments, parallel alignment. should be 2**4 factor weaker
m_i[0] = 2
m_j[0] = 1
r_i = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 2
mag_force2 = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
ratio = mag_force/mag_force2
assert np.allclose(mag_force2,np.array([-(12/(2**4))*1e-7,0,0]))

#half unit distance separation, unit magnetic moments, parallel alignment. should be 2**4 factor stronger
m_i[0] = 2
m_j[0] = 1
r_i = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 1/2
mag_force3 = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
ratio = mag_force/mag_force3
assert np.allclose(mag_force3,np.array([-(12*2**4)*1e-7,0,0]))

#unit distance, unit moments, perpendicular orientation\
m_i[0] = 1
m_j[0] = 0
m_j[1] = 1
r_i = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)

#unit distance, one unit moment, one zero moment
m_i[0] = 1
m_j = np.zeros((3,),dtype=np.float64)
r_i = np.zeros((3,),dtype=np.float64)
r_j = np.zeros((3,),dtype=np.float64)
r_i[0] = 1
mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
print('end')