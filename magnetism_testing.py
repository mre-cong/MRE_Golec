import numpy as np
import cupy as cp
import cupyx
import magnetism
import matplotlib.pyplot as plt
import os
import scipy.special as sci
import time
import numpy.random as rand
import simulate
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
# m_i = np.zeros((3,),dtype=np.float64)
# r_i = np.zeros((3,),dtype=np.float64)
# m_j = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# assert np.allclose(mag_force,np.array([0,0,0]))
# #unit distance separation, unit magnetic moments, attractive parallel alignment
# m_i[0] = 1
# m_j[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# assert np.allclose(mag_force,np.array([-6e-7,0,0]))

# #unit distance separation, unit magnetic moments, repulsive parallel alignment
# r_i[0] = 0
# r_i[1] = 1
# m_i[0] = 1
# m_j[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# assert np.allclose(mag_force,np.array([0,3e-7,0]))

# #unit distance separation, unit magnetic moments, anti-parallel alignment
# m_i[0] = -1
# m_j[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# assert np.allclose(mag_force,np.array([0,-3e-7,0]))

# #unit distance separation, unit magnetic moment and double magnetic moment, parallel alignment
# m_i[0] = 2
# m_j[0] = 1
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# assert np.allclose(mag_force,np.array([-12e-7,0,0]))

# #twice unit distance separation, unit magnetic moments, parallel alignment. should be 2**4 factor weaker
# m_i[0] = 2
# m_j[0] = 1
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 2
# mag_force2 = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# ratio = mag_force/mag_force2
# assert np.allclose(mag_force2,np.array([-(12/(2**4))*1e-7,0,0]))

# #half unit distance separation, unit magnetic moments, parallel alignment. should be 2**4 factor stronger
# m_i[0] = 2
# m_j[0] = 1
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1/2
# mag_force3 = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# ratio = mag_force/mag_force3
# assert np.allclose(mag_force3,np.array([-(12*2**4)*1e-7,0,0]))

# #unit distance, unit moments, perpendicular orientation\
# m_i[0] = 1
# m_j[0] = 0
# m_j[1] = 1
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)

# #unit distance, one unit moment, one zero moment
# m_i[0] = 1
# m_j = np.zeros((3,),dtype=np.float64)
# r_i = np.zeros((3,),dtype=np.float64)
# r_j = np.zeros((3,),dtype=np.float64)
# r_i[0] = 1
# mag_force = magnetism.get_dip_dip_force(m_i,m_j,r_i,r_j)
# print('end')

def main():
    """Testing magnetization for a single particle, plotting a hysteresis loop"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 1./mu0
    n_field_steps = 1000
    if n_field_steps != 1:
        H_step = H_mag/(n_field_steps-1)
    else:
        H_step = H_mag/(n_field_steps)
    #polar angle, aka angle wrt the z axis, range 0 to pi
    Hext_theta_angle = np.pi/2
    Bext_theta_angle = Hext_theta_angle*360/(2*np.pi)
    Hext_phi_angle = 0#(2*np.pi/360)*15#30
    Bext_phi_angle = Hext_phi_angle*360/(2*np.pi)
    Hext_series_magnitude = np.arange(H_step,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,2] = Hext_series_magnitude*np.cos(Hext_theta_angle)
    chi = 66#131
    Ms = 1.9e6
    particle_radius = 1.5e-6
    l_e = 1e-6
    beta = 6.734260376702891e-09
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particle_posns = np.array([[0,0,0]],dtype=np.float64)
    # particle_posns = np.array([[0,0,0],[9,0,0]],dtype=np.float64)
    num_particles = particle_posns.shape[0]
    if num_particles == 1:
        separation = None
    elif num_particles == 2:
        separation = np.linalg.norm(particle_posns[0,:]-particle_posns[1,:])
    else:
        num_separations = int(sci.binom(num_particles,2))
        separations = np.zeros((num_separations,))
        counter = 0
        for i in range(num_particles):
            for j in range(i,num_particles):
                separations[counter] = np.linalg.norm(particle_posns[i,:]-particle_posns[j,:])
                counter +=1
        separation = np.mean(separations)
    magnetization = np.zeros((Hext_series.shape[0],num_particles,3))
    mag_forces = np.zeros((Hext_series.shape[0],num_particles,3))
    M32bit = np.zeros((Hext_series.shape[0],num_particles,3),dtype=np.float32)
    mag_forces32bit = np.zeros((Hext_series.shape[0],num_particles,3),dtype=np.float32)
    for i, Hext in enumerate(Hext_series):
        magnetization[i] = magnetism.get_magnetization_iterative_normalized(Hext,particle_posns,particle_radius,chi,Ms,l_e)
        mag_forces[i] = magnetism.get_dip_dip_forces_normalized(magnetization[i],particle_posns,particle_radius,l_e)
        mag_forces[i] *= beta/(particle_mass*(np.power(l_e,4)))
        M32bit[i] = magnetism.get_magnetization_iterative_normalized_32bit(np.float32(Hext),np.float32(particle_posns),np.float32(particle_radius),np.float32(chi),np.float32(Ms),np.float32(l_e))
        mag_forces32bit[i] = magnetism.get_dip_dip_forces_normalized_32bit(M32bit[i],np.float32(particle_posns),np.float32(particle_radius),np.float32(l_e))
        # magnetic_scaling_factor = beta/(particle_mass*(np.power(l_e,4)))
        mag_forces32bit[i] *= np.float32(beta/(particle_mass*(np.power(l_e,4))))
    normalized_magnetization = magnetization/Ms
    system_magnetization = np.sum(normalized_magnetization,axis=1)/num_particles
    system_magnetization = np.squeeze(system_magnetization)
    Bext_series = mu0*Hext_series
    Bext_series_norm = np.linalg.norm(Bext_series,axis=1)
    nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
    unit_Bext_series = Bext_series[nonzero_field_value_indices[0]]/Bext_series_norm[nonzero_field_value_indices[0]]
    parallel_magnetization = np.dot(system_magnetization,unit_Bext_series)
    fig, ax = plt.subplots()
    ax.plot(Bext_series_norm,parallel_magnetization,'.')
    ax.set_xlabel('B Field (T)')
    ax.set_ylabel('Normalized Magnetization')
    ax.set_title(f'Magnetization versus Applied Field\n{num_particles} Particles, Separation {separation}\nTheta {Bext_theta_angle} Phi {Bext_phi_angle}')
    fig.show()
    save_dir = '/mnt/c/Users/bagaw/Desktop/MRE/magnetization_testing/'
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)
    savename = save_dir + f'{num_particles}_particles_magnetization_chi_{chi}_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
    plt.savefig(savename)
    print(f'32bit and 64bit magnetization match?:{np.allclose(magnetization,M32bit)}')
    print(f'32bit and 64bit magnetic forces match?:{np.allclose(mag_forces,mag_forces32bit)}')
    fig, axs = plt.subplots(3)
    axs[0].plot(Bext_series_norm,mag_forces[:,0,0],'r.')
    axs[0].plot(Bext_series_norm,mag_forces32bit[:,0,0],'b.')
    axs[0].set_xlabel('B Field (T)')
    axs[0].set_ylabel('Magnetic Force X-Dir')
    axs[1].plot(Bext_series_norm,mag_forces[:,0,1],'r.')
    axs[1].plot(Bext_series_norm,mag_forces32bit[:,0,1],'b.')
    axs[1].set_xlabel('B Field (T)')
    axs[1].set_ylabel('Magnetic Force Y-Dir')
    axs[2].plot(Bext_series_norm,mag_forces[:,0,2],'r.')
    axs[2].plot(Bext_series_norm,mag_forces32bit[:,0,2],'b.')
    axs[2].set_xlabel('B Field (T)')
    axs[2].set_ylabel('Magnetic Force Z-Dir')
    fig.legend(labels=['64bit','32bit'])
    savename = save_dir + f'{num_particles}_particles_chi_{chi}_mag_forces_64bit_vs_32bit_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
    plt.savefig(savename)


# cpdef np.ndarray[np.float32_t, ndim=2] get_magnetization_iterative_normalized_32bit(float[:] Hext, float[:,::1] particle_posns, float particle_radius, float chi, float Ms, float l_e):
#     """Get the magnetization of the particles based on the total effective field at the center of each particle. Particle_radius is the radius in meters"""
#     cdef int i
#     cdef int j
#     cdef int count
#     cdef int max_iters = 5
#     cdef np.ndarray[np.float32_t, ndim=2] current_M = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
#     cdef np.ndarray[np.float32_t, ndim=2] next_M = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
#     cdef float[3] init_M = np.empty((3,),dtype=np.float32)
#     #for each particle, get the initial magnetization due to the external field
#     init_M = get_magnetization_32bit(Hext,chi,Ms)
#     for i in range(particle_posns.shape[0]):
#         current_M[i,0] = init_M[0]
#         current_M[i,1] = init_M[1]
#         current_M[i,2] = init_M[2]
#     #for each particle, get the magnetization due to each other particle, keeping track of current and next values of magnetization
#     #starts with getting the dipolar field due to each particle at each other particle's position
#     cdef np.ndarray[np.float32_t, ndim=2] H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
#     cdef float[3] B_dip = np.zeros((3,),dtype=np.float32)
#     #TODO do things better, so you don't recalculate rij vectors unnecessarily. requires making adjustments to the function that calculates dipole fields to take rij vector as an argument
#     cdef float[3] r_i = np.empty((3,),dtype=np.float32)
#     cdef float[3] r_j = np.empty((3,),dtype=np.float32)
#     cdef float[3] m_j = np.empty((3,),dtype=np.float32)
#     cdef float particle_V = (4/3)*np.pi*pow(particle_radius,3)
#     cdef float[3] H_tot = np.empty((3,),dtype=np.float32)
#     for count in range(max_iters):
#         for i in range(particle_posns.shape[0]):
#             #get particle i position and particle j position, don't calculate field for itself
#             for j in range(particle_posns.shape[0]):
#                 if i == j:
#                     pass
#                 else:
#                     r_i[0] = particle_posns[i,0]
#                     r_i[1] = particle_posns[i,1]
#                     r_i[2] = particle_posns[i,2]
#                     r_j[0] = particle_posns[j,0]
#                     r_j[1] = particle_posns[j,1]
#                     r_j[2] = particle_posns[j,2]
#                     m_j[:] = particle_V*current_M[j,:]
#                     B_dip = get_dipole_field_normalized_32bit(r_i,r_j,m_j,l_e)
#                     H_dip[i,0] += B_dip[0]/mu0
#                     H_dip[i,1] += B_dip[1]/mu0
#                     H_dip[i,2] += B_dip[2]/mu0
#             H_tot[:] = Hext[:] + H_dip[i,:]
#             next_M[i,:] = get_magnetization_32bit(H_tot,chi,Ms)
#         H_dip = np.zeros((particle_posns.shape[0],3),dtype=np.float32)
#         current_M = next_M
#     return next_M

# #carbonyl iron parameters Ms = 1990 kA/m, chi_initial = 131
# cdef np.ndarray[np.float32_t, ndim=1] get_magnetization_32bit(float[:] H, float chi, float Ms):
#     cdef float H_mag = sqrt(dot_prod_32bit(H,H))
#     if H_mag == 0:
#         return np.zeros((3,),dtype=np.float32)
#     cdef float M_mag = Ms*chi*H_mag/(Ms + chi*H_mag)
#     cdef np.ndarray[np.float32_t, ndim=1] M = np.empty((3,),dtype=np.float32)
#     cdef int i
#     cdef float[3] H_hat
#     for i in range(3):
#         H_hat[i] = H[i]/H_mag
#         M[i] = M_mag*H_hat[i]
#     return M

magnetization_kernel = cp.RawKernel(r'''
extern "C" __global__                                    
void get_magnetization(const float Ms, const float chi, const float particle_volume, const float* H, float* magnetic_moment, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        //get magnetization based on total Hfield at each particle center
        float H_mag = sqrtf(H[3*tid]*H[3*tid] + H[3*tid+1]*H[3*tid+1] + H[3*tid+2]*H[3*tid+2]);
        //printf("H_mag: %f\n",H_mag);
        //printf("particle volume: %f\n", particle_volume);
        //printf("Ms + chi*H_mag: %f\n",Ms + chi*H_mag);
        //printf("M/Ms: %f\n",chi*H_mag/(Ms + chi*H_mag));
        float m_mag = particle_volume*(Ms*chi*H_mag/(Ms + chi*H_mag));
        //printf("m_mag: %f\n",m_mag);
        float H_hat[3];
        H_hat[0] = H[3*tid]/H_mag;
        H_hat[1] = H[3*tid+1]/H_mag;
        H_hat[2] = H[3*tid+2]/H_mag;
        //printf("H_hat: %f, %f, %f\n",H_hat[0],H_hat[1],H_hat[2]);
        magnetic_moment[3*tid] = m_mag*H_hat[0];
        magnetic_moment[3*tid+1] = m_mag*H_hat[1];
        magnetic_moment[3*tid+2] = m_mag*H_hat[2];
    }
    }
    ''', 'get_magnetization')

separation_vectors_kernel = cp.RawKernel(r'''
extern "C" __global__                                         
void get_separation_vectors(const float* particle_posns, float* separation_vectors, float* separation_vectors_inv_magnitude, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        float ri[3];
        int separation_vector_idx;
        int posn_idx;
        int inv_magnitude_idx;
        ri[0] = particle_posns[3*tid];
        ri[1] = particle_posns[3*tid+1];
        ri[2] = particle_posns[3*tid+2];
        for (int i = 0; i < size_particles; i++)
        {
            if (i != tid)
            {
            separation_vector_idx = tid*3*size_particles+3*i;
            inv_magnitude_idx = tid*size_particles + i;
            posn_idx = 3*i;
            separation_vectors[separation_vector_idx] = ri[0] - particle_posns[posn_idx];                                         
            separation_vectors[separation_vector_idx+1] = ri[1] - particle_posns[posn_idx+1];                                                                
            separation_vectors[separation_vector_idx+2] = ri[2] - particle_posns[posn_idx+2];                                                                
            separation_vectors_inv_magnitude[inv_magnitude_idx] = rnorm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2]);
            }
        }
    }
    }
    ''', 'get_separation_vectors')

# cpdef np.ndarray[np.float32_t, ndim=1] get_dipole_field_normalized_32bit(float[:] r_i, float[:] r_j,  float[:] m, float l_e):
#     """Get the B-Field at a point i due to a dipole at point j"""
#     cdef float[3] rij
#     cdef int i
#     for i in range(3):
#         rij[i] = r_i[i] - r_j[i]
#     cdef float rij_mag = sqrt(dot_prod_32bit(rij,rij))
#     cdef float[3] rij_hat
#     for i in range(3):
#         rij_hat[i] = rij[i]/rij_mag
#     cdef float m_dot_r_hat = dot_prod_32bit(m,rij_hat)
#     cdef np.ndarray[np.float32_t, ndim=1] B = np.empty((3,),dtype=np.float32)
#     cdef float prefactor = mu0_32bit/(4*np.pi*pow(rij_mag,3)*pow(l_e,3))
#     for i in range(3):
#         B[i] = prefactor*(3*m_dot_r_hat*rij_hat[i] - m[i])
#     return B

dipole_field_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
extern "C" __global__
void get_dipole_field(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float* magnetic_moment, const float inv_l_e, float* Htot, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float prefactor;
        float r_hat[3];
        float m_dot_r_hat;
        int separation_vector_idx;
        int inv_magnitude_idx;
        //float INV_4PI = 1/(4*3.141592654f);
        for (int i = 0; i < size_particles; i++)
        {
            separation_vector_idx = tid*3*size_particles+3*i;
            inv_magnitude_idx = tid*size_particles + i;
            r_hat[0] = separation_vectors[separation_vector_idx]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[1] = separation_vectors[separation_vector_idx+1]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[2] = separation_vectors[separation_vector_idx+2]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            m_dot_r_hat = magnetic_moment[3*i]*r_hat[0] + magnetic_moment[3*i+1]*r_hat[1] + magnetic_moment[3*i+2]*r_hat[2];
            prefactor = INV_4PI*powf(separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e,3);
            Htot[3*tid] += prefactor*(3*m_dot_r_hat*r_hat[0] - magnetic_moment[3*i]);
            Htot[3*tid+1] += prefactor*(3*m_dot_r_hat*r_hat[1] - magnetic_moment[3*i+1]);
            Htot[3*tid+2] += prefactor*(3*m_dot_r_hat*r_hat[2] - magnetic_moment[3*i+2]);
        }
    }
    }
    ''', 'get_dipole_field')

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.float32_t, ndim=2] get_dip_dip_forces_normalized_32bit_v2(float[:,::1] moments, float[:,::1] particle_posns, float particle_radius, float l_e):
#     """Get the dipole-dipole interaction forces for each particle pair, returning the vector sum of the dipole forces acting on each dipole"""
#     cdef int N_particles = particle_posns.shape[0]
#     cdef np.ndarray[np.float32_t, ndim=2] forces = np.zeros((N_particles,3),dtype=np.float32)
#     cdef int i
#     cdef int j
#     cdef float[3] r_i = np.empty((3,),dtype=np.float32)
#     cdef float[3] r_j = np.empty((3,),dtype=np.float32)
#     cdef float[3] force = np.empty((3,),dtype=np.float32)
#     cdef float[3] wca_force = np.empty((3,),dtype=np.float32)#additional repulsive force to keep the particles from collapsing on one another
#     for i in range(N_particles):
#         for j in range(i+1,N_particles):
#             r_i[0] = particle_posns[i,0]
#             r_i[1] = particle_posns[i,1]
#             r_i[2] = particle_posns[i,2]
#             r_j[0] = particle_posns[j,0]
#             r_j[1] = particle_posns[j,1]
#             r_j[2] = particle_posns[j,2]
#             force = get_dip_dip_force_32bit(moments[i,:],moments[j,:],r_i,r_j)
#             wca_force = get_particle_wca_force_normalized_32bit(r_i,r_j,particle_radius,l_e)
#             forces[i,0] += force[0]
#             forces[i,1] += force[1]
#             forces[i,2] += force[2]
#             forces[j,0] -= force[0]
#             forces[j,1] -= force[1]
#             forces[j,2] -= force[2]
#             forces[i,0] += wca_force[0]
#             forces[i,1] += wca_force[1]
#             forces[i,2] += wca_force[2]
#             forces[j,0] -= wca_force[0]
#             forces[j,1] -= wca_force[1]
#             forces[j,2] -= wca_force[2]
#     return forces

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[np.float32_t, ndim=1] get_particle_wca_force_normalized_32bit(float[:] r_i, float[:] r_j, float particle_radius,float l_e):
#     """Get a repulsive force between the particles that is supposed to prevent volume overlap. particle_radius is the radius of the particle in meters"""
#     cdef float wca_mag
#     cdef float SURFACE_TO_SURFACE_SPACING = 1e-7
#     cdef float sigma = (2*particle_radius+SURFACE_TO_SURFACE_SPACING)#/l_e#want there to be some volume/space between the particles even if they are strongly attracted
#     cdef float cutoff_length = pow(2,(1/6))*sigma
#     cdef float[3] rij
#     cdef int i
#     for i in range(3):
#         rij[i] = (r_i[i] - r_j[i])#/l_e
#     cdef float rij_mag =  sqrt(dot_prod_32bit(rij,rij))
#     cdef np.ndarray[np.float32_t, ndim=1] force = np.empty((3,),dtype=np.float32)
#     cdef float eps_constant = (1e-7)*4*pow(np.pi,2)*pow(1.9e6,2)*pow(1.5e-6,3)/72#mu0*pi*(Ms**2)*(R**3)/72
#     if rij_mag <= cutoff_length:#if the distance between particles has shrunk to 2^(1/6)*10% or less of their combined radii, we want to introduce an additional repulsive force to prevent volume collapse of the particles on top of one another
#         wca_mag = get_wca_force_32bit(eps_constant,rij_mag,sigma)
#         for i in range(3):
#             force[i] += wca_mag * rij[i] / rij_mag
#     return force
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef float get_wca_force_32bit(float eps_constant, float r, float sigma) nogil:
#     cdef float sigma_over_separation = sigma/r
#     # potential = 4*eps_constant*(pow(sigma_over_separation,12) - pow(sigma_over_separation,6))
#     cdef float force_mag = 4*eps_constant*(12*pow(sigma_over_separation,13)/sigma - 6* pow(sigma_over_separation,7)/sigma)
#     return force_mag
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[np.float32_t, ndim=1] get_dip_dip_force_32bit(float[:] m_i, float[:] m_j, float[:] r_i, float[:] r_j):
#     """Get the force on dipole i due to dipole j"""
#     cdef float[3] rij
#     cdef int i
#     for i in range(3):
#         rij[i] = r_i[i] - r_j[i]
#     cdef float rij_mag = sqrt(dot_prod_32bit(rij,rij))
#     # cdef float[3] rij_hat
#     # for i in range(3):
#     #     rij_hat[i] = rij[i]/rij_mag
#     cdef np.ndarray[np.float32_t, ndim=1] force = np.empty((3,),dtype=np.float32)
#     cdef float mi_dot_r = dot_prod_32bit(m_i,rij)
#     cdef float mj_dot_r = dot_prod_32bit(m_j,rij)
#     cdef float m_dot_m = dot_prod_32bit(m_i,m_j)
#     cdef float prefactor = 3*mu0_32bit/(4*np.pi*pow(rij_mag,5))
#     for i in range(3):
#         force[i] = prefactor*(mj_dot_r*m_i[i] + mi_dot_r*m_j[i] + m_dot_m*rij[i] - 5*rij[i]*mi_dot_r*mj_dot_r/pow(rij_mag,2))
#     return force


dipole_force_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
float SURFACE_TO_SURFACE_SPACING = 1e-7;
extern "C" __global__
void get_dipole_force(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float* magnetic_moment, const float l_e, const float inv_l_e, float* force, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float eps_constant = (1e-7)*4*powf(3.141592654f,2)*powf(1.9e6,2)*powf(1.5e-6,3)/72;
        float particle_radius = 1.5e-6;
        float sigma = (2*particle_radius+SURFACE_TO_SURFACE_SPACING);
        float cutoff_length = powf(2,(1/6))*sigma;
        float prefactor;
        float r_hat[3];
        float mi_dot_r_hat;
        float mj_dot_r_hat;
        float m_dot_m;
        float mi[3];
        mi[0] = magnetic_moment[3*tid];
        mi[1] = magnetic_moment[3*tid+1];
        mi[2] = magnetic_moment[3*tid+2];
        float mj[3];
        int separation_vector_idx;
        int inv_magnitude_idx;
        float mu0_over_four_pi = 1e-7;
        float force_temp_var;
        float rijmag;
        float sigma_over_separation;
        for (int j = 0; j < size_particles; j++)
        {
            if (tid != j)
            {
                mj[0] = magnetic_moment[3*j];
                mj[1] = magnetic_moment[3*j+1];
                mj[2] = magnetic_moment[3*j+2];
                separation_vector_idx = tid*3*size_particles+3*j;
                inv_magnitude_idx = tid*size_particles + j;
                r_hat[0] = separation_vectors[separation_vector_idx]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                r_hat[1] = separation_vectors[separation_vector_idx+1]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                r_hat[2] = separation_vectors[separation_vector_idx+2]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                mi_dot_r_hat = mi[0]*r_hat[0] + mi[1]*r_hat[1] + mi[2]*r_hat[2];
                mj_dot_r_hat = mj[0]*r_hat[0] + mj[1]*r_hat[1] + mj[2]*r_hat[2];
                m_dot_m = mi[0]*mj[0] + mi[1]*mj[1] + mi[2]*mj[2];
                //printf("mi_dot_r_hat = %.8f e-6\nmj_dot_r_hat = %.8f e-6\nm_dot_m = %.8f e-16\n",mi_dot_r_hat*1e6,mj_dot_r_hat*1e6,m_dot_m*1e16);
                prefactor = 3*mu0_over_four_pi*powf(separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e,4);
                //printf("prefactor = %f\n",prefactor);
                force_temp_var = prefactor*(mj_dot_r_hat*mi[0] + mi_dot_r_hat*mj[0] + m_dot_m*r_hat[0] - 5*r_hat[0]*mi_dot_r_hat*mj_dot_r_hat);
                //printf("tid = %i, j = %i, force_temp_var x = %.8f e-6\n",tid,j,force_temp_var*1e6);
                force[3*tid] += force_temp_var;
                force_temp_var = prefactor*(mj_dot_r_hat*mi[1] + mi_dot_r_hat*mj[1] + m_dot_m*r_hat[1] - 5*r_hat[1]*mi_dot_r_hat*mj_dot_r_hat);
                //printf("force_temp_var y = %f\n",force_temp_var);
                force[3*tid+1] += force_temp_var;
                force_temp_var = prefactor*(mj_dot_r_hat*mi[2] + mi_dot_r_hat*mj[2] + m_dot_m*r_hat[2] - 5*r_hat[2]*mi_dot_r_hat*mj_dot_r_hat);
                //printf("force_temp_var z = %f\n",force_temp_var);
                force[3*tid+2] += force_temp_var;
                rijmag = norm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2])*l_e;
                //printf("rijmag = %f\n",rijmag);
                if (rijmag <= cutoff_length)
                {
                    sigma_over_separation = sigma*separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e;
                    force_temp_var = 4*eps_constant*(12*powf(sigma_over_separation,13) - 6*powf(sigma_over_separation,7))/sigma;
                    //printf("inside WCA force calculation bit\ntid = %i, j = %i, force_temp_var = %f\n",tid,j,force_temp_var);
                    force[3*tid] += force_temp_var*r_hat[0];
                    force[3*tid+1] += force_temp_var*r_hat[1];
                    force[3*tid+2] += force_temp_var*r_hat[2];
                }
            }
        }
    }
    }
    ''', 'get_dipole_force')

def get_magnetization_iterative(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # normalized_magnetization_kernel((grid_size,),(block_size,),(Ms,chi,Hext_vector,magnetic_moment,size_particles))
    # magnetization_kernel used the particle volume to return the magnetic moment. the normalized approach normalizes by the saturation magnetization. the issue with using the particle volume to convert from magnetization to magnetic moment was that the result for low field values evaluated to zero.
    #the issue may have actually been that i didn't use 32 bit floats for chi, Ms, and l_e
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
    host_magnetic_moments = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))
    # host_element_forces = cp.asnumpy(cupy_element_forces)
    # host_spring_forces = cp.asnumpy(cupy_spring_forces)
    # host_composite_element_spring_forces = cp.asnumpy(cupy_composite_element_spring_forces)
    # return host_composite_element_spring_forces# host_element_forces, host_spring_forces
    return host_magnetic_moments

def get_magnetization_iterative_iteration_testing(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))

    return host_magnetic_moments

def get_magnetization_iterative_v2(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moments = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # normalized_magnetization_kernel((grid_size,),(block_size,),(Ms,chi,Hext_vector,magnetic_moment,size_particles))
    # magnetization_kernel used the particle volume to return the magnetic moment. the normalized approach normalizes by the saturation magnetization. the issue with using the particle volume to convert from magnetization to magnetic moment was that the result for low field values evaluated to zero.
    #the issue may have actually been that i didn't use 32 bit floats for chi, Ms, and l_e
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moments,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moments,size_particles))
        cupy_stream.synchronize()
    return magnetic_moments

def get_magnetic_forces_composite(Hext,particles,particle_posns,Ms,chi,particle_volume,beta,particle_mass,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moments = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # normalized_magnetization_kernel((grid_size,),(block_size,),(Ms,chi,Hext_vector,magnetic_moment,size_particles))
    # magnetization_kernel used the particle volume to return the magnetic moment. the normalized approach normalizes by the saturation magnetization. the issue with using the particle volume to convert from magnetization to magnetic moment was that the result for low field values evaluated to zero.
    #the issue may have actually been that i didn't use 32 bit floats for chi, Ms, and l_e
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moments,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moments,size_particles))
        cupy_stream.synchronize()
    inv_l_e = np.float32(1/l_e)
    force = cp.zeros((3*size_particles,1),dtype=cp.float32,order='C')
    dipole_force_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,l_e,inv_l_e,force,size_particles))
    force *= np.float32(beta/particle_mass)
    return force

def get_magnetic_forces(particles,separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,l_e,beta,particle_mass):
    """Return magnetic forces"""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    inv_l_e = np.float32(1/l_e)
    force = cp.zeros((3*size_particles,1),dtype=cp.float32,order='C')
    dipole_force_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,l_e,inv_l_e,force,size_particles))
    force *= np.float32(beta/particle_mass)
    # print(f'scaling factor = {np.float32(beta/particle_mass)}')
    return force

def gpu_testing_force_calc():
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.1/mu0
    n_field_steps = 200
    if n_field_steps != 1:
        H_step = H_mag/(n_field_steps-1)
    else:
        H_step = H_mag/(n_field_steps)
    #polar angle, aka angle wrt the z axis, range 0 to pi
    Hext_theta_angle = np.pi/2
    Bext_theta_angle = Hext_theta_angle*360/(2*np.pi)
    Hext_phi_angle = (2*np.pi/360)*15#30
    Bext_phi_angle = Hext_phi_angle*360/(2*np.pi)
    Hext_series_magnitude = np.arange(H_step,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    Bext_series_magnitude = Hext_series_magnitude*mu0
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,2] = Hext_series_magnitude*np.cos(Hext_theta_angle)
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = 1.5e-6
    l_e = np.float32(1e-6)
    beta = 6.734260376702891e-09
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    num_nodes_per_particle = 8
    particles_per_axis = np.array([10,10,10])#np.array([3,3,3])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = 7.8
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation
                particle_posns[particle_counter,1] = j * separation
                particle_posns[particle_counter,2] = k * separation
                particle_counter += 1
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))

    mag_forces32bit = np.zeros((num_particles,3),dtype=np.float32)
    magnetic_moments = np.zeros((Hext_series.shape[0],num_particles,3),dtype=np.float32)
    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e)
    
    
    field_idx = int(n_field_steps/2)
    Hext = Hext_series[int(n_field_steps/2)]

    host_magnetic_moments = magnetic_moments[field_idx]
    device_magnetic_moments = cp.asarray(cp.reshape(magnetic_moments[field_idx],(3*num_particles,1)),dtype=cp.float32,order='C')

    num_streaming_multiprocessors = 14
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(device_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))

    N_iterations = 10
    execution_gpu = cupyx.profiler.benchmark(get_magnetic_forces,(particles,separation_vectors,separation_vectors_inv_magnitude,device_magnetic_moments,l_e,beta,particle_mass),n_repeat=N_iterations)
    # execution_gpu = cupyx.profiler.benchmark(get_magnetization_iterative,(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e),n_repeat=N_iterations)
    delta_gpu_naive = np.sum(execution_gpu.gpu_times)

    start = time.perf_counter()
    for i in range(N_iterations):
        mag_forces32bit = magnetism.get_dip_dip_forces_normalized_32bit_v2(host_magnetic_moments,np.float32(particle_posns*l_e),np.float32(particle_radius),np.float32(l_e))
        mag_forces32bit *= np.float32(beta/particle_mass)#np.float32(beta/(particle_mass*(np.power(l_e,4))))
        # print(f'cpu scaling factor = {np.float32(beta/particle_mass)}')
        # perf_m32bit = magnetism.get_magnetization_iterative_normalized_32bit(np.float32(Hext),np.float32(particle_posns),np.float32(particle_radius),np.float32(chi),np.float32(Ms),np.float32(l_e))
    end = time.perf_counter()
    delta_cy = end-start

    gpu_mag_force = get_magnetic_forces(particles,separation_vectors,separation_vectors_inv_magnitude,device_magnetic_moments,l_e,np.float32(beta),np.float32(particle_mass))
    gpu_mag_force = cp.asnumpy(gpu_mag_force)
    gpu_mag_force = np.reshape(gpu_mag_force,(particles.shape[0],3))
    try:
        correctness = np.allclose(gpu_mag_force,mag_forces32bit)
    except Exception as inst:
            print('Exception raised during calculation')
            print(type(inst))
            print(inst)
    print("GPU and CPU based calculations of forces agree?: " + str(correctness))
    if not correctness:
        difference = np.abs(gpu_mag_force-mag_forces32bit)
        max_component_diff = np.max(difference)
        mean_component_diff = np.mean(difference)
        print(f'maximum difference in volume correction force components is {max_component_diff}')
        print(f'mean difference in volume correction force components is {mean_component_diff}')
        max_norm_diff = np.max(np.linalg.norm(difference,axis=1))
        mean_norm_diff = np.mean(np.linalg.norm(difference,axis=1))
        print(f'maximum difference in volume correction force norm is {max_norm_diff}')
        print(f'mean difference in volume correction force norm is {mean_norm_diff}')
    #     print(f'number of NaN entries in the CPU VCF {np.count_nonzero(np.isnan(volume_correction_force))}')
    #     print(f'number of NaN entries in the GPU VCF {np.count_nonzero(np.isnan(host_cupy_forces))}')
    #     print(f'number of NaN entries in the norm of the CPU VCF {np.count_nonzero(np.isnan(np.linalg.norm(volume_correction_force,axis=1)))}')
    #     print(f'number of zero entries in the norm of the CPU VCF {np.count_nonzero(np.isclose(np.linalg.norm(volume_correction_force,axis=1),0))}')
    #     max_norm_percent_diff = np.max(np.nan_to_num(np.linalg.norm(difference,axis=1)/np.linalg.norm(volume_correction_force,axis=1)))
    #     mean_norm_percent_diff = np.mean(np.nan_to_num(np.linalg.norm(difference,axis=1)/np.linalg.norm(volume_correction_force,axis=1)))
    #     print(f'maximum percentage difference in volume correction force norm is {max_norm_percent_diff}')
    #     print(f'mean percentage difference in volume correction force norm is {mean_norm_percent_diff}')
    print("CPU time is {} seconds".format(delta_cy))
    print("GPU time is {} seconds".format(delta_gpu_naive))
    print("GPU is {}x faster than CPU".format(delta_cy/delta_gpu_naive))
    print("End Main")
    # magnetic_moments = magnetic_moments.reshape((Hext_series.shape[0],num_particles,3))
    # magnetization = magnetic_moments/particle_volume
    # if np.allclose(M32bit,magnetization):
    #     print('seems to work')
    # else:
    #     print('mismatch between gpu calculated magnetization and cpu calculated magnetization')
    #     M32bit /= Ms
    #     magnetization /= Ms
    #     difference_norm = np.linalg.norm(M32bit-magnetization)
    #     print(f'difference norm of normalized magnetizations (all particles, all fields, all components): {difference_norm}')
    #     print(f'difference norm per total number of entries (field values * number particles * 3): {difference_norm/np.shape(np.ravel(magnetization))[0]}')
    #     print(f'mean normalized magnetization component percent difference: {np.mean((M32bit-magnetization)/M32bit)}')
    #     fig, ax = plt.subplots()
    #     ax.plot(Bext_series_magnitude,M32bit[:,0,0],label='p1x')
    #     ax.plot(Bext_series_magnitude,M32bit[:,0,1],label='p1y')
    #     ax.plot(Bext_series_magnitude,M32bit[:,0,2],label='p1z')
    #     # ax.plot(Hext_series_magnitude,M32bit[:,1,0],label='p2x')
    #     # ax.plot(Hext_series_magnitude,M32bit[:,1,1],label='p2y')
    #     # ax.plot(Hext_series_magnitude,M32bit[:,1,2],label='p2z')
    #     ax.plot(Bext_series_magnitude,magnetization[:,0,0],label='gpu p1x')
    #     ax.plot(Bext_series_magnitude,magnetization[:,0,1],label='gpu p1y')
    #     ax.plot(Bext_series_magnitude,magnetization[:,0,2],label='gpu p1z')
    #     # ax.plot(Hext_series_magnitude,magnetization[:,1,0],label='gpu p2x')
    #     # ax.plot(Hext_series_magnitude,magnetization[:,1,1],label='gpu p2y')
    #     # ax.plot(Hext_series_magnitude,magnetization[:,1,2],label='gpu p2z')
    #     fig.legend()
    #     plt.show()
    #     print('review plot')
        
    #     system_magnetization = np.sum(M32bit,axis=1)/num_particles
    #     system_magnetization = np.squeeze(system_magnetization)
    #     gpu_system_magnetization = np.sum(magnetization,axis=1)/num_particles
    #     gpu_system_magnetization = np.squeeze(gpu_system_magnetization)
    #     Bext_series = mu0*Hext_series
    #     Bext_series_norm = np.linalg.norm(Bext_series,axis=1)
    #     nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
    #     unit_Bext_series = Bext_series[nonzero_field_value_indices[0]]/Bext_series_norm[nonzero_field_value_indices[0]]
    #     parallel_magnetization = np.dot(system_magnetization,unit_Bext_series)
    #     gpu_parallel_magnetization = np.dot(gpu_system_magnetization,unit_Bext_series)

    #     fig, ax = plt.subplots()
    #     ax.plot(Bext_series_norm,parallel_magnetization,'.',label='CPU')
    #     ax.plot(Bext_series_norm,gpu_parallel_magnetization,'.',label='GPU')
    #     ax.set_xlabel('B Field (T)')
    #     ax.set_ylabel('Normalized Magnetization')
    #     ax.set_title(f'Magnetization versus Applied Field\n{num_particles} Particles, Separation {separation}\nTheta {Bext_theta_angle} Phi {Bext_phi_angle}')
    #     fig.legend()
    #     fig.show()
    #     save_dir = '/mnt/c/Users/bagaw/Desktop/MRE/magnetization_testing/'
    #     if not (os.path.isdir(save_dir)):
    #         os.mkdir(save_dir)
    #     savename = save_dir + f'{num_particles}_particles_magnetization_chi_{chi}_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
    #     plt.savefig(savename)

def gpu_testing():
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.1/mu0
    n_field_steps = 200
    if n_field_steps != 1:
        H_step = H_mag/(n_field_steps-1)
    else:
        H_step = H_mag/(n_field_steps)
    #polar angle, aka angle wrt the z axis, range 0 to pi
    Hext_theta_angle = np.pi/2
    Bext_theta_angle = Hext_theta_angle*360/(2*np.pi)
    Hext_phi_angle = 0#(2*np.pi/360)*15#30
    Bext_phi_angle = Hext_phi_angle*360/(2*np.pi)
    Hext_series_magnitude = np.arange(H_step,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    Bext_series_magnitude = Hext_series_magnitude*mu0
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,2] = Hext_series_magnitude*np.cos(Hext_theta_angle)
    chi = np.float32(131)
    Ms = np.float32(1.9e6)
    particle_radius = 1.5e-6
    l_e = np.float32(1e-6)
    num_nodes_per_particle = 8
    beta = 6.734260376702891e-09
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([2,1,1])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = 7.8
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation
                particle_posns[particle_counter,1] = j * separation
                particle_posns[particle_counter,2] = k * separation
                particle_counter += 1
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    M32bit = np.zeros((Hext_series.shape[0],num_particles,3),dtype=np.float32)
    magnetic_moments = np.zeros((Hext_series.shape[0],num_particles,3),dtype=np.float32)
    for i, Hext in enumerate(Hext_series):
        M32bit[i] = magnetism.get_magnetization_iterative_normalized_32bit(np.float32(Hext),np.float32(particle_posns),np.float32(particle_radius),np.float32(chi),np.float32(Ms),np.float32(l_e))
        magnetic_moments[i] = get_magnetization_iterative(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e)


    field_idx = int(n_field_steps/2)
    Hext = Hext_series[int(n_field_steps/2)]

    host_magnetic_moments = magnetic_moments[field_idx]
    #testing new and old CPU bound magnetic force calculations
    mag_forces32bit_new = np.zeros((num_particles,3),dtype=np.float32)
    mag_forces32bit_old = np.zeros((num_particles,3),dtype=np.float32)
    mag_forces32bit_new = magnetism.get_dip_dip_forces_normalized_32bit_v2(host_magnetic_moments,np.float32(particle_posns*l_e),np.float32(particle_radius),np.float32(l_e))
    mag_forces32bit_new *= np.float32(beta/particle_mass)#np.float32(beta/(particle_mass*(np.power(l_e,4))))


    magnetization_32bit = host_magnetic_moments/particle_volume
    mag_forces32bit_old = magnetism.get_dip_dip_forces_normalized_32bit(magnetization_32bit,np.float32(particle_posns),np.float32(particle_radius),np.float32(l_e))
    #  mag_forces32bit_old = magnetism.get_dip_dip_forces_normalized_32bit(M32bit[field_idx],np.float32(particle_posns),np.float32(particle_radius),np.float32(l_e))
    # magnetic_scaling_factor = beta/(particle_mass*(np.power(l_e,4)))
    mag_forces32bit_old *= np.float32(beta/(particle_mass*(np.power(l_e,4))))

    if not (np.allclose(mag_forces32bit_new,mag_forces32bit_old)):
        print('mismatch between new and old force calculations on cpu')
        print(f'old calculation values:{mag_forces32bit_old}\nnew calculation values:{mag_forces32bit_new}')
    else:
        print('agreement between new and old force calculations on cpu')

    N_iterations = 1    
    execution_gpu = cupyx.profiler.benchmark(get_magnetization_iterative,(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e),n_repeat=N_iterations)
    delta_gpu_naive = np.sum(execution_gpu.gpu_times)

    start = time.perf_counter()
    for i in range(N_iterations):
        perf_m32bit = magnetism.get_magnetization_iterative_normalized_32bit(np.float32(Hext),np.float32(particle_posns),np.float32(particle_radius),np.float32(chi),np.float32(Ms),np.float32(l_e))
    end = time.perf_counter()
    delta_cy = end-start
    # try:
    #     correctness = np.allclose(host_cupy_forces,np.float32(volume_correction_force))
    # except Exception as inst:
    #         print('Exception raised during calculation')
    #         print(type(inst))
    #         print(inst)
    # print("GPU and CPU based calculations of forces agree?: " + str(correctness))
    # if not correctness:
    #     difference = np.abs(host_cupy_forces-volume_correction_force)
    #     max_component_diff = np.max(difference)
    #     mean_component_diff = np.mean(difference)
    #     print(f'maximum difference in volume correction force components is {max_component_diff}')
    #     print(f'mean difference in volume correction force components is {mean_component_diff}')
    #     max_norm_diff = np.max(np.linalg.norm(difference,axis=1))
    #     mean_norm_diff = np.mean(np.linalg.norm(difference,axis=1))
    #     print(f'maximum difference in volume correction force norm is {max_norm_diff}')
    #     print(f'mean difference in volume correction force norm is {mean_norm_diff}')
    #     print(f'number of NaN entries in the CPU VCF {np.count_nonzero(np.isnan(volume_correction_force))}')
    #     print(f'number of NaN entries in the GPU VCF {np.count_nonzero(np.isnan(host_cupy_forces))}')
    #     print(f'number of NaN entries in the norm of the CPU VCF {np.count_nonzero(np.isnan(np.linalg.norm(volume_correction_force,axis=1)))}')
    #     print(f'number of zero entries in the norm of the CPU VCF {np.count_nonzero(np.isclose(np.linalg.norm(volume_correction_force,axis=1),0))}')
    #     max_norm_percent_diff = np.max(np.nan_to_num(np.linalg.norm(difference,axis=1)/np.linalg.norm(volume_correction_force,axis=1)))
    #     mean_norm_percent_diff = np.mean(np.nan_to_num(np.linalg.norm(difference,axis=1)/np.linalg.norm(volume_correction_force,axis=1)))
    #     print(f'maximum percentage difference in volume correction force norm is {max_norm_percent_diff}')
    #     print(f'mean percentage difference in volume correction force norm is {mean_norm_percent_diff}')
    print("CPU time is {} seconds".format(delta_cy))
    print("GPU time is {} seconds".format(delta_gpu_naive))
    print("GPU is {}x faster than CPU".format(delta_cy/delta_gpu_naive))

    magnetic_moments = magnetic_moments.reshape((Hext_series.shape[0],num_particles,3))
    magnetization = magnetic_moments/particle_volume
    if np.allclose(M32bit,magnetization):
        print('seems to work')
    else:
        print('mismatch between gpu calculated magnetization and cpu calculated magnetization')
        M32bit /= Ms
        magnetization /= Ms
        difference_norm = np.linalg.norm(M32bit-magnetization)
        print(f'difference norm of normalized magnetizations (all particles, all fields, all components): {difference_norm}')
        print(f'difference norm per total number of entries (field values * number particles * 3): {difference_norm/np.shape(np.ravel(magnetization))[0]}')
        print(f'mean normalized magnetization component percent difference: {np.mean((M32bit-magnetization)/M32bit)}')
        fig, ax = plt.subplots()
        ax.plot(Bext_series_magnitude,M32bit[:,0,0],label='p1x')
        ax.plot(Bext_series_magnitude,M32bit[:,0,1],label='p1y')
        ax.plot(Bext_series_magnitude,M32bit[:,0,2],label='p1z')
        # ax.plot(Hext_series_magnitude,M32bit[:,1,0],label='p2x')
        # ax.plot(Hext_series_magnitude,M32bit[:,1,1],label='p2y')
        # ax.plot(Hext_series_magnitude,M32bit[:,1,2],label='p2z')
        ax.plot(Bext_series_magnitude,magnetization[:,0,0],label='gpu p1x')
        ax.plot(Bext_series_magnitude,magnetization[:,0,1],label='gpu p1y')
        ax.plot(Bext_series_magnitude,magnetization[:,0,2],label='gpu p1z')
        # ax.plot(Hext_series_magnitude,magnetization[:,1,0],label='gpu p2x')
        # ax.plot(Hext_series_magnitude,magnetization[:,1,1],label='gpu p2y')
        # ax.plot(Hext_series_magnitude,magnetization[:,1,2],label='gpu p2z')
        fig.legend()
        plt.show()
        print('review plot')
        
        system_magnetization = np.sum(M32bit,axis=1)/num_particles
        system_magnetization = np.squeeze(system_magnetization)
        gpu_system_magnetization = np.sum(magnetization,axis=1)/num_particles
        gpu_system_magnetization = np.squeeze(gpu_system_magnetization)
        Bext_series = mu0*Hext_series
        Bext_series_norm = np.linalg.norm(Bext_series,axis=1)
        nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
        unit_Bext_series = Bext_series[nonzero_field_value_indices[0]]/Bext_series_norm[nonzero_field_value_indices[0]]
        parallel_magnetization = np.dot(system_magnetization,unit_Bext_series)
        gpu_parallel_magnetization = np.dot(gpu_system_magnetization,unit_Bext_series)

        fig, ax = plt.subplots()
        ax.plot(Bext_series_norm,parallel_magnetization,'.',label='CPU')
        ax.plot(Bext_series_norm,gpu_parallel_magnetization,'.',label='GPU')
        ax.set_xlabel('B Field (T)')
        ax.set_ylabel('Normalized Magnetization')
        ax.set_title(f'Magnetization versus Applied Field\n{num_particles} Particles, Separation {separation}\nTheta {Bext_theta_angle} Phi {Bext_phi_angle}')
        fig.legend()
        fig.show()
        save_dir = '/mnt/c/Users/bagaw/Desktop/MRE/magnetization_testing/'
        if not (os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        savename = save_dir + f'{num_particles}_particles_magnetization_chi_{chi}_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
        plt.savefig(savename)

def calc_A(rij,inv_rij_mag,num_particles,l_e,chi,particle_volume):
    """Return matrix A containing information about relative dipole positions for calculating dipolar fields, used to solve the system Ax=b for the magnetic moments of the dipoles."""
    A = cp.zeros((num_particles*3,num_particles*3),dtype=cp.float32)
    chi_v = chi*particle_volume

    nij_x = rij[0::3]*inv_rij_mag
    nij_y = rij[1::3]*inv_rij_mag
    nij_z = rij[2::3]*inv_rij_mag

    nij_x = cp.reshape(nij_x,(num_particles,num_particles),order='C')
    nij_y = cp.reshape(nij_y,(num_particles,num_particles),order='C')
    nij_z = cp.reshape(nij_z,(num_particles,num_particles),order='C')

    inv_rij_mag = cp.reshape(inv_rij_mag,(num_particles,num_particles),order='C')
    
    denom = chi_v/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)

    A[0::3,0::3] = -1*(3*nij_x*nij_x-1)*denom
    A[1::3,1::3] = -1*(3*nij_y*nij_y-1)*denom
    A[2::3,2::3] = -1*(3*nij_z*nij_z-1)*denom
    A[0::3,1::3] = -1*(3*nij_y)*nij_x*denom
    A[1::3,0::3] = A[0::3,1::3]
    A[0::3,2::3] = -1*(3*nij_z)*nij_x*denom
    A[2::3,0::3] = A[0::3,2::3]
    A[1::3,2::3] = -1*(3*nij_z)*nij_y*denom
    A[2::3,1::3] = A[1::3,2::3]
    A += cp.eye(num_particles*3,dtype=cp.float32)

    #if you wanted a matrix that you could multiply by a vector of the magnetic moments of the particles to get the dipole fields, an experssion like the one below would transform the final A matrix to the "gamma" matrix
    # gamma_from_A = -1*(A - eye(size(A)))/chi/V;

    # tempxx = -1*(3*nij_x*nij_x-1)*denom
    # tempyy = -1*(3*nij_y*nij_y-1)*denom
    # tempzz = -1*(3*nij_z*nij_z-1)*denom
    # tempxy = -1*(3*nij_y)*nij_x*denom
    # tempxz = -1*(3*nij_z)*nij_x*denom
    # tempyz = -1*(3*nij_z)*nij_y*denom
    return A

def linear_magnetization_solver(particle_posns,Hext,chi,particle_volume,l_e):
    """Assuming a linear magnetization response, find the magnetic moments of a set of mutually magnetizing dipoles in an external magnetic field"""
    num_particles = np.int64(particle_posns.shape[0]/3)
    assert particle_posns.shape[0] == 3*num_particles
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_particles = num_particles
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    A = calc_A(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,chi,particle_volume)
    Hext_vector = cp.tile(Hext,num_particles)
    chi_v = chi*particle_volume
    b = Hext_vector*chi_v
    # b = cp.ones((particle_posns.shape[0]*3,),dtype=cp.float32)*chi*particle_volume
    magnetic_moments = cp.linalg.solve(A,b)
    correctness_check = cp.dot(A,magnetic_moments)
    print(f'consistent answer from linsolve and Ax=b using magnetic moment result?:{cp.allclose(correctness_check,b)}')
    magnetic_moments = cp.asnumpy(magnetic_moments).reshape((num_particles,3))
    return magnetic_moments

def iterative_magnetization_testing():
    """Plot the magnetization vector components vs iteration number to see how iteration number effects values. Do values converge? How much does microstructure/particle placement impact results?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.01/mu0
    n_field_steps = 10
    H_step = H_mag/(n_field_steps)
    # if n_field_steps != 1:
    #     H_step = H_mag/(n_field_steps-1)
    # else:
    #     H_step = H_mag/(n_field_steps)
    #polar angle, aka angle wrt the z axis, range 0 to pi
    Hext_theta_angle = np.pi/2
    Bext_theta_angle = Hext_theta_angle*360/(2*np.pi)
    Hext_phi_angle = 0#(2*np.pi/360)*5#0#(2*np.pi/360)*15#30
    Bext_phi_angle = Hext_phi_angle*360/(2*np.pi)
    Hext_series_magnitude = np.arange(H_step,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    Bext_series_magnitude = Hext_series_magnitude*mu0
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,2] = Hext_series_magnitude*np.cos(Hext_theta_angle)
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = 1.5e-6
    l_e = np.float32(1e-6)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([2,1,1])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = 3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation# + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation# + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation# + rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 20
    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    linear_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        linear_magnetic_moments[i] = linear_magnetization_solver(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,particle_volume,l_e)




    # magnetic_moments = magnetic_moments.reshape((Hext_series.shape[0],num_particles,3))
    magnetization = magnetic_moments/particle_volume
    linear_magnetization = linear_magnetic_moments/particle_volume

    magnetization /= Ms

    linear_magnetization /= Ms
    
    iter_number = np.arange(max_iters)
    field_index = 0
    
    # fig, axs = plt.subplots(1,3)
    # for count in range(particles.shape[0]):
    #     axs[0].plot(iter_number,magnetization[field_index,:,count,0])
    #     axs[1].plot(iter_number,magnetization[field_index,:,count,1])
    #     axs[2].plot(iter_number,magnetization[field_index,:,count,2])
    # # ax.plot(Bext_series_magnitude,magnetization[:,0,0],label='gpu p1x')
    # # ax.plot(Bext_series_magnitude,magnetization[:,0,1],label='gpu p1y')
    # # ax.plot(Bext_series_magnitude,magnetization[:,0,2],label='gpu p1z')
    # axs[0].set_xlabel('iteration number')
    # axs[1].set_xlabel('iteration number')
    # axs[2].set_xlabel('iteration number')
    # axs[0].set_title('X')
    # axs[1].set_title('Y')
    # axs[2].set_title('Z')
    # axs[0].set_ylabel('normalized particle magnetization')

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,linear_magnetization[field_index,:,count,0])
        axs[1,1].plot(iter_number,linear_magnetization[field_index,:,count,1])
        axs[1,2].plot(iter_number,linear_magnetization[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (linear)')

    plt.show()

    
    system_magnetization = np.sum(magnetic_moments[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    fig, ax = plt.subplots()
    ax.plot(iter_number,system_magnetization[:,:])
    ax.set_xlabel('iteration number')
    ax.set_ylabel('normalized system magnetization')
        

    # fig.legend()
    plt.show()
    print('review plot')
    
    # gpu_system_magnetization = np.sum(magnetization,axis=1)/num_particles
    # gpu_system_magnetization = np.squeeze(gpu_system_magnetization)
    # Bext_series = mu0*Hext_series
    # Bext_series_norm = np.linalg.norm(Bext_series,axis=1)
    # nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
    # unit_Bext_series = Bext_series[nonzero_field_value_indices[0]]/Bext_series_norm[nonzero_field_value_indices[0]]
    # gpu_parallel_magnetization = np.dot(gpu_system_magnetization,unit_Bext_series)

    # fig, ax = plt.subplots()
    # ax.plot(Bext_series_norm,gpu_parallel_magnetization,'.',label='GPU')
    # ax.set_xlabel('B Field (T)')
    # ax.set_ylabel('Normalized Magnetization')
    # ax.set_title(f'Magnetization versus Applied Field\n{num_particles} Particles, Separation {separation}\nTheta {Bext_theta_angle} Phi {Bext_phi_angle}')
    # fig.legend()
    # fig.show()
    # save_dir = '/mnt/c/Users/bagaw/Desktop/MRE/magnetization_testing/'
    # if not (os.path.isdir(save_dir)):
    #     os.mkdir(save_dir)
    # savename = save_dir + f'{num_particles}_particles_magnetization_chi_{chi}_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
    # plt.savefig(savename)
if __name__ == "__main__":
    # main()
    # gpu_testing()
    # gpu_testing_force_calc()
    iterative_magnetization_testing()