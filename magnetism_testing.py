import numpy as np
import cupy as cp
import cupyx
import magnetism
import matplotlib.pyplot as plt
import os
import scipy.special as sci
import scipy.optimize
import time
import numpy.random as rand
import simulate
import torch
import torch.optim as opt
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

normalized_magnetization_kernel = cp.RawKernel(r'''
extern "C" __global__                                    
void get_magnetization(const float chi, const float* h, float* magnetization, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        //get magnetization based on total Hfield at each particle center
        float hmag = sqrtf(h[3*tid]*h[3*tid] + h[3*tid+1]*h[3*tid+1] + h[3*tid+2]*h[3*tid+2]);
        float m_mag = (chi*hmag/(1 + chi*hmag));
        float h_hat[3];
        h_hat[0] = h[3*tid]/hmag;
        h_hat[1] = h[3*tid+1]/hmag;
        h_hat[2] = h[3*tid+2]/hmag;
        magnetization[3*tid] = m_mag*h_hat[0];
        magnetization[3*tid+1] = m_mag*h_hat[1];
        magnetization[3*tid+2] = m_mag*h_hat[2];
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

separation_vectors_pbc_kernel = cp.RawKernel(r'''
extern "C" __global__                                         
void get_separation_vectors(const float* particle_posns, float* separation_vectors, float* separation_vectors_inv_magnitude, const int size_particles, const int* num_images, const float* image_translation_vector, float* image_separation_vectors, float* image_separation_vectors_inv_magnitude) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        float ri[3];
        int separation_vector_idx;
        int posn_idx;
        int inv_magnitude_idx;
        int num_image_volumes = (2*num_images[0] + 1) * (2*num_images[1] + 1) * (2*num_images[2] + 1) - 1;
        int num_image_particles = size_particles*num_image_volumes;
        int image_counter = 0;
        int image_separation_vector_idx;
        int image_inv_magnitude_idx;
        ri[0] = particle_posns[3*tid];
        ri[1] = particle_posns[3*tid+1];
        ri[2] = particle_posns[3*tid+2];
        for (int i = 0; i < size_particles; i++)
        {
            posn_idx = 3*i;
            if (i != tid)
            {
            separation_vector_idx = tid*3*size_particles+3*i;
            inv_magnitude_idx = tid*size_particles + i;
            //posn_idx = 3*i;
            separation_vectors[separation_vector_idx] = ri[0] - particle_posns[posn_idx];                                         
            separation_vectors[separation_vector_idx+1] = ri[1] - particle_posns[posn_idx+1];                                                                
            separation_vectors[separation_vector_idx+2] = ri[2] - particle_posns[posn_idx+2];                                                                
            separation_vectors_inv_magnitude[inv_magnitude_idx] = rnorm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2]);
            }
            for (int image_count_x = -1*num_images[0]; image_count_x < num_images[0]+1; image_count_x++)
            {
                for (int image_count_y = -1*num_images[1]; image_count_y < num_images[1]+1; image_count_y++)
                {
                    for (int image_count_z = -1*num_images[2]; image_count_z < num_images[2]+1; image_count_z++)
                    {
                        if ((image_count_x == 0) && (image_count_y == 0) && (image_count_z == 0))
                        {
                        }
                        else
                        {
                            image_separation_vector_idx = tid*3*num_image_particles + 3*image_counter;
                            image_inv_magnitude_idx = tid*num_image_particles + image_counter;
                            image_separation_vectors[image_separation_vector_idx] = ri[0] - (particle_posns[posn_idx] + image_count_x*image_translation_vector[0]);
                            image_separation_vectors[image_separation_vector_idx+1] = ri[1] - (particle_posns[posn_idx+1] + image_count_y*image_translation_vector[1]);
                            image_separation_vectors[image_separation_vector_idx+2] = ri[2] - (particle_posns[posn_idx+2] + image_count_z*image_translation_vector[2]);
                            image_separation_vectors_inv_magnitude[image_inv_magnitude_idx] = rnorm3df(image_separation_vectors[image_separation_vector_idx],image_separation_vectors[image_separation_vector_idx+1],image_separation_vectors[image_separation_vector_idx+2]);
                            image_counter++;
                        }
                    }
                }
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

dipole_field_pbc_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
extern "C" __global__
void get_dipole_field(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float* magnetic_moment, const float inv_l_e, float* Htot, const int size_particles, const float* image_separation_vectors, const float* image_separation_vectors_inv_magnitude, const int* num_images) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float prefactor;
        float r_hat[3];
        float m_dot_r_hat;
        int separation_vector_idx;
        int inv_magnitude_idx;
        int num_image_volumes = (2*num_images[0] + 1) * (2*num_images[1] + 1) * (2*num_images[2] + 1) - 1;
        int num_image_particles = size_particles*num_image_volumes;
        int image_mag_moment_idx;
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
        for (int i = 0; i < num_image_particles; i++)
        {
            separation_vector_idx = tid*3*num_image_particles+3*i;
            inv_magnitude_idx = tid*num_image_particles + i;
            image_mag_moment_idx = i / num_image_volumes;
            //printf("image mag moment index:%i",image_mag_moment_idx);
            r_hat[0] = image_separation_vectors[separation_vector_idx]*image_separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[1] = image_separation_vectors[separation_vector_idx+1]*image_separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[2] = image_separation_vectors[separation_vector_idx+2]*image_separation_vectors_inv_magnitude[inv_magnitude_idx];
            m_dot_r_hat = magnetic_moment[3*image_mag_moment_idx]*r_hat[0] + magnetic_moment[3*image_mag_moment_idx+1]*r_hat[1] + magnetic_moment[3*image_mag_moment_idx+2]*r_hat[2];
            prefactor = INV_4PI*powf(image_separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e,3);
            Htot[3*tid] += prefactor*(3*m_dot_r_hat*r_hat[0] - magnetic_moment[3*image_mag_moment_idx]);
            Htot[3*tid+1] += prefactor*(3*m_dot_r_hat*r_hat[1] - magnetic_moment[3*image_mag_moment_idx+1]);
            Htot[3*tid+2] += prefactor*(3*m_dot_r_hat*r_hat[2] - magnetic_moment[3*image_mag_moment_idx+2]);
        }
    }
    }
    ''', 'get_dipole_field')

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

def get_magnetization_iterative_iteration_testing(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=20,atol=None,rtol=1e-4):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    if atol == None:
        atol = particle_volume*Ms*1e-3
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
        if i > 0:
            difference = host_magnetic_moments[i] - host_magnetic_moments[i-1]
            if np.all(np.abs(np.ravel(difference)) < atol):
                host_magnetic_moments[i:] = host_magnetic_moments[i]
                break

    return host_magnetic_moments

def get_magnetization_iterative_pbc_iteration_testing(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,num_images,image_translation_vector,max_iters=20,atol=None,rtol=1e-4):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    if atol == None:
        atol = particle_volume*Ms*1e-3
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
    num_image_volumes = (2*num_images[0] + 1) * (2*num_images[1] + 1) * (2*num_images[2] + 1) - 1 
    num_images = cp.asarray(num_images,dtype=cp.int32)
    system_volume = np.float32(image_translation_vector[0]*image_translation_vector[1]*image_translation_vector[2]*np.power(l_e,3))
    image_translation_vector = cp.asarray(image_translation_vector,dtype=cp.float32)
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    image_separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*num_image_volumes*3,1),dtype=cp.float32)
    image_separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0]*num_image_volumes,1),dtype=cp.float32)
    
    separation_vectors_pbc_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles,num_images,image_translation_vector,image_separation_vectors,image_separation_vectors_inv_magnitude))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    for i in range(max_iters):
        system_M = cp.sum(cp.reshape(magnetic_moment,(size_particles,3)),axis=0)/system_volume
        Htot = cp.tile(Hext - (1./3.)*system_M,particles.shape[0])
        # Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_pbc_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles,image_separation_vectors,image_separation_vectors_inv_magnitude,num_images))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))
        if i > 0:
            difference = host_magnetic_moments[i] - host_magnetic_moments[i-1]
            if np.all(np.abs(np.ravel(difference)) < atol):
                host_magnetic_moments[i:] = host_magnetic_moments[i]
                break

    return host_magnetic_moments

def get_magnetization_iterative_iteration_testing_w_outputflipping(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    magnetic_moment_history = cp.zeros((3,particles.shape[0],3),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    host_magnetic_field = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
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
    min_iterations = 3
    absolute_tolerance = particle_volume*Ms*5e-3
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        magnetic_moments_flipped = realign_and_scale_magnetic_moments(magnetic_moment,Hext,particle_volume,Ms,size_particles)
        # we want to let the system iterate a few times before checking for oscillations so that the zero initialization doesn't somehow trigger something
        if i >= min_iterations:
            #when we do check for oscillations, because of the way we are recording the history on a running basis (storing the past 3 results) and overwriting results using modulo math for indexing the historical array, we need to ensure we index the correct parts of the historical array so we are comparing the right sets of results to look for oscillations
            #see 2024-07-17 in progress work on linear magnetization solving and magnetizing solving in general.txt line 99-110 to see how the pattern was derived (reproduced below)
            #three histories plus current, numbers represnt iteration number, X's are compared, O's are compared, O is always used for the current value, what is the pattern?
            # 0,1,2 3
            # X,O,X,O
            # 3,1,2 4
            # X,X,O,O
            # 3,4,2 5
            # O,X,X,O
            # 3,4,5 6
            # X,O,X,O -> pattern starts over
            # compare current to history[1], then [2], then [0] (so np.mod(i+1,3))
            #compare history[0] to [2], [1] to [0], [2] to [1], so np.mod(i,3) and np.mod(i+2,3)
            comparison_indices = [np.mod(i+1,3),np.mod(i,3),np.mod(i+2,3)]
            if cp.allclose(magnetic_moment_history[comparison_indices[1]],magnetic_moment_history[comparison_indices[2]],atol=absolute_tolerance) and cp.allclose(magnetic_moments_flipped,magnetic_moment_history[comparison_indices[0]],atol=absolute_tolerance):
                avg_magnetic_moments = (cp.sum(magnetic_moment_history,axis=0)+magnetic_moments_flipped)/4
                host_magnetic_moments[i:] = cp.asnumpy(avg_magnetic_moments).reshape((particles.shape[0],3))
                break
        magnetic_moment_history[np.mod(i,3)] = magnetic_moments_flipped
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moments_flipped).reshape((particles.shape[0],3))
        host_magnetic_field[i] = cp.asnumpy(Htot).reshape((size_particles,3))

    # magnetic_moment = realign_and_scale_magnetic_moments(magnetic_moment,Hext,particle_volume,Ms,size_particles)
    # host_magnetic_moments[-1] = cp.asnumpy(magnetic_moment)

    return host_magnetic_moments, host_magnetic_field

def get_magnetization_iterative_iteration_testing_w_drag(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    drag = 0.5
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    old_magnetic_moment = cp.copy(magnetic_moment)
    max_allowed_change = Ms*particle_volume*0.05
    min_of_max_allowed_change = Ms*particle_volume*1e-5

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
        delta_magnetic_moment = magnetic_moment-old_magnetic_moment
        sign_of_delta = cp.sign(delta_magnetic_moment)
        mask = cp.nonzero(cp.abs(delta_magnetic_moment) > max_allowed_change)[0]
        delta_magnetic_moment[mask] = max_allowed_change*sign_of_delta[mask]
        max_allowed_change *= 0.9
        if max_allowed_change < min_of_max_allowed_change:
            max_allowed_change == min_of_max_allowed_change
        # cp.put(delta_magnetic_moment,mask,max_allowed_change*cp.sign(delta_magnetic_moment))
        magnetic_moment = old_magnetic_moment + delta_magnetic_moment#drag
        old_magnetic_moment = cp.copy(magnetic_moment)
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))

    return host_magnetic_moments

def get_magnetization_iterative_new_drag(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
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

    old_magnetic_moment = cp.copy(magnetic_moment)
    max_allowed_change = Ms*particle_volume*0.5
    min_of_max_allowed_change = Ms*particle_volume*1e-2

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
        delta_magnetic_moment = magnetic_moment-old_magnetic_moment
        sign_of_delta = cp.sign(delta_magnetic_moment)
        mask = cp.nonzero(cp.abs(delta_magnetic_moment) > max_allowed_change)[0]
        delta_magnetic_moment[mask] = max_allowed_change*sign_of_delta[mask]
        max_allowed_change *= 0.9
        if max_allowed_change < min_of_max_allowed_change:
            max_allowed_change == min_of_max_allowed_change
        magnetic_moment = old_magnetic_moment + delta_magnetic_moment
        old_magnetic_moment = cp.copy(magnetic_moment)
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))

    return host_magnetic_moments

def get_magnetization_iterative_iteration_testing_w_averaging(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
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

    old_magnetic_moment = cp.zeros((max_iters,particles.shape[0]*3,1),dtype=cp.float32)

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
        old_magnetic_moment[i] = cp.copy(magnetic_moment)
        if np.mod(i,5) == 0 and i != 0:
            magnetic_moment = cp.reshape(cp.mean(cp.reshape(old_magnetic_moment,(max_iters,particles.shape[0],3))[i-5:i],axis=0),(particles.shape[0]*3,1))
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))

    return host_magnetic_moments

def get_magnetization_iterative_iteration_testing_w_seeded_values(linear_magnetic_moments,Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    # magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    
    magnetic_moment = cp.array(np.reshape(linear_magnetic_moments,(3*size_particles,)))
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        # magnetic_moment = realign_and_scale_magnetic_moments(magnetic_moment,Hext,particle_volume,Ms,size_particles)
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((size_particles,3))
        magnetic_moment.reshape((size_particles*3,))

    return host_magnetic_moments

def get_magnetization_iterative_iteration_testing_w_seeded_values_and_outputflipping(linear_magnetic_moments,Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=5):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    # magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    
    magnetic_moment = cp.array(np.reshape(linear_magnetic_moments,(3*size_particles,)))
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        magnetic_moments_flipped = realign_and_scale_magnetic_moments(magnetic_moment,Hext,particle_volume,Ms,size_particles)
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moments_flipped).reshape((particles.shape[0],3))

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
    
    denom = chi_v/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#particle_volume/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#

    A[0::3,0::3] = -1*(3*nij_x*nij_x-1)*denom
    A[1::3,1::3] = -1*(3*nij_y*nij_y-1)*denom
    A[2::3,2::3] = -1*(3*nij_z*nij_z-1)*denom
    A[0::3,1::3] = -1*(3*nij_y)*nij_x*denom
    A[1::3,0::3] = A[0::3,1::3]
    A[0::3,2::3] = -1*(3*nij_z)*nij_x*denom
    A[2::3,0::3] = A[0::3,2::3]
    A[1::3,2::3] = -1*(3*nij_z)*nij_y*denom
    A[2::3,1::3] = A[1::3,2::3]
    A += cp.eye(num_particles*3,dtype=cp.float32)#/chi

    #if you wanted a matrix that you could multiply by a vector of the magnetic moments of the particles to get the dipole fields, an experssion like the one below would transform the final A matrix to the "gamma" matrix
    # gamma_from_A = -1*(A - eye(size(A)))/chi/V;

    # tempxx = -1*(3*nij_x*nij_x-1)*denom
    # tempyy = -1*(3*nij_y*nij_y-1)*denom
    # tempzz = -1*(3*nij_z*nij_z-1)*denom
    # tempxy = -1*(3*nij_y)*nij_x*denom
    # tempxz = -1*(3*nij_z)*nij_x*denom
    # tempyz = -1*(3*nij_z)*nij_y*denom
    return A

def linear_magnetization_solver(particle_posns,Hext,chi,Ms,particle_volume,l_e):
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
    # chi = 3*chi/(3+chi)
    A = calc_A(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,chi,particle_volume)
    Hext_vector = cp.tile(Hext,num_particles)
    chi_v = chi*particle_volume
    b = Hext_vector*chi_v#*particle_volume#
    # b = cp.ones((particle_posns.shape[0]*3,),dtype=cp.float32)*chi*particle_volume
    magnetic_moments = cp.linalg.solve(A,b)
    # correctness_check = cp.dot(A,magnetic_moments)
    # temp = b - correctness_check
    # absolute_tolerance = 1.6e6*particle_volume*0.005
    # print(f'consistent answer from linsolve and Ax=b using magnetic moment result?:{cp.allclose(correctness_check,b,atol=absolute_tolerance)}')

    # magnetic_moments = realign_and_scale_magnetic_moments(magnetic_moments,Hext,particle_volume,Ms,num_particles)

    Ainv = cp.linalg.inv(A)
    AinvA = cp.matmul(Ainv,A)
    trace_check = cp.trace(AinvA)
    det_check = cp.linalg.det(AinvA)
    # print(f'Trace of A^-1 A is {trace_check}')
    # print(f'Determinant of A^-1 A is {det_check}')

    magnetic_moments = cp.asnumpy(magnetic_moments).reshape((num_particles,3))
    return magnetic_moments, trace_check, det_check

def linear_normalized_magnetization_solver(particle_posns,Hext,chi,Ms,particle_volume,l_e):
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
    # chi = 3*chi/(3+chi)
    A = calc_A(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,chi,particle_volume)
    Hext_vector = cp.tile(Hext,num_particles)
    # chi_v = chi*particle_volume
    # b = Hext_vector*chi_v#*particle_volume#
    chiMs = chi/Ms
    b = Hext_vector*chiMs
    # b = cp.ones((particle_posns.shape[0]*3,),dtype=cp.float32)*chi*particle_volume
    magnetic_moments = cp.linalg.solve(A,b)
    # correctness_check = cp.dot(A,magnetic_moments)
    # temp = b - correctness_check
    # absolute_tolerance = 1.6e6*particle_volume*0.005
    # print(f'consistent answer from linsolve and Ax=b using magnetic moment result?:{cp.allclose(correctness_check,b,atol=absolute_tolerance)}')

    # magnetic_moments = realign_and_scale_magnetic_moments(magnetic_moments,Hext,particle_volume,Ms,num_particles)

    Ainv = cp.linalg.inv(A)
    AinvA = cp.matmul(Ainv,A)
    trace_check = cp.trace(AinvA)
    det_check = cp.linalg.det(AinvA)
    # print(f'Trace of A^-1 A is {trace_check}')
    # print(f'Determinant of A^-1 A is {det_check}')

    magnetic_moments = cp.asnumpy(magnetic_moments).reshape((num_particles,3))
    return magnetic_moments, trace_check, det_check

def realign_and_scale_magnetic_moments(magnetic_moments,Hext,particle_volume,Ms,num_particles):
    """Checks if magnetic moment values are anti-aligned with the external field and flips them, then constrains magnetic moment vector magnitude to saturation value for oversaturated or flipped magnetic moments and returns the resulting magnetic moment components"""
    new_magnetic_moments = cp.copy(cp.reshape(magnetic_moments,(num_particles,3)))

    #check if the magnetic moment is anti-parallel (at least partially) to the external field
    
    Hext_magnitude = cp.linalg.norm(Hext)
    Hext_hat = Hext/Hext_magnitude
    
    alignment_check = cp.einsum('ij,j->i',new_magnetic_moments,Hext_hat)
    alignment_mask = alignment_check < 0
    new_magnetic_moments[alignment_mask] *= -1

    mag_moment_magnitude = cp.linalg.norm(new_magnetic_moments,axis=1)
    max_mag_moment = particle_volume*Ms
    # oversaturated_mask = mag_moment_magnitude/particle_volume/Ms>1
    oversaturated_mask = cp.greater(mag_moment_magnitude/particle_volume/Ms,1)
    scaling_mask = cp.logical_or(alignment_mask,oversaturated_mask)
    if np.any(scaling_mask):
        new_magnetic_moments[scaling_mask] *= max_mag_moment/mag_moment_magnitude[scaling_mask][:,cp.newaxis]
    return new_magnetic_moments

def calc_A_chi_scaled(rij,inv_rij_mag,num_particles,l_e,chi,particle_volume):
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
    
    denom = particle_volume/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#

    A[0::3,0::3] = -1*(3*nij_x*nij_x-1)*denom
    A[1::3,1::3] = -1*(3*nij_y*nij_y-1)*denom
    A[2::3,2::3] = -1*(3*nij_z*nij_z-1)*denom
    A[0::3,1::3] = -1*(3*nij_y)*nij_x*denom
    A[1::3,0::3] = A[0::3,1::3]
    A[0::3,2::3] = -1*(3*nij_z)*nij_x*denom
    A[2::3,0::3] = A[0::3,2::3]
    A[1::3,2::3] = -1*(3*nij_z)*nij_y*denom
    A[2::3,1::3] = A[1::3,2::3]
    A += cp.eye(num_particles*3,dtype=cp.float32)#/chi
    return A

def linear_magnetization_solver_chi_scaled(particle_posns,Hext,chi,particle_volume,l_e):
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

    A = calc_A_chi_scaled(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,chi,particle_volume)
    Hext_vector = cp.tile(Hext,num_particles)
    chi_v = chi*particle_volume
    b = Hext_vector*particle_volume#
    # b = cp.ones((particle_posns.shape[0]*3,),dtype=cp.float32)*chi*particle_volume
    magnetic_moments = cp.linalg.solve(A,b)
    # correctness_check = cp.dot(A,magnetic_moments)
    # print(f'consistent answer from linsolve and Ax=b using magnetic moment result?:{cp.allclose(correctness_check,b)}')
    magnetic_moments = cp.asnumpy(magnetic_moments).reshape((num_particles,3))*chi
    return magnetic_moments

def linear_magnetization_comparison(magnetic_moments,particle_posns,Hext,chi,particle_volume,l_e):
    """Assuming a linear magnetization response, given the magnetic moments of a set of mutually magnetizing dipoles in an external magnetic field, see if it matches the result of a linear magnetization response"""
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
    b = Hext_vector*chi_v#*particle_volume#
    # magnetic_moments = cp.linalg.solve(A,b)
    correctness_check = cp.dot(A,magnetic_moments)
    temp = b - correctness_check
    absolute_tolerance = 1.6e6*particle_volume*0.005
    print(f'consistent answer from linsolve and Ax=b using magnetic moment result?:{cp.allclose(correctness_check,b,atol=absolute_tolerance)}')


def iterative_magnetization_testing():
    """Plot the magnetization vector components vs iteration number to see how iteration number effects values. Do values converge? How much does microstructure/particle placement impact results?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.001/mu0
    n_field_steps = 1
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
    l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([2,2,2])
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
                particle_posns[particle_counter,0] = i * separation #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40
    max_iters_w_drag = 80
    max_iters_modified = 8
    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    magnetic_moments_w_flipping = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    total_field = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    magnetic_moments_w_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)
    linear_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    linear_magnetization_w_chi_scaled = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    seeded_iterative_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    seeded_iterative_magnetic_moments_w_output_flipping = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    averaging_iterative_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        magnetic_moments_w_flipping[i], total_field[i] = get_magnetization_iterative_iteration_testing_w_outputflipping(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        linear_magnetic_moments[i], _, _ = linear_magnetization_solver(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,Ms,particle_volume,l_e)
        # linear_magnetization_w_chi_scaled[i] = linear_magnetization_solver_chi_scaled(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,particle_volume,l_e)
        seeded_iterative_magnetic_moments[i] = get_magnetization_iterative_iteration_testing_w_seeded_values(linear_magnetic_moments[i,0],cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        seeded_iterative_magnetic_moments_w_output_flipping[i] = get_magnetization_iterative_iteration_testing_w_seeded_values_and_outputflipping(linear_magnetic_moments[i,0],cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        magnetic_moments_w_drag[i] = get_magnetization_iterative_iteration_testing_w_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)
        averaging_iterative_magnetic_moments[i] = get_magnetization_iterative_iteration_testing_w_averaging(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)




    # magnetic_moments = magnetic_moments.reshape((Hext_series.shape[0],num_particles,3))
    magnetization = magnetic_moments/particle_volume
    magnetization_w_flipping = magnetic_moments_w_flipping/particle_volume
    linear_magnetization = linear_magnetic_moments/particle_volume
    linear_magnetization_w_chi_scaled = linear_magnetization_w_chi_scaled/particle_volume
    seeded_iterative_magnetization = seeded_iterative_magnetic_moments/particle_volume
    seeded_iterative_magnetization_w_output_flipping = seeded_iterative_magnetic_moments_w_output_flipping/particle_volume
    magnetization_w_drag = magnetic_moments_w_drag/particle_volume
    magnetization_w_averaging = averaging_iterative_magnetic_moments/particle_volume

    magnetization /= Ms
    magnetization_w_flipping /= Ms
    seeded_iterative_magnetization /= Ms
    seeded_iterative_magnetization_w_output_flipping /= Ms
    magnetization_w_drag /= Ms
    magnetization_w_averaging /= Ms

    linear_magnetization /= Ms
    linear_magnetization_w_chi_scaled /= Ms
    
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

    # fig, axs = plt.subplots(2,3)
    # for count in range(particles.shape[0]):
    #     axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
    #     axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
    #     axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
    #     axs[1,0].plot(iter_number,linear_magnetization[field_index,:,count,0])
    #     axs[1,1].plot(iter_number,linear_magnetization[field_index,:,count,1])
    #     axs[1,2].plot(iter_number,linear_magnetization[field_index,:,count,2])

    # axs[1,0].set_xlabel('iteration number')
    # axs[1,1].set_xlabel('iteration number')
    # axs[1,2].set_xlabel('iteration number')
    # axs[0,0].set_title('X')
    # axs[0,1].set_title('Y')
    # axs[0,2].set_title('Z')
    # axs[0,0].set_ylabel('normalized particle magnetization')
    # axs[1,0].set_ylabel('normalized particle magnetization (linear)')

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,linear_magnetization[field_index,:,count,0])
        axs[1,1].plot(iter_number,linear_magnetization[field_index,:,count,1])
        axs[1,2].plot(iter_number,linear_magnetization[field_index,:,count,2])
        axs[2,0].plot(iter_number,linear_magnetization_w_chi_scaled[field_index,:,count,0])
        axs[2,1].plot(iter_number,linear_magnetization_w_chi_scaled[field_index,:,count,1])
        axs[2,2].plot(iter_number,linear_magnetization_w_chi_scaled[field_index,:,count,2])

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (linear)')
    axs[2,0].set_ylabel('normalized particle magnetization (linear but scaled)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,seeded_iterative_magnetization[field_index,:,count,0])
        axs[1,1].plot(iter_number,seeded_iterative_magnetization[field_index,:,count,1])
        axs[1,2].plot(iter_number,seeded_iterative_magnetization[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (seeded)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,seeded_iterative_magnetization_w_output_flipping[field_index,:,count,0])
        axs[1,1].plot(iter_number,seeded_iterative_magnetization_w_output_flipping[field_index,:,count,1])
        axs[1,2].plot(iter_number,seeded_iterative_magnetization_w_output_flipping[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (seeded w output flipping)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,magnetization_w_flipping[field_index,:,count,0])
        axs[1,1].plot(iter_number,magnetization_w_flipping[field_index,:,count,1])
        axs[1,2].plot(iter_number,magnetization_w_flipping[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (w_flipping)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,total_field[field_index,:,count,0])
        axs[0,1].plot(iter_number,total_field[field_index,:,count,1])
        axs[0,2].plot(iter_number,total_field[field_index,:,count,2])
        axs[1,0].plot(iter_number,magnetization_w_flipping[field_index,:,count,0])
        axs[1,1].plot(iter_number,magnetization_w_flipping[field_index,:,count,1])
        axs[1,2].plot(iter_number,magnetization_w_flipping[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('Magnetic Field (A/m)')
    axs[1,0].set_ylabel('normalized particle magnetization (w_flipping)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,0])
        axs[1,1].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,1])
        axs[1,2].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (w drag)')

    plt.show(block=False)
    # comparison_mag_moments = cp.asarray(magnetic_moments_w_drag[field_index,-1]).reshape((num_particles*3,1))
    # linear_magnetization_comparison(comparison_mag_moments,device_particle_posns,cp.asarray(Hext_series[field_index],dtype=cp.float32),chi,particle_volume,l_e)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,magnetization_w_averaging[field_index,:,count,0])
        axs[1,1].plot(iter_number,magnetization_w_averaging[field_index,:,count,1])
        axs[1,2].plot(iter_number,magnetization_w_averaging[field_index,:,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (w averaging)')

    plt.show(block=False)

    iter_number = np.arange(max_iters_w_drag)
    system_magnetization = np.sum(magnetic_moments_w_drag[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    fig, ax = plt.subplots()
    ax.plot(iter_number,system_magnetization[:,:])
    ax.set_xlabel('iteration number')
    ax.set_ylabel('normalized system magnetization')
        

    # fig.legend()
    plt.show(block=False)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.set_title('Normalized Magnetic Moment vectors (iterative)')
    ax.set_xlabel('X (l_e)')
    ax.set_ylabel('Y (l_e)')
    ax.set_zlabel('Z (l_e)')
    ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],magnetization_w_flipping[field_index,-1,:,0],magnetization_w_flipping[field_index,-1,:,1],magnetization_w_flipping[field_index,-1,:,2],pivot='middle',length=10.0)
    # ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],seeded_iterative_magnetization[field_index,-1,:,0],seeded_iterative_magnetization[field_index,-1,:,1],seeded_iterative_magnetization[field_index,-1,:,2],pivot='middle',units='width')
    plt.show(block=False)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.set_title('Normalized Magnetic Moment vectors (linear)')
    ax.set_xlabel('X (l_e)')
    ax.set_ylabel('Y (l_e)')
    ax.set_zlabel('Z (l_e)')
    ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],linear_magnetization[field_index,-1,:,0],linear_magnetization[field_index,-1,:,1],linear_magnetization[field_index,-1,:,2],pivot='middle',length=3.0)

    plt.show()
    print('review plot')

def linear_magnetization_testing():
    """Plot the magnetization vector components vs separation for two particles to see how separation effects values, also chi. At what separation do we see anti-parallel alignment with the external field? At what field value do we see alignment with the external field even at surface-surface separations?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.001/mu0
    n_field_steps = 1
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
    chi = np.float32(70)#np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = 1.5e-6
    l_e = np.float32(1e-6)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([2,2,1])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = 3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40
    min_separation = 3.1e-6
    max_separation = 8e-6
    separations = np.linspace(min_separation,max_separation,num=500)
    num_separations = separations.shape[0]
    linear_magnetic_moments = np.zeros((Hext_series.shape[0],num_separations,num_particles,3),dtype=np.float32)
    linear_normalized_magnetization = np.zeros((Hext_series.shape[0],num_separations,num_particles,3),dtype=np.float32)
    traceA = np.zeros((Hext_series.shape[0],num_separations),dtype=np.float32)
    detA = np.zeros((Hext_series.shape[0],num_separations),dtype=np.float32)
    for separation_counter, separation in enumerate(separations):
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
        for i, Hext in enumerate(Hext_series):
            linear_magnetic_moments[i,separation_counter], traceA[i,separation_counter], detA[i,separation_counter] = linear_magnetization_solver(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,Ms,particle_volume,l_e)
            linear_normalized_magnetization[i,separation_counter], traceA[i,separation_counter], detA[i,separation_counter] = linear_normalized_magnetization_solver(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,Ms,particle_volume,l_e)
        # linear_magnetization_w_chi_scaled[i] = linear_magnetization_solver_chi_scaled(device_particle_posns,cp.asarray(Hext,dtype=cp.float32),chi,particle_volume,l_e)

    linear_magnetization = linear_magnetic_moments/particle_volume
    linear_magnetization /= Ms

    iter_number = np.arange(max_iters)
    field_index = 0
    
    fig, axs = plt.subplots(1,3)
    for count in range(particles.shape[0]):
        axs[0].plot(separations,linear_magnetization[field_index,:,count,0],'.')
        axs[1].plot(separations,linear_magnetization[field_index,:,count,1],'.')
        axs[2].plot(separations,linear_magnetization[field_index,:,count,2],'.')
    axs[0].set_xlabel('separation (m)')
    axs[1].set_xlabel('separation (m)')
    axs[2].set_xlabel('separation (m)')
    axs[0].set_title(r'm_x')
    axs[1].set_title(r'm_y')
    axs[2].set_title(r'm_z')
    axs[0].set_ylabel('normalized particle magnetization from SI linsolve')
    plt.show(block=False)

    fig, axs = plt.subplots(1,3)
    for count in range(particles.shape[0]):
        axs[0].plot(separations,linear_magnetization[field_index,:,count,0]/linear_magnetization[field_index,:,count,1],'.')
        axs[1].plot(separations,linear_magnetization[field_index,:,count,0]/linear_magnetization[field_index,:,count,2],'.')
        axs[2].plot(separations,linear_magnetization[field_index,:,count,1]/linear_magnetization[field_index,:,count,2],'.')
    axs[0].set_xlabel('separation (m)')
    axs[1].set_xlabel('separation (m)')
    axs[2].set_xlabel('separation (m)')
    axs[0].set_title(r'ratio m_x/m_y')
    axs[1].set_title(r'ratio m_x/m_z')
    axs[2].set_title(r'ratio m_y_/m_z')
    axs[0].set_ylabel('normalized particle magnetization ratio from SI linsolve')
    plt.show(block=False)

    fig, axs = plt.subplots(1,3)
    for count in range(particles.shape[0]):
        axs[0].plot(separations,linear_normalized_magnetization[field_index,:,count,0],'.')
        axs[1].plot(separations,linear_normalized_magnetization[field_index,:,count,1],'.')
        axs[2].plot(separations,linear_normalized_magnetization[field_index,:,count,2],'.')
    axs[0].set_xlabel('separation (m)')
    axs[1].set_xlabel('separation (m)')
    axs[2].set_xlabel('separation (m)')
    axs[0].set_title(r'm_x')
    axs[1].set_title(r'm_y')
    axs[2].set_title(r'm_z')
    axs[0].set_ylabel('normalized particle magnetization from normalized linsolve')
    plt.show(block=False)

    fig, axs = plt.subplots(1,3)
    for count in range(particles.shape[0]):
        axs[0].plot(separations,linear_normalized_magnetization[field_index,:,count,0],'.')
        axs[1].plot(separations,detA[field_index,:],'.')
        axs[2].plot(separations,traceA[field_index,:],'.')
    axs[0].set_xlabel('separation (m)')
    axs[1].set_xlabel('separation (m)')
    axs[2].set_xlabel('separation (m)')
    axs[0].set_title(r'm_x')
    axs[1].set_title(r'det $A^-1$A')
    axs[2].set_title(r'trace $A^-1$A')
    axs[0].set_ylabel('normalized particle magnetization from normalized linsolve')
    plt.show(block=False)
    print('End of linear magnetization finding testing')

def iterative_magnetization_w_drag_testing():
    """Plot the magnetization vector components vs iteration number to see how iteration number effects values for different methods, including the basic iterative method, and variants of quasi-drag based methods. Do values converge? How much does microstructure/particle placement impact results?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.01/mu0
    n_field_steps = 1
    H_step = H_mag/(n_field_steps)
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
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    # l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([5,5,5])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40
    max_iters_w_drag = 300

    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    magnetic_moments_w_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)
    magnetic_moments_new_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)

    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        magnetic_moments_w_drag[i] = get_magnetization_iterative_iteration_testing_w_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)
        magnetic_moments_new_drag[i] = get_magnetization_iterative_new_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)

    magnetization = magnetic_moments/particle_volume
    magnetization_w_drag = magnetic_moments_w_drag/particle_volume
    magnetization_new_drag = magnetic_moments_new_drag/particle_volume

    magnetization /= Ms
    magnetization_w_drag /= Ms
    magnetization_new_drag /= Ms

    
    iter_number = np.arange(max_iters)
    field_index = 0

    fig, axs = plt.subplots(3,4)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,0])
        axs[1,1].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,1])
        axs[1,2].plot(np.arange(max_iters_w_drag),magnetization_w_drag[field_index,:,count,2])
        axs[2,0].plot(np.arange(max_iters_w_drag),magnetization_new_drag[field_index,:,count,0])
        axs[2,1].plot(np.arange(max_iters_w_drag),magnetization_new_drag[field_index,:,count,1])
        axs[2,2].plot(np.arange(max_iters_w_drag),magnetization_new_drag[field_index,:,count,2])
        axs[2,3].plot(np.arange(max_iters_w_drag),np.linalg.norm(magnetization_new_drag[field_index,:,count],axis=1))

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('normalized particle magnetization (w drag)')
    axs[2,0].set_ylabel('normalized particle magnetization (new drag)')

    plt.show(block=False)

    iter_number = np.arange(max_iters_w_drag)
    system_magnetization = np.sum(magnetic_moments_new_drag[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    fig, ax = plt.subplots()
    ax.plot(iter_number,system_magnetization[:,:])
    ax.set_xlabel('iteration number')
    ax.set_ylabel('normalized system magnetization')
        
    plt.show(block=False)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.set_title('Normalized Magnetic Moment vectors (iterative w new drag)')
    ax.set_xlabel('X (l_e)')
    ax.set_ylabel('Y (l_e)')
    ax.set_zlabel('Z (l_e)')
    ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],magnetization_new_drag[field_index,-1,:,0],magnetization_new_drag[field_index,-1,:,1],magnetization_new_drag[field_index,-1,:,2],pivot='middle',length=15.0)

    plt.show()
    print('review plot')

def nonlinear_magnetization_fsolve():
    """Plot the magnetization vector components from using scipy.optimize.fsolve() or scipy.optimize.broyden1(). Test how it behaves with and without symmetry, with more particles. Wishlist for PBC on magnetization, so that you have "image" volumes around the real simulated volume that are copies of the real simulated volume (and include a demagnetization field due to an infinite surrounding medium with a continuous magnetization equal to the magnetic moment vector sum of the real volume divided by the system volume, with a shape factor for a sphere (or a cube, but it is infinite in extent in both cases)). See how symmetry, particle number, and separation/presence or absence of noise causes the method to work or fail"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 1.0/mu0
    n_field_steps = 1000
    H_step = H_mag/(n_field_steps)
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
    Hext_series = Hext_series[::-1,:]
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    # l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([3,1,1])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7.8e-6,7.8e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40
    max_iters_w_drag = 300

    # frohlich_kennelly_root_finding(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e)
    Hext = np.float32(Hext_series[0])
    # initial_guess = particle_volume*chi*Hext
    # initial_guess = np.tile(initial_guess,num_particles)
    # initial_guess = np.reshape(initial_guess,(num_particles*3,))
    initial_guess_normalized_magnetization = np.zeros((num_particles*3,))
    initial_guess_normalized_magnetization = chi*Hext_series[0]/Ms
    initial_guess_normalized_magnetization = np.tile(initial_guess_normalized_magnetization,num_particles)
    fixed_point_initial_soln = initial_guess_normalized_magnetization.copy()
    fsolve_initial_soln = initial_guess_normalized_magnetization.copy()
    broyden_initial_soln = initial_guess_normalized_magnetization.copy()
    seeded_gpu_fixedpoint_initial_soln = initial_guess_normalized_magnetization.copy()*particle_volume
    

    fixed_point_magnetization = np.zeros((Hext_series.shape[0],3*num_particles,))
    fsolve_magnetization = np.zeros((Hext_series.shape[0],3*num_particles,))
    broyden_magnetization = np.zeros((Hext_series.shape[0],3*num_particles,))


    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    seeded_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    # magnetic_moments_w_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)
    # magnetic_moments_new_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)

    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        seeded_magnetic_moments[i] = get_magnetization_iterative_iteration_testing_w_seeded_values(np.float32(seeded_gpu_fixedpoint_initial_soln),cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        seeded_gpu_fixedpoint_initial_soln = np.reshape(seeded_magnetic_moments[i,-1],(3*num_particles,))

        # magnetic_moments_w_drag[i] = get_magnetization_iterative_iteration_testing_w_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)
        # magnetic_moments_new_drag[i] = get_magnetization_iterative_new_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)
        try:
            fixed_point_magnetization[i] = scipy.optimize.fixed_point(frohlich_kennelly_fixed_point,x0=np.float32(fixed_point_initial_soln),args=(particle_posns.astype(np.float32),chi,particle_radius,Ms,np.float32(Hext),l_e))
            fixed_point_initial_soln = fixed_point_magnetization[i].copy()
        except:
            fixed_point_magnetization[i] = np.zeros((3*num_particles,))

        fsolve_magnetization[i] = scipy.optimize.fsolve(frohlich_kennelly_root_finding,x0=np.float64(fsolve_initial_soln),args=(particle_posns,np.float64(chi),np.float64(particle_radius),np.float64(Ms),np.float64(Hext),np.float64(l_e)))
        fsolve_initial_soln = fsolve_magnetization[i].copy()

        broyden_magnetization_struct = scipy.optimize.root(frohlich_kennelly_root_finding,x0=np.float64(broyden_initial_soln),args=(particle_posns,np.float64(chi),np.float64(particle_radius),np.float64(Ms),np.float64(Hext),np.float64(l_e)),method='broyden1')
        broyden_magnetization[i] = np.reshape(broyden_magnetization_struct.x,(num_particles*3))
        broyden_initial_soln = broyden_magnetization[i].copy()


    magnetization = magnetic_moments/particle_volume
    seeded_magnetization = seeded_magnetic_moments/particle_volume
    # magnetization_w_drag = magnetic_moments_w_drag/particle_volume
    # magnetization_new_drag = magnetic_moments_new_drag/particle_volume

    magnetization /= Ms
    seeded_magnetization /= Ms
    # magnetization_w_drag /= Ms
    # magnetization_new_drag /= Ms

    # fsolve_magnetization = np.reshape(fsolve_mag_moments,(num_particles,3))/particle_volume
    # fixed_point_magnetization = np.reshape(fixed_point_mag_moments,(num_particles,3))/particle_volume

    # fsolve_magnetization /= Ms
    # fixed_point_magnetization /= Ms

    fsolve_magnetization = np.reshape(fsolve_magnetization,(Hext_series.shape[0],num_particles,3))
    fixed_point_magnetization = np.reshape(fixed_point_magnetization,(Hext_series.shape[0],num_particles,3))
    broyden_magnetization = np.reshape(broyden_magnetization,(Hext_series.shape[0],num_particles,3))
    
    iter_number = np.arange(max_iters)
    field_index = Hext_series.shape[0]-1

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[field_index,count,0])
        axs[1,1].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[field_index,count,1])
        axs[1,2].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[field_index,count,2])
        axs[2,0].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[field_index,count,0])
        axs[2,1].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[field_index,count,1])
        axs[2,2].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[field_index,count,2])

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('(fsolve)')
    axs[2,0].set_ylabel('(fixed_point)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,seeded_magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,seeded_magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,seeded_magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[field_index,count,0])
        axs[1,1].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[field_index,count,1])
        axs[1,2].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[field_index,count,2])

    axs[1,0].set_xlabel('iteration number')
    axs[1,1].set_xlabel('iteration number')
    axs[1,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('(seeded)')
    axs[1,0].set_ylabel('(broyden)')

    plt.show(block=False)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude[::-1],fsolve_magnetization[:,count,0])
        axs[1,1].plot(Bext_series_magnitude[::-1],fsolve_magnetization[:,count,1])
        axs[1,2].plot(Bext_series_magnitude[::-1],fsolve_magnetization[:,count,2])
        axs[2,0].plot(Bext_series_magnitude[::-1],fixed_point_magnetization[:,count,0])
        axs[2,1].plot(Bext_series_magnitude[::-1],fixed_point_magnetization[:,count,1])
        axs[2,2].plot(Bext_series_magnitude[::-1],fixed_point_magnetization[:,count,2])

    axs[2,0].set_xlabel('Applied Field (T)')
    axs[2,1].set_xlabel('Applied Field (T)')
    axs[2,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('(fsolve)')
    axs[2,0].set_ylabel('(fixed_point)')

    plt.show(block=False)

    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude[::-1],broyden_magnetization[:,count,0])
        axs[1,1].plot(Bext_series_magnitude[::-1],broyden_magnetization[:,count,1])
        axs[1,2].plot(Bext_series_magnitude[::-1],broyden_magnetization[:,count,2])

    axs[1,0].set_xlabel('Applied Field (T)')
    axs[1,1].set_xlabel('Applied Field (T)')
    axs[1,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('(seeded)')
    axs[1,0].set_ylabel('(broyden)')

    plt.show(block=False)

    # iter_number = np.arange(max_iters_w_drag)
    # system_magnetization = np.sum(magnetic_moments_new_drag[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    # fig, ax = plt.subplots()
    # ax.plot(iter_number,system_magnetization[:,:])
    # ax.set_xlabel('iteration number')
    # ax.set_ylabel('normalized system magnetization')
        
    # plt.show(block=False)

    # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # default_width,default_height = fig.get_size_inches()
    # fig.set_size_inches(3*default_width,3*default_height)
    # fig.set_dpi(200)
    # ax.set_title('Normalized Magnetic Moment vectors (iterative w new drag)')
    # ax.set_xlabel('X (l_e)')
    # ax.set_ylabel('Y (l_e)')
    # ax.set_zlabel('Z (l_e)')
    # ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],magnetization_new_drag[field_index,-1,:,0],magnetization_new_drag[field_index,-1,:,1],magnetization_new_drag[field_index,-1,:,2],pivot='middle',length=15.0)

    # plt.show()
    print('review plot')

def pbc_nonlinear_magnetization_fsolve():
    """Plot the magnetization vector components from using scipy.optimize.fsolve() or scipy.optimize.broyden1(). Test how it behaves with and without symmetry, with more particles. Wishlist for PBC on magnetization, so that you have "image" volumes around the real simulated volume that are copies of the real simulated volume (and include a demagnetization field due to an infinite surrounding medium with a continuous magnetization equal to the magnetic moment vector sum of the real volume divided by the system volume, with a shape factor for a sphere (or a cube, but it is infinite in extent in both cases)). See how symmetry, particle number, and separation/presence or absence of noise causes the method to work or fail"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = 0.001/mu0
    n_field_steps = 1
    H_step = H_mag/(n_field_steps)
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
    chi = np.float32(70)#np.float32(3)#
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    # l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    particles_per_axis = np.array([2,2,2])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    volume_fraction = 0.20
    fictional_total_volume = num_particles*particle_volume/volume_fraction
    fictional_side_length = np.power(fictional_total_volume,(1/3))
    particle_density = num_particles/fictional_total_volume
    particle_separation_metric = 1/np.power(particle_density,(1/3))
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [particle_separation_metric,particle_separation_metric,particle_separation_metric]#[3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7e-6,7e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] + rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    max_iters = 40
    max_iters_w_drag = 300

    # frohlich_kennelly_root_finding(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e)
    Hext = np.float32(Hext_series[0])
    # initial_guess = particle_volume*chi*Hext
    # initial_guess = np.tile(initial_guess,num_particles)
    # initial_guess = np.reshape(initial_guess,(num_particles*3,))
    num_images = np.array([2,2,2],dtype=np.int32)
    translation_vector = np.array(separation,dtype=np.float64)*particles_per_axis/l_e

    # translation_vector[0] *= 1.0

    initial_guess_normalized_magnetization = np.zeros((num_particles*3,))
    initial_guess_normalized_magnetization = chi*Hext/Ms
    initial_guess_normalized_magnetization = np.tile(initial_guess_normalized_magnetization,num_particles)
    fixed_point_magnetization = np.zeros((3*num_particles,))
    # fixed_point_magnetization = scipy.optimize.fixed_point(pbc_frohlich_kennelly_fixed_point,x0=np.float32(initial_guess_normalized_magnetization),args=(particle_posns.astype(np.float32),chi,particle_radius,Ms,Hext,l_e,num_images,np.float32(translation_vector)),maxiter=1000,)#method='iteration'
    fsolve_magnetization = scipy.optimize.fsolve(pbc_frohlich_kennelly_root_finding,x0=np.float64(initial_guess_normalized_magnetization),args=(particle_posns,np.float64(chi),np.float64(particle_radius),np.float64(Ms),np.float64(Hext),np.float64(l_e),num_images,translation_vector))
    broyden_magnetization = scipy.optimize.root(pbc_frohlich_kennelly_root_finding,x0=np.float64(initial_guess_normalized_magnetization),args=(particle_posns,np.float64(chi),np.float64(particle_radius),np.float64(Ms),np.float64(Hext),np.float64(l_e),num_images,translation_vector),method='broyden1')

    broyden_magnetization = np.reshape(broyden_magnetization.x,(num_particles,3))

    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    pbc_magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    # magnetic_moments_w_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)
    # magnetic_moments_new_drag = np.zeros((Hext_series.shape[0],max_iters_w_drag,num_particles,3),dtype=np.float32)

    for i, Hext in enumerate(Hext_series):
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        pbc_magnetic_moments[i] = get_magnetization_iterative_pbc_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,num_images,translation_vector,max_iters)
        # magnetic_moments_w_drag[i] = get_magnetization_iterative_iteration_testing_w_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)
        # magnetic_moments_new_drag[i] = get_magnetization_iterative_new_drag(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters_w_drag)

    magnetization = magnetic_moments/particle_volume
    pbc_magnetization = pbc_magnetic_moments/particle_volume
    # magnetization_w_drag = magnetic_moments_w_drag/particle_volume
    # magnetization_new_drag = magnetic_moments_new_drag/particle_volume

    magnetization /= Ms
    pbc_magnetization /= Ms
    # magnetization_w_drag /= Ms
    # magnetization_new_drag /= Ms

    # fsolve_magnetization = np.reshape(fsolve_mag_moments,(num_particles,3))/particle_volume
    # fixed_point_magnetization = np.reshape(fixed_point_mag_moments,(num_particles,3))/particle_volume

    # fsolve_magnetization /= Ms
    # fixed_point_magnetization /= Ms

    fsolve_magnetization = np.reshape(fsolve_magnetization,(num_particles,3))
    fixed_point_magnetization = np.reshape(fixed_point_magnetization,(num_particles,3))
    
    iter_number = np.arange(max_iters)
    field_index = 0

    fig, axs = plt.subplots(4,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[count,0])
        axs[1,1].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[count,1])
        axs[1,2].plot(iter_number,np.ones(max_iters,)*fsolve_magnetization[count,2])
        axs[3,0].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[count,0])
        axs[3,1].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[count,1])
        axs[3,2].plot(iter_number,np.ones(max_iters,)*broyden_magnetization[count,2])
        axs[2,0].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,0])
        axs[2,1].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,1])
        axs[2,2].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,2])

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('(fsolve)')
    axs[3,0].set_ylabel('(broyden)')
    axs[2,0].set_ylabel('(fixed_point)')

    plt.show(block=False)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,pbc_magnetization[field_index,:,count,0])
        axs[1,1].plot(iter_number,pbc_magnetization[field_index,:,count,1])
        axs[1,2].plot(iter_number,pbc_magnetization[field_index,:,count,2])
        axs[2,0].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,0])
        axs[2,1].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,1])
        axs[2,2].plot(iter_number,np.ones(max_iters,)*fixed_point_magnetization[count,2])

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('gpu_pbc_fixedpoint')
    axs[2,0].set_ylabel('(fixed_point)')

    plt.show(block=False)

    # iter_number = np.arange(max_iters_w_drag)
    # system_magnetization = np.sum(magnetic_moments_new_drag[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    # fig, ax = plt.subplots()
    # ax.plot(iter_number,system_magnetization[:,:])
    # ax.set_xlabel('iteration number')
    # ax.set_ylabel('normalized system magnetization')
        
    # plt.show(block=False)

    # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # default_width,default_height = fig.get_size_inches()
    # fig.set_size_inches(3*default_width,3*default_height)
    # fig.set_dpi(200)
    # ax.set_title('Normalized Magnetic Moment vectors (Broyden, Peridioc)')
    # ax.set_xlabel('X (l_e)')
    # ax.set_ylabel('Y (l_e)')
    # ax.set_zlabel('Z (l_e)')
    # ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],broyden_magnetization[:,0],broyden_magnetization[:,1],broyden_magnetization[:,2],pivot='middle',length=30.0)

    # plt.show()
    print('review plot')

def frohlich_kennelly_fixed_point(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e):
    mag_moments = magnetism.get_normalized_magnetic_moment_frohlich_kennelly_normalized_posns_32bit(magnetic_moments,Hext,particle_posns,particle_radius,chi,Ms,l_e)
    return mag_moments

def frohlich_kennelly_root_finding(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e):
    result = magnetism.root_finding_normalized_frohlich_kennelly_normalized_posns_64bit(magnetic_moments,Hext,particle_posns,particle_radius,chi,Ms,l_e)
    return result

def pbc_frohlich_kennelly_fixed_point(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e,num_images,translation_vector):
    """with periodic boundary conditions (image volumes)"""
    mag_moments = magnetism.get_normalized_magnetic_moment_frohlich_kennelly_normalized_posns_pbc_32bit(magnetic_moments,Hext,particle_posns,particle_radius,chi,Ms,l_e,num_images,translation_vector)
    return mag_moments

def pbc_frohlich_kennelly_root_finding(magnetic_moments,particle_posns,chi,particle_radius,Ms,Hext,l_e,num_images,translation_vector):
    """with periodic boundary conditions (image volumes)"""
    result = magnetism.root_finding_normalized_frohlich_kennelly_normalized_posns_pbc_64bit(magnetic_moments,Hext,particle_posns,particle_radius,chi,Ms,l_e,num_images,translation_vector)
    return result

def nonlinear_magnetization_energy_minimization_w_constraint():
    """Plot the magnetization vector components from using scipy.optimize.minimize(), different methods that allow nonlinear equality constraints . Test how it behaves with and without symmetry, with more particles. What cases fail? Can a PBC with repeated image volumes be implemented?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = .010/mu0
    n_field_steps = 10
    H_step = H_mag/(n_field_steps)
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
    Hext_series = Hext_series[::-1,:]
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    # l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([3,3,3])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7.8e-6,7.8e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    particle_posns = np.float32(particle_posns)
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40

    Hext = np.float32(Hext_series[0])

    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    energy_optimization_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_alt_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_noself_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_magnetization_slsqp = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    global_energy_optimization_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    energy_optimization_unconstrained_penalty = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_unconstrained_penalty_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_unconstrained_penalty_enhanced = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    energy_optimization_unconstrained_penalty_enhanced_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    fixed_point_energy = np.zeros((Hext_series.shape[0],),dtype=np.float32)
    optimization_energy = np.zeros((Hext_series.shape[0],),dtype=np.float32)


    initial_guess = np.float32(0.9*Hext_series[0,:]/np.linalg.norm(Hext_series[0,:]))#+np.array([0.1,0.1,0.1],dtype=np.float32)
    initial_guess = np.tile(initial_guess,num_particles)
    initial_guess_unconstrained = initial_guess.copy()
    initial_guess_no_self_energy = initial_guess.copy()
    penalty_constant = 1
    for i, Hext in enumerate(Hext_series):
        Hext = np.float32(Hext)
        magnetic_moments[i] = get_magnetization_iterative_iteration_testing(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        # if i == 0:
        #     initial_guess = magnetic_moments[i,-1].reshape((num_particles*3,))/(Ms*particle_volume)
        #     initial_guess_unconstrained = initial_guess.copy()
        res = solve_magnetization_optimization(initial_guess,np.float32(Hext),np.float32(particle_posns),chi,particle_radius,particle_volume,Ms,l_e,num_particles,method='SLSQP')
        # print(res)
        energy_optimization_magnetization[i] = res.x
        initial_guess = energy_optimization_magnetization[i]
        # energy_optimization_alt_constraint_violation[i] = get_frohlich_kenelley_constraint_violation(energy_optimization_magnetization[i],np.float32(Hext),np.float32(particle_posns),chi,particle_radius,Ms,l_e,num_particles)
        # res = solve_magnetization_optimization_no_self_term(initial_guess_no_self_energy,np.float32(Hext),np.float32(particle_posns),chi,particle_radius,particle_volume,Ms,l_e,num_particles,method='SLSQP')
        # energy_optimization_noself_magnetization[i] = res.x
        # initial_guess_no_self_energy = energy_optimization_noself_magnetization[i]
        # res = solve_magnetization_optimization(initial_guess,np.float32(Hext),np.float32(particle_posns),chi,particle_radius,particle_volume,Ms,l_e,num_particles,method='SLSQP')
        # energy_optimization_magnetization_slsqp[i] = res.x
        # res = solve_magnetization_global_optimization(initial_guess,np.float32(Hext),np.float32(particle_posns),chi,particle_radius,particle_volume,Ms,l_e,num_particles)
        # global_energy_optimization_magnetization[i] = res.x

        res = penalty_method_magnetization_finder(initial_guess_unconstrained,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles,penalty_constant,method='BFGS')
        energy_optimization_unconstrained_penalty[i] = res.x
        energy_optimization_unconstrained_penalty_constraint_violation[i] = get_frohlich_kenelley_constraint_violation(energy_optimization_unconstrained_penalty[i],np.float32(Hext),np.float32(particle_posns),chi,particle_radius,Ms,l_e,num_particles)
        #should i use the result with lower penalty constant to feed into the next attempt with stronger penalty? probably, but let's see how they behave with different constants for now
        # initial_guess_unconstrained = energy_optimization_unconstrained_penalty[i]
        res = penalty_method_magnetization_finder(initial_guess_unconstrained,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles,penalty_constant*100,method='BFGS')
        energy_optimization_unconstrained_penalty_enhanced[i] = res.x
        energy_optimization_unconstrained_penalty_enhanced_constraint_violation[i] = get_frohlich_kenelley_constraint_violation(energy_optimization_unconstrained_penalty_enhanced[i],np.float32(Hext),np.float32(particle_posns),chi,particle_radius,Ms,l_e,num_particles)

        initial_guess_unconstrained = energy_optimization_unconstrained_penalty[i]

        fixed_point_energy[i] = get_total_magnetic_energy(np.reshape(magnetic_moments[i,-1]/(Ms*particle_volume),(3*num_particles,)),Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e)
        optimization_energy[i] = get_total_magnetic_energy(energy_optimization_magnetization[i],Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e)

    energy_optimization_magnetization = np.reshape(energy_optimization_magnetization,(Hext_series.shape[0],num_particles,3))
    energy_optimization_constraint_violation = np.reshape(energy_optimization_constraint_violation,(Hext_series.shape[0],num_particles,3))
    energy_optimization_alt_constraint_violation = np.reshape(energy_optimization_alt_constraint_violation,(Hext_series.shape[0],num_particles,3))
    energy_optimization_noself_magnetization = np.reshape(energy_optimization_noself_magnetization,(Hext_series.shape[0],num_particles,3))
    energy_optimization_magnetization_slsqp = np.reshape(energy_optimization_magnetization_slsqp,(Hext_series.shape[0],num_particles,3))

    energy_optimization_unconstrained_penalty = np.reshape(energy_optimization_unconstrained_penalty,(Hext_series.shape[0],num_particles,3))
    energy_optimization_unconstrained_penalty_constraint_violation = np.reshape(energy_optimization_unconstrained_penalty_constraint_violation,(Hext_series.shape[0],num_particles,3))
    energy_optimization_unconstrained_penalty_enhanced = np.reshape(energy_optimization_unconstrained_penalty_enhanced,(Hext_series.shape[0],num_particles,3))
    energy_optimization_unconstrained_penalty_enhanced_constraint_violation = np.reshape(energy_optimization_unconstrained_penalty_enhanced_constraint_violation,(Hext_series.shape[0],num_particles,3))

    # global_energy_optimization_magnetization = np.reshape(global_energy_optimization_magnetization,(Hext_series.shape[0],num_particles,3))

    magnetization = magnetic_moments/particle_volume
    # energy_optimization_magnetization = energy_optimization_magnetic_moments/particle_volume

    magnetization /= Ms
    # energy_optimization_magnetization /= Ms
    
    iter_number = np.arange(max_iters)
    field_index = Hext_series.shape[0]-1
    if field_index < 0:
        field_index = 0

    # fig, axs = plt.subplots(1,3)
    # for count in range(particles.shape[0]):
    #     axs[0].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,0])
    #     axs[1].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,1])
    #     axs[2].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,2])
    # axs[0].set_ylabel('normalized magnetization global optimization')


    fig, axs = plt.subplots(1,3)
    for count in range(particles.shape[0]):
        # axs[0].plot(iter_number,magnetization[field_index,:,count,0])
        # axs[1].plot(iter_number,magnetization[field_index,:,count,1])
        # axs[2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[0].plot(Bext_series_magnitude[::-1],energy_optimization_alt_constraint_violation[:,count,0])
        axs[1].plot(Bext_series_magnitude[::-1],energy_optimization_alt_constraint_violation[:,count,1])
        axs[2].plot(Bext_series_magnitude[::-1],energy_optimization_alt_constraint_violation[:,count,2])
    axs[0].set_ylabel('constraint violation')

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization[:,count,0])
        axs[1,1].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization[:,count,1])
        axs[1,2].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization[:,count,2])
        axs[2,0].plot(Bext_series_magnitude[::-1],energy_optimization_constraint_violation[:,count,0])
        axs[2,1].plot(Bext_series_magnitude[::-1],energy_optimization_constraint_violation[:,count,1])
        axs[2,2].plot(Bext_series_magnitude[::-1],energy_optimization_constraint_violation[:,count,2])
        # axs[2,0].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization_slsqp[:,count,0])
        # axs[2,1].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization_slsqp[:,count,1])
        # axs[2,2].plot(Bext_series_magnitude[::-1],energy_optimization_magnetization_slsqp[:,count,2])
        # axs[2,0].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,0])
        # axs[2,1].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,1])
        # axs[2,2].plot(Bext_series_magnitude[::-1],energy_optimization_noself_magnetization[:,count,2])        

    axs[2,0].set_xlabel('Applied Field (T)')
    axs[2,1].set_xlabel('Applied Field (T)')
    axs[2,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    # axs[0,3].set_title('magnitude normalized')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('constrained energy optimization')
    # axs[2,0].set_ylabel('constrained no self energy')
    axs[2,0].set_ylabel('constraint violation')

    plt.show(block=False)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty[:,count,0])
        axs[1,1].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty[:,count,1])
        axs[1,2].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty[:,count,2])
        axs[2,0].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_constraint_violation[:,count,0])
        axs[2,1].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_constraint_violation[:,count,1])
        axs[2,2].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_constraint_violation[:,count,2])

    axs[2,0].set_xlabel('Applied Field (T)')
    axs[2,1].set_xlabel('Applied Field (T)')
    axs[2,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('unconstrained energy optimization w penalty')
    axs[2,0].set_ylabel('constraint violation')

    plt.show(block=False)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced[:,count,0])
        axs[1,1].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced[:,count,1])
        axs[1,2].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced[:,count,2])
        axs[2,0].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced_constraint_violation[:,count,0])
        axs[2,1].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced_constraint_violation[:,count,1])
        axs[2,2].plot(Bext_series_magnitude[::-1],energy_optimization_unconstrained_penalty_enhanced_constraint_violation[:,count,2])

    axs[2,0].set_xlabel('Applied Field (T)')
    axs[2,1].set_xlabel('Applied Field (T)')
    axs[2,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization')
    axs[1,0].set_ylabel('unconstrained energy optimization w enhanced penalty')
    axs[2,0].set_ylabel('constraint violation')

    plt.show(block=False)


    fig, ax = plt.subplots()
    
    ax.plot(Bext_series_magnitude[::-1],fixed_point_energy,label='fixed point')
    ax.plot(Bext_series_magnitude[::-1],optimization_energy,label='optimization')
    ax.set_ylabel('Energy (normalized)')
    ax.set_xlabel('Applied Field (T)')
    ax.legend()
    plt.show(block=False)

    # plt.show(block=False)

    # iter_number = np.arange(max_iters_w_drag)
    # system_magnetization = np.sum(magnetic_moments_new_drag[field_index,:],axis=1)/(particle_volume*num_particles)/Ms
    # fig, ax = plt.subplots()
    # ax.plot(iter_number,system_magnetization[:,:])
    # ax.set_xlabel('iteration number')
    # ax.set_ylabel('normalized system magnetization')
        
    # plt.show(block=False)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.set_title('Normalized Magnetic Moment vectors (constrained optimization))')
    ax.set_xlabel('X (l_e)')
    ax.set_ylabel('Y (l_e)')
    ax.set_zlabel('Z (l_e)')
    ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],energy_optimization_unconstrained_penalty_enhanced[field_index,:,0],energy_optimization_unconstrained_penalty_enhanced[field_index,:,1],energy_optimization_unconstrained_penalty_enhanced[field_index,:,2],pivot='middle',length=15.0)
    # print('constrained result:',energy_optimization_magnetization[field_index,:])
    # print('SLSQP',energy_optimization_magnetization_slsqp[field_index,:])
    # print('unconstrained penalty result:',energy_optimization_unconstrained_penalty[field_index,:])

    # dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.reshape(energy_optimization_magnetization[field_index],(3*num_particles,)),np.float32(Hext_series[field_index]),np.float32(particle_posns),particle_radius,chi,Ms,l_e)
    # print('result:',dipole_fields)
    # get_frohlich_kenelley_constraint_violation(np.reshape(energy_optimization_magnetization[field_index],(3*num_particles,)),np.float32(Hext_series[field_index]),np.float32(particle_posns),chi,particle_radius,Ms,l_e,num_particles)

    plt.show(block=False)
    print('review plot')

def get_zeeman_energy(magnetic_moments,Hext):
     mu0 = 4*np.pi*1e-7
     individual_energy = -1*np.sum(magnetic_moments*Hext)#-1*np.einsum('ij,j->i',magnetic_moments,Hext)
     total_energy = mu0*np.sum(individual_energy)
     return total_energy

def get_dipolar_energy(magnetic_moments,dipole_fields):
    mu0 = 4*np.pi*1e-7
    individual_energy = -1*np.dot(np.ravel(magnetic_moments),np.ravel(dipole_fields))#-1*np.einsum('ij,hj->i',magnetic_moments,dipole_fields)
    total_energy = mu0*np.sum(individual_energy)/2
    return total_energy

def get_self_energy(magnetic_moments,chi,particle_volume):
    #under the assumption of linear, nonsaturable magnetizability... but could be worked out as an integral equation where chi depends on M, and an integration is done over dM
    #W = (1/(2*chi)) * M^2
    #dW = (1/(2*chi)) * d (M^2)
    #\integral_0^M
    mu0 = 4*np.pi*1e-7
    individual_energy = mu0/(2*chi*particle_volume) * np.einsum('ij,hj->i',magnetic_moments,magnetic_moments)
    total_energy = np.sum(individual_energy)
    return total_energy

def get_total_magnetic_energy(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e):
    mu0 = 4*np.pi*1e-7
    scaling_factor = mu0*np.power(Ms,2)*particle_volume
    dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
    magnetic_moments = np.reshape(normalized_magnetic_moments,(particle_posns.shape[0],3))*Ms*particle_volume
    zeeman_energy = get_zeeman_energy(magnetic_moments,Hext)
    dipole_energy = get_dipolar_energy(magnetic_moments,dipole_fields)
    self_energy = get_self_energy(magnetic_moments,chi,particle_volume)
    total_energy = (zeeman_energy + dipole_energy + self_energy)/scaling_factor
    return total_energy
    
def solve_magnetization_optimization(initial_guess,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles,method='trust-constr'):
    """Function wrapper to run the optimization with constraint, since the constraint can't be passed arguments, they need to be defined in scope (in this function) to be used"""
    def frohlich_kenelley_constraint(normalized_magnetic_moments):
        # tiled_hext = np.tile(Hext,num_particles)/Ms
        dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
        h_tot = (dipole_fields + Hext)/Ms
        htot_norm = np.linalg.norm(h_tot,axis=1)
        htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
        h_tot = np.reshape(h_tot,(3*num_particles,))
        return normalized_magnetic_moments*(1 + chi*htot_norm_repeat) - chi*h_tot

    fk_constraint = scipy.optimize.NonlinearConstraint(frohlich_kenelley_constraint,0,0)
    res = scipy.optimize.minimize(get_total_magnetic_energy,x0=initial_guess,args=(Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e),method=method,constraints=fk_constraint)
    return res

def solve_magnetization_global_optimization(initial_guess,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles):
    """Function wrapper to run the optimization with constraint, since the constraint can't be passed arguments, they need to be defined in scope (in this function) to be used"""
    def frohlich_kenelley_constraint(normalized_magnetic_moments):
        # tiled_hext = np.tile(Hext,num_particles)/Ms
        dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
        h_tot = (dipole_fields + Hext)/Ms
        htot_norm = np.linalg.norm(h_tot,axis=1)
        htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
        h_tot = np.reshape(h_tot,(3*num_particles,))
        return normalized_magnetic_moments*(1 + chi*htot_norm_repeat) - chi*h_tot

    fk_constraint = scipy.optimize.NonlinearConstraint(frohlich_kenelley_constraint,0,0)
    bounds = scipy.optimize.Bounds(-1*np.ones((3*num_particles,)),np.ones((3*num_particles,)))
    res = scipy.optimize.differential_evolution(get_total_magnetic_energy,bounds,x0=initial_guess,args=(Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e),constraints=fk_constraint)
    return res

def get_frohlich_kenelley_constraint_violation(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,Ms,l_e,num_particles):
    # tiled_hext = np.tile(Hext,num_particles)/Ms
    dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
    h_tot = (dipole_fields + Hext)/Ms
    htot_norm = np.linalg.norm(h_tot,axis=1)
    htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    h_tot = np.reshape(h_tot,(3*num_particles,))
    return normalized_magnetic_moments*(1 + chi*htot_norm_repeat) - chi*h_tot

def penalty_method_magnetization_finder(initial_guess,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles,penalty_constant,method='BFGS'):
    def penalty_objective_func(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e):
        obj = get_total_magnetic_energy(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e)
        penalty = penalty_constant*np.sum(np.power(get_frohlich_kenelley_constraint_violation(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,Ms,l_e,num_particles),2))
        return obj + penalty
    res = scipy.optimize.minimize(penalty_objective_func,x0=initial_guess,args=(Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e),method=method)
    return res

def solve_magnetization_optimization_no_self_term(initial_guess,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e,num_particles,method='trust-constr'):
    """Function wrapper to run the optimization with constraint, since the constraint can't be passed arguments, they need to be defined in scope (in this function) to be used"""
    def frohlich_kenelley_constraint(normalized_magnetic_moments):
        # tiled_hext = np.tile(Hext,num_particles)
        dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
        h_tot = (dipole_fields + Hext)/Ms
        htot_norm = np.linalg.norm(h_tot,axis=1)
        htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
        h_tot = np.reshape(h_tot,(3*num_particles,))
        return normalized_magnetic_moments*(Ms + chi*htot_norm_repeat) - chi*h_tot

    fk_constraint = scipy.optimize.NonlinearConstraint(frohlich_kenelley_constraint,0,0)
    res = scipy.optimize.minimize(get_total_magnetic_energy_no_self_term,x0=initial_guess,args=(Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e),method=method,constraints=fk_constraint)
    return res

def get_total_magnetic_energy_no_self_term(normalized_magnetic_moments,Hext,particle_posns,chi,particle_radius,particle_volume,Ms,l_e):
    mu0 = 4*np.pi*1e-7
    scaling_factor = mu0*np.power(Ms,2)*particle_volume
    dipole_fields = magnetism.get_dipole_fields_normalized_posns_32bit(np.float32(normalized_magnetic_moments),Hext,particle_posns,particle_radius,chi,Ms,l_e)
    magnetic_moments = np.reshape(normalized_magnetic_moments,(particle_posns.shape[0],3))*Ms*particle_volume
    zeeman_energy = get_zeeman_energy(magnetic_moments,Hext)
    dipole_energy = get_dipolar_energy(magnetic_moments,dipole_fields)
    total_energy = (zeeman_energy + dipole_energy)//scaling_factor
    return total_energy

def get_normalized_dipole_field_kernel(rij,inv_rij_mag,num_particles,l_e,particle_volume):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    kernel = cp.zeros((num_particles*3,num_particles*3),dtype=cp.float32)

    nij_x = rij[0::3]*inv_rij_mag
    nij_y = rij[1::3]*inv_rij_mag
    nij_z = rij[2::3]*inv_rij_mag

    nij_x = cp.reshape(nij_x,(num_particles,num_particles),order='C')
    nij_y = cp.reshape(nij_y,(num_particles,num_particles),order='C')
    nij_z = cp.reshape(nij_z,(num_particles,num_particles),order='C')

    inv_rij_mag = cp.reshape(inv_rij_mag,(num_particles,num_particles),order='C')
    
    denom = particle_volume/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[2::3,0::3] = kernel[0::3,2::3]
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom
    kernel[2::3,1::3] = kernel[1::3,2::3]
    
    return kernel

def get_normalized_dipole_field_kernel_torch(rij,inv_rij_mag,num_particles,l_e,particle_volume):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    kernel = torch.zeros((num_particles*3,num_particles*3)).cuda()

    nij_x = torch.asarray(rij[0::3]*inv_rij_mag)
    nij_y = torch.asarray(rij[1::3]*inv_rij_mag)
    nij_z = torch.asarray(rij[2::3]*inv_rij_mag)

    nij_x = torch.reshape(nij_x,(num_particles,num_particles))
    nij_y = torch.reshape(nij_y,(num_particles,num_particles))
    nij_z = torch.reshape(nij_z,(num_particles,num_particles))

    inv_rij_mag = torch.reshape(torch.asarray(inv_rij_mag),(num_particles,num_particles))
    
    denom = particle_volume/(4*np.float32(np.pi))*torch.pow(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[2::3,0::3] = kernel[0::3,2::3]
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom
    kernel[2::3,1::3] = kernel[1::3,2::3]
    
    return kernel

def get_normalized_dipole_field_kernel_torch_pbc(rij,inv_rij_mag,num_particles,rij_image,inv_rij_mag_image,num_images,l_e,particle_volume):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    raise NotImplementedError
    kernel = torch.zeros((num_particles*3,num_particles*3)).cuda()

    nij_x = torch.asarray(rij[0::3]*inv_rij_mag)
    nij_y = torch.asarray(rij[1::3]*inv_rij_mag)
    nij_z = torch.asarray(rij[2::3]*inv_rij_mag)

    nij_x = torch.reshape(nij_x,(num_particles,num_particles))
    nij_y = torch.reshape(nij_y,(num_particles,num_particles))
    nij_z = torch.reshape(nij_z,(num_particles,num_particles))

    inv_rij_mag = torch.reshape(torch.asarray(inv_rij_mag),(num_particles,num_particles))
    
    denom = particle_volume/(4*np.float32(np.pi))*torch.pow(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom

    nij_x_image = torch.asarray(rij_image[0::3]*inv_rij_mag_image)
    nij_y_image = torch.asarray(rij_image[1::3]*inv_rij_mag_image)
    nij_z_image = torch.asarray(rij_image[2::3]*inv_rij_mag_image)

    nij_x_image = torch.reshape(nij_x_image,(num_particles*num_images,num_particles*num_images))
    nij_y_image = torch.reshape(nij_y_image,(num_particles*num_images,num_particles*num_images))
    nij_z_image = torch.reshape(nij_z_image,(num_particles*num_images,num_particles*num_images))

    kernel[2::3,1::3] = kernel[1::3,2::3]
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[2::3,0::3] = kernel[0::3,2::3]
    return kernel

def get_dipole_field_kernel(rij,inv_rij_mag,num_particles,l_e):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    kernel = cp.zeros((num_particles*3,num_particles*3),dtype=cp.float32)

    nij_x = rij[0::3]*inv_rij_mag
    nij_y = rij[1::3]*inv_rij_mag
    nij_z = rij[2::3]*inv_rij_mag

    nij_x = cp.reshape(nij_x,(num_particles,num_particles),order='C')
    nij_y = cp.reshape(nij_y,(num_particles,num_particles),order='C')
    nij_z = cp.reshape(nij_z,(num_particles,num_particles),order='C')

    inv_rij_mag = cp.reshape(inv_rij_mag,(num_particles,num_particles),order='C')
    
    denom = 1/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[2::3,0::3] = kernel[0::3,2::3]
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom
    kernel[2::3,1::3] = kernel[1::3,2::3]
    
    return kernel

def performance_testing_nonlinear_magnetization_energy_minimization_w_constraint(plotting_results=False,performance_testing=True,nrepeats=10):
    """Test run time for different optimization methods and implementations of objective function calculation. Plot the magnetization vector components from using scipy.optimize.minimize(), different methods that allow nonlinear equality constraints . Test how it behaves with and without symmetry, with more particles. What cases fail? Can a PBC with repeated image volumes be implemented?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = .001/mu0
    n_field_steps = 1
    H_step = H_mag/(n_field_steps)
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
    Hext_series = Hext_series[::-1,:]
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    # l_e = np.float32(1e-7)
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([3,3,3])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7.8e-6,7.8e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    # save_dir = '/mnt/c/Users/bagaw/Desktop/2024-08-06_kristen_files/'
    # np.savetxt(save_dir+f'{num_particles}_particles_posns.csv',particle_posns,delimiter=',')
    # np.savetxt(save_dir+'Hext_series.csv',Hext_series,delimiter=',')
    
    particle_posns /= l_e
    particle_posns = np.float32(particle_posns)
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40

    Hext = np.float32(Hext_series[0])

    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    bfgs_auglang_magnetization= np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    bfgs_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    bfgs_wgrad_auglang_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    bfgs_wgrad_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    lbfgs_auglang_magnetization= np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    lbfgs_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    lbfgs_wgrad_auglang_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    lbfgs_wgrad_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    cg_auglang_magnetization= np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    cg_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    newtoncg_auglang_magnetization = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    newtoncg_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    initial_guess = np.float64(0.9*Hext_series[0,:]/np.linalg.norm(Hext_series[0,:]))#+np.array([0.1,0.1,0.1],dtype=np.float32)
    initial_guess_cpu = np.tile(initial_guess,num_particles)
    initial_guess_cpu_grad = initial_guess_cpu.copy()
    initial_guess_lbfgs = initial_guess_cpu.copy()
    initial_guess_lbfgs_wgrad = initial_guess_cpu.copy()
    initial_guess_cg = initial_guess_cpu.copy()
    initial_guess_newtoncg = initial_guess_cpu.copy()
    penalty_constant = 1
    lagrange_multipliers = np.zeros((3*num_particles,))

    num_particles = np.int64(num_particles)
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_particles = num_particles
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(device_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    #for testing the runtime performance of different methods
    # nrepeats = 1
    if performance_testing:
        for i, Hext in enumerate(Hext_series):
            Hext = np.float32(Hext)
            fixedpoint_runtime, _ = get_function_runtime(get_magnetization_fixed_point_iteration,nrepeats,cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)

        for i, Hext in enumerate(Hext_series):
            hext = Hext/Ms
            bfgs_wgrad_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS')

            lbfgs_wgrad_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')

            conjgrad_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='CG')

            # newtoncg_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='Newton-CG')
            newtoncg_time = None

        for i, Hext in enumerate(Hext_series):
            hext = Hext/Ms
            bfgs_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization,nrepeats,initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS')

            lbfgs_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization,nrepeats,initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')

            

        print(f'For applied fields {Hext_series*mu0}, {num_particles} particles, and {nrepeats} function evaluations\n')
        print(f'fixed point runtime:{fixedpoint_runtime}')
        print(f'bfgs runtime:{bfgs_time}')
        print(f'bfgs with gradient runtime:{bfgs_wgrad_time}')
        print(f'l-bfgs runtime:{lbfgs_time}')
        print(f'l-bfgs with gradient runtime:{lbfgs_wgrad_time}')
        print(f'conjugate gradient runtime:{conjgrad_time}')
        print(f'Newton-CG runtime:{newtoncg_time}')

        print(f'BFGS Speed up with gradient:{bfgs_time/bfgs_wgrad_time}')
    
    if plotting_results:
        dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
        dipole_field_kernel = np.float64(cp.asnumpy(dipole_field_kernel))
        # for plotting and analyzing results of different methods
        for i, Hext in enumerate(Hext_series):
            Hext = np.float32(Hext)
            hext = Hext/Ms
            magnetic_moments[i] = get_magnetization_fixed_point_iteration(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)

            res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS')
            bfgs_auglang_magnetization[i] = res.x
            initial_guess_cpu = res.x
            bfgs_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_cpu,dipole_field_kernel,chi,hext,num_particles)

            res = solve_magnetization_augmented_lagrangian_optimization_w_grad(initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS')
            bfgs_wgrad_auglang_magnetization[i] = res.x
            initial_guess_cpu_grad = res.x
            bfgs_wgrad_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_cpu_grad,dipole_field_kernel,chi,hext,num_particles)

            res = solve_magnetization_augmented_lagrangian_optimization_w_grad(initial_guess_lbfgs_wgrad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')
            lbfgs_wgrad_auglang_magnetization[i] = res.x
            initial_guess_lbfgs_wgrad = res.x
            lbfgs_wgrad_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_lbfgs_wgrad,dipole_field_kernel,chi,hext,num_particles)

            res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')
            lbfgs_auglang_magnetization[i] = res.x
            initial_guess_lbfgs = res.x
            lbfgs_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_lbfgs,dipole_field_kernel,chi,hext,num_particles)

            res = solve_magnetization_augmented_lagrangian_optimization_w_grad(initial_guess_cg,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='CG')
            cg_auglang_magnetization[i] = res.x
            initial_guess_cg = res.x
            cg_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_cg,dipole_field_kernel,chi,hext,num_particles)

            res = solve_magnetization_augmented_lagrangian_optimization_w_grad(initial_guess_newtoncg,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='Newton-CG')
            newtoncg_auglang_magnetization[i] = res.x
            initial_guess_newtoncg = res.x
            newtoncg_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_newtoncg,dipole_field_kernel,chi,hext,num_particles)

        magnetization = magnetic_moments/particle_volume
        magnetization /= Ms

        xdata = Bext_series_magnitude[::-1]
        plot_soln_and_constraint_violation(bfgs_auglang_magnetization,bfgs_auglang_constraint_violation,xdata,'bfgs')
        plot_soln_and_constraint_violation(bfgs_wgrad_auglang_magnetization,bfgs_wgrad_auglang_constraint_violation,xdata,'bfgs wgrad')
        plot_soln_and_constraint_violation(lbfgs_auglang_magnetization,lbfgs_auglang_constraint_violation,xdata,'lbfgs')
        plot_soln_and_constraint_violation(lbfgs_wgrad_auglang_magnetization,lbfgs_wgrad_auglang_constraint_violation,xdata,'lbfgs wrad')
        plot_soln_and_constraint_violation(cg_auglang_magnetization,cg_auglang_constraint_violation,xdata,'cg')
        plot_soln_and_constraint_violation(newtoncg_auglang_magnetization,newtoncg_auglang_constraint_violation,xdata,'newtoncg')

        # np.savetxt(save_dir+f'{num_particles}_particles_normalized_magnetization.csv',bfgs_agulang,delimiter=',')
        bfgs_auglang_magnetization= bfgs_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))
        bfgs_wgrad_auglang_magnetization= bfgs_wgrad_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))

        field_index = Hext_series.shape[0]-1
        if field_index < 0:
            field_index = 0

        fig, axs = plt.subplots(3,3)
        for count in range(particles.shape[0]):
            axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
            axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
            axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
            axs[1,0].plot(Bext_series_magnitude[::-1],bfgs_auglang_magnetization[:,count,0])
            axs[1,1].plot(Bext_series_magnitude[::-1],bfgs_auglang_magnetization[:,count,1])
            axs[1,2].plot(Bext_series_magnitude[::-1],bfgs_auglang_magnetization[:,count,2])
            axs[2,0].plot(Bext_series_magnitude[::-1],bfgs_wgrad_auglang_magnetization[:,count,0])
            axs[2,1].plot(Bext_series_magnitude[::-1],bfgs_wgrad_auglang_magnetization[:,count,1])
            axs[2,2].plot(Bext_series_magnitude[::-1],bfgs_wgrad_auglang_magnetization[:,count,2])    

        axs[2,0].set_xlabel('Applied Field (T)')
        axs[2,1].set_xlabel('Applied Field (T)')
        axs[2,2].set_xlabel('Applied Field (T)')
        axs[0,0].set_title('X')
        axs[0,1].set_title('Y')
        axs[0,2].set_title('Z')
        axs[0,0].set_ylabel('normalized particle magnetization')
        axs[1,0].set_ylabel('cpu based auglang')
        axs[2,0].set_ylabel('gpu based auglang')

        plt.show(block=False)
        print('end of function')

def plot_soln_and_constraint_violation(soln,constraint_violation,xdata,tag=''):
    fig, axs = plt.subplots(2,3)
    num_particles =np.floor_divide(soln.shape[1],3)
    if np.ndim(soln) != 3:
        soln = soln.reshape((xdata.shape[0],num_particles,3))
    if np.ndim(constraint_violation) != 3:
        constraint_violation = constraint_violation.reshape((xdata.shape[0],num_particles,3))
    for count in range(num_particles):
        axs[0,0].plot(xdata,soln[:,count,0])
        axs[0,1].plot(xdata,soln[:,count,1])
        axs[0,2].plot(xdata,soln[:,count,2])
        axs[1,0].plot(xdata,constraint_violation[:,count,0])
        axs[1,1].plot(xdata,constraint_violation[:,count,1])
        axs[1,2].plot(xdata,constraint_violation[:,count,2])

    axs[1,0].set_xlabel('Applied Field (T)')
    axs[1,1].set_xlabel('Applied Field (T)')
    axs[1,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized particle magnetization' + tag)
    axs[1,0].set_ylabel('constraint violation')
    

    plt.show(block=False)

def get_function_runtime(func,nrepeats,*args,**kwargs):
    start = time.perf_counter()
    for i in range(nrepeats):
        return_val = func(*args,**kwargs)
    end = time.perf_counter()
    delta = end-start
    print("run time is {} seconds".format(delta))
    return delta, return_val

def placeholder_function():
    return 0

def get_magnetization_fixed_point_iteration(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=20,atol=None,rtol=1e-4,initial_soln=None):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    if atol == None:
        atol = particle_volume*Ms*1e-3
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetic_moments = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    if initial_soln == None:
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))
    else:
        magnetic_moment = initial_soln

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    dipole_field_kernel = get_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,size_particles,l_e)

    inv_l_e = np.float32(1/l_e)
    for i in range(max_iters):
        Htot = cp.tile(Hext,particles.shape[0])
        dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetic_moment))
        Htot += dipolar_fields
        # dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        # cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
        host_magnetic_moments[i] = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))
        if i > 0:
            difference = host_magnetic_moments[i] - host_magnetic_moments[i-1]
            if np.all(np.abs(np.ravel(difference)) < atol):
                host_magnetic_moments[i:] = host_magnetic_moments[i]
                break

    return host_magnetic_moments

def get_normalized_magnetization_fixed_point_iteration(hext,particles,particle_posns,Ms,chi,particle_volume,l_e,max_iters=20,atol=None,rtol=1e-4,initial_soln=None):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    if atol == None:
        atol = 1e-2#5e-3
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetization = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    host_magnetizations = np.zeros((max_iters,particles.shape[0],3),dtype=np.float32)
    hext_vector = cp.tile(hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    if type(initial_soln) == type(None):
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,hext_vector,magnetization,size_particles))
        guess_flag = False
    else:
        guess_flag = True
        magnetization = initial_soln.copy()
        # magnetization_history = initial_soln.copy()
        # max_allowed_change = 5e-2

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,size_particles,l_e,particle_volume)
    for i in range(max_iters):
        htot = cp.tile(hext,particles.shape[0])
        dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetization))
        htot += dipolar_fields
        # dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        # cupy_stream.synchronize()
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,htot,magnetization,size_particles))
        cupy_stream.synchronize()
        # if guess_flag:
        #     difference = magnetization - magnetization_history
        #     sign_of_delta = cp.sign(difference)
        #     mask = cp.nonzero(cp.abs(difference) > max_allowed_change)[0]
        #     difference[mask] = max_allowed_change*sign_of_delta[mask]
        #     magnetization = magnetization_history + difference
        host_magnetizations[i] = cp.asnumpy(magnetization).reshape((particles.shape[0],3))
        if i > 0:
            difference = host_magnetizations[i] - host_magnetizations[i-1]
            if np.all(np.abs(np.ravel(difference)) < atol + np.abs(np.ravel(host_magnetizations[i-1]))*rtol):
                host_magnetizations[i:] = host_magnetizations[i]
                break

    return host_magnetizations

def solve_magnetization_augmented_lagrangian_optimization(initial_guess,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS'):
    """Function wrapper to run the optimization with constraint as an unconstrained optimization using an augmented Lagrangian method."""
    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    dipole_field_kernel = cp.asnumpy(dipole_field_kernel)
    max_iterations = 3
    constraint_history = np.zeros((max_iterations,))
    for i in range(max_iterations):
        res = scipy.optimize.minimize(get_normalized_objective_function_auglag,x0=initial_guess,args=(dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers),method=method)
        initial_guess = res.x
        constraint_violations = get_normalized_constraint_violation_auglag(initial_guess,dipole_field_kernel,chi,hext,num_particles)
        lagrange_multipliers += penalty_constant * constraint_violations
        penalty_constant *= 2
        constraint_history[i] = np.sum(np.abs(constraint_violations))
    return res

def solve_magnetization_augmented_lagrangian_optimization_w_grad(initial_guess,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS'):
    """Function wrapper to run the optimization with constraint as an unconstrained optimization using an augmented Lagrangian method."""
    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    dipole_field_kernel = np.float64(cp.asnumpy(dipole_field_kernel))
    max_iterations = 5
    # constraint_history = np.zeros((max_iterations,))
    soln_history = initial_guess.copy()
    for i in range(max_iterations):
        res = scipy.optimize.minimize(get_normalized_objective_and_grad_auglag,x0=initial_guess,args=(dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers),method=method,jac=True)
        initial_guess = res.x
        if np.allclose(initial_guess,soln_history,atol=1e-3):
            break
        else:
            constraint_violations = get_normalized_constraint_violation_auglag(initial_guess,dipole_field_kernel,chi,hext,num_particles)
            lagrange_multipliers += penalty_constant * constraint_violations
            penalty_constant *= 2
            # constraint_history[i] = np.sum(np.abs(constraint_violations))
    # print(f'exited from iteration {i}')
    return res

def get_normalized_objective_function_auglag(normalized_magnetizations,dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers):
    dipole_fields = np.matmul(dipole_field_kernel,normalized_magnetizations)
    energy = get_normalized_total_magnetic_energy(normalized_magnetizations,hext,dipole_fields,chi,num_particles)
    tiled_hext = np.tile(hext,num_particles)
    h_tot = (dipole_fields + tiled_hext)
    htot_norm = np.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
    htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    # h_tot = np.reshape(h_tot,(3*num_particles,))
    constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
    result = energy + (0.5*penalty_constant) *np.sum(constraint_violations*constraint_violations) + np.sum(lagrange_multipliers*constraint_violations)
    return result

def get_normalized_objective_and_grad_auglag(normalized_magnetizations,dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers):
    dipole_fields = np.matmul(dipole_field_kernel,normalized_magnetizations)
    energy = get_normalized_total_magnetic_energy(normalized_magnetizations,hext,dipole_fields,chi,num_particles)
    tiled_hext = np.tile(hext,num_particles)
    h_tot = (dipole_fields + tiled_hext)
    htot_norm = np.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
    htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    # h_tot = np.reshape(h_tot,(3*num_particles,))
    constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
    result = energy + (0.5*penalty_constant) *np.sum(constraint_violations*constraint_violations) + np.sum(lagrange_multipliers*constraint_violations)
    grad = get_normalized_grad_auglag(normalized_magnetizations,constraint_violations,dipole_field_kernel,tiled_hext,dipole_fields,h_tot,htot_norm_repeat,chi,penalty_constant,lagrange_multipliers)
    return (result,grad)


def get_normalized_grad_auglag(normalized_magnetizations,constraint_violations,dipole_field_kernel,tiled_hext,hdip,htot,htot_norm,chi,penalty_constant,lagrange_multipliers):
    inv_chi = 1/chi
    #result from considering the matrix form of the total magnetic energy and taking the gradient
    grad_energy = -tiled_hext + inv_chi*normalized_magnetizations -1*np.matmul(dipole_field_kernel,normalized_magnetizations)
    num_particles = np.floor_divide(normalized_magnetizations.shape[0],3)
    result = np.zeros((3*num_particles,))
    grad_constraints = magnetism.get_grad_constraints(normalized_magnetizations,htot,htot_norm,dipole_field_kernel,chi,num_particles)
    result += grad_energy
    result += penalty_constant*np.matmul(grad_constraints,constraint_violations)
    result += np.matmul(np.transpose(lagrange_multipliers),grad_constraints)
    return result

def get_normalized_grad_auglag_experimental(normalized_magnetizations,constraint_violations,dipole_field_kernel,tiled_hext,hdip,htot,htot_norm,chi,penalty_constant,lagrange_multipliers):
    inv_chi = 1/chi
    #grad_energy = -tiled_hext -(0.5)*hdip +inv_chi*normalized_magnetizations -(0.5)*np.sum(np.matmul(dipole_field_kernel,normalized_magnetizations),axis=1)
    #result from considering the matrix form of the total magnetic energy and taking the gradient
    grad_energy = -tiled_hext + inv_chi*normalized_magnetizations -1*np.matmul(dipole_field_kernel,normalized_magnetizations)
    num_particles = np.floor_divide(normalized_magnetizations.shape[0],3)
    result = np.zeros((3*num_particles,))
    grad_constraints = np.zeros((3*num_particles,3*num_particles))
    for i in range(3*num_particles):
        for j in range(3*num_particles):
            if i == j:
                #assuming i am correct that the partial derivative component is the row index
                grad_constraints[j,i] = 1 + chi*htot_norm[j]
            else:
                x_field_idx = 3*np.floor_divide(j,3)
                grad_constraints[j,i] = chi*(normalized_magnetizations[j]*(1/htot_norm[j])*(htot[x_field_idx]*dipole_field_kernel[x_field_idx,i] + htot[x_field_idx+1]*dipole_field_kernel[x_field_idx+1,i] + htot[x_field_idx+2]*dipole_field_kernel[x_field_idx+2,i]) - dipole_field_kernel[j,i])
    #TODO rewrite using numpy broadcasting rules to avoid the for loop/take advantage of vectorization of operations
    for i in range(3*num_particles):
        #contribution to the gradient from the energy expression
        result[i] += -tiled_hext[i] - (0.5)*hdip[i] + inv_chi*normalized_magnetizations[i] - 0.5*(np.sum(dipole_field_kernel[:,i]*normalized_magnetizations))
        #TODO write out the gradient of the constraints, then write out the contribution to the total gradient of the augmented lagrangian objective function from the penalty method term and the lagrange multiplier term (both of which depend on the gradient of the constraint, so i can calculate that part once...)
        #contribution to the gradient from the penalty expression
        # result[i] += penalty_constant*np.sum(grad_constraints[i,:]*constraint_violations)
        # #contribution to the gradient from the lagrange multiplier expression
        # result[i] += lagrange_multipliers[i]*grad_constraints[i,:]
    assert np.allclose(grad_energy,result)
    grad_constraints_alt = get_grad_constraints(tiled_hext,htot,htot_norm,dipole_field_kernel,normalized_magnetizations,chi)
    # assert np.allclose(grad_constraints,np.transpose(grad_constraints))
    print(grad_constraints_alt)
    print(grad_constraints-grad_constraints_alt)
    print(grad_constraints_alt/grad_constraints)
    assert np.allclose(grad_constraints_alt,np.transpose(grad_constraints_alt))
    # assert np.allclose(grad_constraints,grad_constraints_alt)
    result += penalty_constant*np.matmul(grad_constraints,constraint_violations)
    result += np.matmul(np.transpose(lagrange_multipliers),grad_constraints)
    return result

def get_third_order_tensor_for_grad_constraints(hext,dipole_field_kernel,normalized_magnetizations,htot_norm):
    dim = normalized_magnetizations.shape[0]
    third_order_tensor = np.zeros((dim,dim,dim))
    hext_vector = hext[0:3]
    for i in range(dim):
        submatrix_idx = np.floor_divide(i,3)
        submatrix = dipole_field_kernel[3*submatrix_idx:3*submatrix_idx+3,:]
        # print((np.matmul(hext_vector,submatrix) + np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations))/htot_norm[i])
        third_order_tensor[i,i,:] = (np.matmul(hext_vector,submatrix) + np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations))/htot_norm[i]
        # third_order_tensor[i,i,:] = 0.5*(np.matmul(hext_vector,submatrix) + np.matmul(np.transpose(submatrix),np.transpose(hext_vector))+ np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations) + np.matmul(np.transpose(normalized_magnetizations),np.matmul(np.transpose(submatrix),submatrix)))/htot_norm[i]#hext_vector*submatrix + np.transpose(submatrix)*hext_vector#2*np.matmul(hext_vector,submatrix)+2*np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations)
    return third_order_tensor

def get_third_order_tensor_for_grad_constraints_experimental(hext,dipole_field_kernel,normalized_magnetizations,htot,htot_norm):
    dim = normalized_magnetizations.shape[0]
    third_order_tensor = np.zeros((dim,dim,dim))
    second_order_tensor_intermediate = np.zeros((dim,dim))
    hext_vector = hext[0:3]
    inv_htot_norm = 1/htot_norm
    htot_hat = htot*inv_htot_norm
    second_order_tensor_intermediate = np.matmul(dipole_field_kernel,htot_hat)
    # index_tuple = (np.arange(0,dim,1),np.arange(0,dim,1))
    
    for i in range(dim):
        third_order_tensor[i,i,:] = second_order_tensor_intermediate 
    #     submatrix_idx = np.floor_divide(i,3)
    #     submatrix = dipole_field_kernel[submatrix_idx:submatrix_idx+3,:]
    #     third_order_tensor[i,i,:] = (np.matmul(hext_vector,submatrix) + np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations))/htot_norm[i]
        # third_order_tensor[i,i,:] = 0.5*(np.matmul(hext_vector,submatrix) + np.matmul(np.transpose(submatrix),np.transpose(hext_vector))+ np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations) + np.matmul(np.transpose(normalized_magnetizations),np.matmul(np.transpose(submatrix),submatrix)))/htot_norm[i]#hext_vector*submatrix + np.transpose(submatrix)*hext_vector#2*np.matmul(hext_vector,submatrix)+2*np.matmul(np.matmul(np.transpose(submatrix),submatrix),normalized_magnetizations)
    return third_order_tensor

def get_grad_constraints(hext,htot,htot_norms,dipole_field_kernel,normalized_magnetizations,chi):
    dim = normalized_magnetizations.shape[0]
    grad_constraints= np.zeros(dim)
    third_order_tensor = get_third_order_tensor_for_grad_constraints(hext,dipole_field_kernel,normalized_magnetizations,htot_norms)
    # print('full third order tensor from component-wise expression')
    # print(third_order_tensor)
    # print('first index 0 of third order tensor')
    # print(third_order_tensor[0,:,:])
    # print('second index 0 of third order tensor')
    # print(third_order_tensor[:,0,:])
    # print('third index 0 of third order tensor')
    # print(third_order_tensor[:,:,0])
    # print(third_order_tensor[:,:,1])
    # print(third_order_tensor[:,:,2])
    # print(third_order_tensor[:,:,3])
    # print(third_order_tensor[:,:,4])
    # print(third_order_tensor[:,:,5])
    # third_order_tensor_experimental = get_third_order_tensor_for_grad_constraints_experimental(hext,dipole_field_kernel,normalized_magnetizations,htot,htot_norms)
    # print(third_order_tensor_experimental)
    grad_constraints = np.eye(dim) + chi*(np.eye(dim)*htot_norms + np.einsum('ijk,k->ij',third_order_tensor,normalized_magnetizations) - dipole_field_kernel)#+ np.matmul(third_order_tensor,normalized_magnetizations)
    # grad_constraints_alt = np.eye(dim) + chi*(np.eye(dim)*htot_norms + np.einsum('ijk,k->ij',third_order_tensor_experimental,normalized_magnetizations) - dipole_field_kernel)
    return grad_constraints

def get_normalized_constraint_violation_auglag(normalized_magnetizations,dipole_field_kernel,chi,hext,num_particles):
    dipole_fields = np.matmul(dipole_field_kernel,normalized_magnetizations)
    tiled_hext = np.tile(hext,num_particles)
    h_tot = (dipole_fields + tiled_hext)
    htot_norm = np.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
    htot_norm_repeat = np.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
    return constraint_violations

def get_normalized_total_magnetic_energy(normalized_magnetic_moments,hext,dipole_fields,chi,num_particles):
    # dipole_fields = dipole_field_kernel*normalized_magnetic_moments
    # magnetic_moments = np.reshape(normalized_magnetic_moments,(num_particles,3))
    zeeman_energy = get_normalized_zeeman_energy(normalized_magnetic_moments.reshape((num_particles,3)),hext)
    dipole_energy = get_normalized_dipolar_energy(normalized_magnetic_moments,dipole_fields)
    self_energy = get_normalized_self_energy(normalized_magnetic_moments,chi)
    total_energy = (zeeman_energy + dipole_energy + self_energy)
    return total_energy

def get_normalized_total_magnetic_energy_w_kernel(normalized_magnetic_moments,hext,dipole_field_kernel,chi,num_particles):
    dipole_fields = np.matmul(dipole_field_kernel,normalized_magnetic_moments)
    # hext_tiled = np.tile(hext,num_particles)
    # magnetic_moments = np.reshape(normalized_magnetic_moments,(num_particles,3))
    zeeman_energy = get_normalized_zeeman_energy(normalized_magnetic_moments.reshape((num_particles,3)),hext)
    dipole_energy = get_normalized_dipolar_energy(normalized_magnetic_moments,dipole_fields)
    self_energy = get_normalized_self_energy(normalized_magnetic_moments,chi)
    total_energy = (zeeman_energy + dipole_energy + self_energy)
    return total_energy

def get_normalized_zeeman_energy(normalized_magnetic_moments,hext):
     total_energy = -1*np.sum(normalized_magnetic_moments*hext)
    #  total_energy = np.sum(individual_energy)
     return total_energy

def get_normalized_dipolar_energy(normalized_magnetic_moments,normalized_dipole_fields):
    total_energy = (-0.5)*np.dot(np.ravel(normalized_magnetic_moments),np.ravel(normalized_dipole_fields))
    # total_energy = np.sum(individual_energy)/2
    return total_energy

def get_normalized_self_energy(normalized_magnetic_moments,chi):
    total_energy = 1/(2*chi) * np.dot(np.ravel(normalized_magnetic_moments),np.ravel(normalized_magnetic_moments))#np.einsum('ij,hj->i',normalized_magnetic_moments,normalized_magnetic_moments)
    # total_energy = np.sum(individual_energy)
    return total_energy

def get_normalized_objective_function_auglag_gpu(normalized_magnetizations,dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers):
    device_magnetizations = cp.asarray(normalized_magnetizations,dtype=cp.float32)
    dipole_fields = cp.matmul(dipole_field_kernel,device_magnetizations)
    energy = get_normalized_total_magnetic_energy_w_kernel_gpu(device_magnetizations,hext,dipole_fields,chi,num_particles)
    tiled_hext = cp.tile(hext,num_particles)
    h_tot = (dipole_fields + tiled_hext)
    htot_norm = cp.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
    htot_norm_repeat = cp.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    constraint_violations = device_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
    result = energy + (0.5*penalty_constant) * cp.sum(constraint_violations*constraint_violations) + cp.sum(lagrange_multipliers*constraint_violations)
    result = cp.asnumpy(result).item()
    return result

def get_normalized_total_magnetic_energy_w_kernel_gpu(normalized_magnetic_moments,hext,dipole_fields,chi,num_particles):
    # mu0 = 4*np.pi*1e-7
    # scaling_factor = mu0*np.power(Ms,2)*particle_volume
    # dipole_fields = dipole_field_kernel*normalized_magnetic_moments
    hext_tiled = cp.tile(hext,num_particles)
    zeeman_energy = get_normalized_zeeman_energy_gpu(normalized_magnetic_moments,hext_tiled)
    dipole_energy = get_normalized_dipolar_energy_gpu(normalized_magnetic_moments,dipole_fields)
    self_energy = get_normalized_self_energy_gpu(normalized_magnetic_moments,chi)
    total_energy = (zeeman_energy + dipole_energy + self_energy)#/scaling_factor
    return total_energy

def get_normalized_zeeman_energy_gpu(normalized_magnetic_moments,hext):
     individual_energy = -1*cp.sum(normalized_magnetic_moments*hext)
     total_energy = cp.sum(individual_energy)
     return total_energy

def get_normalized_dipolar_energy_gpu(normalized_magnetic_moments,normalized_dipole_fields):
    total_energy = -(0.5)*cp.dot(normalized_magnetic_moments,normalized_dipole_fields)
    # total_energy = cp.sum(individual_energy)/2
    return total_energy

def get_normalized_self_energy_gpu(normalized_magnetic_moments,chi):
    total_energy = 1/(2*chi) * cp.dot(normalized_magnetic_moments,normalized_magnetic_moments)
    # total_energy = cp.sum(individual_energy)
    return total_energy

def solve_magnetization_augmented_lagrangian_optimization_gpu(initial_guess,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='BFGS'):
    """Function wrapper to run the optimization with constraint as an unconstrained optimization using an augmented Lagrangian method."""
    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    max_iterations = 3
    constraint_history = np.zeros((max_iterations,))
    for i in range(max_iterations):
        res = scipy.optimize.minimize(get_normalized_objective_function_auglag_gpu,x0=initial_guess,args=(dipole_field_kernel,chi,hext,num_particles,penalty_constant,lagrange_multipliers),method=method)
        initial_guess = res.x
        constraint_violations = get_normalized_constraint_violation_auglag(initial_guess,cp.asnumpy(dipole_field_kernel),chi,cp.asnumpy(hext),num_particles)
        lagrange_multipliers += cp.asarray(penalty_constant * constraint_violations)
        penalty_constant *= 2
        constraint_history[i] = np.sum(np.abs(constraint_violations))
    return res

def test_optimizer_bootstrapped_fixedpoint_solver():
    #2024-08-12 D Marchfield
    """Testing a fixed point iteration method that solves for normalized magnetization, and testing using the optimizer solution to bootstrap a seeded fixed point iteration method. can i change fields and use prior magnetizations and still have the fixed point method work (where it has been failing at small separations and 3D positions of particles). can i make small changes to the particle positions and seed with the magnetizations at the same field but nearby particle positions and have the fixed point iterative method work?"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = .010/mu0
    n_field_steps = 100
    H_step = H_mag/(n_field_steps)
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
    # Hext_series = Hext_series[::-1,:]
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([2,2,2])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7.8e-6,7.8e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    nextparticle_posns = particle_posns.copy()
    particle_posns /= l_e
    particle_posns = np.float32(particle_posns)
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40

    Hext = np.float32(Hext_series[0])

    magnetic_moments = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    seeded_magnetization = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)
    normalized_magnetization = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    movement_seeded_magnetization = np.zeros((Hext_series.shape[0],max_iters,num_particles,3),dtype=np.float32)

    lbfgs_auglang_magnetization= np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)
    lbfgs_auglang_constraint_violation = np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    movement_lbfgs_auglang_magnetization= np.zeros((Hext_series.shape[0],num_particles*3),dtype=np.float32)

    initial_guess = np.float64(0.9*Hext_series[0,:]/np.linalg.norm(Hext_series[0,:]))#+np.array([0.1,0.1,0.1],dtype=np.float32)
    initial_guess_cpu = np.tile(initial_guess,num_particles)
    initial_guess_cpu_grad = initial_guess_cpu.copy()
    initial_guess_lbfgs = initial_guess_cpu.copy()
    penalty_constant = 1
    lagrange_multipliers = np.zeros((3*num_particles,))

    num_particles = np.int64(num_particles)
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_particles = num_particles
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(device_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()
    
    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    dipole_field_kernel = np.float64(cp.asnumpy(dipole_field_kernel))
    # for plotting and analyzing results of different methods
    for i, Hext in enumerate(Hext_series):
        Hext = np.float32(Hext)
        hext = Hext/Ms
        magnetic_moments[i] = get_magnetization_fixed_point_iteration(cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)

        normalized_magnetization[i] = get_normalized_magnetization_fixed_point_iteration(cp.asarray(hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)

        
        res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')
        lbfgs_auglang_magnetization[i] = res.x
        initial_guess_lbfgs = res.x
        lbfgs_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_lbfgs,dipole_field_kernel,chi,hext,num_particles)
        # if i == 0:
            # device_initial_soln = cp.array(lbfgs_auglang_magnetization[i],dtype=cp.float32)
        device_initial_soln = cp.array(lbfgs_auglang_magnetization[i],dtype=cp.float32)

        seeded_magnetization[i] = get_normalized_magnetization_fixed_point_iteration(cp.asarray(hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters,initial_soln=device_initial_soln)
        device_initial_soln = cp.array(seeded_magnetization[i,-1].reshape((3*num_particles,)),dtype=cp.float32)

    particle_counter = 0
    dt = 5e-3
    for _ in range(num_particles):
        nextparticle_posns[particle_counter,0] += rng.integers(low=-1,high=2,size=1)*l_e*dt*0.1
        nextparticle_posns[particle_counter,1] += rng.integers(low=-1,high=2,size=1)*l_e*dt*0.1
        nextparticle_posns[particle_counter,2] += rng.integers(low=-1,high=2,size=1)*l_e*dt*0.1
        particle_counter += 1
    nextparticle_posns /= l_e
    nextparticle_posns = np.float32(nextparticle_posns)
    nextdevice_particle_posns = cp.array(nextparticle_posns.astype(np.float32)).reshape((nextparticle_posns.shape[0]*nextparticle_posns.shape[1],1),order='C')
    initial_guess_lbfgs = initial_guess_cpu.copy()

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(nextdevice_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    for i, Hext in enumerate(Hext_series):
        Hext = np.float32(Hext)
        hext = Hext/Ms
        # device_initial_soln = cp.array(seeded_magnetization[i,-1].reshape((3*num_particles,)),dtype=cp.float32)
        device_initial_soln = cp.array(lbfgs_auglang_magnetization[i].reshape((3*num_particles,)),dtype=cp.float32)
        movement_seeded_magnetization[i] = get_normalized_magnetization_fixed_point_iteration(cp.asarray(hext,dtype=cp.float32),particles,nextdevice_particle_posns,Ms,chi,particle_volume,l_e,max_iters,initial_soln=device_initial_soln)

        res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')
        movement_lbfgs_auglang_magnetization[i] = res.x
        initial_guess_lbfgs = res.x
        # movement_lbfgs_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_lbfgs,dipole_field_kernel,chi,hext,num_particles)

        

    magnetization = magnetic_moments/particle_volume
    magnetization /= Ms

    xdata = Bext_series_magnitude#[::-1]
    plot_soln_and_constraint_violation(lbfgs_auglang_magnetization,lbfgs_auglang_constraint_violation,xdata,'lbfgs')

    lbfgs_auglang_magnetization = lbfgs_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))

    movement_lbfgs_auglang_magnetization = movement_lbfgs_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))

    field_index = 0#Hext_series.shape[0]-1
    if field_index < 0:
        field_index = 0
    
    iter_number = np.arange(max_iters)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(iter_number,magnetization[field_index,:,count,0])
        axs[0,1].plot(iter_number,magnetization[field_index,:,count,1])
        axs[0,2].plot(iter_number,magnetization[field_index,:,count,2])
        axs[1,0].plot(iter_number,normalized_magnetization[field_index,:,count,0])
        axs[1,1].plot(iter_number,normalized_magnetization[field_index,:,count,1])
        axs[1,2].plot(iter_number,normalized_magnetization[field_index,:,count,2])
        axs[2,0].plot(iter_number,seeded_magnetization[field_index,:,count,0])
        axs[2,1].plot(iter_number,seeded_magnetization[field_index,:,count,1])
        axs[2,2].plot(iter_number,seeded_magnetization[field_index,:,count,2])

    axs[2,0].set_xlabel('iteration number')
    axs[2,1].set_xlabel('iteration number')
    axs[2,2].set_xlabel('iteration number')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized magnetization (unnormalized fixed point)')
    axs[1,0].set_ylabel('using normalized fixed point')
    axs[2,0].set_ylabel('seeded normalized fixed point')

    plt.show(block=False)

    fig, axs = plt.subplots(4,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude,magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude,magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude,magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude,lbfgs_auglang_magnetization[:,count,0])
        axs[1,1].plot(Bext_series_magnitude,lbfgs_auglang_magnetization[:,count,1])
        axs[1,2].plot(Bext_series_magnitude,lbfgs_auglang_magnetization[:,count,2])
        axs[2,0].plot(Bext_series_magnitude,normalized_magnetization[:,-1,count,0])
        axs[2,1].plot(Bext_series_magnitude,normalized_magnetization[:,-1,count,1])
        axs[2,2].plot(Bext_series_magnitude,normalized_magnetization[:,-1,count,2])
        axs[3,0].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,0])
        axs[3,1].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,1])
        axs[3,2].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,2])
        # axs[0,0].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,0])
        # axs[0,1].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,1])
        # axs[0,2].plot(Bext_series_magnitude[::-1],magnetization[:,-1,count,2])
        # axs[1,0].plot(Bext_series_magnitude[::-1],lbfgs_auglang_magnetization[:,count,0])
        # axs[1,1].plot(Bext_series_magnitude[::-1],lbfgs_auglang_magnetization[:,count,1])
        # axs[1,2].plot(Bext_series_magnitude[::-1],lbfgs_auglang_magnetization[:,count,2])
        # axs[2,0].plot(Bext_series_magnitude[::-1],normalized_magnetization[:,-1,count,0])
        # axs[2,1].plot(Bext_series_magnitude[::-1],normalized_magnetization[:,-1,count,1])
        # axs[2,2].plot(Bext_series_magnitude[::-1],normalized_magnetization[:,-1,count,2])
        # axs[3,0].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,0])
        # axs[3,1].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,1])
        # axs[3,2].plot(Bext_series_magnitude[::-1],seeded_magnetization[:,-1,count,2])

    axs[3,0].set_xlabel('Applied Field (T)')
    axs[3,1].set_xlabel('Applied Field (T)')
    axs[3,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('normalized magnetization (fixed point moments)')
    axs[1,0].set_ylabel('optimizer auglang')
    axs[2,0].set_ylabel('fixed point magnetizations')
    axs[3,0].set_ylabel('seeded fixed point magnetizations')

    plt.show(block=False)

    for field_index in range(Hext_series.shape[0]):
        if np.mod(field_index,10) == 0:
            fig, axs = plt.subplots(2,3)
            for count in range(particles.shape[0]):
                axs[0,0].plot(iter_number,seeded_magnetization[field_index,:,count,0])
                axs[0,1].plot(iter_number,seeded_magnetization[field_index,:,count,1])
                axs[0,2].plot(iter_number,seeded_magnetization[field_index,:,count,2])
                axs[1,0].plot(iter_number,movement_seeded_magnetization[field_index,:,count,0])
                axs[1,1].plot(iter_number,movement_seeded_magnetization[field_index,:,count,1])
                axs[1,2].plot(iter_number,movement_seeded_magnetization[field_index,:,count,2])

            axs[1,0].set_xlabel('iteration number')
            axs[1,1].set_xlabel('iteration number')
            axs[1,2].set_xlabel('iteration number')
            axs[0,0].set_title('X')
            axs[0,1].set_title('Y')
            axs[0,2].set_title('Z')
            axs[0,0].set_ylabel('bootstrapped fixed point')
            axs[1,0].set_ylabel('particle motion fixed point')
            axs[0,0].set_title(f'{Bext_series_magnitude[field_index]*1000} mT')

            plt.show(block=False)

    fig, axs = plt.subplots(3,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,0])
        axs[0,1].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,1])
        axs[0,2].plot(Bext_series_magnitude,seeded_magnetization[:,-1,count,2])
        axs[1,0].plot(Bext_series_magnitude,movement_seeded_magnetization[:,-1,count,0])
        axs[1,1].plot(Bext_series_magnitude,movement_seeded_magnetization[:,-1,count,1])
        axs[1,2].plot(Bext_series_magnitude,movement_seeded_magnetization[:,-1,count,2])
        axs[2,0].plot(Bext_series_magnitude,movement_lbfgs_auglang_magnetization[:,count,0])
        axs[2,1].plot(Bext_series_magnitude,movement_lbfgs_auglang_magnetization[:,count,1])
        axs[2,2].plot(Bext_series_magnitude,movement_lbfgs_auglang_magnetization[:,count,2])

    axs[2,0].set_xlabel('Applied Field (T)')
    axs[2,1].set_xlabel('Applied Field (T)')
    axs[2,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('bootstrapped fixed point')
    axs[1,0].set_ylabel('particle motion fixed point')
    axs[2,0].set_ylabel('optimizer lbfgs')

    plt.show(block=False)

    num_steps = 10
    movement_seeded_magnetization = np.zeros((num_steps,max_iters,num_particles,3),dtype=np.float32)

    movement_lbfgs_auglang_magnetization= np.zeros((num_steps,num_particles*3),dtype=np.float32)
    nextparticle_posns = particle_posns.copy()
    for i in range(num_steps):
        
        particle_counter = 0
        dt = 5e-3
        for _ in range(num_particles):
            nextparticle_posns[particle_counter,0] += rng.integers(low=-1,high=2,size=1)*dt*0.1
            nextparticle_posns[particle_counter,1] += rng.integers(low=-1,high=2,size=1)*dt*0.1
            nextparticle_posns[particle_counter,2] += rng.integers(low=-1,high=2,size=1)*dt*0.1
            particle_counter += 1
        nextparticle_posns = np.float32(nextparticle_posns)
        nextdevice_particle_posns = cp.array(nextparticle_posns.astype(np.float32)).reshape((nextparticle_posns.shape[0]*nextparticle_posns.shape[1],1),order='C')
        initial_guess_lbfgs = initial_guess_cpu.copy()

        separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
        separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
        separation_vectors_kernel((grid_size,),(block_size,),(nextdevice_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
        cupy_stream.synchronize()

        Hext = Hext_series[0]
        Hext = np.float32(Hext)
        hext = Hext/Ms
        # device_initial_soln = cp.array(seeded_magnetization[i,-1].reshape((3*num_particles,)),dtype=cp.float32)
        if i == 0:
            device_initial_soln = cp.array(lbfgs_auglang_magnetization[i].reshape((3*num_particles,)),dtype=cp.float32)
        movement_seeded_magnetization[i] = get_normalized_magnetization_fixed_point_iteration(cp.asarray(hext,dtype=cp.float32),particles,nextdevice_particle_posns,Ms,chi,particle_volume,l_e,max_iters,initial_soln=device_initial_soln)
        device_initial_soln = cp.array(movement_seeded_magnetization[i,-1].reshape((3*num_particles,)),dtype=cp.float32)

        res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,method='L-BFGS-B')
        movement_lbfgs_auglang_magnetization[i] = res.x
        initial_guess_lbfgs = res.x
        # movement_lbfgs_auglang_constraint_violation[i] = get_normalized_constraint_violation_auglag(initial_guess_lbfgs,dipole_field_kernel,chi,hext,num_particles)

    movement_lbfgs_auglang_magnetization = movement_lbfgs_auglang_magnetization.reshape((num_steps,num_particles,3))
    xdata = np.arange(num_steps)
    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(xdata,movement_seeded_magnetization[:,-1,count,0])
        axs[0,1].plot(xdata,movement_seeded_magnetization[:,-1,count,1])
        axs[0,2].plot(xdata,movement_seeded_magnetization[:,-1,count,2])
        axs[1,0].plot(xdata,movement_lbfgs_auglang_magnetization[:,count,0])
        axs[1,1].plot(xdata,movement_lbfgs_auglang_magnetization[:,count,1])
        axs[1,2].plot(xdata,movement_lbfgs_auglang_magnetization[:,count,2])

    axs[1,0].set_xlabel('time steps')
    axs[1,1].set_xlabel('time steps')
    axs[1,2].set_xlabel('time steps')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('particle motion fixed point')
    axs[1,0].set_ylabel('optimizer lbfgs')
    axs[0,0].set_title(f'{Bext_series_magnitude[0]*1000} mT')
    plt.show(block=False)

    print('end of function')

def pytorch_magnetization_augmented_lagrangian_optimization(initial_guess,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3):
    """Function wrapper to run the optimization with constraint as an unconstrained optimization using an augmented Lagrangian method."""
    dipole_field_kernel = get_normalized_dipole_field_kernel_torch(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    max_iterations = 5
    max_steps = 100

    def get_normalized_objective_function_auglag_gpu(normalized_magnetizations):
        dipole_fields = torch.matmul(dipole_field_kernel,normalized_magnetizations)
        energy = get_normalized_total_magnetic_energy_torch(normalized_magnetizations,hext,dipole_fields,chi,num_particles)
        tiled_hext = torch.tile(hext,(num_particles,))
        h_tot = (dipole_fields + tiled_hext)
        htot_norm = torch.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
        htot_norm_repeat = torch.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
        constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
        result = energy + (0.5*penalty_constant) * torch.sum(constraint_violations*constraint_violations) + torch.sum(lagrange_multipliers*constraint_violations)
        return result
    
    def closure():
        if torch.is_grad_enabled():
            lbfgs.zero_grad(set_to_none=True)
        objective = get_normalized_objective_function_auglag_gpu(x_lbfgs)
        if objective.requires_grad:
            objective.backward(retain_graph=True)
        return objective

    x_lbfgs = torch.asarray(initial_guess,requires_grad=True,device='cuda',dtype=torch.float32)#torch.autograd.Variable(torch.Tensor(initial_guess).cuda(), requires_grad=True)
    last_xlbfgs = torch.zeros((3*num_particles,)).cuda()
    last_xlbfgs = last_xlbfgs.copy_(x_lbfgs)
    lbfgs = opt.LBFGS([x_lbfgs],
                    history_size=10,
                    max_iter=4,
                    line_search_fn="strong_wolfe")
    for _ in range(max_iterations):
        for counter in range(max_steps):
            lbfgs.step(closure)
            delta = torch.abs(x_lbfgs-last_xlbfgs)
            if torch.all(torch.less(delta,atol)):
                break
            else:
                last_xlbfgs = last_xlbfgs.copy_(x_lbfgs)
        end_constraint_violations = get_normalized_constraint_violation_auglag_torch(x_lbfgs,dipole_field_kernel,chi,hext,num_particles)
        if torch.all(torch.less(torch.abs(end_constraint_violations),1e-3)):
            break
        with torch.no_grad():
            lagrange_multipliers = lagrange_multipliers + penalty_constant * end_constraint_violations
        penalty_constant *= 2
    return x_lbfgs, end_constraint_violations

def pytorch_magnetization_augmented_lagrangian_stochastic_optimization(initial_guess,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='SGD',learning_rate=0.1):
    """Function wrapper to run the optimization with constraint as an unconstrained optimization using an augmented Lagrangian method."""
    dipole_field_kernel = get_normalized_dipole_field_kernel_torch(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    max_iterations = 5
    max_steps = 100

    def get_normalized_objective_function_auglag_gpu(normalized_magnetizations):
        dipole_fields = torch.matmul(dipole_field_kernel,normalized_magnetizations)
        energy = get_normalized_total_magnetic_energy_torch(normalized_magnetizations,hext,dipole_fields,chi,num_particles)
        tiled_hext = torch.tile(hext,(num_particles,))
        h_tot = (dipole_fields + tiled_hext)
        htot_norm = torch.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
        htot_norm_repeat = torch.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
        constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
        result = energy + (0.5*penalty_constant) * torch.sum(constraint_violations*constraint_violations) + torch.sum(lagrange_multipliers*constraint_violations)
        return result
    
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad(set_to_none=True)
        objective = get_normalized_objective_function_auglag_gpu(x)
        if objective.requires_grad:
            objective.backward(retain_graph=True)
        return objective

    x = torch.asarray(initial_guess,requires_grad=True,device='cuda',dtype=torch.float32)#torch.autograd.Variable(torch.Tensor(initial_guess).cuda(), requires_grad=True)
    last_x = torch.zeros((3*num_particles,)).cuda()
    last_x = last_x.copy_(x)
    if method == 'SGD':
        optimizer = opt.SGD([x],lr=learning_rate)
    elif method == 'Adam':
        optimizer = opt.Adam([x],lr=learning_rate)
    elif method == 'AdamW':
        optimizer = opt.AdamW([x],lr=learning_rate)
    for _ in range(max_iterations):
        for counter in range(max_steps):
            optimizer.step(closure)
            delta = torch.abs(x-last_x)
            if torch.all(torch.less(delta,atol)):
                break
            else:
                last_x = last_x.copy_(last_x)
        end_constraint_violations = get_normalized_constraint_violation_auglag_torch(x,dipole_field_kernel,chi,hext,num_particles)
        if torch.all(torch.less(torch.abs(end_constraint_violations),1e-3)):
            break
        with torch.no_grad():
            lagrange_multipliers = lagrange_multipliers + penalty_constant * end_constraint_violations
        penalty_constant *= 2
    return x, end_constraint_violations

def get_normalized_constraint_violation_auglag_torch(normalized_magnetizations,dipole_field_kernel,chi,hext,num_particles):
    dipole_fields = torch.matmul(dipole_field_kernel,normalized_magnetizations)
    tiled_hext = torch.tile(hext,(num_particles,))
    h_tot = (dipole_fields + tiled_hext)
    htot_norm = torch.linalg.norm(h_tot.reshape(num_particles,3),axis=1)
    htot_norm_repeat = torch.stack((htot_norm,htot_norm,htot_norm),axis=-1).reshape(3*num_particles,)
    constraint_violations = normalized_magnetizations*(1 + chi*htot_norm_repeat) - chi*h_tot
    return constraint_violations

def get_normalized_total_magnetic_energy_torch(normalized_magnetic_moments,hext,dipole_fields,chi,num_particles):
    hext_tiled = torch.tile(hext,(num_particles,))
    zeeman_energy = get_normalized_zeeman_energy_torch(normalized_magnetic_moments,hext_tiled)
    dipole_energy = get_normalized_dipolar_energy_torch(normalized_magnetic_moments,dipole_fields)
    self_energy = get_normalized_self_energy_torch(normalized_magnetic_moments,chi)
    total_energy = (zeeman_energy + dipole_energy + self_energy)#/scaling_factor
    return total_energy

def get_normalized_zeeman_energy_torch(normalized_magnetic_moments,hext):
     individual_energy = -1*torch.sum(normalized_magnetic_moments*hext)
     total_energy = torch.sum(individual_energy)
     return total_energy

def get_normalized_dipolar_energy_torch(normalized_magnetic_moments,normalized_dipole_fields):
    total_energy = -(0.5)*torch.dot(normalized_magnetic_moments,normalized_dipole_fields)
    # total_energy = cp.sum(individual_energy)/2
    return total_energy

def get_normalized_self_energy_torch(normalized_magnetic_moments,chi):
    total_energy = 1/(2*chi) * torch.dot(normalized_magnetic_moments,normalized_magnetic_moments)
    # total_energy = cp.sum(individual_energy)
    return total_energy

def pytorch_optimization():
    """Try using pytorch lbfgs optimization method for the magnetization solver"""
    #choose the maximum field, number of field steps, and field angle
    mu0 = 4*np.pi*1e-7
    H_mag = .010/mu0
    n_field_steps = 5
    H_step = H_mag/(n_field_steps)
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
    # Hext_series = Hext_series[::-1,:]
    chi = np.float32(70)
    Ms = np.float32(1.6e6)
    particle_radius = np.float32(1.5e-6)
    discretization_order = 3
    l_e = np.float32(2*particle_radius/(2*discretization_order + 1))
    num_nodes_per_particle = 8
    particle_mass_density = 7.86e3 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    particles_per_axis = np.array([5,5,5])
    num_particles = particles_per_axis[0]*particles_per_axis[1]*particles_per_axis[2]
    particles = np.zeros((num_particles,num_nodes_per_particle),dtype=np.int64)
    particle_posns = np.zeros((num_particles,3))
    separation = [3.3e-6,3.3e-6,3.3e-6]#[7.8e-6,7.8e-6,7.8e-6]#[3.3e-6,7e-6,7e-6]#3.2e-6#5.1e-6#
    ss = rand.SeedSequence()
    seed = ss.entropy
    rng = rand.default_rng(seed)
    particle_counter = 0
    for i in range(particles_per_axis[0]):
        for j in range(particles_per_axis[1]):
            for k in range(particles_per_axis[2]):
                particle_posns[particle_counter,0] = i * separation[0] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,1] = j * separation[1] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_posns[particle_counter,2] = k * separation[2] #+ rng.integers(low=-1,high=2,size=1)*l_e
                particle_counter += 1
    particle_posns /= l_e
    particle_posns = np.float32(particle_posns)
    device_particle_posns = cp.array(particle_posns.astype(np.float32)).reshape((particle_posns.shape[0]*particle_posns.shape[1],1),order='C')
    num_particles = particle_posns.shape[0]
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    max_iters = 40

    Hext = np.float32(Hext_series[0])

    lbfgs_auglang_magnetization = torch.zeros((Hext_series.shape[0],num_particles*3),device="cuda")
    lbfgs_auglang_constraint_violation = torch.zeros((Hext_series.shape[0],num_particles*3)).cuda()

    sgd_auglang_magnetization = torch.zeros((Hext_series.shape[0],num_particles*3),device="cuda")
    sgd_auglang_constraint_violation = torch.zeros((Hext_series.shape[0],num_particles*3)).cuda()

    adam_auglang_magnetization = torch.zeros((Hext_series.shape[0],num_particles*3),device="cuda")
    adam_auglang_constraint_violation = torch.zeros((Hext_series.shape[0],num_particles*3)).cuda()

    adamw_auglang_magnetization = torch.zeros((Hext_series.shape[0],num_particles*3),device="cuda")
    adamw_auglang_constraint_violation = torch.zeros((Hext_series.shape[0],num_particles*3)).cuda()

    scipy_lbfgs_magnetization = np.zeros((Hext_series.shape[0],num_particles*3))

    initial_guess = np.float64(0.9*Hext_series[0,:]/np.linalg.norm(Hext_series[0,:]))#+np.array([0.1,0.1,0.1],dtype=np.float32)
    initial_guess_lbfgs = np.tile(initial_guess,num_particles)
    initial_guess_sgd = initial_guess_lbfgs.copy()
    initial_guess_adam = initial_guess_lbfgs.copy()
    initial_guess_adamw = initial_guess_lbfgs.copy()
    initial_guess_cpu = initial_guess_lbfgs.copy()
    initial_guess_cpu_grad = initial_guess_lbfgs.copy()
    penalty_constant = 1
    lagrange_multipliers = torch.zeros((3*num_particles,),requires_grad=False).cuda()
    lagrange_multipliers_cpu = np.zeros((3*num_particles,))

    num_particles = np.int64(num_particles)
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_particles = num_particles
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(device_particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()
    
    # dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    # dipole_field_kernel = np.float64(cp.asnumpy(dipole_field_kernel))

    pytorch_lbfgs_time = np.zeros((Hext_series.shape[0],))
    pytorch_sgd_time = np.zeros((Hext_series.shape[0],))
    pytorch_adam_time = np.zeros((Hext_series.shape[0],))
    pytorch_adamw_time = np.zeros((Hext_series.shape[0],))
    nrepeats = 1
    for i, Hext in enumerate(Hext_series):
        hext = torch.asarray(Hext/Ms).cuda()
        print('pytorch lbfgs')
        pytorch_lbfgs_time[i], return_val = get_function_runtime(pytorch_magnetization_augmented_lagrangian_optimization,nrepeats,initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers)
        # lagrange_multipliers = torch.zeros((3*num_particles,)).cuda()
        initial_guess_lbfgs = return_val[0]

        # print('pytorch SGD')
        # pytorch_sgd_time[i], return_val = get_function_runtime(pytorch_magnetization_augmented_lagrangian_stochastic_optimization,nrepeats,initial_guess_sgd,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='SGD')
        # initial_guess_sgd = return_val[0]
        # print('pytorch Adam')
        # pytorch_adam_time[i], return_val = get_function_runtime(pytorch_magnetization_augmented_lagrangian_stochastic_optimization,nrepeats,initial_guess_adam,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='Adam')
        # initial_guess_adam = return_val[0]
        # print('pytorch AdamW')
        # pytorch_adamw_time[i], return_val = get_function_runtime(pytorch_magnetization_augmented_lagrangian_stochastic_optimization,nrepeats,initial_guess_adamw,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='AdamW')
        # initial_guess_adamw = return_val[0]

    for i, Hext in enumerate(Hext_series):
        Hext = np.float32(Hext)
        hext = Hext/Ms
        print('fixed point iteration method')
        fixedpoint_runtime, _ = get_function_runtime(get_normalized_magnetization_fixed_point_iteration,nrepeats,cp.asarray(hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)
        # fixedpoint_runtime, _ = get_function_runtime(get_magnetization_fixed_point_iteration,nrepeats,cp.asarray(Hext,dtype=cp.float32),particles,device_particle_posns,Ms,chi,particle_volume,l_e,max_iters)

    # for i, Hext in enumerate(Hext_series):
    #     hext = Hext/Ms
    #     print('cpu bound bfgs with gradient')
    #     bfgs_wgrad_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers_cpu,method='BFGS')
    #     lagrange_multipliers_cpu = np.zeros((3*num_particles,))

    #     print('cpu bound lbfgs with gradient')
    #     lbfgs_wgrad_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization_w_grad,nrepeats,initial_guess_cpu_grad,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers_cpu,method='L-BFGS-B')
    #     lagrange_multipliers_cpu = np.zeros((3*num_particles,))

    # for i, Hext in enumerate(Hext_series):
    #     hext = Hext/Ms
    #     # print('cpu bound bfgs without gradient')
    #     # bfgs_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization,nrepeats,initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers_cpu,method='BFGS')
    #     lagrange_multipliers_cpu = np.zeros((3*num_particles,))

    #     print('cpu bound lbfgs without gradient')
    #     lbfgs_time, _ = get_function_runtime(solve_magnetization_augmented_lagrangian_optimization,nrepeats,initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers_cpu,method='L-BFGS-B')
    #     lagrange_multipliers_cpu = np.zeros((3*num_particles,))

    fig, ax = plt.subplots()
    ax.plot(Bext_series_magnitude,pytorch_lbfgs_time)
    ax.set_xlabel('Applied Field (T)')
    ax.set_ylabel('time (s)')
    ax.set_title(f'Over {nrepeats} evaluations')

    plt.show(block=False)

    # for plotting and analyzing results of different methods
    initial_guess_lbfgs = initial_guess_cpu.copy()
    initial_guess_sgd = initial_guess_lbfgs.copy()
    initial_guess_adam = initial_guess_lbfgs.copy()
    initial_guess_adamw = initial_guess_lbfgs.copy()
    for i, Hext in enumerate(Hext_series):
        Hext = np.float32(Hext)
        hext = torch.asarray(Hext/Ms).cuda()
        initial_guess_lbfgs, lbfgs_auglang_constraint_violation[i] = pytorch_magnetization_augmented_lagrangian_optimization(initial_guess_lbfgs,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers)
        lbfgs_auglang_magnetization[i] = initial_guess_lbfgs[:]

        # initial_guess_sgd, sgd_auglang_constraint_violation[i] = pytorch_magnetization_augmented_lagrangian_stochastic_optimization(initial_guess_sgd,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='SGD')
        # sgd_auglang_magnetization[i] = initial_guess_sgd[:]

        # initial_guess_adam, adam_auglang_constraint_violation[i] = pytorch_magnetization_augmented_lagrangian_stochastic_optimization(initial_guess_adam,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='Adam')
        # adam_auglang_magnetization[i] = initial_guess_adam[:]

        # initial_guess_adamw, adamw_auglang_constraint_violation[i] = pytorch_magnetization_augmented_lagrangian_stochastic_optimization(initial_guess_adamw,separation_vectors,separation_vectors_inv_magnitude,hext,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers,atol=1e-3,rtol=1e-3,method='AdamW')
        # adamw_auglang_magnetization[i] = initial_guess_adamw[:]

        # lagrange_multipliers = torch.zeros((3*num_particles,)).cuda()
        hext_cpu = Hext/Ms
        lagrange_multipliers_cpu = np.zeros((3*num_particles,))
        res = solve_magnetization_augmented_lagrangian_optimization(initial_guess_cpu,separation_vectors,separation_vectors_inv_magnitude,hext_cpu,chi,particle_volume,l_e,num_particles,penalty_constant,lagrange_multipliers_cpu,method='L-BFGS-B')
        scipy_lbfgs_magnetization[i] = res.x
        initial_guess_cpu = scipy_lbfgs_magnetization[i].copy()


    lbfgs_auglang_magnetization = lbfgs_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))
    host_lbfgs_magnetization = torch.Tensor.cpu(lbfgs_auglang_magnetization)
    host_lbfgs_magnetization = host_lbfgs_magnetization.detach().numpy()

    lbfgs_auglang_constraint_violation = lbfgs_auglang_constraint_violation.reshape((Hext_series.shape[0],num_particles,3))
    host_lbfgs_constraint_violation = torch.Tensor.cpu(lbfgs_auglang_constraint_violation)
    host_lbfgs_constraint_violation = host_lbfgs_constraint_violation.detach().numpy()

    # sgd_auglang_magnetization = sgd_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))
    # host_sgd_magnetization = torch.Tensor.cpu(sgd_auglang_magnetization)
    # host_sgd_magnetization = host_sgd_magnetization.detach().numpy()

    # sgd_auglang_constraint_violation = sgd_auglang_constraint_violation.reshape((Hext_series.shape[0],num_particles,3))
    # host_sgd_constraint_violation = torch.Tensor.cpu(sgd_auglang_constraint_violation)
    # host_sgd_constraint_violation = host_sgd_constraint_violation.detach().numpy()

    # adam_auglang_magnetization = adam_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))
    # host_adam_magnetization = torch.Tensor.cpu(adam_auglang_magnetization)
    # host_adam_magnetization = host_adam_magnetization.detach().numpy()

    # adam_auglang_constraint_violation = adam_auglang_constraint_violation.reshape((Hext_series.shape[0],num_particles,3))
    # host_adam_constraint_violation = torch.Tensor.cpu(adam_auglang_constraint_violation)
    # host_adam_constraint_violation = host_adam_constraint_violation.detach().numpy()

    # adamw_auglang_magnetization = adamw_auglang_magnetization.reshape((Hext_series.shape[0],num_particles,3))
    # host_adamw_magnetization = torch.Tensor.cpu(adamw_auglang_magnetization)
    # host_adamw_magnetization = host_adamw_magnetization.detach().numpy()

    # adamw_auglang_constraint_violation = adamw_auglang_constraint_violation.reshape((Hext_series.shape[0],num_particles,3))
    # host_adamw_constraint_violation = torch.Tensor.cpu(adamw_auglang_constraint_violation)
    # host_adamw_constraint_violation = host_adamw_constraint_violation.detach().numpy()

    scipy_lbfgs_magnetization = scipy_lbfgs_magnetization.reshape((Hext_series.shape[0],num_particles,3))

    pytorch_lbfgs_energy = np.zeros((Hext_series.shape[0],))
    scipy_lbfgs_energy = np.zeros((Hext_series.shape[0],))
    # pytorch_sgd_energy = np.zeros((Hext_series.shape[0],))
    # pytorch_adam_energy = np.zeros((Hext_series.shape[0],))
    # pytorch_adamw_energy = np.zeros((Hext_series.shape[0],))

    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)
    dipole_field_kernel = cp.asnumpy(dipole_field_kernel)

    for i in range(Hext_series.shape[0]):
        hext = Hext_series[i]/Ms
        pytorch_lbfgs_energy[i] = get_normalized_total_magnetic_energy_w_kernel(host_lbfgs_magnetization[i].reshape((3*num_particles,)),hext,dipole_field_kernel,chi,num_particles)
        # pytorch_sgd_energy[i] = get_normalized_total_magnetic_energy_w_kernel(host_sgd_magnetization[i].reshape((3*num_particles,)),hext,dipole_field_kernel,chi,num_particles)
        # pytorch_adam_energy[i] = get_normalized_total_magnetic_energy_w_kernel(host_adam_magnetization[i].reshape((3*num_particles,)),hext,dipole_field_kernel,chi,num_particles)
        # pytorch_adamw_energy[i] = get_normalized_total_magnetic_energy_w_kernel(host_adamw_magnetization[i].reshape((3*num_particles,)),hext,dipole_field_kernel,chi,num_particles)
        scipy_lbfgs_energy[i] = get_normalized_total_magnetic_energy_w_kernel(scipy_lbfgs_magnetization[i].reshape((3*num_particles,)),hext,dipole_field_kernel,chi,num_particles)

    fig, ax = plt.subplots()
    ax.plot(Bext_series_magnitude,pytorch_lbfgs_energy,label='pytorch lbfgs')
    ax.plot(Bext_series_magnitude,scipy_lbfgs_energy,label='scipy lbfgs')
    # ax.plot(Bext_series_magnitude,pytorch_sgd_energy,label='pytorch sgd')
    # ax.plot(Bext_series_magnitude,pytorch_adam_energy,label='pytorch adam')
    # ax.plot(Bext_series_magnitude,pytorch_adamw_energy,label='pytorch adamw')
    ax.set_xlabel('Applied Field (T)')
    ax.set_ylabel('Energy Normalized')
    fig.legend()
    plt.show(block=False)
    
    fig, axs = plt.subplots(2,3)
    for count in range(particles.shape[0]):
        axs[0,0].plot(Bext_series_magnitude,host_lbfgs_magnetization[:,count,0])
        axs[0,1].plot(Bext_series_magnitude,host_lbfgs_magnetization[:,count,1])
        axs[0,2].plot(Bext_series_magnitude,host_lbfgs_magnetization[:,count,2])
        axs[1,0].plot(Bext_series_magnitude,host_lbfgs_constraint_violation[:,count,0])
        axs[1,1].plot(Bext_series_magnitude,host_lbfgs_constraint_violation[:,count,1])
        axs[1,2].plot(Bext_series_magnitude,host_lbfgs_constraint_violation[:,count,2])

    axs[1,0].set_xlabel('Applied Field (T)')
    axs[1,1].set_xlabel('Applied Field (T)')
    axs[1,2].set_xlabel('Applied Field (T)')
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[0,0].set_ylabel('optimizer lbfgs mag')
    axs[1,0].set_ylabel('constraint violation')

    plt.show(block=False)

    # xdata = Bext_series_magnitude#[::-1]
    # plot_soln_and_constraint_violation(host_sgd_magnetization,host_sgd_constraint_violation,xdata,'sgd')
    # plot_soln_and_constraint_violation(host_adam_magnetization,host_adam_constraint_violation,xdata,'Adam')
    # plot_soln_and_constraint_violation(host_adamw_magnetization,host_adamw_constraint_violation,xdata,'AdamW')


    field_index = 0

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.set_title('Normalized Magnetic Moment vectors (iterative)')
    ax.set_xlabel('X (l_e)')
    ax.set_ylabel('Y (l_e)')
    ax.set_zlabel('Z (l_e)')
    ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],host_lbfgs_magnetization[field_index,:,0],host_lbfgs_magnetization[field_index,:,1],host_lbfgs_magnetization[field_index,:,2],pivot='middle',length=10.0)
    # ax.quiver(particle_posns[:,0],particle_posns[:,1],particle_posns[:,2],seeded_iterative_magnetization[field_index,-1,:,0],seeded_iterative_magnetization[field_index,-1,:,1],seeded_iterative_magnetization[field_index,-1,:,2],pivot='middle',units='width')
    plt.show(block=False)
    print('end of function')

if __name__ == "__main__":
    # main()
    # gpu_testing()
    # gpu_testing_force_calc()
    
    # iterative_magnetization_w_drag_testing()
    # iterative_magnetization_testing()

    # linear_magnetization_testing()

    # nonlinear_magnetization_fsolve()
    # pbc_nonlinear_magnetization_fsolve()

    # nonlinear_magnetization_energy_minimization_w_constraint()

    # performance_testing_nonlinear_magnetization_energy_minimization_w_constraint()

    # test_optimizer_bootstrapped_fixedpoint_solver()
    pytorch_optimization()