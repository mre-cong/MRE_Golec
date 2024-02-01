import numpy as np
import magnetism
import matplotlib.pyplot as plt
import os
import scipy.special as sci
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
    chi = 131
    Ms = 1.9e6
    particle_radius = 1.5e-6
    l_e = 1e-6
    beta = 6.734260376702891e-09
    particle_mass_density = 7.86 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    particle_mass = particle_mass_density*((4/3)*np.pi*(particle_radius**3))
    # particle_posns = np.array([[0,0,0]],dtype=np.float64)
    particle_posns = np.array([[0,0,0],[9,0,0]],dtype=np.float64)
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
    savename = save_dir + f'{num_particles}_particles_magnetization_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
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
    savename = save_dir + f'{num_particles}_particles_mag_forces_64bit_vs_32bit_separation_{separation}_Bext_angle_theta_{Bext_theta_angle}_phi_{Bext_phi_angle}.png'
    plt.savefig(savename)

if __name__ == "__main__":
    main()