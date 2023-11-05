"""
Created on Tues October 3 10:20:19 2023

@author: David Marchfield
"""
import numpy as np
import matplotlib.pyplot as plt
import get_volume_correction_force_cy_nogil
import get_spring_force_cy
import magnetism

def get_accel_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag=10,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    # drag = 20
    bc_forces = np.zeros(x0.shape,dtype=float)
    if bc[0] == 'stress':
        for surface in bc[1]:
            # stress times surface area divided by number of vertices on the surface (resulting in the appropriate stress being applied)
            # !!! it seems likely that this is inappropriate, that for each element in the surface, the vertices need to be counted in a way that takes into account vertices shared by elements. right now the even distribution of force but uneven assignment of stiffnesses based on vertices belonging to multple elements means the edges will push in further than the central vertices on the surface... but let's move forward with this method first and see how it does
            if surface == 'left' or surface == 'right':
                surface_area = dimensions[0]*dimensions[2]
            elif surface == 'top' or surface == 'bottom':
                surface_area = dimensions[0]*dimensions[1]
            else:
                surface_area = dimensions[1]*dimensions[2]
            # assuming tension force only, no compression
            if surface == 'right':
                force_direction = np.array([1,0,0])
            elif surface == 'left':
                force_direction = np.array([-1,0,0])
            elif surface == 'top':
                force_direction = np.array([0,0,1])
            elif surface == 'bottom':
                force_direction = np.array([0,0,-1])
            elif surface == 'front':
                force_direction = np.array([0,1,0])
            elif surface == 'back':
                force_direction = np.array([0,-1,0])
            # i need to distinguish between vertices that exist on the corners, edges, and the rest of the vertices on the boundary surface to adjust the force. I also need to understand how to distribute the force. I want to have a sum of forces such that the stress applied is correct, but i need to corners to have a lower magnitude force vector exerted due to the weaker spring stiffness, the edges to have a force magnitude greater than the corners but less than the center
            bc_forces[boundaries[surface]] = force_direction*bc[2]/len(boundaries[surface])*surface_area
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
    else:
        fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    accel = set_fixed_nodes(accel,fixed_nodes)
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
        mag_forces *= beta/(particle_mass*(l_e**4))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
        for particle in particles:
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
    if debug_flag:
        inspect_vcf = volume_correction_force[particles[0],:]
        inspect_springWCA = spring_force[particles[0],:]
        inspect_particle = accel[particles[0],:]
    return accel

def get_particle_center(particle_nodes,node_posns):
    particle_node_posns = node_posns[particle_nodes,:]
    x_max = np.max(particle_node_posns[:,0])
    y_max = np.max(particle_node_posns[:,1])
    z_max = np.max(particle_node_posns[:,2])
    x_min = np.min(particle_node_posns[:,0])
    y_min = np.min(particle_node_posns[:,1])
    z_min = np.min(particle_node_posns[:,2])
    particle_center = np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2],dtype=np.float64)
    return particle_center

def set_fixed_nodes(accel,fixed_nodes):
    for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
        #TODO almost certainly faster to remove the inner loop and just set each value to 0 in order, or using python semantics, just set the row to zero?
        for j in range(3):
            accel[fixed_nodes[i],j] = 0
    return accel

def get_accel_scaled_no_fixed_nodes(y,elements,springs,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    # if bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
    #     #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
    #     if bc[1][0] == 'x':
    #         fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
    #     elif bc[1][0] == 'y':
    #         fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
    #     elif bc[1][0] == 'z':
    #         fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
    # else:
    #     fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
        mag_forces *= beta/(particle_mass*(l_e**4))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            total_torque[i,:] = get_torque_on_particle(particle,accel,x0)
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
    return accel, total_torque

def get_torque_on_particle(particle,accel,node_posns):
    """Given the list of particle nodes, the accelerations and positions of the nodes, calculate the torque acting on the particle relative to the center of the particle"""
    particle_center = get_particle_center(particle,node_posns)
    nodal_torques = np.zeros((particle.shape[0],3))
    nodal_torques = np.cross(node_posns[particle,:] - particle_center,accel[particle,:])
    total_torque = np.sum(nodal_torques,axis=1)
    return total_torque

def get_accel_scaled_alt(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    drag = 20
    bc_forces = np.zeros(x0.shape,dtype=float)
    if bc[0] == 'stress':
        for surface in bc[1]:
            # stress times surface area divided by number of vertices on the surface (resulting in the appropriate stress being applied)
            # !!! it seems likely that this is inappropriate, that for each element in the surface, the vertices need to be counted in a way that takes into account vertices shared by elements. right now the even distribution of force but uneven assignment of stiffnesses based on vertices belonging to multple elements means the edges will push in further than the central vertices on the surface... but let's move forward with this method first and see how it does
            if surface == 'left' or surface == 'right':
                surface_area = dimensions[0]*dimensions[2]
            elif surface == 'top' or surface == 'bottom':
                surface_area = dimensions[0]*dimensions[1]
            else:
                surface_area = dimensions[1]*dimensions[2]
            # assuming tension force only, no compression
            if surface == 'right':
                force_direction = np.array([1,0,0])
            elif surface == 'left':
                force_direction = np.array([-1,0,0])
            elif surface == 'top':
                force_direction = np.array([0,0,1])
            elif surface == 'bottom':
                force_direction = np.array([0,0,-1])
            elif surface == 'front':
                force_direction = np.array([0,1,0])
            elif surface == 'back':
                force_direction = np.array([0,-1,0])
            # i need to distinguish between vertices that exist on the corners, edges, and the rest of the vertices on the boundary surface to adjust the force. I also need to understand how to distribute the force. I want to have a sum of forces such that the stress applied is correct, but i need to corners to have a lower magnitude force vector exerted due to the weaker spring stiffness, the edges to have a force magnitude greater than the corners but less than the center
            bc_forces[boundaries[surface]] = force_direction*bc[2]/len(boundaries[surface])*surface_area
    elif bc[0] == 'strain':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
        for surface in bc[1]:
            pass
    else:
        fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)

    volume_correction_force_alt = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,scaled_kappa,correction_force_el,vectors,avg_vectors, volume_correction_force_alt)
    spring_force_alt = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, scaled_springs_var, spring_force_alt)

    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]

    volume_correction_force_alt *= m_ratio[:,np.newaxis]
    spring_force_alt *= m_ratio[:,np.newaxis]

    vcf_correctness = np.allclose(volume_correction_force,volume_correction_force_alt)
    springs_correctness = np.allclose(spring_force,spring_force_alt)
    print(f'Volume Correction Forces agreeing:{vcf_correctness}')
    print(f'Spring Forces Agreeing:{springs_correctness}')

    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
    mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
    mag_forces_alt = mag_forces * scaled_magnetic_force_coefficient
    mag_forces *= beta/(particle_mass*(l_e**4))

    mag_correctness = np.allclose(mag_forces,mag_forces_alt)
    print(f'magnetic forces agreeing:{mag_correctness}')

    if not vcf_correctness or not springs_correctness or not mag_correctness:
        indices_of_interest = np.nonzero(np.logical_not(np.isclose(spring_force,spring_force_alt)))
        for i in range(len(indices_of_interest[0])):
            index = indices_of_interest[0][i]
            print(f'for index {index}')
            for j in range(3):
                print(f'original force:{spring_force[index,j]}')
            for j in range(3):
                print(f'alternative force:{spring_force_alt[index,j]}')
        possible_springs_connection = np.nonzero(np.isclose(springs[:,0],indices_of_interest[0][0]))
        possible_springs_connection2 = np.nonzero(np.isclose(springs[:,1],indices_of_interest[0][0]))
        possible_springs_connection_alt = np.nonzero(np.isclose(scaled_springs_var[:,0],indices_of_interest[0][0]))
        possible_springs_connection_alt2 = np.nonzero(np.isclose(scaled_springs_var[:,1],indices_of_interest[0][0]))
        posn1 = x0[indices_of_interest[0][0]]
        posn2 = x0[indices_of_interest[0][3]]
        inspecting_springs = springs[possible_springs_connection,:]
        inspecting_springs2 = springs[possible_springs_connection2,:]
        inspecting_springs_alt = scaled_springs_var[possible_springs_connection_alt,:]
        inspecting_springs_alt2 = scaled_springs_var[possible_springs_connection_alt2,:]
        original_vector_sum = np.zeros((3,))
        original_force_vecsum = np.zeros((3,))
        alt_vector_sum = np.zeros((3,))
        alt_force_vecsum = np.zeros((3,))
        for count in range(inspecting_springs.shape[0]):
            i = np.squeeze(inspecting_springs)[count]
            posn1 = x0[int(i[0])]
            posn2 = x0[int(i[1])]
            force_magnitude = i[2]*(np.linalg.norm(posn1 - posn2) - i[3])
            force_i = -1*force_magnitude*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i = force_i*l_e*beta_i[int(i[0])]
            original_vector_sum += accel_i
            original_force_vecsum += force_i
            i_alt = np.squeeze(inspecting_springs_alt)[count]
            force_magnitude_alt = i_alt[2]*(np.linalg.norm(posn1 - posn2) - i_alt[3])
            force_i_alt = -1*force_magnitude_alt*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i_alt = force_i_alt*m_ratio[int(i[0])]
            alt_vector_sum += accel_i_alt
            alt_force_vecsum += force_i_alt
            if not np.allclose(accel_i,accel_i_alt):
                print(f'disagreement for spring force scaled accelerations')
        for count in range(inspecting_springs2.shape[0]):
            i = np.squeeze(inspecting_springs2)[count]
            posn1 = x0[int(i[0])]
            posn2 = x0[int(i[1])]
            force_magnitude = i[2]*(np.linalg.norm(posn1 - posn2) - i[3])
            force_i = -1*force_magnitude*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i = force_i*l_e*beta_i[int(i[1])]
            original_vector_sum += -1*accel_i
            original_force_vecsum += -1*force_i
            i_alt = np.squeeze(inspecting_springs_alt2)[count]
            force_magnitude_alt = i_alt[2]*(np.linalg.norm(posn1 - posn2) - i_alt[3])
            force_i_alt = -1*force_magnitude_alt*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i_alt = force_i_alt*m_ratio[int(i[1])]
            alt_vector_sum += -1*accel_i_alt
            alt_force_vecsum += -1*force_i_alt
            if not np.allclose(accel_i,accel_i_alt):
                print(f'disagreement for spring force scaled accelerations')
        if not np.allclose(original_vector_sum,alt_vector_sum):
            print("vector sums of scaled accelerations due to spring forces don't agree")
        if not np.allclose(original_force_vecsum*l_e*beta_i[int(i[1])],alt_force_vecsum*m_ratio[int(i[1])]):
            print("vector sums of scaled accelerations due to spring forces don't agree when summing forces before finishing scaling")
        pass

    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    if debug_flag:
        inspect_vcf = volume_correction_force[particles[0],:]
        inspect_springWCA = spring_force[particles[0],:]
        inspect_particle = accel[particles[0],:]
    return accel