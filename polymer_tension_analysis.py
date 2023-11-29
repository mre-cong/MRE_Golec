#2023-10-31
#David Marchfield
#For the no particle, polymer only tension strain case, do the analysis (stress-strain curves, effective modulus, strain tensor, stress tensor, displacement vector) and make publication quality plots and visualizations
#potential to calculate a "stress" tensor from from scaled accelerations, calculate forces and then stress tensor field. but thinking about it now, that is really a traction tensor, which is the result of the dot product of the strain tensor with the surface normal. not enough information (underdetermined system of equations) for getting 6 tensor components, because the unit normal is along the cartesian axes in the Lagrangian (reference configuration) description. in the Eulerian (current configuration) description I would need to use the vertices to define the faces and then describe the unit normal. Unless i thin about the forces on the node really acting on the surfaces of an infinitesimal volume element/cube

#TODO because i am dealing with a scaled system, i need to do conversions, either before calculating or after calculating relevant quantities (displacement, displacement gradient, etc.)

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import get_volume_correction_force_cy_nogil
import simulate
import re
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

def plot_element(node_posns,element,springs):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(node_posns[element,0],node_posns[element,1],node_posns[element,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,node_posns,element,springs,spring_color='b')
    plt.show()


#grab function from checkpoint_analysis for getting simulation parameters.
#write separate functions for effective modulus calculation for shearing, rotation, and tension/compression
#write function that calls the separate effective modulus calculation functions based on the series_descriptor
#use series variable in the for i in range() loops
#visualize surfaces and center cuts
#plot linear and nonlinear strain tensors for surfaces and center cuts
#plot stress tensor for surfaces and center cuts
#will need to introduce some form of visualization through the whole volume (all cuts),
# and some form of visualziaiton of the same cut at different strains/fields/timesteps

def main():
    """Read in and perform analysis and visualization workflow on the polymer only (no particle) tension simulation."""
    #First, read in the init file and get the necessary variables from the params "struct"
    #Second, inside a loop over the strains, calculate the necessary vectors and tensors (displacement, deformation gradient tensor, strain tensors (linear and nonlienar), stress tensor (linear), the stress-strain curves based on the applied strains and forces on the boundaries the strain boundary conditions are applied to, and the effective modulus. Plot and save out figures.)
    drag = 10
    discretization_order = 1

    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-08_strain_testing_tension_order_1_drag_20/'
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-08_strain_testing_compression_order_1_drag_20/'
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-15_strain_testing_shearing_order_1_drag_20/'
    sim_dir = output_dir
    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(output_dir+'init.h5')
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir()]
    for i in range(len(subfolders)):
        final_posns, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        # mre.analyze.plot_outer_surfaces_wireframe(initial_node_posns,final_posns,boundary_conditions,output_dir,tag=f"")
        mre.analyze.plot_outer_surfaces(initial_node_posns,final_posns,boundary_conditions,sim_dir,tag=f'{i}')
    
    effective_modulus, stress, strains, secondary_stress = get_effective_modulus(sim_dir)

    force_component = {'x':0,'y':1,'z':2}
    fig, axs = plt.subplots(2)
    axs[0].plot(strains,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
    axs[0].set_title('Stress versus Strain')
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('stress')
    axs[1].plot(strains,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus versus strain')
    axs[1].set_xlabel('strain')
    axs[1].set_ylabel('Effective Modulus')
    plt.show()

    fig, axs = plt.subplots(2)
    axs[0].plot(strains,secondary_stress[:,force_component[strain_direction[1]]],'-o')
    axs[0].set_title('Stress versus Strain')
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('second surface stress')
    force_component = {'x':0,'y':1,'z':2}
    axs[1].plot(strains,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus versus strain')
    axs[1].set_xlabel('strain')
    axs[1].set_ylabel('Effective Modulus')
    plt.show()

    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(output_dir+'init.h5')

    for i in range(len(params[0])):
        if params.dtype.descr[i][0] == 'num_elements':
            num_elements = params[0][i]
            num_nodes = num_elements + 1
        if params.dtype.descr[i][0] == 'poisson_ratio':
            nu = params[0][i]
        if params.dtype.descr[i][0] == 'young_modulus':
            E = params[0][i]
        if params.dtype.descr[i][0] == 'kappa':
            kappa = params[0][i]
        if params.dtype.descr[i][0] == 'scaling_factor':
            beta = params[0][i]
        if params.dtype.descr[i][0] == 'element_length':
            l_e = params[0][i]
        if params.dtype.descr[i][0] == 'particle_mass':
            particle_mass = params[0][i]
        if params.dtype.descr[i][0] == 'particle_radius':
            particle_radius = params[0][i]
        if params.dtype.descr[i][0] == 'particle_Ms':
            Ms = params[0][i]
        if params.dtype.descr[i][0] == 'particle_chi':
            chi = params[0][i]
        if params.dtype.descr[i][0] == 'drag':
            drag = params[0][i]
        if params.dtype.descr[i][0] == 'characteristic_time':
            characteristic_time = params[0][i]

    #getting the Lame parameters from Young's modulus and poisson ratio. see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))

    dimensions = (l_e*np.max(initial_node_posns[:,0]),l_e*np.max(initial_node_posns[:,1]),l_e*np.max(initial_node_posns[:,2]))
    beta_i = beta/node_mass
    strain_max = 0.20
    n_strain_steps = 21
    if n_strain_steps == 1:
        strain_step_size = strain_max
    else:  
        strain_step_size = strain_max/(n_strain_steps-1)
    strains = np.arange(0.0,strain_max+0.01*strain_max,strain_step_size)
    strain_direction = ('x','x')
    stress = np.zeros((n_strain_steps,3))#np.zeros((len(series),))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    # loop over and read in the output files to do analysis and visualization
    for i in range(n_strain_steps):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(output_dir+f'output_{i}.h5')
        Hext = applied_field
        
        #TODO adjust the read output file function to return a proper boundary conditions variable
        boundary_conditions = (str(boundary_conditions[0][0]),(str(boundary_conditions[0][1]),str(boundary_conditions[0][2])),boundary_conditions[0][3])
        # getting the residual accelerations plotted? maybe not part of the official workflow
        # a_var = simulate.get_accel_scaled(checkpoint_solution,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)

        #getting the accelerations of the boundaries to determine what would be necessary forces to hold them in place
        N_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
        y = np.zeros((6*N_nodes,))
        y[:3*N_nodes] = np.reshape(final_posns,(3*N_nodes,))
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag=10)
        # need to convert the acceleration variable back to forces acting on each node, at least for the boundaries of interest when doing the effective modulus calculation
        #need to decide which boundaries are involved, using the strain_type variable and strain_direction variables. can check both boundaries to see how they compare with regards to the forces/stress involved
        # if strain_type == 'tension' or strain_type == 'compression':
        #     if strain_direction[0] == 'x':

        #forces that must act on the boundaries for them to be in this position
        first_bdry_forces = -1*end_accel[boundaries['left']]/beta_i[boundaries['left'],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries['right']]/beta_i[boundaries['right'],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[1]*dimensions[2])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[1]*dimensions[2])
        print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[0])-np.abs(second_bdry_stress[0])}')
        stress[i] = first_bdry_stress
        secondary_stress[i] = second_bdry_stress

    force_component = {'x':0,'y':1,'z':2}
    fig, axs = plt.subplots(2)
    axs[0].plot(strains,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
    axs[0].set_title('Stress versus Strain')
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('stress')
    for i in range(np.shape(strains)[0]):
        if strains[i] == 0 and np.isclose(np.linalg.norm(stress[i,:]),0):
            effective_modulus[i] = E
        else:
            effective_modulus[i] = np.abs(stress[i,force_component[strain_direction[1]]]/strains[i])
    axs[1].plot(strains,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus versus strain')
    axs[1].set_xlabel('strain')
    axs[1].set_ylabel('Effective Modulus')
    plt.show()

    fig, axs = plt.subplots(2)
    axs[0].plot(strains,secondary_stress[:,force_component[strain_direction[1]]],'-o')
    axs[0].set_title('Stress versus Strain')
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('second surface stress')
    force_component = {'x':0,'y':1,'z':2}
    for i in range(np.shape(strains)[0]):
        if strains[i] == 0 and np.isclose(np.linalg.norm(secondary_stress[i,:]),0):
            effective_modulus[i] = E
        else:
            effective_modulus[i] = np.abs(secondary_stress[i,force_component[strain_direction[1]]]/strains[i])
    axs[1].plot(strains,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus versus strain')
    axs[1].set_xlabel('strain')
    axs[1].set_ylabel('Effective Modulus')
    plt.show()
        # # visualize residual acceleration with pcolormesh or i guess the scatter plot with color for depth could be modified
        # # use argsort to find the strongest acceleration norm nodes, and find out how many of them there are, and where they are (so that you can visualize the parts of the simulation volume that have high accelerations)

        # index = 4
        # cut_type = 'xy'
        # subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,end_accel,index,output_dir,tag="residual_accelerations")

        # displacement = get_displacement_field(initial_node_posns,final_posns)

        
        # xdisplacement_3D, ydisplacement_3D, zdisplacement_3D = get_component_3D_arrays(displacement,num_nodes)
        # xdisplacement_gradient = np.gradient(xdisplacement_3D)
        # ydisplacement_gradient = np.gradient(ydisplacement_3D)
        # zdisplacement_gradient = np.gradient(zdisplacement_3D)
        # # xshape = int(num_nodes[0])
        # # yshape = int(num_nodes[1])
        # # zshape = int(num_nodes[2])
        # # deformation gradient. F = I + \frac{\partial u}{\partial X}, where u is the displacement: u = x - X, where x is the current position, and X is the reference configuration of the point of the body

        # gradu = get_gradu(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,num_nodes)
        # strain_tensor = get_strain_tensor(gradu)
        # green_strain_tensor = get_green_strain_tensor(gradu)

        # stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)

        # index = 13
        # cut_type = 'xy'
        # displacements = (xdisplacement_3D,ydisplacement_3D,zdisplacement_3D)
        # # subplot_cut_pcolormesh_displacement(cut_type,initial_node_posns,displacements,index,output_dir,tag="")

        # subplot_cut_pcolormesh_strain(cut_type,initial_node_posns,green_strain_tensor,index,output_dir,tag="nonlinear")

        # subplot_cut_pcolormesh_strain(cut_type,initial_node_posns,strain_tensor,index,output_dir,tag="")

        # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,index,output_dir,tag="stress")

        # # plot_cut_pcolormesh_displacement(cut_type,initial_node_posns,xdisplacement_3D,index,output_dir,tag="")

        # #i'm fairly confident that i have correctly calculated the gradient of the displacement field, the deformation gradient, and the linear and nonlinear strain tensors. i know that in the case of small displacements the two strain tensors should coincide. i also know that if there were rigid body rotations the strain linear tensor would no longer be useful, and I would need to use the deformation tensor to calculate F F^T (or F^T F) and get V V^T or (U^T U) and then find the square root of the matrix, or else use the iterative algorithm for polar decompositions to go from the deformation gradient to the rotation matrix, then calculate the inverse of the rotation matrix and pre or post multiply F to get V or U. V and U are good strain tensors even for large deformations and rigid body rotations, where the lienar strain tensor can have large differences for the same deformations with and without rigid body rotations. However, the Green strain tensor is supposed to not be corrupted by rigid body rotations.

        # #don't forget that shear terms (off diagonal) of strain are half of the engineering shear values because they are components in a strain tensor

        # #i suppose the next step would be some visualization of the displacement field as a quiver plot and as a color plot (pmeshcolor, pcolormesh?). then I think i need to try and get a stress tensor

        # #I can run strain simulations to try and get effective moduli. to get moduli as a function of position I need to be more thoughtful. the stiffness is a fourth rank tensor, with symmetry arguments, and for isotropic homogeneous media, there are only 2 independent elements. I still need to better understand the symmetry arguments, and see if i can go from the strain functions to stress using the stiffness tensor. or if i can use the forces on the nodes, can i get the stress tensor directly? I need to review my notes from days in the prior few weeks. I can draw imaginary planes through the node positions and then calculate force per (undeformed) area. is that appropriate? the normal example is an infinitessimal cube volume, where the direction and magnitude of forces on opposite faces have to balance to avoid a rotation (angular momentum balance, same for linear momentum (or torque/force))

def get_effective_modulus(sim_dir):
    """Given a simulation directory, calculate and plot the stress-strain curve and effective modulus versus strain."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    #below is a way of getting the subfolders in the simulationd directory, and then a way to traverse/grab the checkpoint files in order. probably don't need the checkpoint stuff, but if i need to traverse the subfolders...
    # with os.scandir(sim_dir) as dirIterator:
    #     subfolders = [f.path for f in dirIterator if f.is_dir()]
    # for subfolder in subfolders:
    #     with os.scandir(subfolder+'/') as dirIterator:
    #         checkpoint_files = [f.path for f in dirIterator if f.is_file() and f.name.startswith('checkpoint')]
    #     for checkpoint_file in checkpoint_files:
    #         fn_w_type = checkpoint_file.split('/')[-1]
    #         fn = fn_w_type.split('.')[0]
    #         checkpoint_number = np.append(checkpoint_number,np.array([int(count) for count in re.findall(r'\d+',fn)]))
    #     sort_indices = np.argsort(checkpoint_number)
    #     checkpoint_file = checkpoint_files[sort_indices[-1]]
    #     solution, applied_field, boundary_conditions, i = mre.initialize.read_checkpoint_file(checkpoint_file)
    #     boundary_conditions = format_boundary_conditions(boundary_conditions)
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain_type = boundary_conditions[0]
    strain_direction = boundary_conditions[1]
    if strain_type == 'tension' or strain_type == 'compression':
        effective_modulus, stress, strains, secondary_stress = get_tension_compression_modulus(sim_dir,strain_direction)
    elif strain_type == 'shearing':
        effective_modulus, stress, strains, secondary_stress = get_shearing_modulus(sim_dir,strain_direction)
    elif strain_type == 'torsion':
        get_torsion_modulus(sim_dir,strain_direction)
    return effective_modulus, stress, strains, secondary_stress


def get_tension_compression_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied."""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    strains = series
    n_strain_steps = len(series)
    stress = np.zeros(n_strain_steps,3)
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    force_component = {'x':0,'y':1,'z':2}
    N_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    y = np.zeros((6*N_nodes,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        Hext = applied_field
        y[:3*N_nodes] = np.reshape(final_posns,(3*N_nodes,))
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
        if strain_direction[0] == 'x':
            #forces that must act on the boundaries for them to be in this position
            relevant_boundaries = ('left','right')
            dimension_indices = (1,2)
        elif strain_direction[0] == 'y':
            relevant_boundaries = ('front','back')
            dimension_indices = (0,2)
        elif strain_direction[0] == 'z':
            relevant_boundaries = ('top','bot')
            dimension_indices = (0,1)
        first_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[1]]]/beta_i[boundaries[relevant_boundaries[1]],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[force_component[strain_direction[1]]])-np.abs(second_bdry_stress[force_component[strain_direction[1]]])}')
        stress[i] = first_bdry_stress
        secondary_stress[i] = second_bdry_stress
    for i in range(np.shape(strains)[0]):
        if strains[i] == 0 and np.isclose(np.linalg.norm(stress[i,:]),0):
            effective_modulus[i] = E
        else:
            effective_modulus[i] = np.abs(stress[i,force_component[strain_direction[1]]]/strains[i])
    return effective_modulus, stress, strains, secondary_stress

def get_shearing_modulus(sim_dir,strain_direction):
    """Calculate a shear modulus, using the shear strain (shearing angle) and the force applied to the sheared surface in the shearing direction to get a shear stress."""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    #nonlinear shear strains are defined as tangent of the angle opened up by the shearing. linear shear strain is the linear, small angle approximation of tan theta ~= theta
    strains = np.tan(series)
    n_strain_steps = len(series)
    stress = np.zeros((n_strain_steps,3))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    force_component = {'x':0,'y':1,'z':2}
    N_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    y = np.zeros((6*N_nodes,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        Hext = applied_field
        y[:3*N_nodes] = np.reshape(final_posns,(3*N_nodes,))
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
        if strain_direction[0] == 'x':
            relevant_boundaries = ('left','right')
            dimension_indices = (1,2)
            if strain_direction[1] == 'y':
                pass
            elif strain_direction[1] == 'z':
                pass
        elif strain_direction[0] == 'y':
            relevant_boundaries = ('front','back')
            dimension_indices = (0,2)
            if strain_direction[1] == 'x':
                pass
            elif strain_direction[1] == 'z':
                pass
        elif strain_direction[0] == 'z':
            relevant_boundaries = ('top','bot')
            dimension_indices = (0,1)
            if strain_direction[1] == 'x':
                pass
            elif strain_direction[1] == 'y':
                pass
        first_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[1]]]/beta_i[boundaries[relevant_boundaries[1]],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[force_component[strain_direction[1]]])-np.abs(second_bdry_stress[force_component[strain_direction[1]]])}')
        stress[i] = first_bdry_stress
        secondary_stress[i] = second_bdry_stress
    for i in range(np.shape(strains)[0]):
        if strains[i] == 0 and np.isclose(np.linalg.norm(stress[i,:]),0):
            effective_modulus[i] = E/3
        else:
            effective_modulus[i] = np.abs(stress[i,force_component[strain_direction[1]]]/strains[i])
    return effective_modulus, stress, strains, secondary_stress

def get_torsion_modulus(sim_dir,strain_direction):
    """Calculate a torsion modulus, in analogy to the shearing modulus, where strain is defined by the twist angle. torsion forces don't quite make sense, so instead the correct calculation is the torque applied to the surface divided by the twist angle (or the derivative of torque wrt the twist angle)"""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if strain_direction[1] == 'CW':
        pass
    elif strain_direction[1] == 'CCW':
        pass

def format_boundary_conditions(boundary_conditions):
    boundary_conditions = (str(boundary_conditions[0][0])[1:],(str(boundary_conditions[0][1])[1:],str(boundary_conditions[0][2])[1:]),boundary_conditions[0][3])
    boundary_conditions = (boundary_conditions[0][1:-1],(boundary_conditions[1][0][1:-1],boundary_conditions[1][1][1:-1]),boundary_conditions[2])
    return boundary_conditions

def read_in_simulation_parameters(sim_dir):
    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(sim_dir+'init.h5')

    for i in range(len(params[0])):
        if params.dtype.descr[i][0] == 'num_elements':
            num_elements = params[0][i]
            num_nodes = num_elements + 1
        if params.dtype.descr[i][0] == 'poisson_ratio':
            nu = params[0][i]
        if params.dtype.descr[i][0] == 'young_modulus':
            E = params[0][i]
        if params.dtype.descr[i][0] == 'kappa':
            kappa = params[0][i]
        if params.dtype.descr[i][0] == 'scaling_factor':
            beta = params[0][i]
        if params.dtype.descr[i][0] == 'element_length':
            l_e = params[0][i]
        if params.dtype.descr[i][0] == 'particle_mass':
            particle_mass = params[0][i]
        if params.dtype.descr[i][0] == 'particle_radius':
            particle_radius = params[0][i]
        if params.dtype.descr[i][0] == 'particle_Ms':
            Ms = params[0][i]
        if params.dtype.descr[i][0] == 'particle_chi':
            chi = params[0][i]
        if params.dtype.descr[i][0] == 'drag':
            drag = params[0][i]
        if params.dtype.descr[i][0] == 'characteristic_time':
            characteristic_time = params[0][i]

    dimensions = (l_e*np.max(initial_node_posns[:,0]),l_e*np.max(initial_node_posns[:,1]),l_e*np.max(initial_node_posns[:,2]))
    beta_i = beta/node_mass
    N_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    k = mre.initialize.get_spring_constants(E, l_e)
    k = np.array(k)

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions

def get_isotropic_medium_stress(shear_modulus,lame_lambda,strain):
    """stress for homogeneous isotropic material defined by Hooke's law in 3D"""
    stress = np.zeros((np.shape(strain)))
    #print(f'The shape of the result of the trace function on the strain tensor variable is {np.shape(np.trace(strain,axis1=3,axis2=4))}')
    stress[:,:,:,0,0] = 2*shear_modulus*strain[:,:,:,0,0] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,1,1] = 2*shear_modulus*strain[:,:,:,1,1] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,2,2] = 2*shear_modulus*strain[:,:,:,2,2] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,0,1] = 2*shear_modulus*strain[:,:,:,0,1]
    stress[:,:,:,0,2] = 2*shear_modulus*strain[:,:,:,0,2]
    stress[:,:,:,1,2] = 2*shear_modulus*strain[:,:,:,1,2]
    stress[:,:,:,1,0] = 2*shear_modulus*strain[:,:,:,1,0]
    stress[:,:,:,2,0] = 2*shear_modulus*strain[:,:,:,2,0]
    stress[:,:,:,2,1] = 2*shear_modulus*strain[:,:,:,2,1]
    return stress
    
def get_strain_tensor(gradu):
    """Calculate the symmetric, linear strain tensor at each point on the grid representing the initial node positions using the gradient of the displacement field."""
    strain_tensor = np.zeros(np.shape(gradu))
    strain_tensor[:,:,:,0,0] = gradu[:,:,:,0,0]
    strain_tensor[:,:,:,1,1] = gradu[:,:,:,1,1]
    strain_tensor[:,:,:,2,2] = gradu[:,:,:,2,2]
    strain_tensor[:,:,:,0,1] = 0.5*(gradu[:,:,:,0,1] + gradu[:,:,:,1,0])
    strain_tensor[:,:,:,0,2] = 0.5*(gradu[:,:,:,0,2] + gradu[:,:,:,2,0])
    strain_tensor[:,:,:,1,2] = 0.5*(gradu[:,:,:,1,2] + gradu[:,:,:,2,1])
    strain_tensor[:,:,:,1,0] = strain_tensor[:,:,:,0,1]
    strain_tensor[:,:,:,2,0] = strain_tensor[:,:,:,0,2]
    strain_tensor[:,:,:,2,1] = strain_tensor[:,:,:,1,2]
    return strain_tensor

def get_green_strain_tensor(gradu):
    """Calculate the symmetric, nonlinear strain tensor at each point on the grid representing the initial node positions using the gradient of the displacement field."""
    strain_tensor = np.zeros(np.shape(gradu))
    strain_tensor[:,:,:,0,0] = gradu[:,:,:,0,0] + 0.5*(np.power(gradu[:,:,:,0,0],2) + np.power(gradu[:,:,:,1,0],2) + np.power(gradu[:,:,:,2,0],2))
    strain_tensor[:,:,:,1,1] = gradu[:,:,:,1,1] + 0.5*(np.power(gradu[:,:,:,0,1],2) + np.power(gradu[:,:,:,1,1],2) + np.power(gradu[:,:,:,2,1],2))
    strain_tensor[:,:,:,2,2] = gradu[:,:,:,2,2] + 0.5*(np.power(gradu[:,:,:,0,2],2) + np.power(gradu[:,:,:,1,2],2) + np.power(gradu[:,:,:,2,2],2))
    strain_tensor[:,:,:,0,1] = 0.5*(gradu[:,:,:,0,1] + gradu[:,:,:,1,0]) + 0.5*(gradu[:,:,:,0,0]*gradu[:,:,:,0,1] + gradu[:,:,:,1,0]*gradu[:,:,:,1,1] + gradu[:,:,:,2,0]*gradu[:,:,:,2,1])
    strain_tensor[:,:,:,0,2] = 0.5*(gradu[:,:,:,0,2] + gradu[:,:,:,2,0]) + 0.5*(gradu[:,:,:,0,0]*gradu[:,:,:,0,2] + gradu[:,:,:,1,0]*gradu[:,:,:,1,2] + gradu[:,:,:,2,0]*gradu[:,:,:,2,2])
    strain_tensor[:,:,:,1,2] = 0.5*(gradu[:,:,:,1,2] + gradu[:,:,:,2,1]) + 0.5*(gradu[:,:,:,0,1]*gradu[:,:,:,0,2] + gradu[:,:,:,1,1]*gradu[:,:,:,1,2] + gradu[:,:,:,2,1]*gradu[:,:,:,2,2])
    strain_tensor[:,:,:,1,0] = strain_tensor[:,:,:,0,1]
    strain_tensor[:,:,:,2,0] = strain_tensor[:,:,:,0,2]
    strain_tensor[:,:,:,2,1] = strain_tensor[:,:,:,1,2]
    return strain_tensor

def get_gradu(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,dimensions):
    """Get the tensor at each point on the grid representing the initial node positions which is the gradient of the displacement field"""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    gradu = np.zeros((xshape,yshape,zshape,3,3))
    gradu[:,:,:,0,0] = xdisplacement_gradient[0][:,:,:]
    gradu[:,:,:,1,1] = ydisplacement_gradient[1][:,:,:]
    gradu[:,:,:,2,2] = zdisplacement_gradient[2][:,:,:]
    gradu[:,:,:,0,1] = xdisplacement_gradient[1][:,:,:]
    gradu[:,:,:,0,2] = xdisplacement_gradient[2][:,:,:]
    gradu[:,:,:,1,2] = ydisplacement_gradient[2][:,:,:]
    gradu[:,:,:,1,0] = ydisplacement_gradient[0][:,:,:]
    gradu[:,:,:,2,0] = zdisplacement_gradient[0][:,:,:]
    gradu[:,:,:,2,1] = zdisplacement_gradient[1][:,:,:]
    return gradu

def get_deformation_gradient(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,dimensions):
    """Calculate the deformation gradient at each point on the grid representing the initial node positions. Equal to gradu plus the identity matrix"""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    deformation_gradient = np.zeros((xshape,yshape,zshape,3,3))
    deformation_gradient[:,:,:,0,0] = xdisplacement_gradient[0][0,0,0] + 1
    deformation_gradient[:,:,:,1,1] = ydisplacement_gradient[1][0,0,0] + 1
    deformation_gradient[:,:,:,2,2] = zdisplacement_gradient[2][0,0,0] + 1
    deformation_gradient[:,:,:,0,1] = xdisplacement_gradient[1][0,0,0]
    deformation_gradient[:,:,:,0,2] = xdisplacement_gradient[2][0,0,0]
    deformation_gradient[:,:,:,1,2] = ydisplacement_gradient[2][0,0,0]
    deformation_gradient[:,:,:,1,0] = ydisplacement_gradient[0][0,0,0]
    deformation_gradient[:,:,:,2,0] = zdisplacement_gradient[0][0,0,0]
    deformation_gradient[:,:,:,2,1] = zdisplacement_gradient[1][0,0,0]
    return deformation_gradient

def get_displacement_field(initial_posns,final_posns):
    """Calculate the displacement of each node from its initial position."""
    displacement = final_posns - initial_posns
    return displacement

def transform_to_3D_array(array,dimensions):
    """Given a 1D vector of node positions, or similarly structured per node values, and convert to a 3D array mapped to the grid of initial node positions for plotting and analysis. Dimensions is tuple or array of number of nodes along each direction (x,y,z)."""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    array_3D = np.zeros((xshape,yshape,zshape))
    nodes_per_column = zshape
    nodes_per_plane = zshape*xshape
    for j in range(yshape):
        for i in range(xshape):
            array_3D[i,j,:] = array[(j*nodes_per_plane)+(i*nodes_per_column):(j*nodes_per_plane)+(i+1)*nodes_per_column]
    return array_3D

def get_component_3D_arrays(array,dimensions):
    xarray_3D = transform_to_3D_array(array[:,0],dimensions)
    yarray_3D = transform_to_3D_array(array[:,1],dimensions)
    zarray_3D = transform_to_3D_array(array[:,2],dimensions)
    return xarray_3D, yarray_3D, zarray_3D
#i want to try to use some of the structure of this function for plotting the displacement field, but i'm not sure the best way to plot the displacement field. i can use pcolormesh. i can use quiver. i should probably try those two before using scatter and color of markers. i want to use the initial node positions, since they are on a grid, to show the deformation of the initial node positions to the final node positions. i want to plot displacement components as separate plots on a subplot figure, and the displacement norm as well... different cuts would be possible for each component/norm.

def subplot_cut_pcolormesh_vectorfield(cut_type,eq_node_posns,vectorfield,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the vectorfield components of some property of the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)
    #if the shape has 2 members, reshape the vectorfield variable to a 3D grid of values
    if len(np.shape(vectorfield)) ==  2:
        xvectorfield_3D, yvectorfield_3D, zvectorfield_3D = get_component_3D_arrays(vectorfield,dimensions)
        vectorfield = (xvectorfield_3D,yvectorfield_3D,zvectorfield_3D)
    #if the shape has 3 members, assume the vectorfield variable is already in the appropriate format of a 3D grid of values for plotting with pcolormesh
    elif len(np.shape(vectorfield)) == 3:
        pass
    else:
        raise ValueError(f'shape of vector field {np.shape(vectorfield)} must be 2D or 3D')

    component_dict = {0:'x',1:'y',2:'z',3:'norm'}
    for i in range(4):
        row = np.floor_divide(i,2)
        col = i%2
        ax = axs[row,col]
        if i != 3:
            vectorfield_component = vectorfield[i]
        else:
            vectorfield_component = np.sqrt(np.power(vectorfield[0],2) +np.power(vectorfield[1],2) + np.power(vectorfield[2],2)) 
        if cut_type == 'xy':
            Z = vectorfield_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = vectorfield_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = vectorfield_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z)
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(tag+ f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    # plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_' + tag + '_vectorfield_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_cut_pcolormesh_tensorfield(cut_type,eq_node_posns,tensorfield,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the symmetric tensor components of some tensor field defined at the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    if len(np.shape(tensorfield)) != 5:
        raise ValueError(f'shape of tensor field {np.shape(tensorfield)} must be 5D')
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)

    component_dict = {0:'xx',1:'yy',2:'zz',3:'xy',4:'xz',5:'yz'}
    for i in range(6):
        row = np.floor_divide(i,3)
        col = i%3
        ax = axs[row,col]
        if i == 0:
            tensor_component = tensorfield[:,:,:,0,0]
        elif i == 1:
            tensor_component = tensorfield[:,:,:,1,1]
        elif i == 2:
            tensor_component = tensorfield[:,:,:,2,2]
        elif i == 3:
            tensor_component = tensorfield[:,:,:,0,1]
        elif i == 4:
            tensor_component = tensorfield[:,:,:,0,2]
        elif i == 5:
            tensor_component = tensorfield[:,:,:,1,2]
        if cut_type == 'xy':
            Z = tensor_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = tensor_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = tensor_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z)
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(tag+f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_'+tag+'_tensorfield_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_residual_acceleration_hist(a_norms,output_dir):
    """Plot a histogram of the acceleration of the nodes. Intended for analyzing the behavior at the end of simulations that are ended before convergence criteria are met."""
    max_accel = np.max(a_norms)
    mean_accel = np.mean(a_norms)
    rms_accel = np.sqrt(np.sum(np.power(a_norms,2))/np.shape(a_norms)[0])
    counts, bins = np.histogram(a_norms, bins=20)
    fig,ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    ax.hist(bins[:-1], bins, weights=counts)
    sigma = np.std(a_norms)
    mu = mean_accel
    ax.set_title(f'Residual Acceleration Histogram\nMaximum {max_accel}\nMean {mean_accel}\n$\sigma={sigma}$\nRMS {rms_accel}')
    ax.set_xlabel('acceleration norm')
    ax.set_ylabel('counts')
    savename = output_dir +'node_residual_acceleration_hist.png'
    plt.savefig(savename)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()