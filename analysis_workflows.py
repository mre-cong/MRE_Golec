#2023-11-29
#David Marchfield
#Establishing the distinct workflows for different types of simulations and analyses via pseudocode. Followed by implementation of necessary component functions and visualizations.

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import simulate
import re
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

#Three distinct cases: 
#   (1) No particles, or particles with no applied magnetic field (magnetic field independent material property simulation)
#   (2) Particles with applied magnetic field and free boundary conditions (magnetic hystersis simulation)
#   (3) Particles with applied magnetic field and strain boundary conditions (magnetic field dependent material property simulation)

# Case (1):
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in
#   stress array variable initialized
#   effective modulus array variable initialized
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       node positions are scaled to SI units using l_e variable for visualization, along with the dimensions of the simulated volume
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
#       visualizations of cuts through the center of the volume are generated and saved out
#       visualizations of cuts through the particle centers and edges are generated and saved out
#       node positions are used to calculate nodal displacement
#       nodal displacement is used to calculated displacement gradient
#       displacement gradient is used to calculate linear and nonlinear strain tensors
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
#       stress and strain tensors are visualized for the outer surfaces
#       stress and strain tensors are visualized for cuts through the center of the volume
#       if particles present:
#          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures

# Case (2):
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
#   magnetization vectors array variable initialized
#   particle separation array variable initialized
#   in a loop, output files are read in and manipulated
#       node positions and particles variable are read in
#       particle variable used to determine particle centers
#       particle centers and Hext used to calculate particle magnetizations
#       overall magnetization calculated from particle magnetizations
#       node positions are scaled to SI units
#       particle separations are calculated from particle centers and scaling
#       overall magnetization vector and particle separation are saved out to respective array variables
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
#       visualizations of cuts through the center of the volume are generated and saved out
#       visualizations of cuts through the particle centers and edges are generated and saved out
#       node positions are used to calculate nodal displacement
#       nodal displacement is used to calculated displacement gradient
#       displacement gradient is used to calculate linear and nonlinear strain tensors
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
#       stress and strain tensors are visualized for the outer surfaces
#       stress and strain tensors are visualized for cuts through the center of the volume
#       stress and strain tensors are visualized for cuts through particle centers and edges if particles present
#   outside the loop:
#   figure with 2 subplots showing the "hysteresis curve" (magnetization parallel to the applied field direction) and the particle separation are generated and saved out
#   magnetization vector array and particle separation array variables saved out to table or csv file for reproducing or modifying the figure

# Case (3) :
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
#   magnetization vectors array variable initialized
#   particle separation array variable initialized
#   stress array variable initialized
#   effective modulus array variable initialized
#   magnetization vectors array variable initialized
#   particle separation array variable initialized
#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       particles variable read in
#       particle variable used to determine particle centers
#       particle centers and Hext used to calculate particle magnetizations
#       overall magnetization calculated from particle magnetizations
#       node positions are scaled to SI units using l_e variable for visualization, along with the dimensions of the simulated volume
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
#       visualizations of cuts through the center of the volume are generated and saved out
#       visualizations of cuts through the particle centers and edges are generated and saved out
#       node positions are used to calculate nodal displacement
#       nodal displacement is used to calculated displacement gradient
#       displacement gradient is used to calculate linear and nonlinear strain tensors
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
#       stress and strain tensors are visualized for the outer surfaces
#       stress and strain tensors are visualized for cuts through the center of the volume
#       stress and strain tensors are visualized for cuts through particle centers and edges if particles present
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures
#   figure with 2 subplots showing the "hysteresis curve" (magnetization parallel to the applied field direction) and the particle separation are generated and saved out
#   magnetization vector array and particle separation array variables saved out to table or csv file for reproducing or modifying the figure

def analysis_case1(sim_dir):
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (1), simulations with no applied magnetic field, with or without particles, for analyzing the effective modulus for some type of strain"""
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, N_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
#   stress array variable initialized
    stress = np.zeros(series.shape[0])
#   effective modulus array variable initialized
    effective_modulus = np.zeros(series.shape[0])
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
    #see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))
#   in a loop, output files are read in and manipulated
    for i in range(series.shape[0]):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       node positions are scaled to SI units using l_e variable for visualization, along with the dimensions of the simulated volume
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
#       visualizations of cuts through the center of the volume are generated and saved out
#       visualizations of cuts through the particle centers and edges are generated and saved out
#       node positions are used to calculate nodal displacement
#       nodal displacement is used to calculated displacement gradient
#       displacement gradient is used to calculate linear and nonlinear strain tensors
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
#       stress and strain tensors are visualized for the outer surfaces
#       stress and strain tensors are visualized for cuts through the center of the volume
#       if particles present:
#          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures
    pass

def analysis_case2():
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (2), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization without an applied strain"""
    pass

def analysis_case3():
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (2), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization, and the effective modulus for an applied strain"""
    pass

def main():
    print('main')

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

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, N_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions

def format_boundary_conditions(boundary_conditions):
    boundary_conditions = (str(boundary_conditions[0][0])[1:],(str(boundary_conditions[0][1])[1:],str(boundary_conditions[0][2])[1:]),boundary_conditions[0][3])
    boundary_conditions = (boundary_conditions[0][1:-1],(boundary_conditions[1][0][1:-1],boundary_conditions[1][1][1:-1]),boundary_conditions[2])
    return boundary_conditions

def get_effective_modulus(sim_dir):
    """Given a simulation directory, calculate and plot the stress-strain curve and effective modulus versus strain."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
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
    #TODO finish this function. shearing in different directions from the same surface is a different modulus (there is anisotropy, or should assume there is). use the direction of the shearing for title, labels, and save name for the figures generated (though that may not occur in this function)
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
    #TODO Implement a calculation of a torsion modulus. May not be used in thesis, and so while the ability to do torsion strain simulations has been implemented, this may not be addressed
    """Calculate a torsion modulus, in analogy to the shearing modulus, where strain is defined by the twist angle. torsion forces don't quite make sense, so instead the correct calculation is the torque applied to the surface divided by the twist angle (or the derivative of torque wrt the twist angle)"""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if strain_direction[1] == 'CW':
        pass
    elif strain_direction[1] == 'CCW':
        pass

def checkpoint_traversal(sim_dir):
    """This is a way of getting the subfolders in the simulation directory, and then a way to traverse those subfolders and grab the checkpoint files within in order for some calculation or visualization"""
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir()]
    for subfolder in subfolders:
        with os.scandir(subfolder+'/') as dirIterator:
            checkpoint_files = [f.path for f in dirIterator if f.is_file() and f.name.startswith('checkpoint')]
        for checkpoint_file in checkpoint_files:
            fn_w_type = checkpoint_file.split('/')[-1]
            fn = fn_w_type.split('.')[0]
            checkpoint_number = np.append(checkpoint_number,np.array([int(count) for count in re.findall(r'\d+',fn)]))
        sort_indices = np.argsort(checkpoint_number)
        checkpoint_file = checkpoint_files[sort_indices[-1]]
        solution, applied_field, boundary_conditions, i = mre.initialize.read_checkpoint_file(checkpoint_file)
        boundary_conditions = format_boundary_conditions(boundary_conditions)

if __name__ == "__main__":
    main()
    sim_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-15_strain_testing_shearing_order_1_drag_20/'
    analysis_case1(sim_dir)