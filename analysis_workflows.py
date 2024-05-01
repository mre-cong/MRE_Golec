#2023-11-29
#David Marchfield
#Establishing the distinct workflows for different types of simulations and analyses via pseudocode. Followed by implementation of necessary component functions and visualizations.

import numpy as np
import scipy.special as sci
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
# plt.switch_backend('TkAgg')
plt.switch_backend('Agg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import magnetism
import simulate
import re
import cupy as cp
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

cut_types = ('xy','xz','yz')

def analysis_case1(sim_dir):
    #TODO Where should visualizations be saved out? how should they be named (relative to the output file/boundary conditions)? If I want SI units, I need new versions of functions with small modifications to the labels, and i need to iterate on the function until i get thesis quality figures
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (1), simulations with no applied magnetic field, with or without particles, for analyzing the effective modulus for some type of strain"""
#   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['modulus','stress','strain','cuts','outer_surfaces']
    figure_subtypes = ['center', 'particle', 'outer_surface']
    for figure_type in figure_types:
        if not (os.path.isdir(output_dir+figure_type+'/')):
          os.mkdir(output_dir+figure_type+'/')
        if figure_type == 'stress' or figure_type =='strain' or figure_type == 'cuts':
            for figure_subtype in figure_subtypes:
                if not (figure_type == 'cuts' and figure_subtype == 'outer_surface'):
                    if not (os.path.isdir(output_dir+figure_type+'/'+figure_subtype+'/')):
                        os.mkdir(output_dir+figure_type+'/'+figure_subtype+'/')
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
#   find the indices corresponding to the outer surfaces of the simulated volume for plotting and visualization
    surf_indices = (0,int(num_nodes[0]-1),0,int(num_nodes[1]-1),0,int(num_nodes[2]-1))
    surf_type = ('left','right','front','back','bottom','top')
#   find indices corresponding to the "center" of the simulated volume for plotting and visualization, corresponding to cut_types values
    center_indices = (int((num_nodes[2]-1)/2),int((num_nodes[1]-1)/2),int((num_nodes[0]-1)/2))
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
    #see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))
#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
    if 'stress' in sim_dir:
        effective_modulus, stress, strains, strain_direction = get_effective_modulus_stress_sim(sim_dir)
    elif 'strain' in sim_dir:
        effective_modulus, stress, strains, strain_direction = get_effective_modulus_strain_series(sim_dir)
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
    subplot_stress_strain_modulus(stress,strains,strain_direction,effective_modulus,output_dir+'modulus/',tag="")
#   in a loop, output files are read in and manipulated
    for i in range(series.shape[0]):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
#       node positions are scaled to SI units using l_e variable for visualization
        si_final_posns = final_posns*l_e
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#TODO Issue with using contours for abritrary simulations. if the surfaces don't have contours, that is, differences in the "depth" from point to point, then there are no contour levels that can be defined, and the thing fails. i can use a try/except clause, but that may be bad style/practice. I'm not sure of the right way to handle this. I suppose if it is shearing or torsion I should expect that this may not be a useful figure to generate anyway, so i could use the boundary_conditions variable first element
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if ((boundary_conditions[0] == "tension" or boundary_conditions[0] == "compression" or boundary_conditions[0] == "free") and boundary_conditions[2] != 0) or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_tiled_outer_surfaces_contours_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{series[i]}")
            except:
                print('contour plotting of outer surfaces failed due to lack of variation (no contour levels could be generated)')
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
        mre.analyze.plot_outer_surfaces_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{series[i]}")
#       visualizations of cuts through the center of the volume are generated and saved out
        mre.analyze.plot_center_cuts_surf(initial_node_posns,final_posns,l_e,output_dir+'cuts/center/',tag=f"3D_strain_{series[i]}")
        # mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{series[i]}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"strain_{series[i]}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{series[i]}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')
#       visualizations of cuts through the particle centers and edges are generated and saved out
        if particles.shape[0] != 0:
            mre.analyze.plot_particle_centric_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/particle/',tag=f"series_{i}")
            mre.analyze.plot_particle_centric_cuts_surf(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/particle/',tag=f"series_{i}")
#       node positions are used to calculate nodal displacement
        displacement_field = get_displacement_field(initial_node_posns,final_posns)
#       nodal displacement is used to calculated displacement gradient
        gradu = get_gradu(displacement_field,num_nodes)
#       displacement gradient is used to calculate linear and nonlinear strain tensors
        strain_tensor = get_strain_tensor(gradu)
        green_strain_tensor = get_green_strain_tensor(gradu)
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
        stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)
#       stress and strain tensors are visualized for the outer surfaces
        for surf_idx,surface in zip(surf_indices,surf_type):
            if surface == 'left' or surface == 'right':
                cut_type = 'yz'
            elif surface == 'front' or surface == 'back':
                cut_type = 'xz'
            elif surface == 'top' or surface == 'bottom':
                cut_type = 'xy'
            tag = surface+'_surface_strain_' + f'{series[i]}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'strain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'nonlinearstrain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,surf_idx,output_dir+'stress/outer_surface/',tag=tag+'stress')
#       stress and strain tensors are visualized for cuts through the center of the volume
        for cut_type,center_idx in zip(cut_types,center_indices):
            tag = 'center_'  f'{series[i]}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'strain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'nonlinearstrain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,center_idx,output_dir+'stress/center/',tag=tag+'stress')
#       if particles present:
#          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
        if particles.shape[0] != 0:
            centers = np.zeros((particles.shape[0],3))
            for i, particle in enumerate(particles):
                tag=f"particle{i+1}_edge_" + f'strain_{series[i]}_'
                centers[i,:] = simulate.get_particle_center(particle,initial_node_posns)
                particle_node_posns = initial_node_posns[particle,:]
                x_max = np.max(particle_node_posns[:,0])
                y_max = np.max(particle_node_posns[:,1])
                z_max = np.max(particle_node_posns[:,2])
                x_min = np.min(particle_node_posns[:,0])
                y_min = np.min(particle_node_posns[:,1])
                z_min = np.min(particle_node_posns[:,2])
                edge_indices = ((z_max,z_min),(y_max,y_min),(x_max,x_min))
                #TODO switchover to plotting tensorfields from this surf plot, but utilize the appropriate indices, cut types, and generate useful tags
                for cut_type,layer_indices in zip(cut_types,edge_indices):
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,int(layer_indices[0]),output_dir+'strain/particle/',tag=tag+'strain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,int(layer_indices[1]),output_dir+'strain/particle/',tag='second'+tag+'strain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,int(layer_indices[0]),output_dir+'strain/particle/',tag=tag+'nonlinearstrain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,int(layer_indices[1]),output_dir+'strain/particle/',tag='second'+tag+'nonlinearstrain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,int(layer_indices[0]),output_dir+'stress/particle/',tag=tag+'stress')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,int(layer_indices[1]),output_dir+'stress/particle/',tag='second'+tag+'stress')
            tag='particle_centers_'+ f'strain_{series[i]}_'
            layers = (int((centers[0,2]+centers[1,2])/2),int((centers[0,1]+centers[1,1])/2),int((centers[0,0]+centers[1,0])/2))
            for cut_type,layer in zip(cut_types,layers):
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'strain')
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'nonlinearstrain')
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,layer,output_dir+'stress/particle/',tag=tag+'stress')
#   outside the loop:
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures

def analysis_case2():

    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (2), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization without an applied strain"""
    pass

def analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=False):
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (3), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization, stress and strain tensors, and the effective modulus for an applied field"""
    #   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['modulus','particle_behavior','stress','strain','cuts','outer_surfaces']
    figure_subtypes = ['center', 'particle', 'outer_surface']
    for figure_type in figure_types:
        if not (os.path.isdir(output_dir+figure_type+'/')):
          os.mkdir(output_dir+figure_type+'/')
        if figure_type == 'stress' or figure_type =='strain' or figure_type == 'cuts':
            for figure_subtype in figure_subtypes:
                if not (figure_type == 'cuts' and figure_subtype == 'outer_surface'):
                    if not (os.path.isdir(output_dir+figure_type+'/'+figure_subtype+'/')):
                        os.mkdir(output_dir+figure_type+'/'+figure_subtype+'/')
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if gpu_flag:
        initial_node_posns = np.float64(initial_node_posns)
        beta_i = np.float64(beta_i)
        springs_var = np.float64(springs_var)
        kappa = np.float64(kappa)
        beta = np.float64(beta)
        l_e = np.float64(l_e)
        particle_mass = np.float64(particle_mass)
        particle_radius = np.float64(particle_radius)
        Ms = np.float64(Ms)
        chi = np.float64(chi)
#   find the indices corresponding to the outer surfaces of the simulated volume for plotting and visualization
    surf_indices = (0,int(num_nodes[0]-1),0,int(num_nodes[1]-1),0,int(num_nodes[2]-1))
    surf_type = ('left','right','front','back','bottom','top')
#   find indices corresponding to the "center" of the simulated volume for plotting and visualization, corresponding to cut_types values
    center_indices = (int((num_nodes[2]-1)/2),int((num_nodes[1]-1)/2),int((num_nodes[0]-1)/2))
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
    #see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))

    #get the the applied field associated with each output file
    Hext_series = get_applied_field_series(sim_dir)
    num_output_files = get_num_output_files(sim_dir)
    #get the particle separations and overall magnetizations and plot them
    plot_particle_behavior_flag = True
    if plot_particle_behavior_flag and particles.shape[0] != 0:
        particle_separations = plot_particle_behavior(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series)
        num_particles = particles.shape[0]
        clustering_distance = 4
        particle_separations *= l_e*1e6
        particle_separations_matrix = np.zeros((num_output_files,num_particles,num_particles))
        for output_file_count in np.arange(num_output_files):
            counter = 0
            for i in np.arange(num_particles):
                for j in np.arange(i+1,num_particles):
                    particle_separations_matrix[output_file_count,i,j] = particle_separations[output_file_count,counter]
                    counter += 1
            particle_separations_matrix[output_file_count] += np.transpose(particle_separations_matrix[output_file_count])
    else:
        particle_separations_matrix = None

#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
    #TODO effective modulus calculations for field dependent simulations
    if 'strain' in sim_dir:
        effective_modulus, stress, strain, Bext_series, strain_direction = get_field_dependent_effective_modulus(sim_dir)
        bc_values = strain
        bc_type = 'strain'
    elif 'stress' in sim_dir:
        effective_modulus, differential_modulus, stress, strain, Bext_series, strain_direction = get_field_dependent_effective_modulus_stress_sim(sim_dir)
        differential_modulus[np.isinf(differential_modulus)] = 0
        bc_values = stress
        bc_type = 'stress'
    #how many cluster pairs formed?
    for output_file_count in np.arange(num_output_files):
        cluster_counter = 0
        for i in np.arange(num_particles):
            temp_separations = particle_separations_matrix[output_file_count,i,:]
            cluster_counter += np.count_nonzero(np.less_equal(temp_separations[temp_separations>0],clustering_distance))
            #if we know some particle clustering has ocdurred, how can we determine if a single particle is clustering with multiple particles, and cross reference to determine if a chain has formed, and how many particles make up that chain?
        cluster_counter /= 2
        if cluster_counter != 0:
            print(f'for field {np.round(Hext_series[output_file_count]*mu0,decimals=5)} and {bc_type} {bc_values[output_file_count]} the total number of clusters: {cluster_counter}')
    #on a field by field basis, fit the stress-strain curve to a linear function to extract the effective modulus, ignoring the (0,0) point
    unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1))
    num_fields = unique_fields.shape[0]
    linear_fit_modulus = np.zeros((num_fields,))
    linear_fit_modulus_error = np.zeros((num_fields,))
    if 'tension' in sim_dir or 'compression' in sim_dir:
        modulus_fit_guess = E
    elif 'shearing' in sim_dir:
        modulus_fit_guess = shear_modulus
    for i, field in enumerate(unique_fields):
        relevant_indices = np.isclose(field,np.linalg.norm(Bext_series,axis=1))
        single_field_stress = stress[relevant_indices]
        single_field_strain = strain[relevant_indices]
        if 'stress' in sim_dir:
            fitting_indices = np.nonzero(single_field_stress)
        elif 'strain' in sim_dir:
            fitting_indices = np.nonzero(single_field_strain)
        single_field_stress = single_field_stress[fitting_indices]
        single_field_strain = single_field_strain[fitting_indices]
        if fitting_indices[0].shape[0] >= 2:
            popt, pcov = scipy.optimize.curve_fit(linear_fit_func,single_field_strain,single_field_stress,p0=np.array([modulus_fit_guess,0]))
            linear_fit_modulus[i] = popt[0]
            linear_fit_modulus_error[i] = np.sqrt(np.diag(pcov))[0]
    component_to_index_dict = {'x':0,'y':1,'z':2}
    #strain measure from a quasi average strain tensor by "integrating" the strain tensor components over the volume and dividing by the system volume
    average_strain_tensor = np.zeros((num_output_files,3,3))
    # relevant_eig_val_avg = np.zeros((num_output_files,))
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
#       node positions are used to calculate nodal displacement
        displacement_field = get_displacement_field(initial_node_posns,final_posns)
#       nodal displacement is used to calculated displacement gradient
        gradu = get_gradu(displacement_field,num_nodes)
#       displacement gradient is used to calculate linear strain tensors
        strain_tensor = get_strain_tensor(gradu)
        average_strain_tensor[i] = get_average_tensor_value(strain_tensor,elements,initial_node_posns,tag='strain')
        # 2024-04-11 DBM: in general, can find eigenvalue/eigenvector decomposition of the symmetric matrix to get the strain in the principal directions local for the matrix. in the case where bending is occurring, I am getting unusual strain values (negative, and not small) for what should be small positive tensile strains (due to small positive tensile stresses as boundary conditions). here i am exploring if the volume average of the local principal direction strains along the analagous direction provide useful strain information for moduls calculations
        # my_sum = 0
        # for row in np.arange(strain_tensor.shape[0]):
        #     for col in np.arange(strain_tensor.shape[1]):
        #         for page in np.arange(strain_tensor.shape[2]):
        #             w = scipy.linalg.eigvalsh(strain_tensor[row,col,page,:,:])
        #             relevant_eig_val = w[component_to_index_dict[boundary_conditions[1][1]]]
        #             my_sum += relevant_eig_val
        # relevant_eig_val_avg[i] = my_sum/(strain_tensor.shape[0]*strain_tensor.shape[1]*strain_tensor.shape[2])

    Bext_series = mu0*Hext_series
    Bext_series_magnitude = np.round(np.linalg.norm(mu0*Hext_series,axis=1)*1e3,decimals=3)
    num_unique_fields = np.unique(Bext_series_magnitude).shape[0]
    relative_strain_tensor = np.zeros(np.shape(average_strain_tensor))
    relative_strain_eigvals = np.zeros((num_output_files,))
    for i in range(num_output_files):
        relative_configuration_index = np.mod(i,num_unique_fields)
        relative_strain_tensor[i] = average_strain_tensor[i] - average_strain_tensor[relative_configuration_index]
        # relative_strain_eigvals[i] = relevant_eig_val_avg[i] - relevant_eig_val_avg[relative_configuration_index]
    # if 'stress' in sim_dir:
    #     for i in range(num_output_files):
    #         relative_configuration_index = np.mod(i,num_unique_fields)
    #         relative_strain_tensor[i] = average_strain_tensor[i] - average_strain_tensor[relative_configuration_index]
    #         relative_strain_eigvals[i] = relevant_eig_val_avg[i] - relevant_eig_val_avg[relative_configuration_index]
    # elif 'strain' in sim_dir:
    #     relative_strain_tensor = average_strain_tensor
    #     relative_strain_eigvals = relevant_eig_val_avg
    # else:
    #     raise NotImplementedError
    rve_strain_average = relative_strain_tensor[:,component_to_index_dict[boundary_conditions[1][0]],component_to_index_dict[boundary_conditions[1][1]]]
    strain_comparisons = rve_strain_average - strain
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(num_output_files),strain_comparisons)
    percent_strain_difference = 100*strain_comparisons/strain
    axs[1].plot(np.arange(num_output_files),percent_strain_difference)
    axs[0].set_xlabel('output file number')
    axs[0].set_ylabel('strain metric difference')
    axs[1].set_ylabel('strain metric percentage difference')
    plt.show()
    linear_fit_modulus_from_rve_strain_avg = np.zeros((num_fields,))
    linear_fit_modulus_from_rve_strain_avg_error = np.zeros((num_fields,))
    # linear_fit_modulus_from_eigval_strain_avg = np.zeros((num_fields,))
    # linear_fit_modulus_from_eigval_strain_avg_error = np.zeros((num_fields,))
    for i, field in enumerate(unique_fields):
        relevant_indices = np.isclose(field,np.linalg.norm(Bext_series,axis=1))
        single_field_stress = stress[relevant_indices]
        single_field_strain = rve_strain_average[relevant_indices]
        if 'stress' in sim_dir:
            fitting_indices = np.nonzero(single_field_stress)
        elif 'strain' in sim_dir:
            fitting_indices = np.nonzero(single_field_strain)
        single_field_stress = single_field_stress[fitting_indices]
        single_field_strain = single_field_strain[fitting_indices]
        # single_field_eigval_strain = relative_strain_eigvals[relevant_indices]
        # single_field_eigval_strain = single_field_eigval_strain[fitting_indices]
        if fitting_indices[0].shape[0] >= 2:
            popt, pcov = scipy.optimize.curve_fit(linear_fit_func,single_field_strain,single_field_stress,p0=np.array([modulus_fit_guess,0]))
            linear_fit_modulus_from_rve_strain_avg[i] = popt[0]
            linear_fit_modulus_from_rve_strain_avg_error[i] = np.sqrt(np.diag(pcov))[0]
            # popt, pcov = scipy.optimize.curve_fit(linear_fit_func,single_field_eigval_strain,single_field_stress,p0=np.array([modulus_fit_guess,0]))
            # linear_fit_modulus_from_eigval_strain_avg[i] = popt[0]
            # linear_fit_modulus_from_eigval_strain_avg_error[i] = np.sqrt(np.diag(pcov))[0]
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
    subplot_stress_field_modulus(stress,strain,strain_direction,effective_modulus,Bext_series,output_dir+'modulus/',tag="")
    if 'stress' in sim_dir:
        subplot_stress_field_differentialmodulus(stress,strain,differential_modulus,Bext_series,output_dir+'modulus/',tag="differential_modulus")
        subplot_stress_strain_differentialmodulus_by_field(stress,strain,differential_modulus,Bext_series,output_dir+'modulus/',tag="differential_modulus")
    subplot_stress_strain_modulus_by_field(stress,strain,strain_direction,effective_modulus,Bext_series,output_dir+'modulus/',tag="")
    subplot_stress_strain_linear_fit_modulus_by_field(stress,strain,linear_fit_modulus,linear_fit_modulus_error,Bext_series,output_dir+'modulus/',tag="",particle_separations=particle_separations_matrix)
    subplot_stress_strain_linear_fit_modulus_by_field(stress,rve_strain_average,linear_fit_modulus_from_rve_strain_avg,linear_fit_modulus_from_rve_strain_avg_error,Bext_series,output_dir+'modulus/',tag="rve_strain_avg",particle_separations=particle_separations_matrix)
    # subplot_stress_strain_linear_fit_modulus_by_field(stress,relative_strain_eigvals,linear_fit_modulus_from_eigval_strain_avg,linear_fit_modulus_from_eigval_strain_avg_error,Bext_series,output_dir+'modulus/',tag="eigval_strain_avg")
    # subplot_stress_field_modulus(alternative_stress_measure,strain,strain_direction,alternative_effective_modulus,Bext_series,output_dir+'modulus/',tag="alternative_measures")

#   in a loop, output files are read in and manipulated
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_nodes(initial_node_posns,final_posns,particles,output_dir+'particle_behavior/',tag=f"{i}")
#       node positions are scaled to SI units using l_e variable for visualization
        si_final_posns = final_posns*l_e
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#TODO Issue with using contours for abritrary simulations. if the surfaces don't have contours, that is, differences in the "depth" from point to point, then there are no contour levels that can be defined, and the thing fails. i can use a try/except clause, but that may be bad style/practice. I'm not sure of the right way to handle this. I suppose if it is shearing or torsion I should expect that this may not be a useful figure to generate anyway, so i could use the boundary_conditions variable first element
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if ((boundary_conditions[0] == "tension" or boundary_conditions[0] == "compression" or boundary_conditions[0] == "free") and boundary_conditions[2] != 0) or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_tiled_outer_surfaces_contours_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
            except:
                print('contour plotting of outer surfaces failed due to lack of variation (no contour levels could be generated)')
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
        mre.analyze.plot_outer_surfaces_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
#       visualizations of cuts through the center of the volume are generated and saved out
        mre.analyze.plot_center_cuts_surf(initial_node_posns,final_posns,l_e,output_dir+'cuts/center/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        # mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')
#       visualizations of cuts through the particle centers and edges are generated and saved out
        # if particles.shape[0] != 0:
        #     mre.analyze.plot_particle_centric_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/particle/',tag=f"{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        #     mre.analyze.plot_particle_centric_cuts_surf(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/particle/',tag=f"{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        if stress_strain_flag:# and not np.isclose(boundary_conditions[2],0):
    #       node positions are used to calculate nodal displacement
            displacement_field = get_displacement_field(initial_node_posns,final_posns)
    #       nodal displacement is used to calculated displacement gradient
            gradu = get_gradu(displacement_field,num_nodes)
    #       displacement gradient is used to calculate linear and nonlinear strain tensors
            strain_tensor = get_strain_tensor(gradu)
            green_strain_tensor = get_green_strain_tensor(gradu)
    #       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
            stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)
    # #       use the components of linear stress and strain tensors to get something like an effective modulus on a per node basis
    #         effective_modulus_tensor = stress_tensor/strain_tensor
            # np.nan_to_num(effective_modulus_tensor,copy=False,nan=0.0,posinf=-1,neginf=+1)
    #       stress and strain tensors are visualized for the outer surfaces
            for surf_idx,surface in zip(surf_indices,surf_type):
                if surface == 'left' or surface == 'right':
                    cut_type = 'yz'
                elif surface == 'front' or surface == 'back':
                    cut_type = 'xz'
                elif surface == 'top' or surface == 'bottom':
                    cut_type = 'xy'
                tag = surface+'_surface_' + f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'strain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'nonlinearstrain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,surf_idx,output_dir+'stress/outer_surface/',tag=tag+'stress')
                # # below, used for plotting the per node effective modulus using stress and strain tensor components. a review of the analytical expressions for the stress and strain tensor components (and looking over figures that were generated) shows that the effective modulus calculated this way will not vary over the surface, or with changes in the applied field, due to their (seemingly entirely) linear relationship (between stress and strain components, through the Lame parameters)
                # if surface == 'right':
                #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,effective_modulus_tensor,surf_idx,output_dir+'modulus/',tag=tag+'quasi-modulus')
    #       stress and strain tensors are visualized for cuts through the center of the volume
            for cut_type,center_idx in zip(cut_types,center_indices):
                tag = 'center_' + f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'strain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'nonlinearstrain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,center_idx,output_dir+'stress/center/',tag=tag+'stress')
    #       if particles present:
    #          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
            # if particles.shape[0] != 0:
            #     centers = np.zeros((particles.shape[0],3))
            #     for i, particle in enumerate(particles):
            #         tag=f"particle{i+1}_edge_" + f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
            #         centers[i,:] = simulate.get_particle_center(particle,initial_node_posns)
            #         particle_node_posns = initial_node_posns[particle,:]
            #         x_max = np.max(particle_node_posns[:,0])
            #         y_max = np.max(particle_node_posns[:,1])
            #         z_max = np.max(particle_node_posns[:,2])
            #         x_min = np.min(particle_node_posns[:,0])
            #         y_min = np.min(particle_node_posns[:,1])
            #         z_min = np.min(particle_node_posns[:,2])
            #         edge_indices = ((z_max,z_min),(y_max,y_min),(x_max,x_min))
            #         #TODO switchover to plotting tensorfields from this surf plot, but utilize the appropriate indices, cut types, and generate useful tags
            #         # for cut_type,layer_indices in zip(cut_types,edge_indices):
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,int(layer_indices[0]),output_dir+'strain/particle/',tag=tag+'strain')
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,int(layer_indices[1]),output_dir+'strain/particle/',tag='second'+tag+'strain')
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,int(layer_indices[0]),output_dir+'strain/particle/',tag=tag+'nonlinearstrain')
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,int(layer_indices[1]),output_dir+'strain/particle/',tag='second'+tag+'nonlinearstrain')
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,int(layer_indices[0]),output_dir+'stress/particle/',tag=tag+'stress')
            #         #     subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,int(layer_indices[1]),output_dir+'stress/particle/',tag='second'+tag+'stress')
            #     tag='particle_centers_'+ f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
            #     layers = (int((centers[0,2]+centers[1,2])/2),int((centers[0,1]+centers[1,1])/2),int((centers[0,0]+centers[1,0])/2))
            #     for cut_type,layer in zip(cut_types,layers):
            #         subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'strain')
            #         # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'nonlinearstrain')
            #         # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,layer,output_dir+'stress/particle/',tag=tag+'stress')
#   outside the loop:
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures

def linear_fit_func(x,m,b):
    """Used with scipy.optimize.curve_fit to try and extract the field dependent effective modulus from stress-strain curves"""
    return m*x + b

def analysis_average_stress_strain(sim_dir,gpu_flag=False):
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (3), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization, stress and strain tensors, and the effective modulus for an applied field"""
    #   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['modulus','particle_behavior','stress','strain','cuts','outer_surfaces']
    figure_subtypes = ['center', 'particle', 'outer_surface']
    for figure_type in figure_types:
        if not (os.path.isdir(output_dir+figure_type+'/')):
          os.mkdir(output_dir+figure_type+'/')
        if figure_type == 'stress' or figure_type =='strain' or figure_type == 'cuts':
            for figure_subtype in figure_subtypes:
                if not (figure_type == 'cuts' and figure_subtype == 'outer_surface'):
                    if not (os.path.isdir(output_dir+figure_type+'/'+figure_subtype+'/')):
                        os.mkdir(output_dir+figure_type+'/'+figure_subtype+'/')
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if gpu_flag:
        initial_node_posns = np.float64(initial_node_posns)
        beta_i = np.float64(beta_i)
        springs_var = np.float64(springs_var)
        kappa = np.float64(kappa)
        beta = np.float64(beta)
        l_e = np.float64(l_e)
        particle_mass = np.float64(particle_mass)
        particle_radius = np.float64(particle_radius)
        Ms = np.float64(Ms)
        chi = np.float64(chi)
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
    #see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))

    #get the the applied field associated with each output file
    Hext_series = get_applied_field_series(sim_dir)
    num_output_files = get_num_output_files(sim_dir)
#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables
#     #TODO effective modulus calculations for field dependent simulations
    if 'strain' in sim_dir:
        effective_modulus, stress, strain, Bext_series, strain_direction = get_field_dependent_effective_modulus(sim_dir)
    elif 'stress' in sim_dir:
        effective_modulus, differential_modulus, stress, strain, Bext_series, strain_direction = get_field_dependent_effective_modulus_stress_sim(sim_dir)
# #   outside the loop:
# #   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
# #   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
#     subplot_stress_field_modulus(stress,strain,strain_direction,effective_modulus,Bext_series,output_dir+'modulus/',tag="")
#     subplot_stress_strain_modulus_by_field(stress,strain,strain_direction,effective_modulus,Bext_series,output_dir+'modulus/',tag="")
#     # subplot_stress_field_modulus(alternative_stress_measure,strain,strain_direction,alternative_effective_modulus,Bext_series,output_dir+'modulus/',tag="alternative_measures")

#   in a loop, output files are read in and manipulated
    average_strain_tensor = np.zeros((num_output_files,3,3))
    average_stress_tensor = np.zeros((num_output_files,3,3))
    average_stiffness_matrix = np.zeros((num_output_files,6,6))
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
#       node positions are used to calculate nodal displacement
        displacement_field = get_displacement_field(initial_node_posns,final_posns)
#       nodal displacement is used to calculated displacement gradient
        gradu = get_gradu(displacement_field,num_nodes)
#       displacement gradient is used to calculate linear and nonlinear strain tensors
        strain_tensor = get_strain_tensor(gradu)
        # green_strain_tensor = get_green_strain_tensor(gradu)
#       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
        stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)

        average_stress_tensor[i] = get_average_tensor_value(stress_tensor,elements,initial_node_posns,tag='stress')
        average_strain_tensor[i] = get_average_tensor_value(strain_tensor,elements,initial_node_posns,tag='strain')
        average_stiffness_matrix[i,3,3] = average_stress_tensor[i,0,1]/average_strain_tensor[i,0,1]
        average_stiffness_matrix[i,4,4] = average_stress_tensor[i,0,2]/average_strain_tensor[i,0,2]
        average_stiffness_matrix[i,5,5] = average_stress_tensor[i,1,2]/average_strain_tensor[i,1,2]
        get_average_stiffness_tensor(average_strain_tensor[i],average_stress_tensor[i])
    Bext_series = mu0*Hext_series
    Bext_series_magnitude = np.round(np.linalg.norm(mu0*Hext_series,axis=1)*1e3,decimals=1)
    num_unique_fields = np.unique(Bext_series_magnitude).shape[0]
    relative_strain_tensor = np.zeros(np.shape(average_strain_tensor))
    relative_stress_tensor = np.zeros(np.shape(average_strain_tensor))
    for i in range(num_output_files):
        #2024-04-10 DBM:
        #i need to prove to myself that this is finding/using the correct reference configuration.
        #for any applied field, I want to compare to the zero stress case. i increment in field first, then stress. i'm pretty sure this was not done correctly
        #am i doing this correctly elsewhere? Update, yes it is working correctly.
        relative_configuration_index = np.mod(i,num_unique_fields)
        # print(f'step index: {i}; relative configuration index: {relative_configuration_index}')
        relative_strain_tensor[i] = average_strain_tensor[i] - average_strain_tensor[relative_configuration_index]
        relative_stress_tensor[i] = average_stress_tensor[i] - average_stress_tensor[relative_configuration_index]
        print(relative_stress_tensor[i,1,0])
    fig, axs = plt.subplots(2)
    axs[0].plot(relative_strain_tensor[:,1,0],average_stress_tensor[:,1,0])#axs[0].plot(average_strain_tensor[:,1,0],average_stress_tensor[:,1,0])
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('stress')
    apparent_shearing_modulus = np.abs(average_stress_tensor[:,1,0]/relative_strain_tensor[:,1,0])#apparent_shearing_modulus = average_stress_tensor[:,1,0]/average_strain_tensor[:,1,0]
    axs[1].plot(Bext_series_magnitude,apparent_shearing_modulus)
    axs[1].set_xlabel('B-Field (mT)')
    axs[1].set_ylabel('Modulus')
    plt.show()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].plot(relative_strain_tensor[:,1,0],relative_stress_tensor[:,1,0])#axs[0].plot(average_strain_tensor[:,1,0],average_stress_tensor[:,1,0])
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('stress')
    apparent_shearing_modulus = np.abs(relative_stress_tensor[:,1,0]/relative_strain_tensor[:,1,0])#apparent_shearing_modulus = average_stress_tensor[:,1,0]/average_strain_tensor[:,1,0]
    axs[1].plot(Bext_series_magnitude,apparent_shearing_modulus)
    axs[1].set_xlabel('B-Field (mT)')
    axs[1].set_ylabel('Modulus')
    plt.show()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].plot(relative_strain_tensor[:,1,0],stress[:,0])#axs[0].plot(average_strain_tensor[:,1,0],average_stress_tensor[:,1,0])
    axs[0].set_xlabel('strain')
    axs[0].set_ylabel('stress')
    apparent_shearing_modulus = np.abs(stress[:,0]/relative_strain_tensor[:,1,0])#apparent_shearing_modulus = average_stress_tensor[:,1,0]/average_strain_tensor[:,1,0]
    axs[1].plot(Bext_series_magnitude,apparent_shearing_modulus)
    axs[1].set_xlabel('B-Field (mT)')
    axs[1].set_ylabel('Modulus')
    plt.show()
    plt.close()

def get_average_tensor_value(tensor,elements,initial_node_posns,tag=""):
    """Calculate the average stress/strain tensor for the RVE/system and compare the results."""
    simplest_average = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            simplest_average[i,j] = np.mean(np.ravel(tensor[:,:,:,i,j]))
    simplest_average[1,0] = simplest_average[0,1]
    simplest_average[2,0] = simplest_average[0,2]
    simplest_average[2,1] = simplest_average[1,2]
    return simplest_average

def get_average_stiffness_tensor(strain_tensor,stress_tensor):
    strain_inverse = np.linalg.inv(strain_tensor)
    some_result =  np.tensordot(stress_tensor,strain_inverse,axes=0)
    # print(some_result.shape)
    return some_result

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
    total_num_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    k = mre.initialize.get_spring_constants(E, l_e)
    k = np.array(k)

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions

def format_boundary_conditions(boundary_conditions):
    boundary_conditions = (str(boundary_conditions[0][0])[1:],(str(boundary_conditions[0][1])[1:],str(boundary_conditions[0][2])[1:]),boundary_conditions[0][3])
    boundary_conditions = (boundary_conditions[0][1:-1],(boundary_conditions[1][0][1:-1],boundary_conditions[1][1][1:-1]),boundary_conditions[2])
    return boundary_conditions

def get_effective_modulus_strain_series(sim_dir):
    """Given a simulation directory, calculate and plot the stress-strain curve and effective modulus versus strain. The returned stress variable is the effective stress applied to the surface equivalent to the "probe" in an indentation experiment. The secondary_stress variable is the effective stress applied to the surface held fixed by a table/wall/platform in an experimental measurement. The effective modulus is calculated using the effective stress applied to the "probed" surface."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain_type = boundary_conditions[0]
    strain_direction = boundary_conditions[1]
    if strain_type == 'tension' or strain_type == 'compression':
        effective_modulus, stress, strains, secondary_stress = get_tension_compression_modulus(sim_dir,strain_direction)
        effective_modulus_alt, stress_alt, strains_alt, secondary_stress_alt = get_strain_dependent_tension_compression_modulus(sim_dir,strain_direction)
    elif strain_type == 'shearing':
        effective_modulus, stress, strains, secondary_stress = get_shearing_modulus(sim_dir,strain_direction)
        effective_modulus, stress, strains, secondary_stress = get_strain_dependent_shearing_modulus(sim_dir,strain_direction)
    elif strain_type == 'torsion':
        get_torsion_modulus(sim_dir,strain_direction)
    return effective_modulus, stress, strains, strain_direction

def get_effective_modulus_stress_sim(sim_dir):
    """Given a simulation directory, calculate and plot the stress-strain curve and effective modulus versus stress/strain. The returned strain variable is the "effective" strain. The effective modulus is calculated using the stress applied to the "probed" surface and the effective strain."""
    raise NotImplementedError('This method has not been properly implemented, and should not be used.')
    # _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    # boundary_conditions = format_boundary_conditions(boundary_conditions)
    # stress_type = boundary_conditions[0]
    # stress_direction = boundary_conditions[1]
    # if stress_type == 'tension' or stress_type == 'compression':
    #     effective_modulus, stress, strains = get_tension_compression_modulus(sim_dir,stress_direction)
    # elif stress_type == 'shearing':
    #     effective_modulus, stress, strains = get_shearing_modulus(sim_dir,stress_direction)
    # elif stress_type == 'torsion':
    #     get_torsion_modulus(sim_dir,stress_direction)
    # return effective_modulus, stress, strains, stress_direction

def get_field_dependent_effective_modulus(sim_dir):
    """Given a simulation directory, calculate and plot the effective modulus versus applied field. The returned stress variable is the effective stress applied to the surface equivalent to the "probe" in an indentation experiment. The secondary_stress variable is the effective stress applied to the surface held fixed by a table/wall/platform in an experimental measurement. The effective modulus is calculated using the effective stress applied to the "probed" surface."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain_type = boundary_conditions[0]
    strain_direction = boundary_conditions[1]
    if strain_type == 'tension' or strain_type == 'compression' or strain_type == 'plate_compression' or strain_type == 'plate_compre' or 'tension' in strain_type:
        effective_modulus, stress, strain, Bext_series = get_field_dependent_tension_compression_modulus(sim_dir,strain_direction)
    elif strain_type == 'shearing':
        effective_modulus, stress, strain, Bext_series = get_field_dependent_shearing_modulus(sim_dir,strain_direction)
    elif strain_type == 'torsion':
        get_torsion_modulus(sim_dir,strain_direction)
    return effective_modulus, stress, strain, Bext_series, strain_direction#, alternative_stress_measure, alternative_effective_modulus

def get_field_dependent_effective_modulus_stress_sim(sim_dir):
    """Given a simulation directory, calculate and plot the effective modulus versus applied field. The returned stress variable is the effective strain using the surface equivalent to the "probe" in an indentation experiment. The effective modulus is calculated using the applied stress and effective strain achieved using displacement of the "probed" surface."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    bc_type = boundary_conditions[0]
    bc_direction = boundary_conditions[1]
    if bc_type =='simple_stress_compression' or 'tension' in bc_type:
        effective_modulus, stress, strain, Bext_series = get_field_dependent_tension_compression_modulus(sim_dir,bc_direction,sim_dir+'figures/modulus/')
    # if bc_type == 'tension' or bc_type == 'compression' or bc_type == 'plate_compression' or bc_type == 'plate_compre':
    #     effective_modulus, stress, strain, Bext_series = get_field_dependent_tension_compression_modulus(sim_dir,bc_direction)
    elif bc_type == 'shearing' or ('shearing' in bc_type):
        effective_modulus, stress, strain, Bext_series = get_field_dependent_shearing_modulus(sim_dir,bc_direction)
                # differential modulus should have the same shape as effective modulus (maybe it could be a different shape, but i need to make new plotting functions for the differential modulus). I need to take into account that i am iterating over the applied field avlue and the applied stress values
                # if j < num_fields-1:
                #     differential_modulus[i,j] = (single_field_stress[i+1] - single_field_stress[i])/(single_field_strain[i+1] - single_field_strain[i])
        # differential_modulus = (stress[1:] - stress[0:-1]) / (strain[1:] - strain[0:-1])
    # elif bc_type == 'torsion':
    #     get_torsion_modulus(sim_dir,bc_direction)
    unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1))
    num_fields = unique_fields.shape[0]
    differential_modulus = np.zeros((num_fields,np.unique(stress).shape[0]))
    counter = 0
    for i, field in enumerate(unique_fields):
        relevant_indices = np.isclose(field,np.linalg.norm(Bext_series,axis=1))
        single_field_stress = stress[relevant_indices]
        single_field_strain = strain[relevant_indices]
        for j in range(single_field_stress.shape[0]):
            if j == (single_field_stress.shape[0] - 1):
                differential_modulus[i,j] = (single_field_stress[j] - single_field_stress[j-1])/(single_field_strain[j] - single_field_strain[j-1])
            elif j == 0:
                differential_modulus[i,j] = (single_field_stress[j+1] - single_field_stress[j])/(single_field_strain[j+1] - single_field_strain[j])
            else:
                differential_modulus[i,j] = ((single_field_stress[j+1] - single_field_stress[j])/(single_field_strain[j+1] - single_field_strain[j]) + (single_field_stress[j] - single_field_stress[j-1])/(single_field_strain[j] - single_field_strain[j-1]))/2
            counter += 1
    return effective_modulus, differential_modulus, stress, strain, Bext_series, bc_direction#, alternative_stress_measure, alternative_effective_modulus

def get_strain_dependent_tension_compression_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of strain values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    strains = series
    n_strain_steps = len(series)
    stress = np.zeros((n_strain_steps,3))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        effective_modulus[i], stress[i], strains[i], secondary_stress[i] = get_tension_compression_modulus_v2(sim_dir,i,strain_direction,beta_i, springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions)
    return effective_modulus, stress, strains, secondary_stress

def get_field_dependent_tension_compression_modulus(sim_dir,bc_direction,output_dir=None):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of applied field values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    Hext_series = get_applied_field_series(sim_dir)
    Bext_series = mu0*Hext_series
    num_unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1)).shape[0]
    n_series_steps = np.shape(Bext_series)[0]
    stress = np.zeros((n_series_steps,))
    strain = np.zeros((n_series_steps,))
    effective_modulus = np.zeros((n_series_steps,))
    if 'stress' in sim_dir:
        zero_stress_comparison_values = np.zeros((num_unique_fields,),dtype=np.float32)
    elif 'strain' in sim_dir:
        stress = np.zeros((n_series_steps,),dtype=np.float32)
        strain = np.zeros((n_series_steps,),dtype=np.float32)
        zero_strain_comparison_forces = np.zeros((num_unique_fields,3),dtype=np.float32)
        springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
        elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
        device_beta_i = cp.array(beta_i.astype(np.float32)).reshape((beta_i.shape[0],1),order='C')
        kappa = cp.float32(kappa*(l_e**2))
    for i in range(num_unique_fields):
        final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        assert(np.isclose(np.linalg.norm(mu0*applied_field),np.linalg.norm(Bext_series[i])))
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        if not np.isclose(boundary_conditions[2],0):
            raise ValueError('Unexpected non-zero value for boundary condition while calculating comparison system configuration metric used to define strain or comparison force used to define stress')
        if bc_direction[0] == 'x':
            relevant_boundary = 'right'
        elif bc_direction[0] == 'y':
            relevant_boundary = 'back'
        elif bc_direction[0] == 'z':
            relevant_boundary = 'top'
        if bc_direction[1] == 'x':
            coordinate_index = 0
        elif bc_direction[1] == 'y':
            coordinate_index = 1
        elif bc_direction[1] == 'z':
            coordinate_index = 2
        if 'simple_stress' in boundary_conditions[0]:
            zero_stress_comparison_values[i] = np.mean(final_posns[boundaries[relevant_boundary],coordinate_index],dtype=np.float32)
        elif 'strain' in boundary_conditions[0]:
            final_posns, _, _, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
            posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
            velocities = cp.zeros(posns.shape,order='C',dtype=cp.float32)
            N_nodes = int(posns.shape[0]/3)
            accel = simulate.composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs_var,device_beta_i,drag)
            accel = np.reshape(accel,(N_nodes,3))
            zero_strain_comparison_forces[i] = np.sum(-1*accel[boundaries[relevant_boundary]]/beta_i[boundaries[relevant_boundary],np.newaxis],axis=0)
    for i in range(n_series_steps):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        if 'simple_stress' in boundary_conditions[0]:
            effective_modulus[i], stress[i], strain[i], _ = get_tension_compression_modulus_v2(sim_dir,i,bc_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions,zero_stress_comparison_values,output_dir)
        elif 'strain' in boundary_conditions[0]:
            effective_modulus[i], stress[i], strain[i], _ = get_tension_compression_modulus_v2(sim_dir,i,bc_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions,zero_strain_comparison_forces=zero_strain_comparison_forces,output_dir=output_dir)
    return effective_modulus, stress, strain, Bext_series

def get_tension_compression_modulus_v2(sim_dir,output_file_number,bc_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions,zero_stress_comparison_values=None,zero_strain_comparison_forces=None,output_dir=None):
    """For a given configuration of nodes and particles, calculate the effective modulus, effective stress on the probe surface, and return those values."""
    force_component = {'x':0,'y':1,'z':2}
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    Hext = np.float64(applied_field)
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    y = np.zeros((6*total_num_nodes,))
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    particle_moment_of_inertia = 1
    if bc_direction[0] == 'x':
        #forces that must act on the boundaries for them to be in this position
        relevant_boundaries = ('right','left')
        dimension_indices = (1,2)
    elif bc_direction[0] == 'y':
        relevant_boundaries = ('back','front')
        dimension_indices = (0,2)
    elif bc_direction[0] == 'z':
        relevant_boundaries = ('top','bot')
        dimension_indices = (0,1)
    #if the boundary conditions are strain based, need to get accelerations calculated where we do not set the boundary nodes accelerations to zero
    if boundary_conditions[0] == 'tension' or boundary_conditions[0] == 'compression':
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        first_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[1]]]/beta_i[boundaries[relevant_boundaries[1]],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        stress = first_bdry_stress
        secondary_stress = second_bdry_stress
        strain = boundary_conditions[2]
        if strain == 0:# and np.isclose(np.linalg.norm(stress[i,:]),0):
            effective_modulus = E
        else:
            effective_modulus = np.abs(stress[force_component[bc_direction[1]]]/strain)
    elif 'strain' in boundary_conditions[0]:
        device_beta_i = cp.array(beta_i.astype(np.float32)).reshape((beta_i.shape[0],1),order='C')
        posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
        velocities = cp.zeros(posns.shape,order='C',dtype=cp.float32)
        N_nodes = int(posns.shape[0]/3)
        accel = simulate.composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs_var,device_beta_i,drag)
        accel = np.reshape(accel,(N_nodes,3))
        boundary_forces = np.sum(-1*accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis],axis=0)
        output_file_number_for_comparison = int(np.mod(output_file_number,zero_strain_comparison_forces.shape[0]))
        relative_boundary_forces = boundary_forces - zero_strain_comparison_forces[output_file_number_for_comparison] 
        relative_boundary_stress = relative_boundary_forces[force_component[boundary_conditions[1][1]]]/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        stress = relative_boundary_stress
        strain = boundary_conditions[2]
        secondary_stress = None
        if strain == 0:
            effective_modulus = E
        else:
            effective_modulus = np.abs(stress/strain)
    else:
    #if the boundary conditions are stress based, need to consider the forces acting on the boundary nodes and the enforced stress boundary conditions
        if bc_direction[0] == 'x':
            relevant_boundary = 'right'
        elif bc_direction[0] == 'y':
            relevant_boundary = 'back'
        elif bc_direction[0] == 'z':
            relevant_boundary = 'top'
        if bc_direction[1] == 'x':
            coordinate_index = 0
        elif bc_direction[1] == 'y':
            coordinate_index = 1
        elif bc_direction[1] == 'z':
            coordinate_index = 2
        if boundary_conditions[0] == 'simple_stress_compression' or boundary_conditions[0] == 'stress_compression' or ('simple_stress' in boundary_conditions[0]):
            stress = boundary_conditions[2]
            # or i can try to take into account the deformed configuration having a different surface area, meaning the actual stress value is slightly different
            # undeformed_boundary_surface_area = dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]]/np.power(l_e,2)
            # deformed_boundary_surface_area = np.max(final_posns[boundaries[relevant_boundary],dimension_indices[0]])*np.max(final_posns[boundaries[relevant_boundary],dimension_indices[1]])
            # surface_area_ratio = undeformed_boundary_surface_area/deformed_boundary_surface_area
            # stress *= surface_area_ratio
        elif boundary_conditions[0] == 'plate_compression':
            end_accel = simulate.get_accel_scaled_rotation(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
            stress = np.sum(end_accel[boundaries[relevant_boundary],coordinate_index]/beta_i[boundaries[relevant_boundary]])
        print(f'max coordinate in relevant direction: {np.max(final_posns[boundaries[relevant_boundary],coordinate_index])}')
        print(f'mean surface coordinate in relevant direction: {np.mean(final_posns[boundaries[relevant_boundary],coordinate_index])}')
        output_file_number_for_comparison = int(np.mod(output_file_number,zero_stress_comparison_values.shape[0]))
        comparison_length = zero_stress_comparison_values[output_file_number_for_comparison]
        strain = np.max(final_posns[boundaries[relevant_boundary],coordinate_index])/(dimensions[coordinate_index]/l_e) - 1
        #TODO utilize both strain measures and create figures for stress-strain curve and effective modulus versus field
        # strain_from_mean_surface_posn = np.mean(final_posns[boundaries[relevant_boundary],coordinate_index])/(dimensions[coordinate_index]/l_e) - 1
        mean_boundary_posn = np.mean(final_posns[boundaries[relevant_boundary],coordinate_index])
        strain = mean_boundary_posn/comparison_length - 1
        secondary_stress = None
        effective_modulus = np.abs(stress/strain)
        boundary_node_posns = final_posns[boundaries[relevant_boundary],coordinate_index]
        if not output_dir == None:
            plot_boundary_node_posn_hist(boundary_node_posns,output_dir,tag=f"{output_file_number}")
    return effective_modulus, stress, strain, secondary_stress

def get_tension_compression_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied."""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    strains = series
    n_strain_steps = len(series)
    stress = np.zeros((n_strain_steps,3))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    force_component = {'x':0,'y':1,'z':2}
    y = np.zeros((6*total_num_nodes,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        Hext = applied_field
        y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        if strain_direction[0] == 'x':
            #forces that must act on the boundaries for them to be in this position
            relevant_boundaries = ('right','left')
            dimension_indices = (1,2)
        elif strain_direction[0] == 'y':
            relevant_boundaries = ('back','front')
            dimension_indices = (0,2)
        elif strain_direction[0] == 'z':
            relevant_boundaries = ('top','bot')
            dimension_indices = (0,1)
        first_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[1]]]/beta_i[boundaries[relevant_boundaries[1]],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        # print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[force_component[strain_direction[1]]])-np.abs(second_bdry_stress[force_component[strain_direction[1]]])}')
        stress[i] = first_bdry_stress
        secondary_stress[i] = second_bdry_stress
    for i in range(np.shape(strains)[0]):
        if strains[i] == 0:# and np.isclose(np.linalg.norm(stress[i,:]),0):
            effective_modulus[i] = E
        else:
            effective_modulus[i] = np.abs(stress[i,force_component[strain_direction[1]]]/strains[i])
    return effective_modulus, stress, strains, secondary_stress

def get_strain_dependent_shearing_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of strain values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    strains = series
    n_strain_steps = len(series)
    stress = np.zeros((n_strain_steps,3))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        effective_modulus[i], stress[i], strains[i], secondary_stress[i] = get_shearing_modulus_v2(sim_dir,i,strain_direction,beta_i, springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions)
    return effective_modulus, stress, strains, secondary_stress

def get_field_dependent_shearing_modulus(sim_dir,bc_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of applied field values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    Hext_series = get_applied_field_series(sim_dir)
    Bext_series = mu0*Hext_series
    num_unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1)).shape[0]
    n_series_steps = np.shape(Bext_series)[0]
    stress = np.zeros((n_series_steps,))
    strain = np.zeros((n_series_steps,))
    effective_modulus = np.zeros((n_series_steps,))
    zero_stress_comparison_values = np.zeros((num_unique_fields,2))
    for i in range(num_unique_fields):
        final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        assert(np.isclose(np.linalg.norm(applied_field),np.linalg.norm(Bext_series[i])))
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        coordinate_index = [0,0]
        if not np.isclose(boundary_conditions[2],0):
            raise ValueError('Unexpected non-zero value for boundary condition while calculating comparison system configuration metric used to define strain')
        if bc_direction[0] == 'x':
            relevant_boundary = 'right'
            coordinate_index[0] = 0
        elif bc_direction[0] == 'y':
            relevant_boundary = 'back'
            coordinate_index[0] = 1
        elif bc_direction[0] == 'z':
            relevant_boundary = 'top'
            coordinate_index[0] = 2
        if bc_direction[1] == 'x':
            coordinate_index[1] = 0
        elif bc_direction[1] == 'y':
            coordinate_index[1] = 1
        elif bc_direction[1] == 'z':
            coordinate_index[1] = 2
        if 'simple_stress' in boundary_conditions[0]:
            zero_stress_comparison_values[i,0] = np.mean(final_posns[boundaries[relevant_boundary],coordinate_index[0]])
            zero_stress_comparison_values[i,1] = np.mean(final_posns[boundaries[relevant_boundary],coordinate_index[1]])

    for i in range(n_series_steps):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        effective_modulus[i], stress[i], strain[i], _ = get_shearing_modulus_v2(sim_dir,i,bc_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions,zero_stress_comparison_values)
    return effective_modulus, stress, strain, Bext_series#, alternative_stress_measure, alternative_effective_modulus

def get_shearing_modulus_v2(sim_dir,output_file_number,bc_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions,zero_stress_comparison_values=None,output_dir=None):
    #TODO finish this function. shearing in different directions from the same surface is a different modulus (there is anisotropy, or should assume there is). use the direction of the shearing for title, labels, and save name for the figures generated (though that may not occur in this function)
    """Calculate a shear modulus, using the shear strain (shearing angle) and the force applied to the sheared surface in the shearing direction to get a shear stress."""
    #nonlinear shear strains are defined as tangent of the angle opened up by the shearing. linear shear strain is the linear, small angle approximation of tan theta ~= theta
    force_component = {'x':0,'y':1,'z':2}
    y = np.zeros((6*total_num_nodes,))
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    Hext = applied_field
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    if not 'stress' in boundary_conditions[0]:
        strain = np.tan(boundary_conditions[2])
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        if bc_direction[0] == 'x':
            relevant_boundaries = ('right','left')
            dimension_indices = (1,2)
        elif bc_direction[0] == 'y':
            relevant_boundaries = ('back','front')
            dimension_indices = (0,2)
        elif bc_direction[0] == 'z':
            relevant_boundaries = ('top','bot')
            dimension_indices = (0,1)
        first_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
        second_bdry_forces = -1*end_accel[boundaries[relevant_boundaries[1]]]/beta_i[boundaries[relevant_boundaries[1]],np.newaxis]
        first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        second_bdry_stress = np.sum(second_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
        stress = first_bdry_stress
        secondary_stress = second_bdry_stress
        if strain == 0:# and np.isclose(np.linalg.norm(stress),0):
            effective_modulus = E/3
        else:
            effective_modulus = np.abs(stress[force_component[bc_direction[1]]]/strain)
    else:
        #if the boundary conditions are stress based, need to consider the forces acting on the boundary nodes and the enforced stress boundary conditions
        if bc_direction[0] == 'x':
            relevant_boundary = 'right'
            if bc_direction[1] == 'y':
                coordinate_indices = [0,1]
            elif bc_direction[1] == 'z':
                coordinate_indices = [0,2]
        elif bc_direction[0] == 'y':
            relevant_boundary = 'back'
            if bc_direction[1] == 'x':
                coordinate_indices = [1,0]
            elif bc_direction[1] == 'z':
                coordinate_indices = [1,2]
        elif bc_direction[0] == 'z':
            relevant_boundary = 'top'
            if bc_direction[1] == 'x':
                coordinate_indices = [2,0]
            elif bc_direction[1] == 'y':
                coordinate_indices = [2,1]
        stress = boundary_conditions[2]
        #calculate the angle that opens up... invtan(opp/adj)
        output_file_number_for_comparison = int(np.mod(output_file_number,zero_stress_comparison_values.shape[0]))
        comparison_system_length = zero_stress_comparison_values[output_file_number_for_comparison,0]
        comparison_boundary_position = zero_stress_comparison_values[output_file_number_for_comparison,1]
        mean_surface_posn = np.mean(final_posns[boundaries[relevant_boundary],coordinate_indices[0]]) #aka adjacent side length
        print(f'starting surface position in shearing coordinate: {dimensions[coordinate_indices[1]]/l_e/2}')
        print(f'mean surface shearing position: {np.mean(final_posns[boundaries[relevant_boundary],coordinate_indices[1]])}')
        old_mean_surface_shearing_displacement = np.mean(final_posns[boundaries[relevant_boundary],coordinate_indices[1]]) - dimensions[coordinate_indices[1]]/l_e/2 #find the midpoint of the surface (in it's initial configuration) along the shearing direction, and subtract that from the current midpoint of the surface in the shearing direction
        old_strain = np.arctan(old_mean_surface_shearing_displacement/mean_surface_posn)
        #using comparison to system configuration at zero stress but the same applied field to determine the actual impact of the applied stress on the strain of the system
        mean_surface_shearing_displacement = np.mean(final_posns[boundaries[relevant_boundary],coordinate_indices[1]]) - comparison_boundary_position
        strain = np.arctan(mean_surface_shearing_displacement/comparison_system_length)
        effective_modulus = stress/strain
        secondary_stress = None
        boundary_node_posns = final_posns[boundaries[relevant_boundary],coordinate_indices[1]]
        if not output_dir == None:
            plot_boundary_node_posn_hist(boundary_node_posns,output_dir,tag=f"{output_file_number}")
    return effective_modulus, stress, strain, secondary_stress#, first_bdry_forces

def get_shearing_modulus(sim_dir,strain_direction):
    #TODO finish this function. shearing in different directions from the same surface is a different modulus (there is anisotropy, or should assume there is). use the direction of the shearing for title, labels, and save name for the figures generated (though that may not occur in this function)
    """Calculate a shear modulus, using the shear strain (shearing angle) and the force applied to the sheared surface in the shearing direction to get a shear stress."""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    #nonlinear shear strains are defined as tangent of the angle opened up by the shearing. linear shear strain is the linear, small angle approximation of tan theta ~= theta
    strains = np.tan(series)
    n_strain_steps = len(series)
    stress = np.zeros((n_strain_steps,3))
    secondary_stress = np.zeros((n_strain_steps,3))
    effective_modulus = np.zeros((n_strain_steps,))
    force_component = {'x':0,'y':1,'z':2}
    y = np.zeros((6*total_num_nodes,))
    for i in range(len(series)):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        Hext = applied_field
        y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        if strain_direction[0] == 'x':
            relevant_boundaries = ('right','left')
            dimension_indices = (1,2)
            if strain_direction[1] == 'y':
                pass
            elif strain_direction[1] == 'z':
                pass
        elif strain_direction[0] == 'y':
            relevant_boundaries = ('back','front')
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
        # print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[force_component[strain_direction[1]]])-np.abs(second_bdry_stress[force_component[strain_direction[1]]])}')
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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if strain_direction[1] == 'CW':
        pass
    elif strain_direction[1] == 'CCW':
        pass

def get_probe_boundary_forces(sim_dir,output_file_number,strain_direction,beta_i,springs_var,elements,boundaries,dimensions,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag):
    """For a given configuration of nodes and particles, calculate the forces on the probe surface, and return those values."""
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    Hext = np.float64(applied_field)
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    y = np.zeros((6*total_num_nodes,))
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    particle_moment_of_inertia = 1
    end_accel = simulate.get_accel_scaled_rotation(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
    # end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
    if strain_direction[0] == 'x':
        #forces that must act on the boundaries for them to be in this position
        relevant_boundaries = ('right','left')
    elif strain_direction[0] == 'y':
        relevant_boundaries = ('back','front')
    elif strain_direction[0] == 'z':
        relevant_boundaries = ('top','bot')
    boundary_forces = -1*end_accel[boundaries[relevant_boundaries[0]]]/beta_i[boundaries[relevant_boundaries[0]],np.newaxis]
    return boundary_forces, end_accel/beta_i[:,np.newaxis]

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

#used for getting stress, strain tensors, displacement vector field, and displacement gradient tensor field
def get_displacement_field(initial_posns,final_posns):
    """Calculate the displacement of each node from its initial position."""
    displacement = final_posns - initial_posns
    return displacement

def get_gradu(displacement,dimensions):
    """Get the tensor at each point on the grid representing the initial node positions which is the gradient of the displacement field"""
    xdisplacement_3D, ydisplacement_3D, zdisplacement_3D = get_component_3D_arrays(displacement,dimensions)
    xdisplacement_gradient = np.gradient(xdisplacement_3D)
    ydisplacement_gradient = np.gradient(ydisplacement_3D)
    zdisplacement_gradient = np.gradient(zdisplacement_3D)
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
    """Calculate the symmetric, nonlinear strain tensor at each point on the grid representing the initial node positions using the gradient of the displacement field. The Langrangian finite strain tensor, aka Green-Lagrangian strain tensor or Green--St-Venant strain tensor"""
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

#manipulating arrays to get them in an appropriate shape for certain types of visualization, like wireframe and surface plots

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

#Stress, strain, and vector field visualization functions
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
            vectorfield_component = np.sqrt(np.power(vectorfield[0],2) + np.power(vectorfield[1],2) + np.power(vectorfield[2],2))
        if cut_type == 'xy':
            Z = vectorfield_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif cut_type == 'xz':
            Z = vectorfield_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        elif cut_type == 'yz':
            Z = vectorfield_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        color_dimension = Z
        color_min, color_max = color_dimension.min(), color_dimension.max()
        colorbar_limit = np.max([np.abs(color_min),np.abs(color_max)])
        colorbar_max = colorbar_limit
        colorbar_min = -1*colorbar_limit
        norm = matplotlib.colors.Normalize(colorbar_min,colorbar_max)
        my_cmap = cm.ScalarMappable(norm=norm)
        my_cmap.set_array([])
        # fig.colorbar(img,ax=ax)
        img = ax.pcolormesh(X,Y,Z,norm=norm)
        cbar = fig.colorbar(img,ax=ax)
        cbar.ax.tick_params(labelsize=25)
        # fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(f' {component_dict[i]} ')
        # ax.set_title(tag+ f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        format_figure(ax)
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
    #want to store the colorbar limits used in each image, so that colorbar limits in each row of images are the same
    colorbar_limits = np.zeros((6,))
    component_dict = {0:'xx',1:'yy',2:'zz',3:'xy',4:'xz',5:'yz'}
    #go through the 6 types of figures to make, and find out the colorbar limits for each figure, so that the proper limits can be used when the plots are created
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
        elif cut_type == 'xz':
            Z = tensor_component[:,index,:]
        elif cut_type == 'yz':
            Z = tensor_component[index,:,:]
        color_dimension = Z
        color_min, color_max = color_dimension.min(), color_dimension.max()
        colorbar_limit = np.max([np.abs(color_min),np.abs(color_max)])
        colorbar_limits[i] = colorbar_limit
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
        elif cut_type == 'xz':
            Z = tensor_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        elif cut_type == 'yz':
            Z = tensor_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        if i >= 0 and i < 3:
            colorbar_limit = np.max(colorbar_limits[:3])
        else:
            colorbar_limit = np.max(colorbar_limits[3:])
        colorbar_max = colorbar_limit
        colorbar_min = -1*colorbar_limit
        norm = matplotlib.colors.Normalize(colorbar_min,colorbar_max)
        my_cmap = cm.ScalarMappable(norm=norm)
        my_cmap.set_array([])
        # fig.colorbar(my_cmap,ax=ax)
        img = ax.pcolormesh(X,Y,Z,norm=norm)
        cbar = fig.colorbar(img,ax=ax)
        cbar.ax.tick_params(labelsize=25)
        if 'stress' in tag:
            cbar.ax.set_ylabel('Stress (Pa)', rotation=270,fontsize=20)
            if i == 0:
                ax.set_title(r'$\sigma_{xx}$')
            elif i == 1:
                ax.set_title(r'$\sigma_{yy}$')
            elif i == 2:
                ax.set_title(r'$\sigma_{zz}$')
            elif i == 3:
                ax.set_title(r'$\sigma_{xy}$')
            elif i == 4:
                ax.set_title(r'$\sigma_{xz}$')
            elif i == 5:
                ax.set_title(r'$\sigma_{yz}$')
        elif 'strain' in tag:
            if i == 0:
                ax.set_title(r'$\epsilon_{xx}$')
            elif i == 1:
                ax.set_title(r'$\epsilon_{yy}$')
            elif i == 2:
                ax.set_title(r'$\epsilon_{zz}$')
            elif i == 3:
                ax.set_title(r'$\epsilon_{xy}$')
            elif i == 4:
                ax.set_title(r'$\epsilon_{xz}$')
            elif i == 5:
                ax.set_title(r'$\epsilon_{yz}$')
        # ax.set_title(tag+f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.axis('equal')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        format_figure(ax)
    # plt.show()
    fig.tight_layout()
    savename = output_dir + f'subplots_{cut_type}_cut_pcolormesh_'+tag+'_tensorfield_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_strain_modulus(stress,strain,strain_direction,effective_modulus,output_dir,tag=""):
    """Generate figure with subplots of stress-strain curve and effective modulus versus strain."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    force_component = {'x':0,'y':1,'z':2}
    axs[0].plot(strain,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
    axs[0].set_title('Stress vs Strain')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress')
    axs[1].plot(strain,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus vs Strain')
    axs[1].set_xlabel('Strain')
    axs[1].set_ylabel('Effective Modulus')
    format_figure(axs[0])
    format_figure(axs[1])
    # plt.show()
    fig.tight_layout()
    savename = output_dir + f'subplots_stress-strain_effective_modulus_'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_field_modulus(stress,strains,strain_direction,effective_modulus,Bext_series,output_dir,tag=""):
    """Generate figure with subplots of stress-field curve and effective modulus versus field."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    force_component = {'x':0,'y':1,'z':2}
    Bext_magnitude = np.linalg.norm(Bext_series,axis=1)
    plotting_indices = np.abs(strains) > 0
    stress = stress[plotting_indices]
    Bext_magnitude = Bext_magnitude[plotting_indices]
    effective_modulus = effective_modulus[plotting_indices]
    axs[0].plot(Bext_magnitude,np.abs(stress),'-o')
    # axs[0].plot(Bext_magnitude,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
    axs[0].set_title('Stress vs Applied Field')
    axs[0].set_xlabel('Applied Field (T)')
    axs[0].set_ylabel('Stress')
    axs[1].plot(Bext_magnitude,effective_modulus,'-o')
    axs[1].set_title('Effective Modulus vs Applied Field')
    axs[1].set_xlabel('Applied Field (T)')
    axs[1].set_ylabel('Effective Modulus')
    format_figure(axs[0])
    format_figure(axs[1])
    # plt.show()
    fig.tight_layout()
    savename = output_dir + f'subplots_stress-field_effective_modulus_'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_field_differentialmodulus(stress,strains,differential_modulus,Bext_series,output_dir,tag=""):
    """Generate figure with subplots of stress-field curve and effective modulus versus field."""
    fig, ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    Bext_magnitude = np.unique(np.linalg.norm(Bext_series,axis=1))
    for i in range(differential_modulus.shape[0]):
        ax.plot(Bext_magnitude[i]*np.ones((differential_modulus.shape[1],)),differential_modulus[i],'-o')
    ax.set_title('Effective Modulus vs Applied Field')
    ax.set_xlabel('Applied Field (T)')
    ax.set_ylabel('Effective Modulus')
    format_figure(ax)
    # plt.show()
    fig.tight_layout()
    savename = output_dir + f'subplots_stress-field_effective_modulus_'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_strain_modulus_by_field(stress,strains,strain_direction,effective_modulus,Bext_series,output_dir,tag=""):
    """Generate figure with subplots of stress-field curve and effective modulus versus field."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    force_component = {'x':0,'y':1,'z':2}
    Bext_magnitude = np.linalg.norm(Bext_series,axis=1)
    unique_field_values = np.unique(Bext_magnitude)

    # plotting_indices = np.abs(strains) > 0
    # stress = stress[plotting_indices]
    # Bext_magnitude = Bext_magnitude[plotting_indices]
    # effective_modulus = effective_modulus[plotting_indices]
    for unique_value in unique_field_values:
        plotting_indices = np.nonzero(np.isclose(unique_value,Bext_magnitude))
        plotting_indices = plotting_indices[0]
        plotting_strains = strains[plotting_indices]
        plotting_stresses = stress[plotting_indices]
        plotting_effective_modulus = effective_modulus[plotting_indices]
        if unique_value != 0:
            marker = 'o'
        else:
            marker = 's'
        linestyle = '-'
        axs[0].plot(plotting_strains,np.abs(plotting_stresses),marker=marker,linestyle=linestyle,label=f'{np.round(unique_value*1000,decimals=2)} (mT)')
        axs[1].plot(np.abs(plotting_stresses),plotting_effective_modulus,marker=marker,linestyle=linestyle)
        # axs[0].plot(plotting_strains,np.abs(plotting_stresses[:,force_component[strain_direction[1]]]),'-o',label=f'{np.round(unique_value*1000)} (mT)')
        # axs[1].plot(np.abs(plotting_stresses[:,force_component[strain_direction[1]]]),plotting_effective_modulus,'-o')
    axs[0].set_title('Stress vs Strain')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress')
    axs[1].set_title('Effective Modulus vs Stress')
    axs[1].set_xlabel('Stress (Pa)')
    axs[1].set_ylabel('Effective Modulus')
    format_figure(axs[0])
    format_figure(axs[1])
    # plt.show()
    fig.tight_layout()
    fig.legend()
    savename = output_dir + f'subplots_stress-strain_effective_modulus_labeled_lines'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_strain_differentialmodulus_by_field(stress,strains,differential_modulus,Bext_series,output_dir,tag=""):
    """Generate figure with subplots of stress-strain curve and differential modulus versus field."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    Bext_magnitude = np.linalg.norm(Bext_series,axis=1)
    unique_field_values = np.unique(Bext_magnitude)
    for i, unique_value in enumerate(unique_field_values):
        relevant_indices = np.isclose(unique_value,np.linalg.norm(Bext_series,axis=1))
        plotting_stresses = stress[relevant_indices]
        plotting_strains = strains[relevant_indices]
        plotting_differential_modulus = differential_modulus[i]
        if i == 0:
            marker = '-x'
        else:
            marker = '-o'
        axs[0].plot(plotting_strains,np.abs(plotting_stresses),marker,label=f'{np.round(unique_value*1000)} (mT)')
        axs[1].plot(np.abs(plotting_stresses),plotting_differential_modulus[:plotting_stresses.shape[0]],marker)
    axs[0].set_title('Stress vs Strain')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress')
    axs[1].set_title('Effective Modulus vs Stress')
    axs[1].set_xlabel('Stress (Pa)')
    axs[1].set_ylabel('Effective Modulus')
    ylimits = (0.9*np.min(differential_modulus),1.1*np.max(differential_modulus))
    axs[1].set_ylim(ylimits)
    format_figure(axs[0])
    format_figure(axs[1])
    # plt.show()
    fig.tight_layout()
    fig.legend()
    savename = output_dir + f'subplots_stress-strain_effective_modulus_labeled_lines'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def subplot_stress_strain_linear_fit_modulus_by_field(stress,strains,linear_fit_modulus,linear_fit_modulus_error,Bext_series,output_dir,tag="",particle_separations=None):
    """Generate figure with subplots of stress-strain curve and modulus found by linear fit of the stress-strain curve versus field."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    Bext_magnitude = np.linalg.norm(Bext_series,axis=1)
    unique_field_values = np.unique(Bext_magnitude)
    for i, unique_value in enumerate(unique_field_values):
        relevant_indices = np.isclose(unique_value,np.linalg.norm(Bext_series,axis=1))
        plotting_stresses = stress[relevant_indices]
        plotting_indices = np.nonzero(plotting_stresses)
        plotting_stresses = plotting_stresses[plotting_indices]
        plotting_strains = strains[relevant_indices]
        plotting_strains = plotting_strains[plotting_indices]
        if i == 0:
            marker = 's'
        else:
            marker = 'o'
        axs[0].plot(plotting_strains,np.abs(plotting_stresses),marker=marker,linestyle='-',label=f'{np.round(unique_value*1000)} (mT)')
    axs[1].errorbar(np.round(1000*np.unique(Bext_magnitude),decimals=2),linear_fit_modulus,marker=marker,linestyle='-',yerr=linear_fit_modulus_error)
    axs[0].set_title('Stress vs Strain')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress')
    axs[1].set_title('Effective Modulus vs Field')
    axs[1].set_xlabel('B Field (mT)')
    axs[1].set_ylabel('Effective Modulus')
    ylimits = (0.9*np.min(linear_fit_modulus),1.1*np.max(linear_fit_modulus))
    axs[1].set_ylim(ylimits)
    format_figure(axs[0])
    format_figure(axs[1])
    # plt.show()
    fig.tight_layout()
    fig.legend()
    savename = output_dir + f'subplots_stress-strain_linear_fit_modulus_labeled_lines'+tag+'.png'
    plt.savefig(savename)
    plt.close()

def format_figure(ax,title_size=30,label_size=30,tick_size=30):
    """Given the axis handle, adjust the font sizes of the title, axis labels, and tick labels."""
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(ax.get_xlabel(),fontsize=label_size)
    ax.set_ylabel(ax.get_ylabel(),fontsize=label_size)
    # ax.set_xlabel("\n"+ax.get_xlabel(),fontsize=label_size)
    # ax.xaxis.set_label_coords(0.5,-0.1)
    # ax.set_ylabel("\n"+ax.get_ylabel(),fontsize=label_size)
    # ax.yaxis.set_label_coords(-0.1,0.5)
    ax.set_title(ax.get_title(),fontsize=title_size)

def format_figure_3D(ax,title_size=30,label_size=30,tick_size=30):
    """Given the axis handle, adjust the font sizes of the title, axis labels, and tick labels."""
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel("\n"+ax.get_xlabel(),fontsize=label_size)
    ax.set_ylabel("\n"+ax.get_ylabel(),fontsize=label_size)
    ax.set_title(ax.get_title(),fontsize=title_size)
    ax.set_zlabel("\n"+ax.get_zlabel(),fontsize=label_size)
    # ax.zaxis.set_label_coords(1.1,0.5)

def get_num_output_files(sim_dir):
    """Get the number of output files, since the series variable in current implementations will only contain the applied strains, and not the applied fields. used for properly reading in output files during analysis, and naming figures"""
    with os.scandir(sim_dir) as dirIterator:
        output_files = [f.path for f in dirIterator if f.is_file() and f.name.startswith('output')]
    num_output_files = len(output_files)
    return num_output_files

def get_applied_field_series(sim_dir):
    """Read in the output files to get the external magnetic fields applied during each simulation step"""
    num_output_files = get_num_output_files(sim_dir)
    Hext_series = np.zeros((num_output_files,3),dtype=np.float64)
    for i in range(num_output_files):
        _, Hext, _, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        Hext_series[i,:] = Hext
    return Hext_series

def get_applied_strain_series(sim_dir):
    """Read in the simulation directory and parse the subfolders to find the applied strain values"""
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir() and f.name.startswith('strain')]
    for folder_num, subfolder in enumerate(subfolders):
        fn = subfolder.split('/')[-1]
        fn_components = fn.split('_')
        strain_indices = np.zeros((len(subfolders),),dtype=np.int64)
        field_indices = np.zeros((len(subfolders),),dtype=np.int64)
        output_file_indices = np.zeros((len(subfolders),),dtype=np.int64)
        for i in range(len(fn_components)):
            if fn_components[i] == 'strain':
                strain_indices[folder_num] = int(fn_components[i+1])
            if fn_components[i] == 'field':
                field_indices[folder_num] = int(fn_components[i+1])
        #the associated output file for each subfolder
    max_field_index = np.max(field_indices)
    for folder_num in range(len(subfolders)):
        output_file_indices[folder_num] = strain_indices[folder_num]*(max_field_index+1)+field_indices[folder_num]
    print(fn_components)

def get_particle_separation(posns,particles):
    """Calculate the particle centers and the magnitude of the center-to-center separation."""
    num_particles = particles.shape[0]
    centers = np.zeros((num_particles,3))
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_separations,))
    for i, particle in enumerate(particles):
        centers[i] = simulate.get_particle_center(particle,posns)
    counter = 0
    for i in range(num_particles):
        for j in range(i+1,num_particles):
            separations[counter] = np.linalg.norm(centers[i]-centers[j])
            counter += 1
    return separations

def get_magnetization(Hext,particle_posns,particle_radius,chi,Ms,l_e):
    """Get the overall system magnetization as a vector sum of the magnetizations of the particles."""
    magnetizations = magnetism.get_magnetization_iterative_normalized(Hext,particle_posns,particle_radius,chi,Ms,l_e)
    normalized_magnetizations = magnetizations/Ms
    overall_magnetization = np.sum(normalized_magnetizations,axis=0)/magnetizations.shape[0]
    return overall_magnetization

def get_magnetization_gpu(posns,particles,Hext,Ms,chi,particle_volume,l_e):
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = simulate.get_particle_center(particle,posns)
    particle_volume = np.float32(particle_volume)
    l_e = np.float32(l_e)
    Ms = np.float32(Ms)
    chi = np.float32(chi)
    Hext = Hext.astype(np.float32)
    magnetic_moments = simulate.get_magnetization_iterative(Hext,particles,cp.array(particle_centers.astype(np.float32)).reshape((particle_centers.shape[0]*particle_centers.shape[1],1),order='C'),Ms,chi,particle_volume,l_e)
    particle_magnetizations = magnetic_moments/particle_volume
    normalized_magnetizations = particle_magnetizations/Ms
    overall_magnetization = np.sum(normalized_magnetizations,axis=0)/particle_magnetizations.shape[0]
    return overall_magnetization

def plot_particle_behavior(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series,gpu_flag=True):
    """Plot the particle behavior as a function of applied field (particle separation and sytem magnetization)"""
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    magnetization = np.zeros((num_output_files,3))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        separations[i,:] = get_particle_separation(final_posns,particles)
        if gpu_flag:
            particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
            magnetization[i,:] = get_magnetization_gpu(final_posns,particles,Hext,Ms,chi,particle_volume,l_e)
        else:
            #temporarily doing casting to deal with the analysis of simulations ran using gpu calculations (where 32bit floats must be used)
            Hext = np.float64(Hext)
            #for each particle, find the position of the center
            particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
            for j, particle in enumerate(particles):
                particle_centers[j,:] = simulate.get_particle_center(particle,final_posns)
                magnetization[i,:] = get_magnetization(Hext,particle_centers,particle_radius,chi,Ms,l_e)
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    # fig.tight_layout()
    axs[0].plot(np.linalg.norm(mu0*Hext_series,axis=1),separations*l_e,'o')
    axs[0].set_xlabel('Applied Field (T)')
    axs[0].set_ylabel('Particle Separation (m)')
    #find the unit vector describing the direction along which the external magnetic field is applied
    Bext_series = mu0*Hext_series
    nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
    Bext_unit_vector = Bext_series[nonzero_field_value_indices[0],:]/np.linalg.norm(Bext_series[nonzero_field_value_indices[0],:])
    magnetization_along_applied_field = np.dot(magnetization,Bext_unit_vector)
    axs[1].plot(np.linalg.norm(mu0*Hext_series,axis=1),magnetization_along_applied_field,'o')
    axs[1].set_xlabel('Applied Field (T)')
    axs[1].set_ylabel('Normalized System Magnetization')
    format_figure(axs[0])
    format_figure(axs[1])
    # fig.show()
    savename = sim_dir + 'figures/particle_behavior/' + f'particle_separation_magnetization.png'
    plt.savefig(savename)
    plt.close()
    return separations

def plot_particle_behavior_hysteresis(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series,gpu_flag=True):
    """Plot the particle behavior as a function of applied field (particle separation and sytem magnetization)"""
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    magnetization = np.zeros((num_output_files,3))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        #temporarily doing casting to deal with the analysis of simulations ran using gpu calculations (where 32bit floats must be used)
        separations[i,:] = get_particle_separation(final_posns,particles)
        if gpu_flag:
            particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
            magnetization[i,:] = get_magnetization_gpu(final_posns,particles,Hext,Ms,chi,particle_volume,l_e)
        else:
            Hext = np.float64(Hext)
            #for each particle, find the position of the center
            particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
            for j, particle in enumerate(particles):
                particle_centers[j,:] = simulate.get_particle_center(particle,final_posns)
            magnetization[i,:] = get_magnetization(Hext,particle_centers,particle_radius,chi,Ms,l_e)
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    # fig.tight_layout()
    B_field_norms = np.linalg.norm(mu0*Hext_series,axis=1)
    turning_point_index = np.argwhere(np.max(B_field_norms)==B_field_norms)[0][0]
    upward_leg = axs[0].plot(B_field_norms[:turning_point_index+1],separations[:turning_point_index+1]*l_e,'o-')
    upward_leg[0].set_label('Upward Leg')
    downward_leg = axs[0].plot(B_field_norms[turning_point_index:],separations[turning_point_index:]*l_e,'x--')
    downward_leg[0].set_label('Downward Leg')
    axs[0].set_xlabel('Applied Field (T)')
    axs[0].set_ylabel('Particle Separation (m)')
    axs[0].legend()
    #find the unit vector describing the direction along which the external magnetic field is applied
    Bext_series = mu0*Hext_series
    nonzero_field_value_indices = np.where(np.linalg.norm(Bext_series,axis=1)>0)[0]
    Bext_unit_vector = Bext_series[nonzero_field_value_indices[0],:]/np.linalg.norm(Bext_series[nonzero_field_value_indices[0],:])
    magnetization_along_applied_field = np.dot(magnetization,Bext_unit_vector)
    upward_leg, = axs[1].plot(B_field_norms[:turning_point_index+1],magnetization_along_applied_field[:turning_point_index+1],'o-')
    # upward_leg.set_label('Upward Leg')
    downward_leg, = axs[1].plot(B_field_norms[turning_point_index:],magnetization_along_applied_field[turning_point_index:],'x--')
    # downward_leg.set_label('Downward Leg')
    axs[1].set_xlabel('Applied Field (T)')
    axs[1].set_ylabel('Normalized System Magnetization')
    format_figure(axs[0])
    format_figure(axs[1])
    # fig.show()
    savename = sim_dir + 'figures/particle_behavior/' + f'particle_separation_magnetization.png'
    plt.savefig(savename)
    plt.close()

def plot_surface_node_force_vector(sim_dir,output_dir,file_number,initial_node_posns,final_posns,strain_direction,beta_i,springs_var,elements,boundaries,dimensions,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,boundary_conditions,force_plate_flag=False):
    """Calculate the forces acting on the "probed" boundary nodes and plot the force vector components and norm in a subplot. Option force_plate_flag to calculate and plot the forces acting on a fictional "force plate" for "plate based" stress boundary conditions"""
    #TODO make into a function the plotting of the forces acting on the probed surface necessary to keep it held fixed in place
    #plotting the force components acting on the probed surface
    i = file_number
    Lx = initial_node_posns[:,0].max()
    Ly = initial_node_posns[:,1].max()
    Lz = initial_node_posns[:,2].max()
    layers = (Lx,Ly,Lz)
    boundary_forces, all_forces = get_probe_boundary_forces(sim_dir,i,strain_direction,beta_i,springs_var,elements,boundaries,dimensions,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag)
    plate_posn = boundary_conditions[2]
    plate_orientation = boundary_conditions[1][0]
    global_index_interacting_nodes, plate_force = simulate.get_plate_force(final_posns,plate_posn,plate_orientation,boundaries)
    plate_force /= beta_i[global_index_interacting_nodes,np.newaxis]
    #go from forces acting on the boundary to forces that would need to be acting to keep the boudnary in place
    all_forces *= -1
    if boundary_conditions[1][0] == 'x':
        index = int(Lx)
        cut_type = 'yz'
    elif boundary_conditions[1][0] == 'y':
        index = int(Ly)
        cut_type = 'xz'
    elif boundary_conditions[1][0] == 'z':
        index = int(Lz)
        cut_type = 'xy'
    subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,all_forces,index,output_dir+'modulus/',tag=f"probed_surface_force_series{i}")
    if force_plate_flag:
        plate_forces = np.zeros(np.shape(all_forces))
        plate_forces[global_index_interacting_nodes] = plate_force
        subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,plate_forces,index,output_dir+'modulus/',tag=f"probe_plate_force_series{i}")

def time_step_comparison(dir_one,dir_two):
    with open(dir_one + 'timesteps.npy','rb') as data:
        delta_t_one = np.load(data)
    with open(dir_two + 'timesteps.npy','rb') as data:
        delta_t_two = np.load(data)
    if delta_t_one.shape[0] == delta_t_two.shape[0]:
        print(f'are the step sizes the same for all entries?:{np.allclose(np.array(delta_t_one),np.array(delta_t_two))}')
        fig, axs = plt.subplots(2)
        axs[0].plot(np.arange(delta_t_one.shape[0]),delta_t_one-delta_t_two)
        axs[0].set_xlabel('integration number')
        axs[0].set_ylabel('difference in step size')
        axs[1].plot(np.arange(delta_t_one.shape[0]),delta_t_one)
        axs[1].plot(np.arange(delta_t_one.shape[0]),delta_t_two)
        axs[1].set_xlabel('integration number')
        axs[1].set_ylabel('step size')
        fig.legend(['dir_one','dir_two'])
        plt.show()
    else:
        print(f'different number of steps executed for each\nfirst directory steps:{delta_t_one.shape[0]}\nsecond directory steps:{delta_t_two.shape[0]}')
        fig, ax = plt.subplots()
        line1, = ax.plot(np.arange(delta_t_one.shape[0]),delta_t_one)
        line2, = ax.plot(np.arange(delta_t_two.shape[0]),delta_t_two)
        ax.set_xlabel('integration number')
        ax.set_ylabel('step size')
        line1.set_label('dir_one')
        line2.set_label('dir_two')
        ax.legend()
        plt.show()

def cpu_vs_gpu_time_step_comparison(sim_dir_one,sim_dir_two):
    with os.scandir(sim_dir_one) as dirIterator:
        subfolders = [f.name for f in dirIterator if f.is_dir()]
    for subfolder in subfolders:    
        if not 'figure' in subfolder:
            time_step_comparison(sim_dir_one+subfolder+'/',sim_dir_two+subfolder+'/')

def cpu_vs_gpu_solution_comparison(sim_dir_one,sim_dir_two):
    if 'gpu_True' in sim_dir_one:
        gpu_dir = sim_dir_one
        cpu_dir = sim_dir_two
    elif 'gpu_True' in sim_dir_two:
        gpu_dir = sim_dir_two
        cpu_dir = sim_dir_one
    with os.scandir(gpu_dir) as dirIterator:
        subfolders = [f.name for f in dirIterator if f.is_dir()]
    i = 0
    for subfolder in subfolders:    
        if not 'figure' in subfolder:
            print(f'beginning comparison for subfolder:{subfolder}')
            compare_per_step_solution_vectors(gpu_dir,cpu_dir,subfolder)
            compare_per_step_acceleration_vectors(gpu_dir,cpu_dir,subfolder,output_file_number=i)
            # gpu_acceleration_norm = get_per_step_acceleration_norms(gpu_dir,subfolder,output_file_number=i)
            # cpu_acceleration_norm = get_per_step_acceleration_norms(cpu_dir,subfolder,output_file_number=i)
            i += 1
            # fig, ax = plt.subplots()
            # line1, = ax.plot(np.arange(gpu_acceleration_norm.shape[0]),gpu_acceleration_norm)
            # line2, = ax.plot(np.arange(gpu_acceleration_norm.shape[0]),cpu_acceleration_norm[:gpu_acceleration_norm.shape[0]])
            # ax.set_xlabel('integration step')
            # ax.set_ylabel('acceleration_norm')
            # line1.set_label('gpu')
            # line2.set_label('cpu')
            # ax.legend()
            # plt.show()

def get_per_step_acceleration_norms(sim_dir,subfolder,output_file_number):
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    particle_moment_of_inertia = np.float32((2/5)*particle_mass*np.power(particle_radius,2))
    scaled_moment_of_inertia = np.float32(particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2)))
    with open(sim_dir + subfolder + '/solutions.npy','rb') as data:
        solutions = np.load(data)
        _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
    acceleration_norm = np.zeros((solutions.shape[0],))
    if 'gpu_True' in sim_dir:
        beta_i = np.float32(beta_i)
        beta = np.float32(beta)
        drag = np.float32(drag)
        Hext = np.float32(Hext)
        particles = np.int32(particles)
        for key in boundaries:
            boundaries[key] = np.int32(boundaries[key])
        dimensions = np.float32(dimensions)
        kappa = np.float32(kappa*(l_e**2))
        l_e = np.float32(l_e)
        particle_radius = np.float32(particle_radius)
        particle_mass = np.float32(particle_mass)
        chi = np.float32(chi)
        Ms = np.float32(Ms)
        elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
        springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
        boundary_conditions = (boundary_conditions[0],boundary_conditions[1],np.float32(boundary_conditions[2]))
        for i in range(solutions.shape[0]):
            a_var = simulate.get_accel_scaled_GPU(solutions[i],elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
            acceleration_norm[i] = np.linalg.norm(a_var)#np.linalg.norm(a_var,axis=1)
    else:
        for i in range(solutions.shape[0]):
            a_var = simulate.get_accel_scaled_rotation(solutions[i],elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
            acceleration_norm[i] = np.linalg.norm(a_var)#np.linalg.norm(a_var,axis=1)
    return acceleration_norm

def compare_per_step_acceleration_vectors(sim_dir_one,sim_dir_two,subfolder,output_file_number):
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir_one)
    particle_moment_of_inertia = np.float32((2/5)*particle_mass*np.power(particle_radius,2))
    scaled_moment_of_inertia = np.float32(particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2)))
    with open(sim_dir_one + subfolder + '/solutions.npy','rb') as data:
        solutions_one = np.load(data)
    with open(sim_dir_two + subfolder + '/solutions.npy','rb') as data:
        solutions_two = np.load(data)
    if 'gpu_True' in sim_dir_one:
        cpu_solutions = solutions_two
        gpu_solutions = solutions_one
        a_var_cpu = np.zeros((solutions_two.shape[0],total_num_nodes,3))
        a_var_gpu = np.zeros((solutions_one.shape[0],total_num_nodes,3))
        _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir_two+f'output_{output_file_number}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
    else:
        cpu_solutions = solutions_one
        gpu_solutions = solutions_two
        a_var_cpu = np.zeros((solutions_one.shape[0],total_num_nodes,3))
        a_var_gpu = np.zeros((solutions_two.shape[0],total_num_nodes,3))
        _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir_one+f'output_{output_file_number}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
    for i in range(cpu_solutions.shape[0]):
        a_var_cpu[i] = simulate.get_accel_scaled_rotation(cpu_solutions[i],elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
    beta_i = np.float32(beta_i)
    beta = np.float32(beta)
    drag = np.float32(drag)
    Hext = np.float32(Hext)
    particles = np.int32(particles)
    for key in boundaries:
        boundaries[key] = np.int32(boundaries[key])
    dimensions = np.float32(dimensions)
    kappa = np.float32(kappa*(l_e**2))
    l_e = np.float32(l_e)
    particle_radius = np.float32(particle_radius)
    particle_mass = np.float32(particle_mass)
    chi = np.float32(chi)
    Ms = np.float32(Ms)
    elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
    springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
    boundary_conditions = (boundary_conditions[0],boundary_conditions[1],np.float32(boundary_conditions[2]))
    for i in range(gpu_solutions.shape[0]):
        a_var_gpu[i] = simulate.get_accel_scaled_GPU(gpu_solutions[i],elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
    smallest_number_of_solutions = np.min([solutions_one.shape[0],solutions_two.shape[0]])
    for i in range(smallest_number_of_solutions):
        print(f'Acceleration vectors are close to being the same?:{np.allclose(a_var_gpu[i],a_var_cpu[i])}')
        if not np.allclose(a_var_gpu[i],a_var_cpu[i]):
            accel_diff = a_var_gpu[i] - a_var_cpu[i]
            print(f'max acceleration component difference is:{np.max(np.abs(accel_diff))}')
            print(f'mean acceleration component difference is:{np.mean(np.abs(accel_diff))}')

def compare_per_step_solution_vectors(sim_dir_one,sim_dir_two,subfolder):
    with open(sim_dir_one + subfolder + '/solutions.npy','rb') as data:
        solutions_one = np.load(data)
    with open(sim_dir_two + subfolder + '/solutions.npy','rb') as data:
        solutions_two = np.load(data)
    smallest_number_of_solutions = np.min([solutions_one.shape[0],solutions_two.shape[0]])
    solutions_difference = solutions_one[:smallest_number_of_solutions] - solutions_two[:smallest_number_of_solutions]
    for i in range(smallest_number_of_solutions):
        print(f'Solutions are close to being the same?:{np.allclose(solutions_one[i],solutions_two[i])}')
        if not np.allclose(solutions_one[i],solutions_two[i]):
            solutions_difference = solutions_one[i] - solutions_two[i]
            print(f'max solution component difference is:{np.max(np.abs(solutions_difference))}')
            print(f'mean solution component difference is:{np.mean(np.abs(solutions_difference))}')

def compare_repeat_calculation_gpu_acceleration_vectors(sim_dir,subfolder,output_file_number):
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    particle_moment_of_inertia = np.float32((2/5)*particle_mass*np.power(particle_radius,2))
    scaled_moment_of_inertia = np.float32(particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2)))
    n_repititions = 10
    with open(sim_dir + subfolder + '/solutions.npy','rb') as data:
        solutions = np.load(data)
        a_var = np.zeros((n_repititions,total_num_nodes,3))
        _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
    beta_i = np.float32(beta_i)
    beta = np.float32(beta)
    drag = np.float32(drag)
    Hext = np.float32(Hext)
    particles = np.int32(particles)
    for key in boundaries:
        boundaries[key] = np.int32(boundaries[key])
    dimensions = np.float32(dimensions)
    kappa = np.float32(kappa*(l_e**2))
    l_e = np.float32(l_e)
    particle_radius = np.float32(particle_radius)
    particle_mass = np.float32(particle_mass)
    chi = np.float32(chi)
    Ms = np.float32(Ms)
    elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
    springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
    boundary_conditions = (boundary_conditions[0],boundary_conditions[1],np.float32(boundary_conditions[2]))
    for i in range(solutions.shape[0]):
        for j in range(n_repititions):
            a_var[j] = simulate.get_accel_scaled_GPU(solutions[i],elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        for j in range(n_repititions):
            for k in range(j,n_repititions):
                print(f'maximum acceleration vector component {np.max(a_var[j])}')
                print(f'Acceleration vectors are close to being the same?:{np.allclose(a_var[j],a_var[k])}')
                accel_diff = a_var[j] - a_var[k]
                print(f'max acceleration component difference is:{np.max(np.abs(accel_diff))}')
                print(f'mean acceleration component difference is:{np.mean(np.abs(accel_diff))}')
                print(f'acceleration component difference standard deviation is:{np.std(accel_diff)}')

def gpu_repeat_accel_calculation_comparison(sim_dir):
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.name for f in dirIterator if f.is_dir()]
    i = 0
    for subfolder in subfolders:    
        if not 'figure' in subfolder:
            print(f'beginning comparison for subfolder:{subfolder}')
            compare_repeat_calculation_gpu_acceleration_vectors(sim_dir,subfolder,i)
            i += 1

def temp_hysteresis_analysis(sim_dir,gpu_flag=False):
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For hysteresis simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization"""
    #   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['modulus','particle_behavior','stress','strain','cuts','outer_surfaces']
    figure_subtypes = ['center', 'particle', 'outer_surface']
    for figure_type in figure_types:
        if not (os.path.isdir(output_dir+figure_type+'/')):
          os.mkdir(output_dir+figure_type+'/')
        if figure_type == 'stress' or figure_type =='strain' or figure_type == 'cuts':
            for figure_subtype in figure_subtypes:
                if not (figure_type == 'cuts' and figure_subtype == 'outer_surface'):
                    if not (os.path.isdir(output_dir+figure_type+'/'+figure_subtype+'/')):
                        os.mkdir(output_dir+figure_type+'/'+figure_subtype+'/')
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    if gpu_flag:
        initial_node_posns = np.float64(initial_node_posns)
        beta_i = np.float64(beta_i)
        springs_var = np.float64(springs_var)
        kappa = np.float64(kappa)
        beta = np.float64(beta)
        l_e = np.float64(l_e)
        particle_mass = np.float64(particle_mass)
        particle_radius = np.float64(particle_radius)
        Ms = np.float64(Ms)
        chi = np.float64(chi)
#   find the indices corresponding to the outer surfaces of the simulated volume for plotting and visualization
    surf_indices = (0,int(num_nodes[0]-1),0,int(num_nodes[1]-1),0,int(num_nodes[2]-1))
    surf_type = ('left','right','front','back','bottom','top')
#   find indices corresponding to the "center" of the simulated volume for plotting and visualization, corresponding to cut_types values
    center_indices = (int((num_nodes[2]-1)/2),int((num_nodes[1]-1)/2),int((num_nodes[0]-1)/2))
#   lambda and mu (Lame parameters) are calculated from Young's modulus and Poisson's ratio variables from the init.h5 file
    #see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))

    #get the the applied field associated with each output file
    Hext_series = get_applied_field_series(sim_dir)
    num_output_files = get_num_output_files(sim_dir)
    #get the particle separations and overall magnetizations and plot them
    plot_particle_behavior_hysteresis(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series)

# #   in a loop, output files are read in and manipulated
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_nodes(initial_node_posns,final_posns,particles,output_dir+'particle_behavior/',tag=f"{i}")
        if ((boundary_conditions[0] == "tension" or boundary_conditions[0] == "compression" or boundary_conditions[0] == "free") and boundary_conditions[2] != 0) or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_tiled_outer_surfaces_contours_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"field_{i}_field_{np.round(mu0*Hext,decimals=4)}")
            except:
                print('contour plotting of outer surfaces failed due to lack of variation (no contour levels could be generated)')
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
        mre.analyze.plot_outer_surfaces_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"field_{i}_field_{np.round(mu0*Hext,decimals=4)}")
#       visualizations of cuts through the center of the volume are generated and saved out
        mre.analyze.plot_center_cuts_surf(initial_node_posns,final_posns,l_e,output_dir+'cuts/center/',tag=f"field_{i}_field_{np.round(mu0*Hext,decimals=4)}")
        # mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"field_{i}_field_{np.round(mu0*Hext,decimals=4)}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"field_{i}_field_{np.round(mu0*Hext,decimals=4)}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')

def plot_boundary_node_posn_hist(boundary_node_posns,output_dir,tag=""):
    """Plot a histogram of the boundary node positions. Intended for analyzing the variation in boundary surface position relevant to the strain calculation."""
    max_posn = np.max(boundary_node_posns)
    min_posn = np.min(boundary_node_posns)
    mean_posn = np.mean(boundary_node_posns)
    rms_posn = np.sqrt(np.sum(np.power(boundary_node_posns,2))/np.shape(boundary_node_posns)[0])
    counts, bins = np.histogram(boundary_node_posns, bins=40)
    fig,ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax.hist(boundary_node_posns,bins=30)
    # ax.hist(bins[:-1], bins, weights=counts)
    sigma = np.std(boundary_node_posns)
    mu = mean_posn
    ax.set_title(f'Boundary Node Position Histogram\nMaximum {max_posn}\n Minimum {min_posn}\nMean {mean_posn}\n$\sigma={sigma}$\nRMS {rms_posn}')
    ax.set_xlabel('node position (l_e)')
    ax.set_ylabel('counts')
    savename = output_dir +'boundary_node_position_'+tag+'_hist.png'
    mre.analyze.format_figure(ax)
    plt.savefig(savename)
    plt.close()

if __name__ == "__main__":
    main()
    # sim_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-15_strain_testing_shearing_order_1_drag_20/'
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-21_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0.05 0.   0.  ]/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-20_strain_testing_torsion_order_0_drag_20/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-20_2particle_freeboundaries_order_0_drag20/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-25_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_E_900000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-31_field_dependent_modulus_strain_shearing_direction('x', 'y')_order_2_E_900000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-28_field_dependent_modulus_strain_compression_direction('x', 'x')_order_2_E_900000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-26_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_E_900000.0_Bext_angle_90.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-01_field_dependent_modulus_strain_shearing_direction('x', 'y')_order_2_E_900000.0_Bext_angle_90.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-26_field_dependent_modulus_strain_tension_direction('y', 'y')_order_2_E_900000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-27_field_dependent_modulus_strain_tension_direction('y', 'y')_order_2_E_900000.0_Bext_angle_90.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-12-29_field_dependent_modulus_strain_compression_direction('y', 'y')_order_2_E_900000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-09_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_90000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-09_field_dependent_modulus_strain_tension_direction('z', 'z')_order_1_E_90000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-10_field_dependent_modulus_strain_shearing_direction('x', 'y')_order_1_E_30000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_0_Bext_angle_0.0_particle_rotations_nodal_WCA_off/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-09_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.25_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_nodal_WCA_off/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-22_field_dependent_modulus_strain_plate_compression_direction('x', 'x')_order_0_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    #cpu case that was profiled
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-27_field_dependent_modulus_strain_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    #gpu case (calculation of elastic and vcf forces) that was profiled
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-26_field_dependent_modulus_strain_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-29_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_0_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-29_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_0_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    #rerunning simulation that was used for profiling the cpu and gpu approaches on the 27th and 26th respectively, to see if the changes to the codebase since then have introduced any bugs/errors
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-29_field_dependent_modulus_strain_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    #gpu simulations with simple stress
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-01_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-01_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_False_tf_300/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-02_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    #simulations ran with low number of integrations and only one round of integrating, to compare solution vectors, acceleration vectors, and time steps for cpu vs gpu implementations
    # sim_dir_one = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_False_tf_300/"
    # sim_dir_two = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # #simulations ran with low number of integrations and only one round of integrating, to compare solution vectors, acceleration vectors, and time steps for cpu vs gpu implementations with the particle rotations turned off
    # sim_dir_one = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_off_gpu_False_tf_300/"
    # sim_dir_two = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_off_gpu_True_tf_300/"
    # #simulations ran with low number of integrations and only one round of integrating, to compare solution vectors, acceleration vectors, and time steps for cpu vs gpu implementations with the VCF turned off (nu=0.25)
    # sim_dir_one = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.25_Bext_angle_0.0_particle_rotations_gpu_False_tf_300/"
    # sim_dir_two = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.25_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # #simulations ran with low number of integrations and only one round of integrating, to compare solution vectors, acceleration vectors, and time steps for cpu vs gpu implementations with the VCF turned off (nu=0.25) and the springs turned off (E=0)
    # sim_dir_one = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_0_nu_0.25_Bext_angle_0.0_particle_rotations_gpu_False_tf_300/"
    # sim_dir_two = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-05_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_0_nu_0.25_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu_repeat_accel_calculation_comparison(sim_dir_two)
    # cpu_vs_gpu_time_step_comparison(sim_dir_one,sim_dir_two)
    # cpu_vs_gpu_solution_comparison(sim_dir_one,sim_dir_two)
    
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-08_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation with "random" particle placement (but only two particles)
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-09_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-09_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used for hysteresis simulation, with new (partially implemented) batch job driving function and simulation running function
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-12_2_particle_hysteresis_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # analysis_case1(sim_dir)
    # ran a series of different discretization orders doing the hysteresis simulation. for this set of results the particle placement was not working as expected... so the initial particle separation was not always the 9 microns that was desired. that has since been fixed (as of 2024-02-13)
    # for i in range(6):
    #     sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-12_2_particle_hysteresis_order_{i}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    #     temp_hysteresis_analysis(sim_dir,gpu_flag=False)

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-13_2_particle_hysteresis_order_{1}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-14_2_particle_hysteresis_order_{0}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    # sim_dir_one = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-14_2_particle_hysteresis_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # sim_dir_two = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-14_2_particle_hysteresis_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # compare_fixed_time_step_solutions_hysteresis(sim_dir_one,sim_dir_two)

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-14_2_particle_hysteresis_order_4_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    # shearing stress simulation at multiple fields
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-18_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_5_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # shearing stress simulation at a 2 non-zero fields and 2 non-zero stresses, plus the zero stress and zero field cases
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-22_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # shearing stress simulation at a 5 non-zero fields and 2 non-zero stresses, plus the zero stress and zero field cases. 3 particle chain
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-23_3_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"

    # shearing/tension without particles.
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-26_0_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_None_gpu_True_stepsize_5.e-3/"

    # shearing, two particles, incomplete run, but long run time/simulation time elapsed
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-26_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    
    #tension, two particles, attractive
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-27_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #compression, two particles, attractive, no particle rotations
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('z', 'z')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #compression, two particles, attractive, particle rotations via scipy.transform.Rotation, 3 different time step sizes
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-04_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-04_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_1.25e-3/"

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-01_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # adjustment to the treatment of particle as rigid body. track particle orientation and relative to particle center vectors poinmting to particle nodes. at the end of an integration round, use the orientation and particle center, as well as the relative to particle center vectors to adjust the particle node positions, to maintain the rigid body shape
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-12_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # high order of discretization test run to determine run times and test behavior
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-12_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('z', 'z')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    #tension case
    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-14_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 8 particles, cubic crystal lattice arrangement, 3% volume ratio. Error in particle placement lead to a configuration that was not exactly a cubic lattice, but the results here are interesting, since the particle clusters do not all form at once (at a single field value), but rather occur at different field values, causing multiple "phase transitions". the collective behavior of the particle clustering is important, seemingly the small differences in the value at which clustering occurs plays a role in both the hysteresis and the effective stiffness behavior. this will need to be reran with the fixed particle placement, but also suggests that adding noise to the placement, or intentionally altering separation along different axes, would produce interesting results for comparison.
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-18_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, cubic crystal lattice arrangement, 3% volume ratio. particle placement very broken, but still interesting to see the impact. looks a bit like a periodic boundary structure simulation might look, but handling periodic boundary conditions if there are particles crossing the boundaries is something i haven't considered, and I have not figured out how to handle the magnetic interactions if using periodic boudnary conditions.
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-19_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_4_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #8 particles, cubic lattice, 3% volume ratio, fixed particle placement. only 0 stress applied, just looking at the field dependent particle behavior
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-20_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #8 particles, cubic lattice, 3% volume ratio, fixed particle placement. multiple stress values applied to analyze effective modulus
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-20_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, cubic lattice, 3% volume ratio, fixed particle placement. multiple stress values applied to analyze effective modulus
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-21_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)
    # analysis_average_stress_strain(sim_dir,gpu_flag=True)

    #27 particles, up to +/-2 volume elements as noise added to periodic particle placement, two different RNG seeds
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-29_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_starttime_14-17_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-03-29_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_starttime_23-23_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, no noise, hysteresis
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-02_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #2 particles, no noise, hysteresis, no scipy.Rotation or rigid body attempts, instead setting particle-particle node springs to have stiffness based on the particle modulus (actually just 100*polymer modulus because the actual ratio was so large it broke things, as small displacements led to huge accelerations)
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-04_2_particle_hysteresis_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #2 particles, no noise, hysteresis, no scipy.Rotation or rigid body attempts, instead setting particle-particle node springs to have stiffness based on the particle modulus (actually 10000*polymer modulus because the actual ratio was so large it broke things, as small displacements led to huge accelerations). had originally set polymer-particle connections to be the average of the stiffness of the two types, but now set to polymer stiffness. could be adjusted to something like 10* polymer stiffness... maybe. issue was with the acceleration calculations used for checking convergence criteria. higher stiffness meant even small displacements from equilbibrium length lead to outsized accelerations that didn't actually influence particle displacement or polymer displacement... just vibration. polymer-particle connections were also high stiffness, so despite trying to "remove" the particle vibrations from the convergence check, the stiffer polymer around the particles left higher residual accelerations/vibrations
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-04_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #27 particles, noise, hysteresis. particle-particle nodes have stiffer connections, no scipy.Rotation. no fixed nodes
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-04_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_16-41_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #27 particles, no noise, hysteresis. cutoff for WCA set to particle diameter + 100 nm. no fixed nodes. anisotropy_factor = [0.7,1.3,1.3]
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-05_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #45 particles [5,3,3], noise, hysteresis. cutoff for WCA set to particle diameter + 100 nm. no fixed nodes. anisotropy_factor = [0.8,1.13ish,1.13ish]
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-05_45_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_15-34_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #45 particles [5,3,3], regular no noise, shear stress xy, was supposed to be only 0.01 stress, but implementation errors with setting stress boundary conditions during initialization led to stress values up to 1.01 in steps of 0.01. stopped early. may end up removing some of the dataset
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-06_45_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, tension in x direction, field along x. small stress values. is the zero field modulus still higher than the field on, despite intuition suggesting that it should be stiffer with the field on?
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-08_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, tension in x direction, field along x. small stress values. no field
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-09_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_None_gpu_True_stepsize_5.e-3/"

    #2 particle along x, tension in x direction, field along x. smallish stress values. non zero fields field
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-09_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #64 particle (4,4,4), regular, no noise, tension in x direction, field along x. smallish stress values. non zero fields field
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-09_64_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along x, tension along z
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along x
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along y
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('y', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along z
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-11_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along x, strain based simulation. only attempted to implement new strain based calculation of modulus for tension/compression cases, and it still needs to be modified
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('y', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 2 particle stress, field along z, tension along x, double the maximum number of integration rounds
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 8 particle (2x2x2), regular. field along z, tension along x
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-14_8_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #27 particles (3x3x3), regular, field along x, tension along x
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-14_27_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    #125 particles (5x5x5), regular, field along x, tension along x. Results prior to fully testing the new gpu implementations of distributing the magnetic force, setting fixed nodes, and (not relevant to this sim) applying stress to boundary nodes
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-22_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle, field along x, tension along x, still haven't fully tested new implementations of gpu kernels
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-23_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #2 particle, field along x, hystersis, after testing/debugging and fixing new implementations of gpu kernels
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-26_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particle, field along z, hysteresis
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-27_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particle, field along x, hysteresis
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-26_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, noisy placement (+/-1 volume elements off regular possible in each direction), field along x, hysteresis. not guaranteed that all steps reached convergence, but still probably "close enough." Need more refined convergence criteria for these noisy systems, that seem to need more time to settle. that means potentially segregating out the accelerations and velocities of the particles from the rest of the system, and checking the convergence of the rest of the system, then the particle level accelerations and velocities (or changes in position) for convergence testing.
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-27_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_14-40_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, noisy placement (+/-1 volume elements off regular possible in each direction), field along z, hysteresis
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-28_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_starttime_04-27_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, regular, field along x, tension along x, strain bc
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-29_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular, field along x, tension along x, strain bc
    sim_dir = f"/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-04-29_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)