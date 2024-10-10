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
from mre.analyze import format_figure, format_figure_3D
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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
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
        pass
    elif 'strain' in sim_dir:
        pass
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out

#   in a loop, output files are read in and manipulated
    for i in range(boundary_condition_series.shape[0]):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
#       node positions are scaled to SI units using l_e variable for visualization
        si_final_posns = final_posns*l_e
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#TODO Issue with using contours for abritrary simulations. if the surfaces don't have contours, that is, differences in the "depth" from point to point, then there are no contour levels that can be defined, and the thing fails. i can use a try/except clause, but that may be bad style/practice. I'm not sure of the right way to handle this. I suppose if it is shearing or torsion I should expect that this may not be a useful figure to generate anyway, so i could use the boundary_conditions variable first element
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if ((boundary_conditions[0] == "tension" or boundary_conditions[0] == "compression" or boundary_conditions[0] == "free") and boundary_conditions[2] != 0) or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_tiled_outer_surfaces_contours_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{boundary_condition_series[i]}")
            except:
                print('contour plotting of outer surfaces failed due to lack of variation (no contour levels could be generated)')
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
        mre.analyze.plot_outer_surfaces_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{boundary_condition_series[i]}")
#       visualizations of cuts through the center of the volume are generated and saved out
        mre.analyze.plot_center_cuts_surf(initial_node_posns,final_posns,l_e,output_dir+'cuts/center/',tag=f"3D_strain_{boundary_condition_series[i]}")
        # mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{series[i]}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"strain_{boundary_condition_series[i]}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{boundary_condition_series[i]}")
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
            tag = surface+'_surface_strain_' + f'{boundary_condition_series[i]}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'strain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'nonlinearstrain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,surf_idx,output_dir+'stress/outer_surface/',tag=tag+'stress')
#       stress and strain tensors are visualized for cuts through the center of the volume
        for cut_type,center_idx in zip(cut_types,center_indices):
            tag = 'center_'  f'{boundary_condition_series[i]}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'strain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'nonlinearstrain')
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,center_idx,output_dir+'stress/center/',tag=tag+'stress')
#       if particles present:
#          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
        if particles.shape[0] != 0:
            centers = np.zeros((particles.shape[0],3))
            for i, particle in enumerate(particles):
                tag=f"particle{i+1}_edge_" + f'strain_{boundary_condition_series[i]}_'
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
            tag='particle_centers_'+ f'strain_{boundary_condition_series[i]}_'
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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
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
    Bext_series = mu0*Hext_series
    num_output_files = get_num_output_files(sim_dir)
    #get the particle separations and overall magnetizations and plot them
    plot_particle_behavior_flag = True
    if plot_particle_behavior_flag and particles.shape[0] != 0:
        particle_separations = plot_particle_behavior(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series)
        num_particles = particles.shape[0]
        # particle_separations_matrix = check_clustering(num_particles,num_output_files,Hext_series,particle_separations,l_e,clustering_distance=4)
    else:
        particle_separations_matrix = None

#   in a loop, output files are read in and manipulated
#       node positions and boundary conditions are used to calculate forces happening at relevant fixed boundaries
#       forces are scaled to SI units
#       surface areas are calculated and used to convert boundary forces to stresses along relevant direction
#       effective modulus is calculated from stress and strain
#       effective modulus and stress are saved to respective array variables

    plot_particle_strain_response(sim_dir,num_output_files,particles,Bext_series,boundary_condition_series,l_e)

    unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1))
    num_fields = unique_fields.shape[0]
    num_boundary_conditions = np.floor_divide(num_output_files,num_fields)
    particle_separations = np.zeros((num_fields,num_boundary_conditions))
    for i in range(num_fields):
        particle_separations[i] = particle_separations_matrix[i::num_fields,0,1]
    particle_orientations = get_particle_orientation(sim_dir,num_output_files,particles)


    # fig, ax = plt.subplots()
    # ax.plot(boundary_condition_series,particle_separations[-3,:]-particle_separations[-3,0],'.',label='80mT')
    # ax.plot(boundary_condition_series,particle_separations[-2,:]-particle_separations[-2,0],'^',label='120mT')
    # ax.plot(boundary_condition_series,particle_separations[-1,:]-particle_separations[-1,0],'o',label='140mT')
    # fig.legend()
    # ax.set_xlabel('applied strain')
    # ax.set_ylabel(r'change in particle separation ($\mu$m)')
    # format_figure(ax)
    # plt.show()
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

    Bext_series_magnitude = np.round(np.linalg.norm(mu0*Hext_series,axis=1)*1e3,decimals=3)
    num_unique_fields = np.unique(Bext_series_magnitude).shape[0]

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

def quadratic_fit_func(x,a,b,c):
    """Used with scipy.optimize.curve_fit to try and extract the field dependent effective modulus from energy density vs strain curves"""
    # return a*x + c
    return (1./2.)*a*np.power(x,2) + b*x + c

def quadratic_no_linear_term_fit_func(x,a,c):
    """Used with scipy.optimize.curve_fit to try and extract the field dependent effective modulus from energy density vs strain curves"""
    return (1./2.)*a*np.power(x,2) + c

def shearing_quadratic_fit_func(x,a,c):
    """Used with scipy.optimize.curve_fit to try and extract the field dependent effective modulus from energy density vs strain curves"""
    return a*np.power(x/2,2) + c

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
    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
    for i in range(len(params)):
        if params.dtype.descr[i][0] == 'num_elements':
            num_elements = params[i]
            num_nodes = num_elements + 1
        if params.dtype.descr[i][0] == 'poisson_ratio':
            nu = params[i]
        if params.dtype.descr[i][0] == 'young_modulus':
            E = params[i]
        if params.dtype.descr[i][0] == 'kappa':
            kappa = params[i]
        if params.dtype.descr[i][0] == 'scaling_factor':
            beta = params[i]
        if params.dtype.descr[i][0] == 'element_length':
            l_e = params[i]
        if params.dtype.descr[i][0] == 'particle_mass':
            particle_mass = params[i]
        if params.dtype.descr[i][0] == 'particle_radius':
            particle_radius = params[i]
        if params.dtype.descr[i][0] == 'particle_Ms':
            Ms = params[i]
        if params.dtype.descr[i][0] == 'particle_chi':
            chi = params[i]
        if params.dtype.descr[i][0] == 'drag':
            drag = params[i]
        if params.dtype.descr[i][0] == 'characteristic_time':
            characteristic_time = params[i]

    dimensions = (l_e*np.max(initial_node_posns[:,0]),l_e*np.max(initial_node_posns[:,1]),l_e*np.max(initial_node_posns[:,2]))
    beta_i = beta/node_mass
    total_num_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    k = mre.initialize.get_spring_constants(E, l_e)
    k = np.array(k)

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions

def format_boundary_conditions(boundary_conditions):
    boundary_conditions = (str(boundary_conditions[0][0])[1:],(str(boundary_conditions[0][1])[1:],str(boundary_conditions[0][2])[1:]),boundary_conditions[0][3])
    boundary_conditions = (boundary_conditions[0][1:-1],(boundary_conditions[1][0][1:-1],boundary_conditions[1][1][1:-1]),boundary_conditions[2])
    return boundary_conditions


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


def format_figure(ax,title_size=30,label_size=30,tick_size=22):
    """Given the axis handle, adjust the font sizes of the title, axis labels, and tick labels."""
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(ax.get_xlabel(),fontsize=label_size)
    ax.set_ylabel(ax.get_ylabel(),fontsize=label_size)
    # ax.set_xlabel("\n"+ax.get_xlabel(),fontsize=label_size)
    # ax.xaxis.set_label_coords(0.5,-0.1)
    # ax.set_ylabel("\n"+ax.get_ylabel(),fontsize=label_size)
    # ax.yaxis.set_label_coords(-0.1,0.5)
    ax.set_title(ax.get_title(),fontsize=title_size)

def format_figure_3D(ax,title_size=30,label_size=30,tick_size=22):
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
    return separations, centers

def get_magnetization(Hext,particle_posns,particle_radius,chi,Ms,l_e):
    """Get the overall system magnetization as a vector sum of the magnetizations of the particles."""
    magnetizations = magnetism.get_magnetization_iterative_normalized(Hext,particle_posns,particle_radius,chi,Ms,l_e)
    normalized_magnetizations = magnetizations/Ms
    overall_magnetization = np.sum(normalized_magnetizations,axis=0)/magnetizations.shape[0]
    return overall_magnetization

def get_magnetization_gpu(particle_posns,particles,Hext,Ms,chi,particle_volume,l_e):
    num_particles = particles.shape[0]
    # particle_centers = np.empty((num_particles,3),dtype=np.float32)
    # for i, particle in enumerate(particles):
    #     particle_centers[i,:] = simulate.get_particle_center(particle,posns)
    particle_volume = np.float32(particle_volume)
    l_e = np.float32(l_e)
    Ms = np.float32(Ms)
    chi = np.float32(chi)
    Hext = Hext.astype(np.float32)
    hext = cp.asarray(Hext/Ms)
    # particle_centers = particle_posns
    particle_posns = cp.asarray(particle_posns,dtype=cp.float32).reshape((3*num_particles,))
    magnetization, _, _, return_code = simulate.get_normalized_magnetization_fixed_point_iteration(hext,num_particles,particle_posns,chi,particle_volume,l_e)
    magnetization = cp.asnumpy(magnetization).reshape((num_particles,3))
    # magnetic_moments = simulate.get_magnetization_iterative(Hext,particles,cp.array(particle_centers.astype(np.float32)).reshape((particle_centers.shape[0]*particle_centers.shape[1],1),order='C'),Ms,chi,particle_volume,l_e)
    # particle_magnetizations = magnetic_moments/particle_volume
    # normalized_magnetizations = particle_magnetizations/Ms
    overall_magnetization = np.sum(magnetization,axis=0)/num_particles
    return overall_magnetization

def plot_particle_behavior(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series):
    """Plot the particle behavior as a function of applied field (particle separation and sytem magnetization)"""
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    magnetization = np.zeros((num_output_files,3))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        separations[i,:], particle_posns = get_particle_separation(final_posns,particles)
        particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
        magnetization[i,:] = get_magnetization_gpu(particle_posns,particles,Hext,Ms,chi,particle_volume,l_e)
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

def plot_particle_behavior_hysteresis(sim_dir,num_output_files,particles,particle_radius,chi,Ms,l_e,Hext_series):
    """Plot the particle behavior as a function of applied field (particle separation and sytem magnetization)"""
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    magnetization = np.zeros((num_output_files,3))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        #temporarily doing casting to deal with the analysis of simulations ran using gpu calculations (where 32bit floats must be used)
        separations[i,:], particle_posns = get_particle_separation(final_posns,particles)
        particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
        magnetization[i,:] = get_magnetization_gpu(particle_posns,particles,Hext,Ms,chi,particle_volume,l_e)
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

def check_clustering(num_particles,num_output_files,Hext_series,particle_separations,l_e,clustering_distance=4):
    """Given some threshold separation in microns, print out the number of clusters formed by particles whose center-to-center distances are below the threshold for each output file."""
    particle_separations *= l_e*1e6
    particle_separations_matrix = np.zeros((num_output_files,num_particles,num_particles))
    for output_file_count in np.arange(num_output_files):
        counter = 0
        for i in np.arange(num_particles):
            for j in np.arange(i+1,num_particles):
                particle_separations_matrix[output_file_count,i,j] = particle_separations[output_file_count,counter]
                counter += 1
        particle_separations_matrix[output_file_count] += np.transpose(particle_separations_matrix[output_file_count])

    for output_file_count in np.arange(num_output_files):
        cluster_counter = 0
        for i in np.arange(num_particles):
            temp_separations = particle_separations_matrix[output_file_count,i,:]
            cluster_counter += np.count_nonzero(np.less_equal(temp_separations[temp_separations>0],clustering_distance))
            #if we know some particle clustering has ocurred, how can we determine if a single particle is clustering with multiple particles, and cross reference to determine if a chain has formed, and how many particles make up that chain?
        cluster_counter /= 2
        if cluster_counter != 0:
            print(f'for field {np.round(Hext_series[output_file_count]*mu0,decimals=5)} and output file {[output_file_count]} the total number of clusters: {cluster_counter}')
            # print(f'for field {np.round(Hext_series[output_file_count]*mu0,decimals=5)} and {bc_type} {bc_values[output_file_count]} the total number of clusters: {cluster_counter}')
    return particle_separations_matrix

def plot_particle_strain_response(sim_dir,num_output_files,particles,Bext_series,boundary_condition_series,l_e):
    """Make a composite plot of particle orientation, separation, and positions as well as their change with respect to applied strain. Intended for two particle simulations."""
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    particle_posns = np.zeros((num_output_files,num_particles,3))
    particle_orientation = np.zeros((num_output_files,num_particles,2))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        separations[i,:], particle_posns[i,:] = get_particle_separation(final_posns,particles)
        particle_orientation[i,:] = get_particle_orientation(particles,final_posns)

    unique_fields = np.unique(np.linalg.norm(Bext_series,axis=1))
    num_fields = unique_fields.shape[0]
    num_boundary_conditions = boundary_condition_series.shape[0]
    unique_field_particle_separations = np.zeros((num_fields,num_boundary_conditions))
    unique_field_particle_separation_change = np.zeros((num_fields,num_boundary_conditions))
    unique_field_particle_posns = np.zeros((num_fields,num_boundary_conditions,num_particles,3))
    unique_field_particle_posn_change = np.zeros((num_fields,num_boundary_conditions,num_particles,3))
    unique_field_particle_orientations = np.zeros((num_fields,num_boundary_conditions,num_particles,2))
    unique_field_particle_orientation_change = np.zeros((num_fields,num_boundary_conditions,num_particles,2))

    for i in range(num_fields):
        unique_field_particle_separations[i] = 1e6*l_e*separations[i::num_fields].ravel()
        unique_field_particle_separation_change[i] = unique_field_particle_separations[i,:] - unique_field_particle_separations[i,0]
        unique_field_particle_posns[i] = 1e6*l_e*particle_posns[i::num_fields]
        unique_field_particle_posn_change[i] = unique_field_particle_posns[i,:] - unique_field_particle_posns[i,0]
        unique_field_particle_orientations[i] = particle_orientation[i::num_fields]
        unique_field_particle_orientation_change[i] = unique_field_particle_orientations[i,:] - unique_field_particle_orientations[i,0]


    fig, axs = plt.subplots(1,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    unique_markers = ['^','s','d','v','o','+','*']
    for i in range(np.floor_divide(num_fields,2)):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0].plot(boundary_condition_series,unique_field_particle_separations[i],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1].plot(boundary_condition_series,unique_field_particle_separation_change[i],unique_markers[np.mod(i,len(unique_markers))])

    axs[0].set_xlabel('applied strain')
    axs[1].set_xlabel('applied strain')
    axs[0].set_ylabel(r'particle separation ($\mu$m)')
    axs[1].set_ylabel(r'change in particle separation ($\mu$m)')

    fig.legend()

    format_figure(axs[0])
    format_figure(axs[1])
    savename = f'particle_separation_strain_response0.png'
    save_dir = sim_dir+'/figures/'
    plt.savefig(save_dir+savename)

    fig, axs = plt.subplots(1,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    for i in range(np.floor_divide(num_fields,2),num_fields):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0].plot(boundary_condition_series,unique_field_particle_separations[i],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1].plot(boundary_condition_series,unique_field_particle_separation_change[i],unique_markers[np.mod(i,len(unique_markers))])

    axs[0].set_xlabel('applied strain')
    axs[1].set_xlabel('applied strain')
    axs[0].set_ylabel(r'particle separation ($\mu$m)')
    axs[1].set_ylabel(r'change in particle separation ($\mu$m)')

    fig.legend()

    format_figure(axs[0])
    format_figure(axs[1])
    savename = f'particle_separation_strain_response1.png'
    plt.savefig(save_dir+savename)

    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    for i in range(np.floor_divide(num_fields,2)):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0,0].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1,0].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,1].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,1].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,2].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,2],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,2].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,2],unique_markers[np.mod(i,len(unique_markers))])
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[1,0].set_xlabel('applied strain')
    axs[1,1].set_xlabel('applied strain')
    axs[1,2].set_xlabel('applied strain')
    axs[0,0].set_ylabel(r'particle posn ($\mu$m)')
    axs[1,0].set_ylabel(r'change in posn ($\mu$m)')

    fig.legend()

    format_figure(axs[0,0])
    format_figure(axs[0,1])
    format_figure(axs[1,0])
    format_figure(axs[1,1])
    format_figure(axs[0,2])
    format_figure(axs[1,2])
    savename = f'particle_posn_strain_response0.png'
    plt.savefig(save_dir+savename)

    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    for i in range(np.floor_divide(num_fields,2),num_fields):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0,0].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1,0].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,1].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,1].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,2].plot(boundary_condition_series,unique_field_particle_posns[i,:,:,2],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,2].plot(boundary_condition_series,unique_field_particle_posn_change[i,:,:,2],unique_markers[np.mod(i,len(unique_markers))])
    axs[0,0].set_title('X')
    axs[0,1].set_title('Y')
    axs[0,2].set_title('Z')
    axs[1,0].set_xlabel('applied strain')
    axs[1,1].set_xlabel('applied strain')
    axs[1,2].set_xlabel('applied strain')
    axs[0,0].set_ylabel(r'particle posn ($\mu$m)')
    axs[1,0].set_ylabel(r'change in posn ($\mu$m)')

    fig.legend()

    format_figure(axs[0,0])
    format_figure(axs[0,1])
    format_figure(axs[1,0])
    format_figure(axs[1,1])
    format_figure(axs[0,2])
    format_figure(axs[1,2])
    savename = f'particle_posn_strain_response1.png'
    plt.savefig(save_dir+savename)


    fig, axs = plt.subplots(2,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    for i in range(np.floor_divide(num_fields,2)):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0,0].plot(boundary_condition_series,unique_field_particle_orientations[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1,0].plot(boundary_condition_series,unique_field_particle_orientation_change[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,1].plot(boundary_condition_series,unique_field_particle_orientations[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,1].plot(boundary_condition_series,unique_field_particle_orientation_change[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
    axs[0,0].set_title('polar')
    axs[0,1].set_title('azimuthal')
    axs[1,0].set_xlabel('applied strain')
    axs[1,1].set_xlabel('applied strain')
    axs[0,0].set_ylabel(r'particle orientation (rad)')
    axs[1,0].set_ylabel(r'change in particle orientation (rad)')

    fig.legend()

    format_figure(axs[0,0])
    format_figure(axs[0,1])
    format_figure(axs[1,0])
    format_figure(axs[1,1])
    savename = f'particle_orientation_strain_response0.png'
    plt.savefig(save_dir+savename)

    fig, axs = plt.subplots(2,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    for i in range(np.floor_divide(num_fields,2),num_fields):
        field = np.round(unique_fields[i]*1000,decimals=0)
        axs[0,0].plot(boundary_condition_series,unique_field_particle_orientations[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))],label=f'{field}mT')
        axs[1,0].plot(boundary_condition_series,unique_field_particle_orientation_change[i,:,:,0],unique_markers[np.mod(i,len(unique_markers))])
        axs[0,1].plot(boundary_condition_series,unique_field_particle_orientations[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
        axs[1,1].plot(boundary_condition_series,unique_field_particle_orientation_change[i,:,:,1],unique_markers[np.mod(i,len(unique_markers))])
    axs[0,0].set_title('polar')
    axs[0,1].set_title('azimuthal')
    axs[1,0].set_xlabel('applied strain')
    axs[1,1].set_xlabel('applied strain')
    axs[0,0].set_ylabel(r'particle orientation (rad)')
    axs[1,0].set_ylabel(r'change in particle orientation (rad)')

    fig.legend()

    format_figure(axs[0,0])
    format_figure(axs[0,1])
    format_figure(axs[1,0])
    format_figure(axs[1,1])
    savename = f'particle_orientation_strain_response1.png'
    plt.savefig(save_dir+savename)

def get_particle_orientation(particles,node_posns):
    """Get the particle orientation as (polar,azimuthal) angle pairs."""
    num_particles = particles.shape[0]
    particle_orientation = np.zeros((num_particles,2))
    rij_vectors = node_posns[particles[:,0]] - node_posns[particles[:,1]]
    rij_magnitudes = np.linalg.norm(rij_vectors,axis=1)
    zaxis = np.array([0,0,1])
    xaxis = np.array([1,0,0])
    particle_orientation[:,0] = np.arccos(np.dot(rij_vectors,zaxis)/rij_magnitudes)
    rij_vectors[:,2] = 0
    rij_magnitudes = np.linalg.norm(rij_vectors,axis=1)
    particle_orientation[:,1] = np.arccos(np.dot(rij_vectors,xaxis)/rij_magnitudes) 
    return particle_orientation

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

def get_zero_strain_reference_forces(sim_dir):
    """Return the forces acting on the relevant boundary for each applied field value at zero strain."""
    _, _, _, _, boundaries, _, _, field_series, _, _ = mre.initialize.read_init_file(sim_dir+'init.h5')

    sim_variables_dict, _ = reinitialize_sim(sim_dir)

    beta_i = sim_variables_dict['beta_i']
    host_beta_i = cp.asnumpy(beta_i)
    elements = sim_variables_dict['elements']
    kappa = sim_variables_dict['kappa']
    springs = sim_variables_dict['springs']
    drag = sim_variables_dict['drag']
    boundaries = sim_variables_dict['boundaries']

    num_unique_fields = field_series.shape[0]
    Bext_series = mu0*field_series

    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    bc_direction = boundary_conditions[1]
    if bc_direction[0] == 'x':
        relevant_boundary = 'right'
    elif bc_direction[0] == 'y':
        relevant_boundary = 'back'
    elif bc_direction[0] == 'z':
        relevant_boundary = 'top'
    num_boundary_nodes = boundaries[relevant_boundary].shape[0]
    zero_strain_comparison_forces = np.zeros((num_unique_fields,num_boundary_nodes,3),dtype=np.float32)

    for i in range(num_unique_fields):
        final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        assert(np.isclose(np.linalg.norm(mu0*applied_field),np.linalg.norm(Bext_series[i])))
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        bc_direction = boundary_conditions[1]
        if not np.isclose(boundary_conditions[2],0):
            raise ValueError('Unexpected non-zero value for boundary condition while calculating comparison system configuration metric used to define strain or comparison force used to define stress')
        final_posns, _, _, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
        velocities = cp.zeros(posns.shape,order='C',dtype=cp.float32)
        N_nodes = int(posns.shape[0]/3)
        accel = simulate.composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,beta_i,drag)
        accel = np.reshape(accel,(N_nodes,3))
        zero_strain_comparison_forces[i] = -1*accel[boundaries[relevant_boundary]]/host_beta_i[boundaries[relevant_boundary]]
    
    return zero_strain_comparison_forces

def plot_full_sim_surface_forces(sim_dir):
    """Given the simulation directory, plot the required force vector components at each node on the strained surface to keep the surface fixed (for non-zero strain values)"""
    host_initial_node_posns, mass, _, _, boundaries, _, parameters, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
    sim_variables_dict, _ = reinitialize_sim(sim_dir)

    l_e = sim_variables_dict['element_length']

    num_output_files = get_num_output_files(sim_dir)

    output_dir = sim_dir + 'figures/surface_forces/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    zero_strain_comparison_forces = get_zero_strain_reference_forces(sim_dir)

    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        device_posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')

        plot_surface_forces(output_dir,i,host_initial_node_posns,sim_variables_dict,final_posns,boundary_conditions,zero_strain_comparison_forces)

def plot_surface_forces(output_dir,output_file_number,initial_node_posns,sim_variables_dict,final_posns,boundary_conditions,reference_forces):
    """Calculate the forces acting on the "probed" boundary nodes and plot the force vector components and norm in a subplot."""
    #TODO get the zero-strain value forces. plot both the total force values and the total minus reference (zero-strain) forces for non-zero strain simulations. separate figures for the two.
    dimensions = sim_variables_dict['dimensions']
    l_e = sim_variables_dict['element_length']
    Lx = dimensions[0]/l_e
    Ly = dimensions[0]/l_e
    Lz = dimensions[0]/l_e
    layers = (Lx,Ly,Lz)

    beta_i = sim_variables_dict['beta_i']
    host_beta_i = cp.asnumpy(beta_i)
    elements = sim_variables_dict['elements']
    kappa = sim_variables_dict['kappa']
    springs = sim_variables_dict['springs']
    drag = sim_variables_dict['drag']
    boundaries = sim_variables_dict['boundaries']
    bc_direction = boundary_conditions[1]

    if bc_direction[0] == 'x':
        #forces that must act on the boundaries for them to be in this position
        relevant_boundaries = ('right','left')
    elif bc_direction[0] == 'y':
        relevant_boundaries = ('back','front')
    elif bc_direction[0] == 'z':
        relevant_boundaries = ('top','bot')

    posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
    velocities = cp.zeros(posns.shape,order='C',dtype=cp.float32)
    N_nodes = int(posns.shape[0]/3)
    accel = simulate.composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,beta_i,drag)
    accel = np.reshape(accel,(N_nodes,3))
    boundary_forces = -1*accel[boundaries[relevant_boundaries[0]]]/host_beta_i[boundaries[relevant_boundaries[0]]]
    all_forces = np.zeros(accel.shape)
    all_forces[boundaries[relevant_boundaries[0]]] = boundary_forces
    
    if boundary_conditions[1][0] == 'x':
        index = int(np.round(Lx))
        cut_type = 'yz'
    elif boundary_conditions[1][0] == 'y':
        index = int(np.round(Ly))
        cut_type = 'xz'
    elif boundary_conditions[1][0] == 'z':
        index = int(np.round(Lz))
        cut_type = 'xy'
    subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,all_forces,index,output_dir,tag=f"strained_surface_forces_series{output_file_number}")

    output_file_number_for_comparison = int(np.mod(output_file_number,reference_forces.shape[0]))
    relative_boundary_forces = boundary_forces - reference_forces[output_file_number_for_comparison]
    all_forces = np.zeros(accel.shape)
    all_forces[boundaries[relevant_boundaries[0]]] = relative_boundary_forces
    subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,all_forces,index,output_dir,tag=f"strained_surface_relative_forces_series{output_file_number}")

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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
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

def plot_particles_scatter_sim(sim_dir):
    """Makes 3D scatter plots of the particle nodes for each field + bc combo step output file."""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
    num_output_files = get_num_output_files(sim_dir)
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    if not (os.path.isdir(output_dir+'particle_behavior/')):
        os.mkdir(output_dir+'particle_behavior/')
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_nodes(initial_node_posns,final_posns,particles,output_dir+'particle_behavior/',tag=f"{i}")

def get_energies(sim_dir):
    """For the configurations in the output files, calculate the different energies, and the total energy."""
    # initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
    sim_variables_dict, _ = reinitialize_sim(sim_dir)
    elements = sim_variables_dict['elements']
    kappa = sim_variables_dict['kappa']
    springs = sim_variables_dict['springs']

    particles = sim_variables_dict['particles']
    num_particles = particles.shape[0]
    Ms = sim_variables_dict['Ms']
    chi = sim_variables_dict['chi']
    particle_volume = sim_variables_dict['particle_volume']

    l_e = sim_variables_dict['element_length']

    num_output_files = get_num_output_files(sim_dir)
    total_energy = np.zeros((num_output_files,1),dtype=np.float32)
    spring_energy = np.zeros((num_output_files,1),dtype=np.float32)
    element_energy = np.zeros((num_output_files,1),dtype=np.float32)
    dipole_energy = np.zeros((num_output_files,1),dtype=np.float32)
    wca_energy = np.zeros((num_output_files,1),dtype=np.float32)
    self_energy = np.zeros((num_output_files,1),dtype=np.float32)
    zeeman_energy = np.zeros((num_output_files,1),dtype=np.float32)

    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        device_posns = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')

        particle_posns = simulate.get_particle_posns(particles,final_posns)
        particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
        total_energy[i], spring_energy[i], element_energy[i], dipole_energy[i], wca_energy[i], self_energy[i], zeeman_energy[i] = composite_gpu_energy_calc(device_posns,elements,kappa,springs,particles,particle_posns,num_particles,Hext,Ms,chi,particle_volume,l_e)
    return total_energy, spring_energy, element_energy, dipole_energy, wca_energy, self_energy, zeeman_energy

def plot_energy_figures(sim_dir):
    """Given the energies, plot them versus the applied strains and external fields, etc."""
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    if not (os.path.isdir(output_dir+'energy/')):
        os.mkdir(output_dir+'energy/')
    fig_output_dir = output_dir + 'energy/'
    node_posns, _, _, _, _, particles, parameters, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
    l_e = parameters[7]
    dimensions = np.array([np.max(node_posns[:,0])*l_e,np.max(node_posns[:,1])*l_e,np.max(node_posns[:,2])*l_e])
    total_sim_volume = dimensions[0]*dimensions[1]*dimensions[2]
    particle_volume = (4/3)*np.pi*np.power(parameters['particle_radius'],3)
    print(f'Actual volume fraction: {particles.shape[0]*particle_volume/total_sim_volume}')
    if 'stress' in sim_type:
        xlabel = 'stress (Pa)'
        xscale_factor = 1
    elif 'strain' in sim_type:
        xlabel = 'strain (%)'
        xscale_factor = 100
        if 'shearing' in sim_type:
            xscale_factor = 1
            xlabel='shear strain'
            fit_func = shearing_quadratic_fit_func
        else:
            fit_func = quadratic_no_linear_term_fit_func
    elif 'hysteresis' in sim_type:
        pass

    total_energy, spring_energy, element_energy, dipole_energy, wca_energy, self_energy, zeeman_energy = get_energies(sim_dir)

    num_output_files = get_num_output_files(sim_dir)
    applied_field = np.zeros((num_output_files,3),dtype=np.float32)
    applied_bc = np.zeros((num_output_files,),dtype=np.float32)
    for i in range(num_output_files):
        _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        applied_field[i] = Hext*mu0
        applied_bc[i] = boundary_conditions[2]
    # if 'shearing' in sim_type and 'strain' in sim_type:
    #     applied_bc = np.tan(applied_bc)
    applied_bc *= xscale_factor
    #which figures do i want? I want plots of the total energy, and the individual energies as a function of the boundary condition value, with each line representing a different applied field. I may also want the inverse, (each line a different strain), and I may also want to plot different energies separately, but lets just start
    Bext_magnitude = np.linalg.norm(applied_field,axis=1)
    unique_field_values = np.unique(Bext_magnitude)
    energy_density_plus_wca_plus_self_fit_modulus = np.zeros((unique_field_values.shape[0],))
    energy_density_plus_wca_plus_self_fit_error = np.zeros((unique_field_values.shape[0],))
    subset_modulus_error_dict = dict({})
    for i, unique_value in enumerate(unique_field_values):
        # fig, axs = plt.subplots(1,3)
        # default_width,default_height = fig.get_size_inches()
        # fig.set_size_inches(2*default_width,2*default_height)
        # fig.set_dpi(200)
        relevant_indices = np.isclose(unique_value,np.linalg.norm(applied_field,axis=1))
        plotting_total_energy = total_energy[relevant_indices]
        plotting_spring_energy = spring_energy[relevant_indices]
        plotting_element_energy = element_energy[relevant_indices]
        plotting_dipole_energy = dipole_energy[relevant_indices]
        plotting_wca_energy = wca_energy[relevant_indices]
        plotting_self_energy = self_energy[relevant_indices]
        plotting_zeeman_energy = zeeman_energy[relevant_indices]
        plotting_bc = applied_bc[relevant_indices]

        #check for issues with total energy by observing the trend of the self energy as the strain increases. if the trend changes (goes from increasing to decreasing or vice versa), need to fit to a subset of the data, or not use the dataset at all for effective modulus analysis
        strain_differential_self_energy = np.diff(plotting_self_energy.ravel())
        tmp_var = np.where(strain_differential_self_energy[:-1]*strain_differential_self_energy[1:] < 0)[0]
        if tmp_var.shape[0] == 1 and tmp_var[0] == 0:
            potential_subsets = 1
        else:
            energy_trend_switch_indices = tmp_var[::2] + 2
            potential_subsets = energy_trend_switch_indices.shape[0] + 1
        fig, axs = plt.subplots(2,3)
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(200)
        axs[0,0].plot(plotting_bc,plotting_element_energy,marker='^',linestyle='-',label=f'Element: {np.round(unique_value*1000)} (mT)')
        axs[0,1].plot(plotting_bc,plotting_spring_energy,marker='X',linestyle='-',label=f'Spring: {np.round(unique_value*1000)} (mT)')
        axs[0,2].plot(plotting_bc,plotting_wca_energy,marker='D',linestyle='-',label=f'WCA: {np.round(unique_value*1000)} (mT)')
        axs[1,0].plot(plotting_bc,plotting_zeeman_energy,marker='o',linestyle='-',label=f'Zeeman: {np.round(unique_value*1000)} (mT)')
        axs[1,1].plot(plotting_bc,plotting_dipole_energy,marker='v',linestyle='-',label=f'Dipole: {np.round(unique_value*1000)} (mT)')
        axs[1,2].plot(plotting_bc,plotting_self_energy,marker='P',linestyle='-',label=f'Self: {np.round(unique_value*1000)} (mT)')
        axs[1,0].set_xlabel(xlabel)
        axs[1,1].set_xlabel(xlabel)
        axs[1,2].set_xlabel(xlabel)
        axs[0,0].set_ylabel('Energy (J)')
        axs[1,0].set_ylabel('Energy (J)')
        format_figure(axs[0,0])
        format_figure(axs[0,1])
        format_figure(axs[0,2])
        format_figure(axs[1,0])
        format_figure(axs[1,1])
        format_figure(axs[1,2])
        fig.legend()
        savename = fig_output_dir + f'individual_energies_{np.round(unique_value*1000)}_mT.png'
        plt.savefig(savename)
        plt.close()

        modulus_fit_guess = 9e3

        energy_plus_wca_plus_self_density = np.ravel(plotting_total_energy)/total_sim_volume

        fig, ax = plt.subplots()
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(200)
        ax.plot(plotting_bc,plotting_total_energy,marker='D',linestyle='-',label=f'Total: {np.round(unique_value*1000)} (mT)')
        ax.set_title('System Energy')
        ax.set_ylabel('Energy (J)')
        format_figure(ax)
        ax.set_xlabel(xlabel)

        fig.legend()
        modulus_fit_guess = 9e3
        if potential_subsets == 1:
            popt, pcov = scipy.optimize.curve_fit(fit_func,plotting_bc/xscale_factor,energy_plus_wca_plus_self_density,p0=np.array([modulus_fit_guess,0]))
            energy_density_plus_wca_plus_self_fit_modulus[i] = popt[0]
            energy_density_plus_wca_plus_self_fit_error[i] = np.sqrt(np.diag(pcov))[0]
            ax.plot(plotting_bc,total_sim_volume*fit_func(plotting_bc/xscale_factor,popt[0],popt[1]))
            plt.annotate(f'modulus from fit: {np.round(energy_density_plus_wca_plus_self_fit_modulus[i])}',xy=(10,10),xycoords='figure pixels')
        else:
            subset_start_idx = 0
            for subset_count in range(potential_subsets):
                if subset_count + 1 == potential_subsets:
                    subset_end_idx = plotting_bc.shape[0]
                else:
                    subset_end_idx = energy_trend_switch_indices[subset_count]
                #if there are not at least 3 datapoints to fit to, don't bother trying a fit.
                if subset_end_idx - subset_start_idx < 3:
                    subset_start_idx = subset_end_idx
                    continue
                popt, pcov = scipy.optimize.curve_fit(fit_func,plotting_bc[subset_start_idx:subset_end_idx]/xscale_factor,energy_plus_wca_plus_self_density[subset_start_idx:subset_end_idx],p0=np.array([modulus_fit_guess,0]))
                if subset_count == 0:
                    energy_density_plus_wca_plus_self_fit_modulus[i] = popt[0]
                    energy_density_plus_wca_plus_self_fit_error[i] = np.sqrt(np.diag(pcov))[0]
                else:
                    subset_modulus_error_dict[f'{i}'] = (popt[0],np.sqrt(np.diag(pcov))[0])
                ax.plot(plotting_bc[subset_start_idx:subset_end_idx],total_sim_volume*fit_func(plotting_bc[subset_start_idx:subset_end_idx]/xscale_factor,popt[0],popt[1]))
                plt.annotate(f'modulus from fit: {np.round(energy_density_plus_wca_plus_self_fit_modulus[i])}',xy=(10,10),xycoords='figure pixels')
                if f'{i}' in subset_modulus_error_dict:
                    plt.annotate(f'modulus from alt fit: {np.round(subset_modulus_error_dict[f"{i}"][0])}',xy=(10,25),xycoords='figure pixels')
                subset_start_idx = subset_end_idx
        savename = fig_output_dir + f'total_energy_{np.round(unique_value*1000)}_mT.png'
        plt.savefig(savename)
        plt.close()

    #if there were not at least 3 data points for the first subset of data to fit to, the modulus would be zero, and this will break the search for outliers. so we need to correct for that here, by getting the true smallest modulus
    nonzero_modulus_indices = np.nonzero(energy_density_plus_wca_plus_self_fit_modulus)[0]
    smallest_modulus_magnitude = np.min(np.abs(energy_density_plus_wca_plus_self_fit_modulus[nonzero_modulus_indices]))
    outlier_mask = np.logical_or(energy_density_plus_wca_plus_self_fit_modulus<=0,energy_density_plus_wca_plus_self_fit_modulus>(100*smallest_modulus_magnitude))
    non_outlier_mask = np.logical_not(outlier_mask)
    fig, ax = plt.subplots()
    ax.errorbar(unique_field_values[non_outlier_mask]*1000,energy_density_plus_wca_plus_self_fit_modulus[non_outlier_mask],linestyle='-',marker='o',yerr=energy_density_plus_wca_plus_self_fit_error[non_outlier_mask])

    #now i need to grab the other subsets fits that were done, and get those results on the same figure.
    subset_modulus = np.zeros(energy_density_plus_wca_plus_self_fit_modulus.shape)
    subset_error = np.zeros(energy_density_plus_wca_plus_self_fit_modulus.shape)
    for key in subset_modulus_error_dict.keys():
        subset_modulus[int(key)] = subset_modulus_error_dict[key][0]
        subset_error[int(key)] = subset_modulus_error_dict[key][1]
    nonzero_indices = np.nonzero(subset_modulus)[0]
    if nonzero_indices.shape[0] != 0:
        ax.errorbar(1000*unique_field_values[nonzero_indices],subset_modulus[nonzero_indices],marker='s',yerr=subset_error[nonzero_indices])
    
    ax.set_xlabel(f'Applied Field (mT)')
    ax.set_ylabel(f'Modulus (Pa)')
    plt.annotate(f'modulus minimum: {np.round(np.min(energy_density_plus_wca_plus_self_fit_modulus[non_outlier_mask]))}',xy=(10,10),xycoords='figure pixels')
    savename = fig_output_dir + 'energy_plus_wca_plus_self_density_fit_modulus.png'
    plt.savefig(savename)
    plt.close()

def reinitialize_sim(sim_dir):
    #first figure out the simulation type
    node_posns, mass, springs_var, elements, boundaries, particles, parameters, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
    # figure out the field sequence and stress/strain sequence if necessary
    # figure out where things were interrupted
    num_output_files = mre.analyze.get_num_output_files(sim_dir)
    continuation_index = num_output_files

    #extract argument values from the parameters variable
    young_modulus = parameters[1]
    poisson_ratio = parameters[2]
    anisotropy_factor = parameters[3]
    spring_stiffness = parameters[4]
    kappa = parameters[5]
    drag = parameters[6]
    l_e = parameters[7]
    Ms = parameters[9]
    chi = parameters[10]
    particle_radius = parameters[11]
    particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
    particle_mass = parameters[12]
    max_integration_rounds = parameters[13]
    max_integration_steps = parameters[14]
    time_step = parameters[15]
    tolerance = parameters[16]
    beta = parameters[17]
    beta_i = beta/mass
    characteristic_mass = parameters[18]
    characteristic_time = parameters[19]
    dimensions = np.array([np.max(node_posns[:,0])*l_e,np.max(node_posns[:,1])*l_e,np.max(node_posns[:,2])*l_e])
    Hext_series = field_series

    _, _, boundary_condition, _ = mre.initialize.read_output_file(sim_dir+f'output_{continuation_index-1}.h5')
    boundary_condition = format_boundary_conditions(boundary_condition)
    print(boundary_condition)
    bc_direction = boundary_condition[1]

    # convert all the necessary variables to the appropriate types (32 bit floats, or integers) and move necessary variables to gpu memory
    #this requires reading in the variables from the parameters variable, and repacking the sim_variables_dict
    x0 = cp.array(node_posns.astype(np.float32)).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
    beta_i = cp.array(beta_i.astype(np.float32)).reshape((beta_i.shape[0],1),order='C')
    beta = np.float32(beta)
    drag = np.float32(drag)
    if 'stress' in sim_type:
        stresses = np.float32(boundary_condition_series)
    if 'strain' in sim_type:
        strains = np.float32(boundary_condition_series)
    Hext_series = np.float32(Hext_series)
    particles = np.int32(particles)
    for key in boundaries:
        boundaries[key] = np.int32(boundaries[key])
    dimensions = np.float32(dimensions)
    kappa = cp.float32(kappa*(l_e**2))
    l_e = np.float32(l_e)
    particle_radius = np.float32(particle_radius)
    particle_volume = np.float32(particle_volume)
    particle_mass = np.float32(particle_mass)
    chi = np.float32(chi)
    Ms = np.float32(Ms)
    elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
    springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
    step_size = cp.float32(time_step)#cp.float32(0.01)

    #pack into sim_variables_dict for passing to the run_*_sim() function. those functions need flags for knowing if a simulation is being continued or extended
    sim_variables_dict = dict({})
    sim_variables_dict['output_dir'] = sim_dir
    sim_variables_dict['initial_posns'] = x0

    sim_variables_dict['Hext_series'] = Hext_series

    sim_variables_dict['springs'] = springs_var
    sim_variables_dict['elements'] = elements
    sim_variables_dict['dimensions'] = dimensions
    sim_variables_dict['boundaries'] = boundaries
    sim_variables_dict['kappa'] = kappa
    sim_variables_dict['element_length'] = l_e
    sim_variables_dict['beta'] = beta
    sim_variables_dict['beta_i'] = beta_i
    sim_variables_dict['drag'] = drag

    sim_variables_dict['particles'] = particles
    sim_variables_dict['particle_radius'] = particle_radius
    sim_variables_dict['particle_volume'] = particle_volume
    sim_variables_dict['particle_mass'] = particle_mass
    sim_variables_dict['chi'] = chi
    sim_variables_dict['Ms'] = Ms

    if 'strain' in sim_type:
        sim_variables_dict['strains'] = strains
    elif 'stress' in sim_type:
        sim_variables_dict['stresses'] = stresses

    sim_variables_dict['boundary_condition_type'] = sim_type
    sim_variables_dict['boundary_condition_direction'] = bc_direction

    sim_variables_dict['max_integrations'] = max_integration_rounds
    sim_variables_dict['max_integration_steps'] = max_integration_steps
    sim_variables_dict['tolerance'] = tolerance
    sim_variables_dict['step_size'] = step_size

    sim_variables_dict['gpu_flag'] = True 
    sim_variables_dict['particle_rotation_flag'] = True
    sim_variables_dict['persistent_checkpointing_flag'] = True
    sim_variables_dict['plotting_flag'] = False
    sim_variables_dict['criteria_flag'] = False

    sim_variables_dict['continuation_index'] = continuation_index

    my_sim = mre.initialize.Simulation(young_modulus,poisson_ratio,kappa,spring_stiffness,drag,l_e,dimensions[0],dimensions[1],dimensions[2],particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integration_rounds,max_integration_steps,step_size,tolerance,anisotropy_factor)

    return sim_variables_dict, my_sim

def composite_gpu_energy_calc(posns,cupy_elements,kappa,cupy_springs,particles,particle_posns,num_particles,Hext,Ms,chi,particle_volume,l_e):
    """Combining gpu kernels to calculate different energies"""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    size_elements = int(cupy_elements.shape[0]/8)
    block_size = 128
    element_grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    element_energies = cp.zeros((size_elements,1),dtype=cp.float32)
    simulate.scaled_element_energy_kernel((element_grid_size,),(block_size,),(cupy_elements,posns,kappa,element_energies,size_elements))
    cupy_stream.synchronize()

    size_springs = int(cupy_springs.shape[0]/4)
    spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    spring_energies = cp.zeros((size_springs,1),dtype=cp.float32)

    simulate.scaled_spring_energy_kernel((spring_grid_size,),(block_size,),(cupy_springs,posns,spring_energies,size_springs))
    cupy_stream.synchronize()
    if num_particles != 0:
        magnetization, htot, _ = simulate.get_normalized_magnetization_and_total_field(Hext/Ms,num_particles,particle_posns,chi,particle_volume,l_e)
        magnetic_moments = magnetization*Ms*particle_volume
        Htot = htot*Ms
        # magnetic_moments, Htot = simulate.get_magnetization_iterative_and_total_field(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e)
        dipole_grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        dipole_energies = cp.zeros((num_particles,1),dtype=cp.float32)
        Hext_norm = np.linalg.norm(Hext)
        Hext = cp.array(Hext,dtype=cp.float32,order='C')
        simulate.dipole_energy_kernel((dipole_grid_size,),(block_size,),(magnetic_moments,Htot,Hext,dipole_energies,num_particles))
        cupy_stream.synchronize()
        
        host_magnetic_moments = cp.asnumpy(magnetic_moments)
        # self_energy = np.sum(host_magnetic_moments*host_magnetic_moments)*mu0/2/chi/particle_volume
        self_energy = np.sum(host_magnetic_moments*host_magnetic_moments)*mu0/2/chi/particle_volume/Ms*(Ms+chi*Hext_norm)

        zeeman_energy = -1*mu0*np.sum(np.dot(host_magnetic_moments.reshape((num_particles,3)),cp.asnumpy(Hext)))

        #get separation vectors for wca energy
        separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
        separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
        simulate.separation_vectors_kernel((dipole_grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,num_particles))
        cupy_stream.synchronize()

        wca_energies = cp.zeros((num_particles,1),dtype=cp.float32)
        particle_radius = np.float32(np.power((3./4)*(1/np.pi)*particle_volume,1/3))
        inv_l_e = np.float32(1/l_e)
        simulate.wca_energy_kernel((dipole_grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,particle_radius,l_e,inv_l_e,wca_energies,num_particles))
        cupy_stream.synchronize()

        #factor of two because we count the interaction of each dipole with each other dipole, which results in a double counting of the interaction energy for each pair
        dipole_energy = cp.asnumpy(cp.sum(dipole_energies,0))/2

        wca_energy = cp.asnumpy(cp.sum(wca_energies,0))/2
    else:
        zeeman_energy = 0
        self_energy = 0
        dipole_energy = 0
        wca_energy = 0

    #sum energies, and scale back to SI units as necessary
    element_energy = cp.asnumpy(cp.sum(element_energies,0))
    element_energy *= l_e#multiply by l_e^3, the volume of the undeformed unit cell, and divide by l_e^2 to rescale the constant kappa#np.power(l_e,3)/np.power(l_e,2)

    spring_energy = cp.asnumpy(cp.sum(spring_energies,0))
    spring_energy *= l_e

    total_energy = element_energy + spring_energy + dipole_energy + zeeman_energy + self_energy + wca_energy
    return total_energy, spring_energy, element_energy, dipole_energy, wca_energy, self_energy, zeeman_energy

def plot_outer_surfaces_and_center_cuts(sim_dir,gpu_flag=False):
    """Given the folder containing simulation output, generate figures of the outer surfaces and cuts through the center from each output file."""
    #   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['particle_behavior','cuts','outer_surfaces']
    figure_subtypes = ['center','particle', 'outer_surface']
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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
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

    #how many cluster pairs formed?
    for output_file_count in np.arange(num_output_files):
        cluster_counter = 0
        for i in np.arange(num_particles):
            temp_separations = particle_separations_matrix[output_file_count,i,:]
            cluster_counter += np.count_nonzero(np.less_equal(temp_separations[temp_separations>0],clustering_distance))
            #if we know some particle clustering has ocurred, how can we determine if a single particle is clustering with multiple particles, and cross reference to determine if a chain has formed, and how many particles make up that chain?
        cluster_counter /= 2
        if cluster_counter != 0:
            print(f'for field {np.round(Hext_series[output_file_count]*mu0,decimals=5)} and {sim_type} {boundary_condition_series[np.floor_divide(output_file_count,field_series.shape[0])]} the total number of clusters: {cluster_counter}')

#   in a loop, output files are read in and manipulated
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_nodes(initial_node_posns,final_posns,particles,output_dir+'particle_behavior/',tag=f"{i}")
#       node positions are scaled to SI units using l_e variable for visualization
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
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,l_e,output_dir+'cuts/center/',tag=f"stress_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')

def plot_strain_tensor_field(sim_dir):
    """Given the folder containing simulation output, calculate relevant quantities and generate figures.
    
    For case (3), simulations with particles and applied magnetic fields, for analyzing the particle motion and magnetization, stress and strain tensors, and the effective modulus for an applied field"""
    #   if a directory to save the visualizations doesn't exist, make it
    output_dir = sim_dir+'figures/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    figure_types = ['particle_behavior','strain']
    figure_subtypes = ['center', 'outer_surface']
    for figure_type in figure_types:
        if not (os.path.isdir(output_dir+figure_type+'/')):
          os.mkdir(output_dir+figure_type+'/')
        if figure_type =='strain':
            for figure_subtype in figure_subtypes:
                if not (os.path.isdir(output_dir+figure_type+'/'+figure_subtype+'/')):
                    os.mkdir(output_dir+figure_type+'/'+figure_subtype+'/')
#   user provides directory containing simulation files, including init.h5, output_i.h5 files
#   init.h5 is read in and simulation parameters are extracted
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, field_series, boundary_condition_series, sim_type, dimensions = read_in_simulation_parameters(sim_dir)
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

#   in a loop, output files are read in and manipulated
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_nodes(initial_node_posns,final_posns,particles,output_dir+'particle_behavior/',tag=f"{i}")
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
            tag = surface+'_surface_' + f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'strain')
#       stress and strain tensors are visualized for cuts through the center of the volume
        for cut_type,center_idx in zip(cut_types,center_indices):
            tag = 'center_' + f'boundary_condition_{boundary_conditions[2]}_field_{np.round(mu0*Hext,decimals=4)}_'
            subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'strain')

def plot_mr_effect_figure(directory_file,output_dir):
    """Given a text file containing newline separated directories containing results for the analysis, extract the effective moduli and generate a figure showing the volume fraction dependence of the effective modulus and MR Effect."""
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    directories = []
    vol_fractions = []
    zero_field_effective_modulus = []
    effective_modulus = []
    mr_effect = []
    with open(directory_file,'r') as filehandle:
        for line in filehandle:
            line = line.strip()
            directories.append(line)
    for directory in directories:
        if directory[-1] != '/':
            directory += '/'
        sim_dir = directory
        node_posns, _, _, _, _, particles, parameters, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
        l_e = parameters[7]
        if 'shearing' in sim_type:
            fit_func = shearing_quadratic_fit_func
        else:
            fit_func = quadratic_no_linear_term_fit_func
        dimensions = np.array([np.max(node_posns[:,0])*l_e,np.max(node_posns[:,1])*l_e,np.max(node_posns[:,2])*l_e])
        total_sim_volume = dimensions[0]*dimensions[1]*dimensions[2]
        particle_volume = (4/3)*np.pi*np.power(parameters['particle_radius'],3)
        actual_vol_fraction = particles.shape[0]*particle_volume/total_sim_volume
        vol_fractions.append(actual_vol_fraction)

        total_energy, spring_energy, element_energy, dipole_energy, wca_energy, self_energy, zeeman_energy = get_energies(sim_dir)

        num_output_files = get_num_output_files(sim_dir)
        applied_field = np.zeros((num_output_files,3),dtype=np.float32)
        applied_bc = np.zeros((num_output_files,),dtype=np.float32)
        for i in range(num_output_files):
            _, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
            boundary_conditions = format_boundary_conditions(boundary_conditions)
            applied_field[i] = Hext*mu0
            applied_bc[i] = boundary_conditions[2]

        Bext_magnitude = np.linalg.norm(applied_field,axis=1)
        unique_field_values = np.unique(Bext_magnitude)
        energy_density_plus_wca_plus_self_fit_modulus = np.zeros((unique_field_values.shape[0],))
        energy_density_plus_wca_plus_self_fit_error = np.zeros((unique_field_values.shape[0],))
        subset_modulus_error_dict = dict({})
        for i, unique_value in enumerate(unique_field_values):
            relevant_indices = np.isclose(unique_value,np.linalg.norm(applied_field,axis=1))
            plotting_total_energy = total_energy[relevant_indices]
            # plotting_spring_energy = spring_energy[relevant_indices]
            # plotting_element_energy = element_energy[relevant_indices]
            # plotting_dipole_energy = dipole_energy[relevant_indices]
            # plotting_wca_energy = wca_energy[relevant_indices]
            plotting_self_energy = self_energy[relevant_indices]
            # plotting_zeeman_energy = zeeman_energy[relevant_indices]
            plotting_bc = applied_bc[relevant_indices]

            #check for issues with total energy by observing the trend of the self energy as the strain increases. if the trend changes (goes from increasing to decreasing or vice versa), need to fit to a subset of the data, or not use the dataset at all for effective modulus analysis
            strain_differential_self_energy = np.diff(plotting_self_energy.ravel())
            tmp_var = np.where(strain_differential_self_energy[:-1]*strain_differential_self_energy[1:] < 0)[0]
            if tmp_var.shape[0] == 1 and tmp_var[0] == 0:
                potential_subsets = 1
            else:
                energy_trend_switch_indices = tmp_var[::2] + 2
                potential_subsets = energy_trend_switch_indices.shape[0] + 1
            # energy_trend_switch_indices = np.where(strain_differential_self_energy[:-1]*strain_differential_self_energy[1:] < 0)[0][::2] + 2

            energy_plus_wca_plus_self_density = np.ravel(plotting_total_energy)/total_sim_volume
            modulus_fit_guess = 9e3
            if potential_subsets == 1:
                popt, pcov = scipy.optimize.curve_fit(fit_func,plotting_bc,energy_plus_wca_plus_self_density,p0=np.array([modulus_fit_guess,0]))
                energy_density_plus_wca_plus_self_fit_modulus[i] = popt[0]
                energy_density_plus_wca_plus_self_fit_error[i] = np.sqrt(np.diag(pcov))[0]
            else:
                subset_start_idx = 0
                for subset_count in range(potential_subsets):
                    if subset_count + 1 == potential_subsets:
                        subset_end_idx = plotting_bc.shape[0]
                    else:
                        subset_end_idx = energy_trend_switch_indices[subset_count]
                    #if there are not at least 3 datapoints to fit to, don't bother trying a fit.
                    if subset_end_idx - subset_start_idx < 3:
                        subset_start_idx = subset_end_idx
                        continue
                    popt, pcov = scipy.optimize.curve_fit(fit_func,plotting_bc[subset_start_idx:subset_end_idx],energy_plus_wca_plus_self_density[subset_start_idx:subset_end_idx],p0=np.array([modulus_fit_guess,0]))
                    if subset_count == 0:
                        energy_density_plus_wca_plus_self_fit_modulus[i] = popt[0]
                        energy_density_plus_wca_plus_self_fit_error[i] = np.sqrt(np.diag(pcov))[0]
                    else:
                        subset_modulus_error_dict[f'{i}'] = (popt[0],np.sqrt(np.diag(pcov))[0])
                    subset_start_idx = subset_end_idx
        if '0' in subset_modulus_error_dict:
            zero_field_modulus = subset_modulus_error_dict['0'][0]
        else:
            zero_field_modulus = energy_density_plus_wca_plus_self_fit_modulus[0]
        if '1' in subset_modulus_error_dict:
            nonzero_field_modulus = subset_modulus_error_dict['1'][0]
        else:
            nonzero_field_modulus = energy_density_plus_wca_plus_self_fit_modulus[1]
        zero_field_effective_modulus.append(zero_field_modulus)
        effective_modulus.append(nonzero_field_modulus)
        mr_effect.append((nonzero_field_modulus/zero_field_modulus-1)*100)
        #*** if the fitting is done, now you need to add the values to the appropriate lists. then you need to sort the lists, or convert to np.ndarrays and sort them based on the volume fraction. this is to avoid line drawing issues when making the figure. need to consider the subset fitting, if it is present, do you use it instead of the default fitting value? probably. also, don't forget to save out the dataset used to generate the final figure. probably need to add a mkdir call at the top of this function to have a directory to save out the figure and datasets to. 
    vol_fractions = np.array(vol_fractions)
    zero_field_effective_modulus = np.array(zero_field_effective_modulus)
    effective_modulus = np.array(effective_modulus)
    mr_effect = np.array(mr_effect)
    sorted_indices = np.argsort(vol_fractions)
    vol_fractions = vol_fractions[sorted_indices]
    zero_field_effective_modulus = zero_field_effective_modulus[sorted_indices]
    effective_modulus = effective_modulus[sorted_indices]
    mr_effect = mr_effect[sorted_indices]
    fig, axs = plt.subplots(3,1)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(200)
    savename = output_dir + f'mr_effect.png'
    axs[2].set_xlabel('Vol. Fraction')
    axs[0].set_ylabel(r'$G_{B=0}$ Pa')
    nonzero_field_value = np.max(unique_field_values)
    axs[1].set_ylabel(r'$G_{B!=0}$ Pa')
    axs[2].set_ylabel('MR Effect (%)')
    axs[0].plot(vol_fractions,zero_field_effective_modulus)
    axs[1].plot(vol_fractions,effective_modulus)
    axs[2].plot(vol_fractions,mr_effect)
    format_figure(axs[0])
    format_figure(axs[1])
    format_figure(axs[2])
    plt.savefig(savename)
    plt.close()

if __name__ == "__main__":
    main()    
    results_directory = '/mnt/c/Users/bagaw/Desktop/MRE/two_particle/'
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation
    # sim_dir = results_directory + f"/2024-02-08_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation with "random" particle placement (but only two particles)
    # sim_dir = results_directory + f"/2024-02-09_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used simulation
    # sim_dir = results_directory + f"/2024-02-09_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True_tf_300/"
    # gpu based acceleration calculation and gpu based leapfrog integrator used for hysteresis simulation, with new (partially implemented) batch job driving function and simulation running function
    # sim_dir = results_directory + f"/2024-02-12_2_particle_hysteresis_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # analysis_case1(sim_dir)
    # ran a series of different discretization orders doing the hysteresis simulation. for this set of results the particle placement was not working as expected... so the initial particle separation was not always the 9 microns that was desired. that has since been fixed (as of 2024-02-13)
    # for i in range(6):
    #     sim_dir = results_directory + f"/2024-02-12_2_particle_hysteresis_order_{i}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    #     temp_hysteresis_analysis(sim_dir,gpu_flag=False)

    # sim_dir = results_directory + f"/2024-02-13_2_particle_hysteresis_order_{1}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # sim_dir = results_directory + f"/2024-02-14_2_particle_hysteresis_order_{0}_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_gpu_True/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    # sim_dir_one = results_directory + f"/2024-02-14_2_particle_hysteresis_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # sim_dir_two = results_directory + f"/2024-02-14_2_particle_hysteresis_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # compare_fixed_time_step_solutions_hysteresis(sim_dir_one,sim_dir_two)

    # sim_dir = results_directory + f"/2024-02-14_2_particle_hysteresis_order_4_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    # shearing stress simulation at multiple fields
    # sim_dir = results_directory + f"/2024-02-18_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_5_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # shearing stress simulation at a 2 non-zero fields and 2 non-zero stresses, plus the zero stress and zero field cases
    # sim_dir = results_directory + f"/2024-02-22_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # shearing stress simulation at a 5 non-zero fields and 2 non-zero stresses, plus the zero stress and zero field cases. 3 particle chain
    # sim_dir = results_directory + f"/2024-02-23_3_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"

    # shearing/tension without particles.
    # sim_dir = results_directory + f"/2024-02-26_0_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_None_gpu_True_stepsize_5.e-3/"

    # shearing, two particles, incomplete run, but long run time/simulation time elapsed
    # sim_dir = results_directory + f"/2024-02-26_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    
    #tension, two particles, attractive
    # sim_dir = results_directory + f"/2024-02-27_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = results_directory + f"/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #compression, two particles, attractive, no particle rotations
    # sim_dir = results_directory + f"/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = results_directory + f"/2024-02-28_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('z', 'z')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #compression, two particles, attractive, particle rotations via scipy.transform.Rotation, 3 different time step sizes
    # sim_dir = results_directory + f"/2024-03-04_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    # sim_dir = results_directory + f"/2024-03-04_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_1.25e-3/"

    # sim_dir = results_directory + f"/2024-03-01_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_2.5e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # adjustment to the treatment of particle as rigid body. track particle orientation and relative to particle center vectors poinmting to particle nodes. at the end of an integration round, use the orientation and particle center, as well as the relative to particle center vectors to adjust the particle node positions, to maintain the rigid body shape
    # sim_dir = results_directory + f"/2024-03-12_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # high order of discretization test run to determine run times and test behavior
    # sim_dir = results_directory + f"/2024-03-12_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('x', 'x')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    # sim_dir = results_directory + f"/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_compression_direction('z', 'z')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    #tension case
    # sim_dir = results_directory + f"/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    # sim_dir = results_directory + f"/2024-03-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # sim_dir = results_directory + f"/2024-03-14_2_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_6_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 8 particles, cubic crystal lattice arrangement, 3% volume ratio. Error in particle placement lead to a configuration that was not exactly a cubic lattice, but the results here are interesting, since the particle clusters do not all form at once (at a single field value), but rather occur at different field values, causing multiple "phase transitions". the collective behavior of the particle clustering is important, seemingly the small differences in the value at which clustering occurs plays a role in both the hysteresis and the effective stiffness behavior. this will need to be reran with the fixed particle placement, but also suggests that adding noise to the placement, or intentionally altering separation along different axes, would produce interesting results for comparison.
    sim_dir = results_directory + f"/2024-03-18_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, cubic crystal lattice arrangement, 3% volume ratio. particle placement very broken, but still interesting to see the impact. looks a bit like a periodic boundary structure simulation might look, but handling periodic boundary conditions if there are particles crossing the boundaries is something i haven't considered, and I have not figured out how to handle the magnetic interactions if using periodic boudnary conditions.
    sim_dir = results_directory + f"/2024-03-19_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_4_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #8 particles, cubic lattice, 3% volume ratio, fixed particle placement. only 0 stress applied, just looking at the field dependent particle behavior
    sim_dir = results_directory + f"/2024-03-20_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_2_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #8 particles, cubic lattice, 3% volume ratio, fixed particle placement. multiple stress values applied to analyze effective modulus
    sim_dir = results_directory + f"/2024-03-20_8_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, cubic lattice, 3% volume ratio, fixed particle placement. multiple stress values applied to analyze effective modulus
    sim_dir = results_directory + f"/2024-03-21_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)
    # analysis_average_stress_strain(sim_dir,gpu_flag=True)

    #27 particles, up to +/-2 volume elements as noise added to periodic particle placement, two different RNG seeds
    sim_dir = results_directory + f"/2024-03-29_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_starttime_14-17_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)
    sim_dir = results_directory + f"/2024-03-29_27_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_starttime_23-23_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #27 particles, no noise, hysteresis
    sim_dir = results_directory + f"/2024-04-02_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #2 particles, no noise, hysteresis, no scipy.Rotation or rigid body attempts, instead setting particle-particle node springs to have stiffness based on the particle modulus (actually just 100*polymer modulus because the actual ratio was so large it broke things, as small displacements led to huge accelerations)
    sim_dir = results_directory + f"/2024-04-04_2_particle_hysteresis_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #2 particles, no noise, hysteresis, no scipy.Rotation or rigid body attempts, instead setting particle-particle node springs to have stiffness based on the particle modulus (actually 10000*polymer modulus because the actual ratio was so large it broke things, as small displacements led to huge accelerations). had originally set polymer-particle connections to be the average of the stiffness of the two types, but now set to polymer stiffness. could be adjusted to something like 10* polymer stiffness... maybe. issue was with the acceleration calculations used for checking convergence criteria. higher stiffness meant even small displacements from equilbibrium length lead to outsized accelerations that didn't actually influence particle displacement or polymer displacement... just vibration. polymer-particle connections were also high stiffness, so despite trying to "remove" the particle vibrations from the convergence check, the stiffer polymer around the particles left higher residual accelerations/vibrations
    sim_dir = results_directory + f"/2024-04-04_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #27 particles, noise, hysteresis. particle-particle nodes have stiffer connections, no scipy.Rotation. no fixed nodes
    sim_dir = results_directory + f"/2024-04-04_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_16-41_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #27 particles, no noise, hysteresis. cutoff for WCA set to particle diameter + 100 nm. no fixed nodes. anisotropy_factor = [0.7,1.3,1.3]
    sim_dir = results_directory + f"/2024-04-05_27_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #45 particles [5,3,3], noise, hysteresis. cutoff for WCA set to particle diameter + 100 nm. no fixed nodes. anisotropy_factor = [0.8,1.13ish,1.13ish]
    sim_dir = results_directory + f"/2024-04-05_45_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_15-34_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #45 particles [5,3,3], regular no noise, shear stress xy, was supposed to be only 0.01 stress, but implementation errors with setting stress boundary conditions during initialization led to stress values up to 1.01 in steps of 0.01. stopped early. may end up removing some of the dataset
    sim_dir = results_directory + f"/2024-04-06_45_particle_field_dependent_modulus_stress_simple_stress_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, tension in x direction, field along x. small stress values. is the zero field modulus still higher than the field on, despite intuition suggesting that it should be stiffer with the field on?
    sim_dir = results_directory + f"/2024-04-08_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, tension in x direction, field along x. small stress values. no field
    sim_dir = results_directory + f"/2024-04-09_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_None_gpu_True_stepsize_5.e-3/"

    #2 particle along x, tension in x direction, field along x. smallish stress values. non zero fields field
    sim_dir = results_directory + f"/2024-04-09_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #64 particle (4,4,4), regular, no noise, tension in x direction, field along x. smallish stress values. non zero fields field
    sim_dir = results_directory + f"/2024-04-09_64_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along x, tension along z
    sim_dir = results_directory + f"/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along x
    sim_dir = results_directory + f"/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along y
    sim_dir = results_directory + f"/2024-04-10_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('y', 'y')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along z
    sim_dir = results_directory + f"/2024-04-11_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle along x, field along z, tension along x, strain based simulation. only attempted to implement new strain based calculation of modulus for tension/compression cases, and it still needs to be modified
    sim_dir = results_directory + f"/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    sim_dir = results_directory + f"/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('y', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    sim_dir = results_directory + f"/2024-04-12_2_particle_field_dependent_modulus_strain_strain_tension_direction('z', 'z')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 2 particle stress, field along z, tension along x, double the maximum number of integration rounds
    sim_dir = results_directory + f"/2024-04-13_2_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    # 8 particle (2x2x2), regular. field along z, tension along x
    sim_dir = results_directory + f"/2024-04-14_8_particle_field_dependent_modulus_stress_simple_stress_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_drag_1_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #27 particles (3x3x3), regular, field along x, tension along x
    sim_dir = results_directory + f"/2024-04-14_27_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)
    #125 particles (5x5x5), regular, field along x, tension along x. Results prior to fully testing the new gpu implementations of distributing the magnetic force, setting fixed nodes, and (not relevant to this sim) applying stress to boundary nodes
    sim_dir = results_directory + f"/2024-04-22_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle, field along x, tension along x, still haven't fully tested new implementations of gpu kernels
    sim_dir = results_directory + f"/2024-04-23_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #2 particle, field along x, hystersis, after testing/debugging and fixing new implementations of gpu kernels
    sim_dir = results_directory + f"/2024-04-26_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particle, field along z, hysteresis
    sim_dir = results_directory + f"/2024-04-27_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particle, field along x, hysteresis
    sim_dir = results_directory + f"/2024-04-26_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, noisy placement (+/-1 volume elements off regular possible in each direction), field along x, hysteresis. not guaranteed that all steps reached convergence, but still probably "close enough." Need more refined convergence criteria for these noisy systems, that seem to need more time to settle. that means potentially segregating out the accelerations and velocities of the particles from the rest of the system, and checking the convergence of the rest of the system, then the particle level accelerations and velocities (or changes in position) for convergence testing.
    sim_dir = results_directory + f"/2024-04-27_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_14-40_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, noisy placement (+/-1 volume elements off regular possible in each direction), field along z, hysteresis
    sim_dir = results_directory + f"/2024-04-28_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_starttime_04-27_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #125 particles, regular, field along x, tension along x, strain bc
    sim_dir = results_directory + f"/2024-04-29_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular, field along z, tension along x, strain bc
    sim_dir = results_directory + f"/2024-04-29_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular noisy, field along x, tension along x, strain bc
    sim_dir = results_directory + f"/2024-05-01_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_16-53_stepsize_5.e-3/"
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #2 particles along x, field along x, tension along x, strain bc with boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-03_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_5_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #2 particles along x, field along z, tension along x, strain bc with boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-03_2_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_5_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular, field along x, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-06_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular, field along z, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-07_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular noisy, field along x, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-09_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_16-55_stepsize_5.e-3/"
    # plot_full_sim_surface_forces(sim_dir)
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular noisy, field along z, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    sim_dir = results_directory + f"/2024-05-10_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_starttime_11-33_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, regular noisy, field along x, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    #both smaller and larger strains, for exploring fitting to energy density vs strain for effective modulus vs field analysis
    sim_dir = results_directory + f"/2024-06-13_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_14-38_stepsize_5.e-3/"
    # plot_full_sim_surface_forces(sim_dir)
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particles, 1MPa polymer (closer to natural rubber), regular noisy, field along x, tension along x, strain bc. boundary motion allowed for 0 strain to set reference configurations for non-zero strains
    #both smaller and larger strains, for exploring fitting to energy density vs strain for effective modulus vs field analysis
    sim_dir = results_directory + f"/2024-06-15_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_1000000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_18-05_stepsize_5.e-3/"
    # plot_full_sim_surface_forces(sim_dir)
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particle, regular noisy, 9e3 Pa modulus, field along x, compression along x, strain bc. boudnary motion allowed at 0 strain. fewer small strains, more large strains. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-19_125_particle_field_dependent_modulus_strain_strain_compression_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_starttime_17-56_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=False,gpu_flag=True)

    #125 particle, regular noisy, 1e6 Pa modulus, field along x, compression along x, strain bc. boudnary motion allowed at 0 strain. fewer small strains, more large strains. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-20_125_particle_field_dependent_modulus_strain_strain_compression_direction('x', 'x')_order_3_E_1000000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_starttime_14-49_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)


    #2 particle, 9e3 Pa modulus, field along x, compression along x, strain bc. boudnary motion allowed at 0 strain. .post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-24_2_particle_field_dependent_modulus_strain_strain_compression_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #2 particle, 9e3 Pa modulus, field along x, shearing xy, strain bc. boudnary motion allowed at 0 strain. .post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-24_3_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #125 particle, regular placement, 9e3 Pa modulus, field along x, shearing xy, strain bc. boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-24_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #125 particle, regular noisy placement, 9e3 Pa modulus, field along x, shearing xy, strain bc. boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-25_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_starttime_20-48_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #125 particle, anisotropic regular placement [0.7,~,~], 9e3 Pa modulus, field along x, shearing xy, strain bc. boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + f"/2024-06-27_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)

    #125 particle, anisotropic regular placement [0.7,~,~], 9e3 Pa modulus, field along z, shearing xy, strain bc. boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + "/2024-06-28_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # analysis_case3(sim_dir,stress_strain_flag=True,gpu_flag=True)
    # plot_outer_surfaces_and_center_cuts(sim_dir,gpu_flag=True)

    #125 particle, anisotropic regular placement [0.7,~,~], 9e3 Pa modulus, field along x, hysteresis, bottom fixed bc. post-imlpementation of gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + "/2024-07-01_125_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir,gpu_flag=True)

    #20 particle (5x2x2), anisotropic regular noisy placement [0.8,~,~], 9e3 Pa modulus, field along x, shearing xy, strain bc. boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + "/2024-07-10_20_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_5_E_9000.0_nu_0.47_Bext_angle_0.0_regular_anisotropic_noisy_vol_frac_0.1_starttime_12-19_stepsize_1.e-3/"
    # plot_energy_figures(sim_dir)

    #20 particle (5x2x2), anisotropic regular noisy placement [0.8,~,~], 9e3 Pa modulus, field along x, shearing xy, strain bc, 0 field only (actually zero, not just near zero). boundary motion allowed at 0 strain. post-imlpementation of cupy built-ins for strained boundary net force calculation and gpu based particle position finding using positions of nodes making up central voxel of each particle
    sim_dir = results_directory + "/2024-07-10_20_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_5_E_9000.0_nu_0.47_Bext_angle_90_regular_anisotropic_noisy_vol_frac_0.1_starttime_16-28_stepsize_1.e-3/"
    # plot_energy_figures(sim_dir)
    # plot_strain_tensor_field(sim_dir)

    # 2 particle hysteresis, field along z, particles along z. first sim in a while. going to try and write a short report on this set of results
    sim_dir = results_directory + "/2024-08-21_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_1.e-3/"
    # temp_hysteresis_analysis(sim_dir)

    sim_dir = results_directory + "/2024-08-23_2_particle_hysteresis_order_3_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # temp_hysteresis_analysis(sim_dir)

    sim_dir = results_directory + "/2024-05-10_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_starttime_11-33_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "/2024-06-28_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "/2024-06-13_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_14-38_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "/2024-06-24_2_particle_field_dependent_modulus_strain_strain_compression_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "/2024-06-24_3_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "/2024-06-25_125_particle_field_dependent_modulus_strain_strain_shearing_direction('x', 'y')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_profiling_starttime_20-48_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "2024-09-03_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # analysis_case3(sim_dir)
    # plot_energy_figures(sim_dir)
    
    sim_dir = results_directory + "2024-09-03_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_90000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-03_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_900000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-04_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-06_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-06_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "2024-09-09_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_5_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "2024-09-09_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.06_stepsize_5.e-3/"
    # analysis_case3(sim_dir)
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-09_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angle_90_regular_vol_frac_0.06_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    # #fields perpendicular to particle axis, along shearing direction
    # sim_dir = results_directory + "2024-09-14_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angles_90.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-14_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angles_90.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # #fields perp to particle axis, perp to shearing direction
    # sim_dir = results_directory + "2024-09-16_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angles_90.0_90.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-16_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angles_90.0_90.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # #8 particle isotropic shearing
    # sim_dir = results_directory + "2024-09-16_8_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angles_90.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-17_8_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-17_8_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-17_8_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_18000.0_nu_0.47_Bext_angles_90.0_0.0_regular_vol_frac_0.03_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    # 0 particle strain sims with different dimensions, cubic shape first
    sim_dir = results_directory + "2024-09-17_0_particle_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    sim_dir = results_directory + "2024-09-17_0_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_3_E_9000.0_nu_0.47_[20 10 10]_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    # volume fraction variation, for generating a plot of MR effect versus volume fraction. two particles, shearing, field along particle axis
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.02_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.04000000000000001_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.06000000000000001_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.08000000000000002_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.10000000000000002_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.12000000000000002_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.14_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.16_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-20_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.18000000000000002_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    # sim_dir = results_directory + "2024-09-21_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.2_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)

    sim_dir = results_directory + "2024-09-24_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.005_stepsize_5.e-3/"
    sim_dir = results_directory + "2024-09-24_2_particle_field_dependent_modulus_strain_shearing_direction('z', 'x')_order_7_E_9000.0_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_0.01_stepsize_5.e-3/"
    # plot_energy_figures(sim_dir)
    directory_file = '/mnt/c/Users/bagaw/Desktop/MRE/mr_effect_volfrac.txt'
    output_dir = '/mnt/c/Users/bagaw/Desktop/MRE/MR_effect/'
    plot_mr_effect_figure(directory_file,output_dir)

    print('Exiting')