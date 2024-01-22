#2023-11-29
#David Marchfield
#Establishing the distinct workflows for different types of simulations and analyses via pseudocode. Followed by implementation of necessary component functions and visualizations.

import numpy as np
import scipy.special as sci
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
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
        mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{series[i]}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{series[i]}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{series[i]}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')
#       visualizations of cuts through the particle centers and edges are generated and saved out
        if particles.shape[0] != 0:
            mre.analyze.plot_particle_centric_cuts_wireframe(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/particle/',tag=f"series_{i}")
            mre.analyze.plot_particle_centric_cuts_surf(initial_node_posns,final_posns,particles,output_dir+'cuts/particle/',tag=f"series_{i}")
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

def analysis_case3(sim_dir,stress_strain_flag=True):
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
    #TODO effective modulus calculations for field dependent simulations
    effective_modulus, stress, strain, Bext_series, strain_direction = get_field_dependent_effective_modulus(sim_dir)
#   outside the loop:
#   strains are gotten from init.h5 and/or the boundary_conditions variable in every output_i.h5 file in the loop
#   figure with 2 subplots showing the stress-strain curve of the simulated volume and the effective modulus as a function of strain is generated and saved out
    subplot_stress_field_modulus(stress,strain,strain_direction,effective_modulus,Bext_series,output_dir+'modulus/',tag="")
    # subplot_stress_field_modulus(alternative_stress_measure,strain,strain_direction,alternative_effective_modulus,Bext_series,output_dir+'modulus/',tag="alternative_measures")
    #get the the applied field associated with each output file
    Hext_series = get_applied_field_series(sim_dir)
    num_output_files = get_num_output_files(sim_dir)
    #get the particle separations and overall magnetizations
    num_particles = particles.shape[0]
    num_separations = int(sci.binom(num_particles,2))
    separations = np.zeros((num_output_files,num_separations))
    magnetization = np.zeros((num_output_files,3))
    for i in range(num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        separations[i,:] = get_particle_separation(final_posns,particles)
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

#   in a loop, output files are read in and manipulated
    for i in range(num_output_files):#range(6,num_output_files):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        #TODO make into a function the plotting of the forces acting on the probed surface necessary to keep it held fixed in place
        #plotting the force components acting on the probed surface
        Lx = initial_node_posns[:,0].max()
        Ly = initial_node_posns[:,1].max()
        Lz = initial_node_posns[:,2].max()
        layers = (Lx,Ly,Lz)
        boundary_forces, all_forces = get_probe_boundary_forces(sim_dir,i,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag)
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
#       node positions are scaled to SI units using l_e variable for visualization
        si_final_posns = final_posns*l_e
#       visualizations of the outer surface as contour plots in a tiled layout are generated and saved out
#TODO Issue with using contours for abritrary simulations. if the surfaces don't have contours, that is, differences in the "depth" from point to point, then there are no contour levels that can be defined, and the thing fails. i can use a try/except clause, but that may be bad style/practice. I'm not sure of the right way to handle this. I suppose if it is shearing or torsion I should expect that this may not be a useful figure to generate anyway, so i could use the boundary_conditions variable first element
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if ((boundary_conditions[0] == "tension" or boundary_conditions[0] == "compression" or boundary_conditions[0] == "free") and boundary_conditions[2] != 0) or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_tiled_outer_surfaces_contours_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{boundary_conditions[2]}_field_{mu0*Hext}")
            except:
                print('contour plotting of outer surfaces failed due to lack of variation (no contour levels could be generated)')
#       visualizations of the outer surface as a 3D plot using surfaces are generated and saved out
        mre.analyze.plot_outer_surfaces_si(initial_node_posns,final_posns,l_e,output_dir+'outer_surfaces/',tag=f"strain_{boundary_conditions[2]}_field_{mu0*Hext}")
#       visualizations of cuts through the center of the volume are generated and saved out
        mre.analyze.plot_center_cuts_surf_si(initial_node_posns,final_posns,l_e,particles,output_dir+'cuts/center/',plot_3D_flag=True,tag=f"3D_strain_{boundary_conditions[2]}_field_{mu0*Hext}")
        mre.analyze.plot_center_cuts_wireframe(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{boundary_conditions[2]}_field_{mu0*Hext}")
        #If there is a situation in which some depth variation could be occurring (so that contour levels could be created), try to make a contour plot. potential situations include, applied tension or compression strains with non-zero values, and the presence of an external magnetic field and magnetic particles
        if (boundary_conditions[2] != 0 and boundary_conditions[0] != "free" and boundary_conditions[0] != "shearing" and boundary_conditions[0] != "torsion") or (np.linalg.norm(Hext) != 0 and particles.shape[0] != 0):
            try:
                mre.analyze.plot_center_cuts_contour(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/center/',tag=f"strain_{boundary_conditions[2]}_field_{mu0*Hext}")
            except:
                print('contour plotting of volume center cuts failed due to lack of variation (no contour levels could be generated)')
#       visualizations of cuts through the particle centers and edges are generated and saved out
        if particles.shape[0] != 0:
            mre.analyze.plot_particle_centric_cuts_wireframe(initial_node_posns,final_posns,particles,boundary_conditions,output_dir+'cuts/particle/',tag=f"series_{i}")
            mre.analyze.plot_particle_centric_cuts_surf(initial_node_posns,final_posns,particles,output_dir+'cuts/particle/',tag=f"series_{i}")
        if stress_strain_flag:
    #       node positions are used to calculate nodal displacement
            displacement_field = get_displacement_field(initial_node_posns,final_posns)
    #       nodal displacement is used to calculated displacement gradient
            gradu = get_gradu(displacement_field,num_nodes)
    #       displacement gradient is used to calculate linear and nonlinear strain tensors
            strain_tensor = get_strain_tensor(gradu)
            green_strain_tensor = get_green_strain_tensor(gradu)
    #       linear strain tensor and Lame parameters are used to calculate the linear stress tensor
            stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)
    #       use the components of linear stress and strain tensors to get something like an effective modulus on a per node basis
            effective_modulus_tensor = stress_tensor/strain_tensor
            np.nan_to_num(effective_modulus_tensor,copy=False,nan=0.0,posinf=-1,neginf=+1)
    #       stress and strain tensors are visualized for the outer surfaces
            for surf_idx,surface in zip(surf_indices,surf_type):
                if surface == 'left' or surface == 'right':
                    cut_type = 'yz'
                elif surface == 'front' or surface == 'back':
                    cut_type = 'xz'
                elif surface == 'top' or surface == 'bottom':
                    cut_type = 'xy'
                tag = surface+'_surface_strain_' + f'strain_{boundary_conditions[2]}_field_{mu0*Hext}_'
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'strain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,surf_idx,output_dir+'strain/outer_surface/',tag=tag+'nonlinearstrain')
                # subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,surf_idx,output_dir+'stress/outer_surface/',tag=tag+'stress')
                if surface == 'right':
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,effective_modulus_tensor,surf_idx,output_dir+'modulus/',tag=tag+'quasi-modulus')
    #       stress and strain tensors are visualized for cuts through the center of the volume
            for cut_type,center_idx in zip(cut_types,center_indices):
                tag = 'center_'  f'strain_{boundary_conditions[2]}_field_{mu0*Hext}_'
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'strain')
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,center_idx,output_dir+'strain/center/',tag=tag+'nonlinearstrain')
                subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,center_idx,output_dir+'stress/center/',tag=tag+'stress')
    #       if particles present:
    #          stress and strain tensors are visualized for cuts through particle centers and edges if particles present
            if particles.shape[0] != 0:
                centers = np.zeros((particles.shape[0],3))
                for i, particle in enumerate(particles):
                    tag=f"particle{i+1}_edge_" + f'strain_{boundary_conditions[2]}_field_{mu0*Hext}_'
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
                tag='particle_centers_'+ f'strain_{boundary_conditions[2]}_field_{mu0*Hext}_'
                layers = (int((centers[0,2]+centers[1,2])/2),int((centers[0,1]+centers[1,1])/2),int((centers[0,0]+centers[1,0])/2))
                for cut_type,layer in zip(cut_types,layers):
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'strain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,green_strain_tensor,layer,output_dir+'strain/particle/',tag=tag+'nonlinearstrain')
                    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,layer,output_dir+'stress/particle/',tag=tag+'stress')
#   outside the loop:
#   table or csv file with stress, strain, and effective modulus values are saved out for potential reconstruction or modification of figures

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

def get_field_dependent_effective_modulus(sim_dir):
    """Given a simulation directory, calculate and plot the effective modulus versus applied field. The returned stress variable is the effective stress applied to the surface equivalent to the "probe" in an indentation experiment. The secondary_stress variable is the effective stress applied to the surface held fixed by a table/wall/platform in an experimental measurement. The effective modulus is calculated using the effective stress applied to the "probed" surface."""
    _, _, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_0.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain_type = boundary_conditions[0]
    strain_direction = boundary_conditions[1]
    if strain_type == 'tension' or strain_type == 'compression':
        effective_modulus, stress, strain, Bext_series = get_field_dependent_tension_compression_modulus(sim_dir,strain_direction)
    elif strain_type == 'shearing':
        effective_modulus, stress, strain, Bext_series, alternative_stress_measure, alternative_effective_modulus = get_field_dependent_shearing_modulus(sim_dir,strain_direction)
    elif strain_type == 'torsion':
        get_torsion_modulus(sim_dir,strain_direction)
    return effective_modulus, stress, strain, Bext_series, strain_direction#, alternative_stress_measure, alternative_effective_modulus

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

def get_field_dependent_tension_compression_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of applied field values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    Hext_series = get_applied_field_series(sim_dir)
    Bext_series = mu0*Hext_series
    n_series_steps = np.shape(Bext_series)[0]
    stress = np.zeros((n_series_steps,3))
    strain = np.zeros((n_series_steps,))
    effective_modulus = np.zeros((n_series_steps,))
    for i in range(n_series_steps):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        effective_modulus[i], stress[i], strain[i], _ = get_tension_compression_modulus_v2(sim_dir,i,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions)
    return effective_modulus, stress, strain, Bext_series

def get_tension_compression_modulus_v2(sim_dir,output_file_number,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions):
    """For a given configuration of nodes and particles, calculate the effective modulus, effective stress on the probe surface, and return those values."""
    force_component = {'x':0,'y':1,'z':2}
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    Hext = applied_field
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain = boundary_conditions[2]
    y = np.zeros((6*total_num_nodes,))
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
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
    stress = first_bdry_stress
    secondary_stress = second_bdry_stress
    if strain == 0:# and np.isclose(np.linalg.norm(stress[i,:]),0):
        effective_modulus = E
    else:
        effective_modulus = np.abs(stress[force_component[strain_direction[1]]]/strain)
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
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
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

def get_field_dependent_shearing_modulus(sim_dir,strain_direction):
    """Calculate a tension/compression modulus (Young's modulus), considering the stress on both surfaces that would be necessary to achieve the strain applied for a series of applied field values."""
    _, beta_i, springs_var, elements, boundaries, particles, _, total_num_nodes, E, _, _, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, _, series, _, dimensions = read_in_simulation_parameters(sim_dir)
    Hext_series = get_applied_field_series(sim_dir)
    Bext_series = mu0*Hext_series
    n_series_steps = np.shape(Bext_series)[0]
    stress = np.zeros((n_series_steps,3))
    alternative_stress_measure = np.zeros((n_series_steps,3))
    strain = np.zeros((n_series_steps,))
    effective_modulus = np.zeros((n_series_steps,))
    if strain_direction[0] == 'x':
        relevant_boundary = 'right'
        dimension_indices = (1,2)
    elif strain_direction[0] == 'y':
        relevant_boundary = 'back'
        dimension_indices = (0,2)
    elif strain_direction[0] == 'z':
        relevant_boundary = 'top'
        dimension_indices = (0,1)
    bdry_forces = np.zeros((n_series_steps,len(boundaries[relevant_boundary]),3))
    alternative_effective_modulus = np.zeros((n_series_steps,))
    for i in range(n_series_steps):# should be for i in range(len(series)):, but i had incorrectly saved out the strain series magnitudes and instead saved a field series
        effective_modulus[i], stress[i], strain[i], _ = get_shearing_modulus_v2(sim_dir,i,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions)
        # effective_modulus[i], stress[i], strain[i], _, bdry_forces[i] = get_shearing_modulus_v2(sim_dir,i,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions)
    # below was part of an attempt to subtract out contributions to the forces on the boundary surface due to the particle motion, but the results demonstrate how poor of an idea that was. i had thought it might show more reasonable effective modulus values, i was wrong.
    # num_sims_zero_strain = int(n_series_steps/2)
    # force_component = {'x':0,'y':1,'z':2}
    # for i in range(num_sims_zero_strain):
    #     adjusted_boundary_forces = bdry_forces[i + num_sims_zero_strain] - bdry_forces[i]
    #     alternative_stress_measure[i+num_sims_zero_strain] = np.sum(adjusted_boundary_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
    #     alternative_effective_modulus[i+num_sims_zero_strain] = np.abs(alternative_stress_measure[i+num_sims_zero_strain,force_component[strain_direction[1]]]/strain[i+num_sims_zero_strain])
    return effective_modulus, stress, strain, Bext_series#, alternative_stress_measure, alternative_effective_modulus

def get_shearing_modulus_v2(sim_dir,output_file_number,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag,dimensions):
    #TODO finish this function. shearing in different directions from the same surface is a different modulus (there is anisotropy, or should assume there is). use the direction of the shearing for title, labels, and save name for the figures generated (though that may not occur in this function)
    """Calculate a shear modulus, using the shear strain (shearing angle) and the force applied to the sheared surface in the shearing direction to get a shear stress."""
    #nonlinear shear strains are defined as tangent of the angle opened up by the shearing. linear shear strain is the linear, small angle approximation of tan theta ~= theta
    force_component = {'x':0,'y':1,'z':2}
    y = np.zeros((6*total_num_nodes,))
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain = np.tan(boundary_conditions[2])
    Hext = applied_field
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
    if strain_direction[0] == 'x':
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
    stress = first_bdry_stress
    secondary_stress = second_bdry_stress
    if strain == 0:# and np.isclose(np.linalg.norm(stress),0):
        effective_modulus = E/3
    else:
        effective_modulus = np.abs(stress[force_component[strain_direction[1]]]/strain)
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
        end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
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

def get_probe_boundary_forces(sim_dir,output_file_number,strain_direction,beta_i,springs_var,elements,boundaries,particles,total_num_nodes,E,kappa,beta,l_e,particle_mass,particle_radius,Ms,chi,drag):
    """For a given configuration of nodes and particles, calculate the forces on the probe surface, and return those values."""
    final_posns, applied_field, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{output_file_number}.h5')
    Hext = applied_field
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    y = np.zeros((6*total_num_nodes,))
    y[:3*total_num_nodes] = np.reshape(final_posns,(3*total_num_nodes,))
    end_accel, _ = simulate.get_accel_scaled_no_fixed_nodes(y,elements,springs_var,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag)
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
        ax.axis('equal')
        ax.set_title(f' {component_dict[i]} ')
        # ax.set_title(tag+f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
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
    axs[0].plot(Bext_magnitude,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
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
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-10_field_dependent_modulus_strain_shearing_direction('x', 'y')_order_1_E_30000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_0_Bext_angle_0.0_particle_rotations_nodal_WCA_off/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-09_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.25_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-01-12_field_dependent_modulus_strain_tension_direction('x', 'x')_order_1_E_9000.0_nu_0.47_Bext_angle_0.0_particle_rotations_nodal_WCA_off/"
    analysis_case3(sim_dir,stress_strain_flag=True)
    # analysis_case1(sim_dir)