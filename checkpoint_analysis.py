#2023-11-06
#David Marchfield
#using persistent checkpointing (keeping checkpoints at the end of each integration run), read in the checkpoint files to calculate analogous simulation criteria as done in the SimCriteria class. Visualize the solution vectors. Implement reading in and restarting an integration/simulation from the most recent checkpoint.
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import get_volume_correction_force_cy_nogil
import simulate
import re
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

def main():
    """Read in and perform analysis and visualization workflow on the simulation checkpoint files."""
    #First, read in the init file and get the necessary variables from the params "struct"
    #Second, inside a loop over the strains/fields, calculate the simulation criteria and visualize the evolution of the system
    drag = 10
    discretization_order = 3

    sim_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-05_2particle_larger_WCA_cutoff_order_{discretization_order}_drag_{drag}/'

    extend_from_checkpoint(sim_dir,max_integrations=20,max_integration_steps=2000)

    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(sim_dir+'init.h5')

    # print(f'{params.dtype}')

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
    N_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])

    with os.scandir(output_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir()]
    for subfolder in subfolders:
        with os.scandir(subfolder+'/') as dirIterator:
            checkpoint_files = [f.path for f in dirIterator if f.is_file() and f.name.startswith('checkpoint')]
        for checkpoint_file in checkpoint_files:
            solution, applied_field, boundary_conditions, i = mre.initialize.read_checkpoint_file(checkpoint_file)
            boundary_conditions = (str(boundary_conditions[0][0]),(str(boundary_conditions[0][1]),str(boundary_conditions[0][2])),boundary_conditions[0][3])
            current_posns = np.reshape(solution[:3*N_nodes],(N_nodes,3))
            if i == 0:
                tag = '1st_configuration'
            elif i == 1:
                tag = '2nd_configuration'
            elif i == 2:
                tag = '3rd_configuration'
            else:
                tag = f'{i+1}th_configuration'
            # overlayed wirecuts are a bad idea, it is too busy, and i can't see anything useful. it is possible that htere are ways, like downsampling appropriately, that would make it a useful visualization, but i'm not convinced right now that it is the correct way to move forward
            # mre.analyze.plot_overlayed_center_cuts_wireframe(initial_node_posns,current_posns,particles,boundary_conditions,subfolder+'/',tag)
            mre.analyze.plot_center_cuts_surf(initial_node_posns,current_posns,particles,boundary_conditions,subfolder+'/',tag)
            mre.analyze.plot_center_cuts_wireframe(initial_node_posns,current_posns,particles,boundary_conditions,subfolder+'/',tag)
            mre.analyze.plot_center_cuts_contour(initial_node_posns,current_posns,particles,boundary_conditions,subfolder+'/',tag)
            # The original center cut plotting using scatter and only plotting edge springs was taking a significant amount of time to run, for reasons i am unsure of. maybe the selectivity of spring plotting, or maybe the way i am trying to get the nodes and node coordinates for particular cuts. the use of tricontourf was an attempt to get useful visualizations without waiting hours for basic visualizations.
            # mre.analyze.plot_center_cuts(initial_node_posns,current_posns,springs_var,particles,boundary_conditions,subfolder+'/',tag)

def continue_from_checkpoint(sim_dir,max_integrations,max_integration_steps):
    """Continue from the most recent checkpoint of some interrupted simulation, with the possibility to extend a simulation past the originally intended number of allowed integrations"""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor = read_in_simulation_parameters(sim_dir)
    determine_current_field_and_boundary_conditions()
    pickup_looping_from_current_point()
    simulate_scaled()
    print_relevant_information_on_status_and_on_completion()

def extend_from_checkpoint(sim_dir,max_integrations,max_integration_steps):
    """Extend a simulation from the most recent checkpoint of some simulation or simulation step (applied strain or field)"""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor = read_in_simulation_parameters(sim_dir)
    checkpoint_number = np.empty((0,),dtype=np.int64)
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
        print(boundary_conditions)
    # determine_current_field_and_boundary_conditions()
    # pickup_looping_from_current_point()
    # simulate_scaled()
    # print_relevant_information_on_status_and_on_completion()
    
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

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor

if __name__ == "__main__":
    main()