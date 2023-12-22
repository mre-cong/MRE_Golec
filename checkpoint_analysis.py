#2023-11-06
#David Marchfield
#using persistent checkpointing (keeping checkpoints at the end of each integration run), read in the checkpoint files to calculate analogous simulation criteria as done in the SimCriteria class. Visualize the solution vectors. Implement reading in and restarting an integration/simulation from the most recent checkpoint.
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')
plt.switch_backend('TkAgg')
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

def checkpoints_analysis(sim_dir):
    """Read in the simulation files from the directory, and then read in the checkpoint files to perform an analysis on the simulation criteria and/or the effective stress and effective modulus of the material in order to study how the values converge (or not) as the simulation progresses"""
    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(sim_dir+'init.h5')
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir()]
    for subfolder in subfolders:
        with os.scandir(subfolder+'/') as dirIterator:
            checkpoint_files = [f.path for f in dirIterator if f.is_file() and f.name.startswith('checkpoint')]
        #sort the checkpoint files so we analyze them in order of progressing through the simulation/integration
        if len(checkpoint_files) != 0:
            checkpoint_number = np.empty((0,),dtype=np.int64)
            for checkpoint_file in checkpoint_files:
                fn_w_type = checkpoint_file.split('/')[-1]
                fn = fn_w_type.split('.')[0]
                checkpoint_number = np.append(checkpoint_number,np.array([int(count) for count in re.findall(r'\d+',fn)]))
            sort_indices = np.argsort(checkpoint_number)
            sorted_checkpoint_files = []
            for i in range(len(sort_indices)):
                sorted_checkpoint_files.append(checkpoint_files[sort_indices[i]])
            get_checkpoints_tension_compression_modulus_stress(sim_dir,sorted_checkpoint_files)

def subplot_checkpoints_stress_strain_modulus(stress,checkpoints,strain_direction,effective_modulus,output_dir,tag=""):
    """Generate figure with subplots of stress-strain curve and effective modulus versus strain."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    force_component = {'x':0,'y':1,'z':2}
    axs[0].plot(checkpoints,np.abs(stress[:,force_component[strain_direction[1]]]),'-o')
    # axs[0].set_title('Stress vs Checkpoint Number')
    axs[0].set_ylabel('Stress')
    axs[0].set_xlabel('Checkpoint Number')
    axs[1].plot(checkpoints,effective_modulus,'-o')
    # axs[1].set_title('Effective Modulus vs Checkpoint Number')
    axs[1].set_xlabel('Checkpoint Number')
    axs[1].set_ylabel('Effective Modulus')
    plt.show()
    fig.tight_layout()
    # savename = output_dir + f'subplots_checkpoints_stress-strain_effective_modulus_'+tag+'.png'
    # plt.savefig(savename)
    plt.close()

def get_checkpoints_tension_compression_modulus_stress(sim_dir,checkpoint_files):
    effective_modulus = np.zeros((len(checkpoint_files),))
    effective_stress = np.zeros((len(checkpoint_files),3))
    num_checkpoints = len(checkpoint_files)
    checkpoints = np.arange(0,num_checkpoints,1)
    for i, checkpoint_file in enumerate(checkpoint_files):
        effective_modulus[i], effective_stress[i], strain, strain_direction = get_checkpoint_tension_compression_modulus_stress(sim_dir,checkpoint_file)
    subplot_checkpoints_stress_strain_modulus(effective_stress,checkpoints,strain_direction,effective_modulus,output_dir="",tag="")

def get_checkpoint_tension_compression_modulus_stress(sim_dir,checkpoint_file):
    """Read in the checkpoint file and calculate the effective stress on the "probe" surface and, if applicable, the effective modulus"""
    solution, applied_field, boundary_conditions, i = mre.initialize.read_checkpoint_file(checkpoint_file)
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    strain = boundary_conditions[2]
    strain_direction = boundary_conditions[1]
    Hext = applied_field
    force_component = {'x':0,'y':1,'z':2}
    y = solution
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
    first_bdry_stress = np.sum(first_bdry_forces,axis=0)/(dimensions[dimension_indices[0]]*dimensions[dimension_indices[1]])
    # print(f'Difference in stress from opposite surfaces is {np.abs(first_bdry_stress[force_component[strain_direction[1]]])-np.abs(second_bdry_stress[force_component[strain_direction[1]]])}')
    stress = first_bdry_stress
    if strain == 0:# and np.isclose(np.linalg.norm(stress[i,:]),0):
        effective_modulus = E
    else:
        effective_modulus = np.abs(stress[force_component[strain_direction[1]]]/strain)
    return effective_modulus, stress, strain, strain_direction

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

def main():
    """Read in and perform analysis and visualization workflow on the simulation checkpoint files."""
    #First, read in the init file and get the necessary variables from the params "struct"
    #Second, inside a loop over the strains/fields, calculate the simulation criteria and visualize the evolution of the system
    drag = 10
    discretization_order = 3

    sim_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-05_2particle_larger_WCA_cutoff_order_{discretization_order}_drag_{drag}/'
    # extend_from_checkpoints(sim_dir,max_integrations=20,max_integration_steps=2000)

    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(sim_dir+'init.h5')
    with os.scandir(sim_dir) as dirIterator:
        subfolders = [f.path for f in dirIterator if f.is_dir()]
    for i in range(len(subfolders)):
        final_posns, Hext, boundary_conditions, _ = mre.initialize.read_output_file(sim_dir+f'output_{i}.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        mre.analyze.plot_particle_centric_cuts_surf(initial_node_posns,final_posns,particles,boundary_conditions,sim_dir,tag="")
        # mre.analyze.plot_particle_centric_cuts_wireframe(initial_node_posns,final_posns,particles,boundary_conditions,sim_dir,tag="")
        # mre.analyze.plot_outer_surfaces(initial_node_posns,final_posns,boundary_conditions,sim_dir,tag=f'field_{i}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}')
        # mre.analyze.plot_tiled_outer_surfaces_contours(initial_node_posns,final_posns,sim_dir,tag=f'field_{i}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}')
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

    with os.scandir(sim_dir) as dirIterator:
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
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    determine_current_field_and_boundary_conditions()
    pickup_looping_from_current_point()
    simulate_scaled()
    print_relevant_information_on_status_and_on_completion()

def extend_from_checkpoints(sim_dir,max_integrations,max_integration_steps):
    """Extend a simulation from the most recent checkpoint of some simulation or simulation step (applied strain or field)"""
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
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
    # TODO: What if I directly provided the checkpoint file to start from? That is the correct way to approach this, as a function structured that way will be reusable for other cases, including needing to go through each subfolder in the simulation directory, or some subset of them    
    # determine_current_field_and_boundary_conditions()
    # pickup_looping_from_current_point()
    # simulate_scaled()
    # print_relevant_information_on_status_and_on_completion()

def extend_from_checkpoint(checkpoint_file,max_integrations,max_integration_steps,tolerance,subfolder_flag=True):
    """Extend a simulation from a checkpoint file"""
    tmp_var = checkpoint_file.split('/')
    if subfolder_flag:
        if tmp_var[-1] == '':
            sim_dir = checkpoint_file[:-1*(len(tmp_var[-2])+len(tmp_var[-3]))-1]
        elif tmp_var[-1] != '':
            sim_dir = checkpoint_file[:-1*(len(tmp_var[-1])+len(tmp_var[-2]))-1]
        if tmp_var[-1] == '':
            output_dir = checkpoint_file[:-1*(len(tmp_var[-2]))-1]
        elif tmp_var[-1] != '':
            output_dir = checkpoint_file[:-1*(len(tmp_var[-1]))-1]
    else:
        if tmp_var[-1] == '':
            sim_dir = checkpoint_file[:-1*(len(tmp_var[-2]))-1]
        elif tmp_var[-1] != '':
            sim_dir = checkpoint_file[:-1*(len(tmp_var[-1]))-1]
        output_dir = sim_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions = read_in_simulation_parameters(sim_dir)
    solution, applied_field, boundary_conditions, i = mre.initialize.read_checkpoint_file(checkpoint_file)
    boundary_conditions = format_boundary_conditions(boundary_conditions)
    t_f = 30
    start = time.time()
    sol, return_status = simulate.extend_simulate_scaled(solution,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,applied_field,particle_radius,particle_mass,chi,Ms,drag,initial_node_posns,output_dir,i,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True)
    end = time.time()
    delta = end - start
    x0 = np.reshape(sol[:initial_node_posns.shape[0]*initial_node_posns.shape[1]],initial_node_posns.shape)
    print('took %.2f seconds to simulate' % delta)
    # TODO: get the output file numbering from the series variable and the applied field (or boundary conditions, depending on the series_descriptor). couild overwrite the old outut file if it exists, by trying to read it in and adding the time to simulate from that output file to the time to simulate the extension/continuation
    mre.initialize.write_output_file(str(i)+'extended',x0,applied_field,boundary_conditions,np.array([delta]),output_dir)

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
    total_num_nodes = int(num_nodes[0]*num_nodes[1]*num_nodes[2])
    k = mre.initialize.get_spring_constants(E, l_e)
    k = np.array(k)

    return initial_node_posns, beta_i, springs_var, elements, boundaries, particles, num_nodes, total_num_nodes, E, nu, k, kappa, beta, l_e, particle_mass, particle_radius, Ms, chi, drag, characteristic_time, series, series_descriptor, dimensions
if __name__ == "__main__":
    # sim_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-05_2particle_larger_WCA_cutoff_order_3_drag_10/'
    # extend_from_checkpoint(sim_dir+'field_2_Bext_0.15/checkpoint19.h5',max_integrations=20,max_integration_steps=2500,tolerance=1e-4)
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-21_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0. 0. 0.]/"
    checkpoints_analysis(sim_dir)
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-21_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0.05 0.   0.  ]/"
    checkpoints_analysis(sim_dir)
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-23_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0.1 0.  0. ]/"
    checkpoints_analysis(sim_dir)
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-24_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0.15 0.   0.  ]/"
    checkpoints_analysis(sim_dir)
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-11-25_field_dependent_modulus_strain_tension_direction('x', 'x')_order_2_drag_20_Bext_[0.2 0.  0. ]/"
    checkpoints_analysis(sim_dir)
    main()