# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:19:19 2022

@author: bagaw
"""

#2023-02-14: I am trying to create versions of the Golec method which utilize cythonized functions to compare performance to a pure python, cython plus GPU via cuPy, and GPU Version via cuPy implementations
# along with cythonizing functions (which will happen instages, since profiling is separate from benchmarking... i'll need to do both)
# I will be altering the original implementation logic in places (such as the enforcement of boundary conditions, their instantiation, and the types of boundary conditions that can be handled)

#I would like to implement the 3d hybrid mass spring system from Golec et al 2020 paper, in the simplest case, a single cubic unit cell

#pseudo code:
    #determine the vertices of the unit cell(s) based on the volume/dimensions of the system of interest and the level ofdiscretization desired
    #calculate a connectivity matrix that represents the presence of a spring (whether linear or non-linear) connecting particle i and particle j with a non-zero value (the stiffness constant) in row i column j if particle i and j are connected by a spring. this is a symmetric matrix, size N x N where N is the number of vertices, with many zero entries
    #calculate the magnitude of the separation vector among particles connected by springs, and create a matrix of the same shape as the connectivity matrix, where the entries are non-zero if the particles are connected by a spring, and the value stored is the magnitude of separation between the particles
    #at this point we have defined the basic set up for the system and can move on to the simulation
    #to run a simulation of any use, we need to define boundary conditions on the system, which means choosing values of displacement (as a vector), or traction (force as a vector), applied to each node on a boundary of the system (technically a displacement gradient could be assigned to the nodes as well, which would be related to the strain, and then to a stress which leads to a traction when the unit normal outward is specified and matrix multiplied to the stress at the boundary point)
    #we then need to choose the method we will utilize for the system, energy minimization, or some form of numerical integration (Verlet method, or whatever else). numerical integration requires assigning mass values to each node, and a damping factor, where we have found the "solution", being the final configuration of nodes/displacements of the nodes for a given boundary condition, arrangement/connectivity, and stiffness values. energy minimization can be done by a conjugate gradient method
    #in either case we need to calculate both energy and force(negative of the gradient of the energy). For the linear spring case the energy is quadratic in the displacement, and the gradient is linear with respect to the displacement. additional energy terms related to the volume preserving forces will also need to be calculated
    #when the method of choice is chosen, we need functions describing the energy, gradient, and the optimization method, and we need to save out the state at each time step if numerically integrating (if we are interested in the dynamics), or the final state of minimization
    
#!!! wishlist
#TODO
# adjust script to use the new spring variable initialization, springs.get_springs():DONE
# post simulation check on forces to determine if convergence has occurred, and to restart the simulationwith the intermediate configuration, looping until convergence criteria are met
# tracking of particle centers:DONE
# magnetic force interaction calculations:DONE
# profiling a two particle system with magnetic interactions
# performance comparison of gpu calculations of spring forces:DONE
# use density values for PDMS blends and carbonyl iron to set the mass values for each node properly. DONE
# use realistic values for the young's modulus DONE
# consider and implement options for particle magnetization (mumax3 results are only useful for the two particle case with particle aligned field): options include froehlich-kennely, hyperbolic tangent (anhysteretic saturable models)

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import time
from datetime import date
import os
import lib_programname
import tables as tb#pytables, for HDF5 interface
# import get_volume_correction_force_cy_nogil
# import get_spring_force_cy
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import springs
# import magnetism
import simulate
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7
#remember, purpose, signature, stub

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

def place_two_particles_normalized(radius,l_e,dimensions,separation):
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Nel_x, Nel_y, Nel_z = dimensions
    # radius = 0.5*l_e# radius = l_e*(4.5)
    assert radius < np.min(dimensions)/2, f"Particle size greater than the smallest dimension of the simulation"
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    #find the center of the simulated system
    center = (np.array([Nel_x,Nel_y,Nel_z])/2)
    #if there are an even number of elements in a direction, need to increment the central position by half an edge length so the particle centers match up with the centers of cubic elements
    if np.mod(Nel_x,2) == 0:
        center[0] += 1/2
    if np.mod(Nel_y,2) == 0:
        center[1] += 1/2
    if np.mod(Nel_z,2) == 0:
        center[2] += 1/2
    #check particle separation to see if it is acceptable or not for the shift in particle placement from the simulation "center" to align with the cubic element centers
    shift_l = np.round(separation/2)
    shift_r = separation - shift_l
    # if np.mod(separation,2) == 1:
    #     shift_l = (separation-1)*1/2
    #     shift_r = (separation+1)*1/2
    # else:
    #     shift_l = separation*1/2
    #     shift_r = shift_l
    particle_nodes = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center-np.array([shift_l,0,0]),dimensions)
    particle_nodes2 = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center+np.array([shift_r,0,0]),dimensions)
    particles = np.vstack((particle_nodes,particle_nodes2))
    return particles

def place_n_particles_normalized(n_particles,radius,l_e,dimensions,separation):
    #TODO Unfinished, intention to place particles with either random distribution or regular/crystal structure like distribution.
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Nel_x, Nel_y, Nel_z = dimensions
    # radius = 0.5*l_e# radius = l_e*(4.5)
    assert radius < np.min(dimensions)/2, f"Particle size greater than the smallest dimension of the simulation"
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    #find the center of the simulated system
    center = (np.array([Nel_x,Nel_y,Nel_z])/2)
    #if there are an even number of elements in a direction, need to increment the central position by half an edge length so the particle centers match up with the centers of cubic elements
    if np.mod(Nel_x,2) == 0:
        center[0] += 1/2
    if np.mod(Nel_y,2) == 0:
        center[1] += 1/2
    if np.mod(Nel_z,2) == 0:
        center[2] += 1/2
    #check particle separation to see if it is acceptable or not for the shift in particle placement from the simulation "center" to align with the cubic element centers
    shift_l = np.round(separation/2)
    shift_r = separation - shift_l
    particle_nodes = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center-np.array([shift_l,0,0]),dimensions)
    particle_nodes2 = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center+np.array([shift_r,0,0]),dimensions)
    particles = np.vstack((particle_nodes,particle_nodes2))
    return particles

def placeholder_particle_placement(radius,l_e,dimensions):
    Nel_x, Nel_y, Nel_z = dimensions
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    #if i want to think about the grid of possible "points" where a particle would be entirely within the internal volume (no voxel of the particle existing at the boundaries of the simulated system), then i need to think about the number of potential voxels in each dimension that could be the center voxel of a particle. if i am placing them "randomly" then i need to ensure that the particles neither overlap, or have immediately adjacent voxels. 
    allowed_elements_x = (Nel_x - 2*radius_voxels).astype(np.int32)
    allowed_elements_y = (Nel_y - 2*radius_voxels).astype(np.int32)
    allowed_elements_z = (Nel_z - 2*radius_voxels).astype(np.int32)
    pass

def run_strain_sim(output_dir,strain_type,strain_direction,strains,Hext,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,particle_rotation_flag=False):
    """Run a simulation applying a series of a particular type of strain to the volume, by passing the strain type as a string (one of the following: tension, compression, shearing, torsion), the strain direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""
    eq_posns = x0.copy()
    total_delta = 0
    # stress = np.zeros(strains.shape,dtype=np.float64)
    for count, strain in enumerate(strains):
        if output_dir[-1] != '/':
            current_output_dir = output_dir + f'/strain_{count}_{strain_type}_{np.round(strain,decimals=3)}/'
        elif output_dir[-1] == '/':
            current_output_dir = output_dir + f'strain_{count}_{strain_type}_{np.round(strain,decimals=3)}/'
        if not (os.path.isdir(current_output_dir)):
            os.mkdir(current_output_dir)
        if strain_type == 'tension':
            if strain < 0:
                strain *= -1
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[0]),strain)
            if strain_direction[0] == 'x':
                x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
            elif strain_direction[0] == 'y':
                x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
            elif strain_direction[0] == 'z':
                x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
        elif strain_type == 'compression':
            if strain >= 1 or strain <= -1:
                raise ValueError('For compressive strains, cannot exceed 100"%" strain')
            if strain > 0:
                strain *= -1
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[0]),strain)
            if strain_direction[0] == 'x':
                x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
            elif strain_direction[0] == 'y':
                x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
            elif strain_direction[0] == 'z':
                x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif strain_type == 'shearing':
            if strain >= np.abs(np.pi/2):
                raise ValueError('For shearing strains, cannot use or exceed an angle of pi/2')
            if strain_direction[0] == 'x':
                if strain_direction[1] == 'x':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in x direction and force in x direction')
                adjacent_length = dimensions[0]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'y':
                    x0[boundaries['right'],1] = eq_posns[boundaries['right'],1] + opposite_length
                elif strain_direction[1] == 'z':
                    x0[boundaries['right'],2] = eq_posns[boundaries['right'],2] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            elif strain_direction[0] == 'y':
                if strain_direction[1] == 'y':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in y direction and force in y direction')
                adjacent_length = dimensions[1]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'x':
                    x0[boundaries['back'],0] = eq_posns[boundaries['back'],0] + opposite_length
                elif strain_direction[1] == 'z':
                    x0[boundaries['back'],2] = eq_posns[boundaries['back'],2] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            elif strain_direction[0] == 'z':
                if strain_direction[1] == 'z':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in z direction and force in z direction')
                adjacent_length = dimensions[2]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'x':
                    x0[boundaries['top'],0] = eq_posns[boundaries['top'],0] + opposite_length
                elif strain_direction[1] == 'y':
                    x0[boundaries['top'],1] = eq_posns[boundaries['top'],1] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[1]),strain)
        elif strain_type == 'torsion':
            #torsion will only be applied to the top surface
            #because the center of the system isn't at the origin, I need to translate the positions so that they are in a coordinate system where the center of the system in the 2D plane 'xy' is at (0,0), then rotate, then translate back to the original coordinate system
            if strain_direction[1] == 'CW':
                starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
                x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + starting_positions[:,1]*np.sin(strain)
                x0[boundaries['top'],1] = -1*starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
                x0[boundaries['top']] += np.array([dimensions[0]/l_e,dimensions[1]/l_e,0])
            elif strain_direction[1] == 'CCW':
                starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
                x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + -1*starting_positions[:,1]*np.sin(strain)
                x0[boundaries['top'],1] = starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
                x0[boundaries['top']] += np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
            else:
                raise ValueError('strain direction for torsion must be one of ("CW", "CCW") for clockwise or counterclockwise rotation of the top surface of the simulated volume')
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[1]),strain)
        else:
            raise ValueError('Strain type not one of the following accepted types ("tension", "compression", "shearing", "torsion")')
        try:
            start = time.time()
            if particle_rotation_flag == False:
                sol, return_status = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
            elif particle_rotation_flag == True:
                sol, return_status = simulate.simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
        end = time.time()
        delta = end - start
        total_delta += delta
        #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
        # end_result = sol.y[:,-1]
        #below getting the solution at the final time, is the solution provided from scipy.integrate.ode. no more issue with memory overhead since this way, instead of using solve_ivp(), doesn' t record the itnermediate states, just spits out the state at the desired time
        end_result = sol
        x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
        print('took %.2f seconds to simulate' % delta)
        mre.initialize.write_output_file(count,x0,Hext,boundary_conditions,np.array([delta]),output_dir)
    return total_delta, return_status

def run_hysteresis_sim(output_dir,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,particle_rotation_flag=False):
    eq_posns = x0.copy()
    total_delta = 0
    for count, Hext in enumerate(Hext_series):
        #TODO better implementation of boundary conditions
        boundary_conditions = ('free',('free','free'),0) 
        if output_dir[-1] != '/':
            current_output_dir = output_dir + f'/field_{count}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        elif output_dir[-1] == '/':
            current_output_dir = output_dir + f'field_{count}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        if not (os.path.isdir(current_output_dir)):
            os.mkdir(current_output_dir)
        try:
            print(f'Running simulation with external magnetic field: ({Hext[0]*mu0}, {Hext[1]*mu0}, {Hext[2]*mu0}) T\n')
            start = time.time()
            if particle_rotation_flag == False:
                sol, return_status = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
            elif particle_rotation_flag == True:
                sol, return_status = simulate.simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        total_delta += delta
        #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
        # end_result = sol.y[:,-1]
        #below getting the solution at the final time, is the solution provided from scipy.integrate.ode. no more issue with memory overhead since this way, instead of using solve_ivp(), doesn't record the itnermediate states, just spits out the state at the desired time
        end_result = sol
        x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
        # posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        # a_var = mre.analyze.get_accelerations_post_simulation_v3(x0,boundaries,springs_var,elements,kappa,l_e,boundary_conditions)
        # end_boundary_forces = a_var[boundaries['right']]*m[boundaries['right'],np.newaxis]
        # boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        # effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,x0,Hext,boundary_conditions,np.array([delta]),output_dir)
        # mre.analyze.post_plot_cut_normalized_hyst(eq_posns,x0,springs_var,particles,Hext,output_dir)
    return total_delta, return_status

def run_field_dependent_strain_sim(output_dir,strain_type,strain_direction,strains,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,particle_rotation_flag=False,gpu_flag=False):
    """Run a simulation applying a series of a particular type of strain to the volume with a series of applied external magnetic fields, by passing the strain type as a string (one of the following: tension, compression, shearing, torsion), the strain direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""
    eq_posns = x0.copy()
    total_delta = 0
    # stress = np.zeros(strains.shape,dtype=np.float64)
    for count, strain in enumerate(strains):
        if strain_type == 'plate_compression':
            if strain >= 1 or strain <= -1:
                raise ValueError('For compressive strains, cannot exceed 100"%" strain')
            if strain > 0:
                strain *= -1
            if strain_direction[0] == 'x':
                plate_posn = dimensions[0]/l_e * (1 + strain)
            elif strain_direction[0] == 'y':
                plate_posn = dimensions[1]/l_e * (1 + strain)
            elif strain_direction[0] == 'z':
                plate_posn = dimensions[2]/l_e * (1 + strain)
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[0]),plate_posn)
        elif strain_type == 'tension':
            if strain < 0:
                strain *= -1
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[0]),strain)
            if strain_direction[0] == 'x':
                x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
            elif strain_direction[0] == 'y':
                x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
            elif strain_direction[0] == 'z':
                x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
        elif strain_type == 'compression':
            if strain >= 1 or strain <= -1:
                raise ValueError('For compressive strains, cannot exceed 100"%" strain')
            if strain > 0:
                strain *= -1
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[0]),strain)
            if strain_direction[0] == 'x':
                x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
            elif strain_direction[0] == 'y':
                x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
            elif strain_direction[0] == 'z':
                x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif strain_type == 'shearing':
            if strain >= np.abs(np.pi/2):
                raise ValueError('For shearing strains, cannot use or exceed an angle of pi/2')
            if strain_direction[0] == 'x':
                if strain_direction[1] == 'x':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in x direction and force in x direction')
                adjacent_length = dimensions[0]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'y':
                    x0[boundaries['right'],1] = eq_posns[boundaries['right'],1] + opposite_length
                elif strain_direction[1] == 'z':
                    x0[boundaries['right'],2] = eq_posns[boundaries['right'],2] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            elif strain_direction[0] == 'y':
                if strain_direction[1] == 'y':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in y direction and force in y direction')
                adjacent_length = dimensions[1]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'x':
                    x0[boundaries['back'],0] = eq_posns[boundaries['back'],0] + opposite_length
                elif strain_direction[1] == 'z':
                    x0[boundaries['back'],2] = eq_posns[boundaries['back'],2] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            elif strain_direction[0] == 'z':
                if strain_direction[1] == 'z':
                    raise ValueError('Cannot have a shear strain applied to a surface with surface normal in z direction and force in z direction')
                adjacent_length = dimensions[2]/l_e
                opposite_length = np.tan(strain)*adjacent_length
                if strain_direction[1] == 'x':
                    x0[boundaries['top'],0] = eq_posns[boundaries['top'],0] + opposite_length
                elif strain_direction[1] == 'y':
                    x0[boundaries['top'],1] = eq_posns[boundaries['top'],1] + opposite_length
                else:
                    raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[1]),strain)
        elif strain_type == 'torsion':
            #torsion will only be applied to the top surface
            #because the center of the system isn't at the origin, I need to translate the positions so that they are in a coordinate system where the center of the system in the 2D plane 'xy' is at (0,0), then rotate, then translate back to the original coordinate system
            if strain_direction[1] == 'CW':
                starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
                x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + starting_positions[:,1]*np.sin(strain)
                x0[boundaries['top'],1] = -1*starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
                x0[boundaries['top']] += np.array([dimensions[0]/l_e,dimensions[1]/l_e,0])
            elif strain_direction[1] == 'CCW':
                starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
                x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + -1*starting_positions[:,1]*np.sin(strain)
                x0[boundaries['top'],1] = starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
                x0[boundaries['top']] += np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
            else:
                raise ValueError('strain direction for torsion must be one of ("CW", "CCW") for clockwise or counterclockwise rotation of the top surface of the simulated volume')
            boundary_conditions = (strain_type,(strain_direction[0],strain_direction[1]),strain)
        else:
            raise ValueError('Strain type not one of the following accepted types ("tension", "compression", "shearing", "torsion")')
        for i, Hext in enumerate(Hext_series):
            if output_dir[-1] != '/':
                current_output_dir = output_dir + f'/strain_{count}_{strain_type}_{np.round(strain,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            elif output_dir[-1] == '/':
                current_output_dir = output_dir + f'strain_{count}_{strain_type}_{np.round(strain,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            if not (os.path.isdir(current_output_dir)):
                os.mkdir(current_output_dir)
            print(f'Running simulation with external magnetic field: ({np.round(Hext[0]*mu0,decimals=2)}, {np.round(Hext[1]*mu0,decimals=2)}, {np.round(Hext[2]*mu0,decimals=2)}) T\n')
            start = time.time()
            try:
                if not particle_rotation_flag:
                    sol, return_status = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
                elif particle_rotation_flag and (not gpu_flag):
                    sol, return_status = simulate.simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag,get_time_flag=True)
                elif gpu_flag:
                    # mempool = cp.get_default_memory_pool()
                    # print(f'Memory used by springs and elements variable in GB: {mempool.used_bytes()/1024/1024/1024}')
                    sol, return_status = simulate.simulate_scaled_gpu(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag,get_time_flag=True)
            except Exception as inst:
                print('Exception raised during simulation')
                print(type(inst))
                print(inst)
            end = time.time()
            delta = end - start
            total_delta += delta
            #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
            # end_result = sol.y[:,-1]
            #below getting the solution at the final time, is the solution provided from scipy.integrate.ode. no more issue with memory overhead since this way, instead of using solve_ivp(), doesn' t record the itnermediate states, just spits out the state at the desired time
            end_result = sol
            x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
            print('took %.2f seconds to simulate' % delta)
            output_file_number = count*Hext_series.shape[0]+i
            mre.initialize.write_output_file(output_file_number,x0,Hext,boundary_conditions,np.array([delta]),output_dir)
            #if we have already run a particular simulation with zero strain and at some field, use that as the starting point for the solution
            if (output_file_number >= (Hext_series.shape[0]-1)) and (output_file_number < Hext_series.shape[0]*len(strains)-1):
                output_file_num_to_reuse = output_file_number-(Hext_series.shape[0]-1)
                x0, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
                print(f'reusing previously calculated solution with B_ext = {mu0*output_file_Hext}')
    return total_delta, return_status

def run_field_dependent_stress_sim(output_dir,bc_type,bc_direction,stresses,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,particle_rotation_flag=False,gpu_flag=False):
    """Run a simulation applying a series of a particular type of strain to the volume with a series of applied external magnetic fields, by passing the strain type as a string (one of the following: tension, compression, shearing, torsion), the strain direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""
    eq_posns = x0.copy()
    total_delta = 0
    # stress = np.zeros(strains.shape,dtype=np.float64)
    for count, stress in enumerate(stresses):
        boundary_conditions = (bc_type,(bc_direction[0],bc_direction[1]),stress)
        for i, Hext in enumerate(Hext_series):
            if output_dir[-1] != '/':
                current_output_dir = output_dir + f'/stress_{count}_{bc_type}_{np.round(stress,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            elif output_dir[-1] == '/':
                current_output_dir = output_dir + f'stress_{count}_{bc_type}_{np.round(stress,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            if not (os.path.isdir(current_output_dir)):
                os.mkdir(current_output_dir)
            print(f'Running simulation with external magnetic field: ({np.round(Hext[0]*mu0,decimals=2)}, {np.round(Hext[1]*mu0,decimals=2)}, {np.round(Hext[2]*mu0,decimals=2)}) T\n')
            start = time.time()
            try:
                if not particle_rotation_flag:
                    sol, return_status = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
                elif particle_rotation_flag and (not gpu_flag):
                    sol, return_status = simulate.simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag,get_time_flag=True)
                elif gpu_flag:
                    # mempool = cp.get_default_memory_pool()
                    # print(f'Memory used by springs and elements variable in GB: {mempool.used_bytes()/1024/1024/1024}')
                    sol, return_status = simulate.simulate_scaled_gpu(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag,get_time_flag=True)
            except Exception as inst:
                print('Exception raised during simulation')
                print(type(inst))
                print(inst)
            end = time.time()
            delta = end - start
            total_delta += delta
            #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
            # end_result = sol.y[:,-1]
            #below getting the solution at the final time, is the solution provided from scipy.integrate.ode. no more issue with memory overhead since this way, instead of using solve_ivp(), doesn' t record the itnermediate states, just spits out the state at the desired time
            end_result = sol
            x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
            print('took %.2f seconds to simulate' % delta)
            output_file_number = count*Hext_series.shape[0]+i
            mre.initialize.write_output_file(output_file_number,x0,Hext,boundary_conditions,np.array([delta]),output_dir)
            #if we have already run a particular simulation with zero strain and at some field, use that as the starting point for the solution
            if (output_file_number >= (Hext_series.shape[0]-1)) and (output_file_number < Hext_series.shape[0]*len(stresses)-1):
                output_file_num_to_reuse = output_file_number-(Hext_series.shape[0]-1)
                x0, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
                print(f'reusing previously calculated solution with B_ext = {mu0*output_file_Hext}')
    return total_delta, return_status

def main():
    E = 1e3
    nu = 0.499
    l_e = 0.1e-0#cubic element side length
    Lx = 1.5e-0
    Ly = 1.1e-0
    Lz = 1.1e-0
    t_f = 30
    dimensions = np.array([Lx,Ly,Lz])
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]
    separation = 5
    radius = 0.5*l_e# radius = l_e*(4.5)
    particles = place_two_particles_normalized(radius,l_e,normalized_dimensions,separation)
    kappa = mre.initialize.get_kappa(E, nu)
    #TODO: update and improve implementation of saving out/checking/reading in initialization files
    #need functionality to check some central directory containing initialization files
    # system_string = f'E_{E}_le_{l_e}_Lx_{Lx}_Ly_{Ly}_Lz_{Lz}'
    # current_dir = os.path.abspath('.')
    # input_dir = current_dir + f'/init_files/{system_string}/'
    # if not (os.path.isdir(input_dir)):#TODO add and statement that checks if the init file also exists?
    #     os.mkdir(input_dir)
    #     node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    #     normalized_posns = mre.initialize.discretize_space(Lx/l_e,Ly/l_e,Lz/l_e,1)
    #     # normalized_posns = node_posns/l_e
    #     elements = springs.get_elements(normalized_posns, dimensions, 1)
    #     boundaries = mre.initialize.get_boundaries(normalized_posns)
    #     k = mre.initialize.get_spring_constants(E, nu, l_e)
    #     node_types = springs.get_node_type(normalized_posns.shape[0],boundaries,dimensions,1)
    #     k = np.array(k,dtype=np.float64)
    #     max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
    #     springs_var = np.empty((max_springs,4),dtype=np.float64)
    #     num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, 1)
    #     springs_var = springs_var[:num_springs,:]
    #     separation = 5
    #     radius = 0.5*l_e# radius = l_e*(4.5)
    #     particles = place_two_particles(radius,l_e,dimensions,separation)
    #     mre.initialize.write_init_file(node_posns,springs_var,elements,particles,boundaries,input_dir)
    # elif os.path.isfile(input_dir+'init.h5'):
    #     node_posns, springs_var, elements, boundaries = mre.initialize.read_init_file(input_dir+'init.h5')
    #     #TODO implement support functions for particle placement to ensure matching to existing grid of points and avoid unnecessary repetition
    #     #radius = l_e*0.5
    #     separation = 5
    #     radius = 0.5*l_e# radius = l_e*(4.5)
    #     particles = place_two_particles(radius,l_e,dimensions,separation)
    #     # #TODO do better at placing multiple particles, make the helper functionality to ensure placement makes sense
    # else:
    #     node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    #     normalized_posns = mre.initialize.discretize_space(Lx,Ly,Lz,1)
    #     # normalized_posns = node_posns/l_e
    #     elements = springs.get_elements(normalized_posns, dimensions, 1)
    #     boundaries = mre.initialize.get_boundaries(normalized_posns)
    #     k = mre.initialize.get_spring_constants(E, nu, l_e)
    #     node_types = springs.get_node_type(normalized_posns.shape[0],boundaries,dimensions,1)
    #     k = np.array(k,dtype=np.float64)
    #     max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
    #     springs_var = np.empty((max_springs,4),dtype=np.float64)
    #     num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, 1)
    #     springs_var = springs_var[:num_springs,:]
    #     separation = 5
    #     radius = 0.5*l_e# radius = l_e*(4.5)
    #     particles = place_two_particles(radius,l_e,dimensions,separation)
    # particles = np.array([])
    # kappa = mre.initialize.get_kappa(E, nu)
    boundary_conditions = ('strain',('left','right'),.05)

    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name.stem + '/tests/2_dip/'
    output_dir = '/mnt/c/Users/bagaw/Desktop/normalization_testing/'
    
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,kappa,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    

    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.05,-0.02)
    Hext = np.array([0,0,0],dtype=np.float64)
    particle_radius = radius
    chi = 131
    Ms = 1.9e6
    drag = 20
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    tolerance = 1e-4
    for count, strain in enumerate(strains):
        #TODO better implementation of boundary conditions
        boundary_conditions = ('strain',('left','right'),strain)
        # boundary_conditions=('free',('free','free'),0)
        if boundary_conditions[0] == 'strain':
        # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
            surface = boundary_conditions[1][1]
            if surface == 'right' or surface == 'left':
                pinned_axis = 0
            elif surface == 'top' or surface == 'bottom':
                pinned_axis = 2
            else:
                pinned_axis = 1
            x0[boundaries[surface],pinned_axis] = normalized_posns[boundaries[surface],pinned_axis] * (1 + boundary_conditions[2])   
        try:
            start = time.time()
            sol = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,normalized_posns,output_dir)
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
        # end_result = sol.y[:,-1]
        #below getting the solution at the final time, is the solution provided from scipy.integrate.ode. no more issue with memory overhead since this way, instead of using solve_ivp(), doesn't record the itnermediate states, just spits out the state at the desired time
        end_result = sol
        x0 = np.reshape(end_result[:normalized_posns.shape[0]*normalized_posns.shape[1]],normalized_posns.shape)
        # posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        # a_var = mre.analyze.get_accelerations_post_simulation_v3(x0,boundaries,springs_var,elements,kappa,l_e,boundary_conditions)
        # end_boundary_forces = a_var[boundaries['right']]*m[boundaries['right'],np.newaxis]
        # boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        # effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,x0,Hext,boundary_conditions,np.array([delta]),output_dir)
        # mre.analyze.post_plot_v2(x0,springs,boundary_conditions,output_dir)
        # mre.analyze.post_plot_v3(node_posns,x0,springs,boundary_conditions,boundaries,output_dir)
        # mre.analyze.post_plot_cut(normalized_posns,x0,springs_var,particles,dimensions,l_e,boundary_conditions,output_dir)
        mre.analyze.post_plot_cut_normalized(normalized_posns,x0,springs_var,particles,boundary_conditions,output_dir)
        # mre.analyze.post_plot_particle(node_posns,x0,particle_nodes,springs,boundary_conditions,output_dir)
    
def main2():
    """Simulating 2 particle hysteresis with particles perfectly aligned"""
    start = time.time()
    E = 9e5
    nu = 0.499
    max_integrations = 4
    max_integration_steps = 5000
    tolerance = 1e-4
    #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    particle_diameter = 3e-6
    #discretization order
    discretization_order = 0
    l_e = (particle_diameter/2) / (discretization_order + 1/2)
    #particle separation
    separation_meters = 9e-6
    separation_volume_elements = int(separation_meters / l_e)
    separation = separation_volume_elements#20#12#4
    particle_radius = (discretization_order + 1/2)*l_e#2.5*l_e# 0.5*l_e# radius = l_e*(4.5)
    #l_e = (3/5)*1e-6#3e-6#cubic element side length
    # Lx = 41*l_e#27*l_e#15*l_e
    # Ly = 23*l_e#17*l_e#11*l_e
    # Lz = 23*l_e#17*l_e#11*l_e
    Lx = separation_meters + particle_diameter + 1.8*separation_volume_elements*l_e
    Ly = particle_diameter * 7
    Lz = Ly
    t_f = 30
    drag = 20
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    print(f'Current node setup of ({N_nodes_x},{N_nodes_y},{N_nodes_z})')
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]

    particles = place_two_particles_normalized(particle_radius,l_e,normalized_dimensions,separation)
    chi = 131
    Ms = 1.9e6
    #TODO: for distributed computing, I can't depend on looking at existing initialization files to extract variables. I'll have to either instantiate them based on command line arguments or an input file containing similar information, or (and this method seems like it is not th ebest for distributed computing) have separate "jobs" that i run locally or distributed to generate the init files, and use those as transferred input files for the main program (actually running the numerical integration to find equilibrium node configurations)
    # particles = np.array([])
    # kappa = mre.initialize.get_kappa(E, nu)
    # boundary_conditions = ('strain',('left','right'),.05)

    # script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    today = date.today()
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_2particle_freeboundaries_order_{discretization_order}_drag{drag}/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    #determine if doing hysteresis loop (up and down legs) or not and choose the maximum field, number of field steps, and field angle
    hysteresis_loop_flag = False
    mu0 = 4*np.pi*1e-7
    H_mag = 0.15/mu0
    n_field_steps = 4
    if n_field_steps != 1:
        H_step = H_mag/(n_field_steps-1)
    else:
        H_step = H_mag/(n_field_steps)
    #polar angle, aka angle wrt the z axis, range 0 to pi
    Hext_theta_angle = np.pi/2
    Hext_phi_angle = (2*np.pi/360)*15#30
    Hext_series_magnitude = np.arange(H_step,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    if hysteresis_loop_flag:
        Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_phi_angle)*np.sin(Hext_theta_angle)
    Hext_series[:,2] = Hext_series_magnitude*np.cos(Hext_theta_angle)

    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    #new beta coefficient without characteristic mass
    beta_new = 4*(np.pi**2)/(k_e*l_e)
    m_ratio = characteristic_mass/m
    scaled_kappa = (l_e**2)*beta_new
    example_scaled_k = k[0]*beta_new*l_e
    scaled_magnetic_force_coefficient = beta/(particle_mass*(l_e**4))
    #using simple harmonic oscillator to calculate a critical drag coefficient, though the actual value likely differs from this one (if i restrict my view to assume, in each cartesian direction, only two springs, resulting in an effective stiffness twice that of a single spring, then the drag coefficient would be \sqrt(2) larger. taking into account the impact of 8 center diagonal springs, and again considering motion in one cartesian direction, the length change is only due to one component (and so we have 8 springs, but the 45 degree angle between the cartesian axis and the axis of the springs reduces the contribution by 1/\sqrt(2))). we also have 8 face diagonal springs, ignoring the 4 that lie in the plane whose normal is parallel to the direction of the displacement of the node, and again a 45 degree angle)
    # drag = beta_new*2*np.sqrt(characteristic_mass*k_e)
    # alt_drag = beta_new*2*np.sqrt(characteristic_mass*(2*k_e + 8 * k[1] / np.sqrt(2) + 8 * k[2]/np.sqrt(3)))
    #TODO functionalize the calculation of the scaled drag coefficient, and test different effective stiffness values. what sort of drag values come out before scaling by beta_new (which is really beta/characteristic_mass) and after (where we want the value to be close to unity)
    my_sim = mre.initialize.Simulation(E,nu,kappa,k,drag,l_e,Lx,Ly,Lz,particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integrations,max_integration_steps)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    field_or_strain_type_string = 'magnetic_field'
    mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,Hext_series,field_or_strain_type_string,output_dir)
    # read_posns,read_mass,read_springs,read_elements, read_boundaries, read_particles, read_parameters, read_series, read_type = mre.initialize.read_init_file(output_dir+'init.h5')
    end = time.time()
    delta = end - start
    print(f'Time to initialize:{delta} seconds\n')
    print(f'Running simulation with dimensions: Lx = {Lx}, Ly = {Ly}, Lz = {Lz}\ndiscretization order = {discretization_order}, l_e = {l_e}')
    # criteria_flag = False
    simulation_time, return_status = run_hysteresis_sim(output_dir,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True,particle_rotation_flag=True)
    my_sim.append_log(f'Simulation took:{simulation_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)

def main_strain():
    """Testing and implementing applied strains with and without particles."""
    start = time.time()
    E = 9e5
    nu = 0.499
    max_integrations = 1
    max_integration_steps = 5000
    tolerance = 1e-4
    #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    particle_diameter = 3e-6
    #discretization order
    discretization_order = 0
    l_e = (particle_diameter/2) / (discretization_order + 1/2)
    # #particle separation
    separation_meters = 9e-6
    separation_volume_elements = int(separation_meters / l_e)
    separation = separation_volume_elements#20#12#4
    particle_radius = (discretization_order + 1/2)*l_e#2.5*l_e# 0.5*l_e# radius = l_e*(4.5)

    Lx = separation_meters + particle_diameter + 1.8*separation_volume_elements*l_e
    Ly = particle_diameter * 7
    Lz = Ly
    # l_e = 1e-6
    t_f = 30
    
    # Lx = 8e-6
    # Ly = 8e-6
    # Lz = 8e-6
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]
    
    particles = np.array([],dtype=np.int32)
    particles = place_two_particles_normalized(particle_radius,l_e,normalized_dimensions,separation)
    chi = 131
    Ms = 1.9e6
    #TODO: for distributed computing, I can't depend on looking at existing initialization files to extract variables. I'll have to either instantiate them based on command line arguments or an input file containing similar information, or (and this method seems like it is not th ebest for distributed computing) have separate "jobs" that i run locally or distributed to generate the init files, and use those as transferred input files for the main program (actually running the numerical integration to find equilibrium node configurations)
    # boundary_conditions = ('strain',('left','right'),.05)

    # script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')

    mu0 = 4*np.pi*1e-7
    # H_mag = 0.0/mu0
    # n_field_steps = 1
    # H_step = H_mag/n_field_steps
    # Hext_angle = (2*np.pi/360)*0#30
    # Hext_series_magnitude = np.arange(H_mag,H_mag + 1,H_step)
    # #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    # Hext_series = np.zeros((len(Hext_series_magnitude),3))
    # Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_angle)
    # Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_angle)
    Hext = np.array([0,0,0],dtype=np.float64)
    Hext_series = Hext
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    beta = np.power(characteristic_time,2)/l_e
    #if i wanted to try and estimate the effective stiffness for small deviations from initial positions of an internal node, i could then calculate a different characteristic time and scaling constant
    effective_stiffness_guess = 2*k_e + 8 * k[1] / np.sqrt(2) + 8 * k[2]/np.sqrt(3)
    other_universe_beta = 4*(np.pi**2)*characteristic_mass/(effective_stiffness_guess*l_e)
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    #new beta coefficient without characteristic mass
    beta_new = 4*(np.pi**2)/(k_e*l_e)
    #using simple harmonic oscillator to calculate a critical drag coefficient, though the actual value likely differs from this one (if i restrict my view to assume, in each cartesian direction, only two springs, resulting in an effective stiffness twice that of a single spring, then the drag coefficient would be \sqrt(2) larger. taking into account the impact of 8 center diagonal springs, and again considering motion in one cartesian direction, the length change is only due to one component (and so we have 8 springs, but the 45 degree angle between the cartesian axis and the axis of the springs reduces the contribution by 1/\sqrt(2))). we also have 8 face diagonal springs, ignoring the 4 that lie in the plane whose normal is parallel to the direction of the displacement of the node, and again a 45 degree angle)
    # drag = beta_new*2*np.sqrt(characteristic_mass*k_e)
    # alt_drag = beta_new*2*np.sqrt(characteristic_mass*(2*k_e + 8 * k[1] / np.sqrt(2) + 8 * k[2]/np.sqrt(3)))
    drag = 20
    my_sim = mre.initialize.Simulation(E,nu,kappa,k,drag,l_e,Lx,Ly,Lz,particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integrations,max_integration_steps)
    my_sim.set_time(t_f)
    field_or_strain_type_string = 'shear_strain'
    strain_type = 'torsion'
    strain_direction = ('z','CCW')
    #shear strain (nonlinear definition) is defined as tangent of the angle opened up. the linear shear strain is simply the angle (which makes sense, the small angle approximation for tangent theta is theta)
    shear_strain_max = np.pi/2/90*1
    strain_max = shear_strain_max
    # strain_max = 0.20
    n_strain_steps = 1
    if n_strain_steps == 1:
        strain_step_size = strain_max
    else:  
        strain_step_size = strain_max/(n_strain_steps-1)
    strains = np.arange(0.0,strain_max+0.01*strain_max,strain_step_size)
    today = date.today()
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_strain_testing_{strain_type}_order_{discretization_order}_drag_{drag}/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    my_sim.write_log(output_dir)
    
    mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,strains,field_or_strain_type_string,output_dir)
    end = time.time()
    delta = end - start
    print(f'Time to initialize:{delta} seconds\n')
    
    simulation_time, return_status = run_strain_sim(output_dir,strain_type,strain_direction,strains,Hext,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True,particle_rotation_flag=True)
    my_sim.append_log(f'Simulation took:{simulation_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)

def main_series_simulations():
    """A series of simulations to run during the winter break while I'm away, with a focus on getting results that can be used for calculating effective moduli dependence on the applied field"""
    youngs_modulus = [3e4]
    discretizations = [1]
    mu0 = 4*np.pi*1e-7
    H_mag = 0.25/mu0
    n_field_steps = 5
    H_step = H_mag/n_field_steps
    Hext_series_magnitude = np.arange(0.0,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    strain_types = ('tension','shearing')#('tension','compression','shearing')
    strain_type_strings = ('tension_strain','shear_strain')#('tension_strain','compressive_strain','shear_strain')
    strain_directions = ((('x','x'),('y','y'),('z','z')),(('x','y'),('x','z')))#((('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))#((('x','x'),('y','y'),('z','z')),(('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))
    Hext_angles = (0,np.pi/2)
    total_sim_num = 0
    for E in youngs_modulus:
        for discretization_order in discretizations:
            for i, strain_type in enumerate(strain_types):
                for strain_direction in strain_directions[i]:
                    field_or_strain_type_string = strain_type_strings[i]
                    for Hext_angle in Hext_angles:
                        Hext_series = np.zeros((len(Hext_series_magnitude),3))
                        Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_angle)
                        Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_angle)
                        print(f'field_or_strain_type_string = {field_or_strain_type_string}\nstrain_type = {strain_type}\nstrain_direction = {strain_direction}\n')
                        print(f"Young's modulus = {E} Pa\ndiscretization order = {discretization_order}\nstrain_direction={strain_direction}\n")
                        main_field_dependent_modulus(discretization_order=discretization_order,separation_meters=9e-6,E=E,nu=0.499,Hext_series=Hext_series,field_or_strain_type_string=field_or_strain_type_string,strain_type=strain_type,strain_direction=strain_direction,max_integrations=25,max_integration_steps=5000,tolerance=1e-4)
                        total_sim_num += 1
    print(total_sim_num)

def experimental_simulation_tests():
    """A simulation with the stiffness constants set to zero and the additional bulk modulus set to zero, used for testing the impact of node-node WCA forces and particle-particle WCA forces for an attractive magnetic field and particle configuration. What kinds of accelerations occur? Where do the particles stop, or do they stop at all? How much does the system oscillate around equilibrium?"""
    youngs_modulus = [9e3]
    discretizations = [0]
    mu0 = 4*np.pi*1e-7
    H_mag = 0.25/mu0
    n_field_steps = 1
    H_step = H_mag/n_field_steps
    Hext_series_magnitude = np.arange(0.0,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    strain_types = ('compression',)#('plate_compression',)#('tension',)#('tension','compression','shearing')
    strain_type_strings = ('compressive_strain',)#('plate_compression',)#('tension_strain',)#('tension_strain','compressive_strain','shear_strain')
    strain_directions = ((('x','x'),),)#((('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))#((('x','x'),('y','y'),('z','z')),(('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))
    Hext_angles = (0,)
    total_sim_num = 0
    for E in youngs_modulus:
        for discretization_order in discretizations:
            for i, strain_type in enumerate(strain_types):
                for strain_direction in strain_directions[i]:
                    field_or_strain_type_string = strain_type_strings[i]
                    for Hext_angle in Hext_angles:
                        Hext_series = np.zeros((len(Hext_series_magnitude),3))
                        Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_angle)
                        Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_angle)
                        print(f'field_or_strain_type_string = {field_or_strain_type_string}\nstrain_type = {strain_type}\nstrain_direction = {strain_direction}\n')
                        print(f"Young's modulus = {E} Pa\ndiscretization order = {discretization_order}\nstrain_direction={strain_direction}\n")
                        main_field_dependent_modulus(discretization_order=discretization_order,separation_meters=9e-6,E=E,nu=0.47,Hext_series=Hext_series,field_or_strain_type_string=field_or_strain_type_string,strain_type=strain_type,strain_direction=strain_direction,max_integrations=3,max_integration_steps=1000,tolerance=1e-4,gpu_flag=False)
                        total_sim_num += 1
    print(total_sim_num)

def experimental_stress_simulation_tests():
    """A simulation with the stiffness constants set to zero and the additional bulk modulus set to zero, used for testing the impact of node-node WCA forces and particle-particle WCA forces for an attractive magnetic field and particle configuration. What kinds of accelerations occur? Where do the particles stop, or do they stop at all? How much does the system oscillate around equilibrium?"""
    youngs_modulus = [9e3]
    discretizations = [1]
    mu0 = 4*np.pi*1e-7
    H_mag = 0.25/mu0
    n_field_steps = 1
    H_step = H_mag/n_field_steps
    Hext_series_magnitude = np.arange(0.0,H_mag + 1,H_step)
    #create a list of applied field magnitudes, going up from 0 to some maximum and back down in fixed intervals
    # Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])
    stress_types = ('simple_stress_compression',)#('simple_stress_shearing',)
    bc_type_strings = ('compression_stress',)#('shearing_stress',)
    bc_directions = ((('x','x'),),)#((('x','y'),),)#((('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))#((('x','x'),('y','y'),('z','z')),(('x','x'),('y','y'),('z','z')),(('x','y'),('x','z'),('y','x'),('y','z'),('z','x'),('z','y')))
    Hext_angles = (0,)
    total_sim_num = 0
    for E in youngs_modulus:
        for discretization_order in discretizations:
            for i, stress_type in enumerate(stress_types):
                for bc_direction in bc_directions[i]:
                    field_or_bc_type_string = bc_type_strings[i]
                    for Hext_angle in Hext_angles:
                        Hext_series = np.zeros((len(Hext_series_magnitude),3))
                        Hext_series[:,0] = Hext_series_magnitude*np.cos(Hext_angle)
                        Hext_series[:,1] = Hext_series_magnitude*np.sin(Hext_angle)
                        print(f'field_or_strain_type_string = {field_or_bc_type_string}\nbc_type = {stress_type}\nbc_direction = {bc_direction}\n')
                        print(f"Young's modulus = {E} Pa\ndiscretization order = {discretization_order}\nbc_direction={bc_direction}\n")
                        main_field_dependent_modulus_stress(discretization_order=discretization_order,separation_meters=9e-6,E=E,nu=0.47,Hext_series=Hext_series,field_or_bc_type_string=field_or_bc_type_string,bc_type=stress_type,bc_direction=bc_direction,max_integrations=2,max_integration_steps=10000,tolerance=1e-4,gpu_flag=True)
                        total_sim_num += 1
    print(total_sim_num)

def main_field_dependent_modulus(discretization_order=1,separation_meters=9e-6,E=9e3,nu=0.499,Hext_series=np.array(np.array([0,0,0],dtype=np.float64)),field_or_strain_type_string = 'shear_strain',strain_type = 'shearing',strain_direction = ('z','x'),max_integrations = 5,max_integration_steps = 5000,tolerance = 1e-4,gpu_flag=False):
    """Running a two particle simulation whose output can be used to calculate an effective modulus, with the end goal being calculation of the dependence of the effective modulus on the applied magnetic field."""
    start = time.time()
    # E = 9e3
    # nu = 0.499
    # max_integrations = 5
    # max_integration_steps = 5000
    # tolerance = 1e-6
    #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    particle_diameter = 3e-6
    #discretization order
    # discretization_order = 1
    l_e = (particle_diameter/2) / (discretization_order + 1/2)
    # #particle separation
    # separation_meters = 9e-6
    separation_volume_elements = int(separation_meters / l_e)
    separation = separation_volume_elements#20#12#4
    particle_radius = (discretization_order + 1/2)*l_e#2.5*l_e# 0.5*l_e# radius = l_e*(4.5)

    Lx = separation_meters + particle_diameter + 1.8*separation_volume_elements*l_e
    Ly = particle_diameter * 7
    Lz = Ly
    # l_e = 1e-6
    t_f = 30
    
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]
    
    # particles = np.array([],dtype=np.int32)
    particles = place_two_particles_normalized(particle_radius,l_e,normalized_dimensions,separation)
    chi = 131
    Ms = 1.9e6

    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')

    mu0 = 4*np.pi*1e-7
    if Hext_series[1,0] != 0:
        Bext_angle = np.arctan(Hext_series[1,1]/Hext_series[1,0])*180/np.pi
    else:
        Bext_angle = 90
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    if (not np.isclose(k_e,0)):
        beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    else:
        beta = 1e-9
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    drag = 20
    my_sim = mre.initialize.Simulation(E,nu,kappa,k,drag,l_e,Lx,Ly,Lz,particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integrations,max_integration_steps)
    my_sim.set_time(t_f)
    #shear strain (nonlinear definition) is defined as tangent of the angle opened up. the linear shear strain is simply the angle (which makes sense, the small angle approximation for tangent theta is theta)
    strain_max = 0.001
    if strain_type =='shearing':
        strain_max = np.arctan(0.01)
    # strain_max = 0.20
    n_strain_steps = 1
    if n_strain_steps == 1:
        strain_step_size = strain_max
    else:  
        strain_step_size = strain_max/(n_strain_steps-1)
    strains = np.arange(0.0,strain_max+0.01*strain_max,strain_step_size)
    today = date.today()
    # output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_field_dependent_modulus_strain_{strain_type}_direction{strain_direction}_order_{discretization_order}_E_{E}_Bext_angle_{Bext_angle}/'
    # if not (os.path.isdir(output_dir)):
    #     os.mkdir(output_dir)
    # my_sim.write_log(output_dir)
    
    # mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,strains,field_or_strain_type_string,output_dir)
    # end = time.time()
    # delta = end - start
    # print(f'Time to initialize:{delta} seconds\n')
    # #first run without the particle rotations
    # simulation_time, return_status = run_field_dependent_strain_sim(output_dir,strain_type,strain_direction,strains,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True,particle_rotation_flag=False)
    # my_sim.append_log(f'Simulation took:{simulation_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)
    #then run with the particle rotations
    today = date.today()
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_field_dependent_modulus_strain_{strain_type}_direction{strain_direction}_order_{discretization_order}_E_{E}_nu_{nu}_Bext_angle_{Bext_angle}_particle_rotations/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    my_sim.write_log(output_dir)
    
    mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,strains,field_or_strain_type_string,output_dir)
    
    if gpu_flag:
        # print(np.can_cast(np.float64,np.float32))
        x0 = np.float32(normalized_posns.copy())
        beta_i = np.float32(beta_i)
        beta = np.float32(beta)
        drag = np.float32(drag)
        strains = np.float32(strains)
        Hext_series = np.float32(Hext_series)
        particles = np.int32(particles)
        for key in boundaries:
            boundaries[key] = np.int32(boundaries[key])
        dimensions = np.float32(dimensions)
        kappa = cp.float32(kappa)
        l_e = np.float32(l_e)
        particle_radius = np.float32(particle_radius)
        particle_mass = np.float32(particle_mass)
        chi = np.float32(chi)
        Ms = np.float32(Ms)
        elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
        springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
    simulation_time, return_status = run_field_dependent_strain_sim(output_dir,strain_type,strain_direction,strains,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True,particle_rotation_flag=True,gpu_flag=gpu_flag)
    my_sim.append_log(f'Simulation took:{simulation_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)

def main_field_dependent_modulus_stress(discretization_order=1,separation_meters=9e-6,E=9e3,nu=0.499,Hext_series=np.array(np.array([0,0,0],dtype=np.float64)),field_or_bc_type_string = 'empty',bc_type = 'shearing',bc_direction = ('z','x'),max_integrations = 5,max_integration_steps = 5000,tolerance = 1e-4,gpu_flag=False):
    """Running a two particle simulation whose output can be used to calculate an effective modulus, with the end goal being calculation of the dependence of the effective modulus on the applied magnetic field."""
    start = time.time()
    # E = 9e3
    # nu = 0.499
    # max_integrations = 5
    # max_integration_steps = 5000
    # tolerance = 1e-6
    #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    particle_diameter = 3e-6
    #discretization order
    # discretization_order = 1
    l_e = (particle_diameter/2) / (discretization_order + 1/2)
    # #particle separation
    # separation_meters = 9e-6
    separation_volume_elements = int(separation_meters / l_e)
    separation = separation_volume_elements#20#12#4
    particle_radius = (discretization_order + 1/2)*l_e#2.5*l_e# 0.5*l_e# radius = l_e*(4.5)

    Lx = separation_meters + particle_diameter + 1.8*separation_volume_elements*l_e
    Ly = particle_diameter * 7
    Lz = Ly
    # l_e = 1e-6
    t_f = 300
    
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]
    
    # particles = np.array([],dtype=np.int32)
    particles = place_two_particles_normalized(particle_radius,l_e,normalized_dimensions,separation)
    chi = 131
    Ms = 1.9e6

    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')

    mu0 = 4*np.pi*1e-7
    if Hext_series[1,0] != 0:
        Bext_angle = np.arctan(Hext_series[1,1]/Hext_series[1,0])*180/np.pi
    else:
        Bext_angle = 90
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    if (not np.isclose(k_e,0)):
        beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    else:
        beta = 1e-9
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    drag = 20
    my_sim = mre.initialize.Simulation(E,nu,kappa,k,drag,l_e,Lx,Ly,Lz,particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integrations,max_integration_steps)
    my_sim.set_time(t_f)
    #shear strain (nonlinear definition) is defined as tangent of the angle opened up. the linear shear strain is simply the angle (which makes sense, the small angle approximation for tangent theta is theta)
    stress_max = 100 #units of Pa aka N/m^2
    n_stress_steps = 1
    if n_stress_steps == 1:
        stress_step_size = stress_max
    else:  
        stress_step_size = stress_max/(n_stress_steps-1)
    stresses = np.arange(0.0,stress_max+0.01*stress_max,stress_step_size)
    if 'compression' in bc_type:
        stresses *= -1
    #then run with the particle rotations
    today = date.today()
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_field_dependent_modulus_stress_{bc_type}_direction{bc_direction}_order_{discretization_order}_E_{E}_nu_{nu}_Bext_angle_{Bext_angle}_particle_rotations_gpu_{gpu_flag}_tf_{t_f}/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    my_sim.write_log(output_dir)
    
    mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,stresses,field_or_bc_type_string,output_dir)
    
    if gpu_flag:
        # print(np.can_cast(np.float64,np.float32))
        x0 = np.float32(normalized_posns.copy())
        beta_i = np.float32(beta_i)
        beta = np.float32(beta)
        drag = np.float32(drag)
        stresses = np.float32(stresses)
        Hext_series = np.float32(Hext_series)
        particles = np.int32(particles)
        for key in boundaries:
            boundaries[key] = np.int32(boundaries[key])
        dimensions = np.float32(dimensions)
        kappa = cp.float32(kappa)
        l_e = np.float32(l_e)
        particle_radius = np.float32(particle_radius)
        particle_mass = np.float32(particle_mass)
        chi = np.float32(chi)
        Ms = np.float32(Ms)
        elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
        springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
    simulation_time, return_status = run_field_dependent_stress_sim(output_dir,bc_type,bc_direction,stresses,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag,max_integrations,max_integration_steps,tolerance,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=True,particle_rotation_flag=True,gpu_flag=gpu_flag)
    my_sim.append_log(f'Simulation took:{simulation_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)

if __name__ == "__main__":
    experimental_stress_simulation_tests()
    # experimental_simulation_tests()
    # main_series_simulations()
    # main_strain()
    # main2()