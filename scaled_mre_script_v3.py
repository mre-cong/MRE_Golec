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
import numpy.random as rand
import cupy as cp
import scipy.special as sci
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
import random

from typing import Any
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
    assert radius < np.min(dimensions*l_e)/2, f"Particle size greater than the smallest dimension of the simulation"
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

def place_n_particles_normalized(n_particles,radius,l_e,dimensions,separation,particle_placement=None,num_particles_along_axes=None,anisotropy_factor=None):
    #TODO Unfinished, intention to additionally implement ability to place particles with either random distribution or regular/crystal structure like distribution with some noise.
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Nel_x, Nel_y, Nel_z = dimensions
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
    if n_particles == 1:
        centers = center
    # elif n_particles == 2:
    #     shift_l_mag = np.round(separation/2)
    #     shift_r_mag = separation - shift_l_mag
    #     shift_l = np.array([shift_l_mag,0,0])
    #     shift_r = np.array([shift_r_mag,0,0])
    #     centers = np.array([center-shift_l,center+shift_r])
    # elif n_particles == 3:
    #     shift = np.array([separation,0,0])
    #     centers = np.array([center-shift,center,center+shift])
    elif particle_placement == 'regular':
        # cubic crystal structure placement. need to find "origin" and then translate properly according to number of particles in each direction
        if num_particles_along_axes == None:
            num_particles_along_x = int(np.power(n_particles,(1/3)))
            num_particles_along_y = num_particles_along_x
            num_particles_along_z = num_particles_along_x
            edge_elements = Nel_x - separation*(num_particles_along_x - 1)
            origin = np.ones((3,))*np.round(edge_elements/2,decimals=1)
            if np.mod(edge_elements,2) == 0:
                origin += 1/2
        else:
            num_particles_along_x = num_particles_along_axes[0]
            num_particles_along_y = num_particles_along_axes[1]
            num_particles_along_z = num_particles_along_axes[2]
            edge_elements_x = Nel_x - separation*(num_particles_along_x - 1)
            edge_elements_y = Nel_y - separation*(num_particles_along_y - 1)
            edge_elements_z = Nel_z - separation*(num_particles_along_z - 1)
            origin = np.ones((3,))
            origin[0] *= np.round(edge_elements_x/2,decimals=1)
            origin[1] *= np.round(edge_elements_y/2,decimals=1)
            origin[2] *= np.round(edge_elements_z/2,decimals=1)
            if np.mod(edge_elements_x,2) == 0:
                origin[0] += 1/2
            if np.mod(edge_elements_y,2) == 0:
                origin[1] += 1/2
            if np.mod(edge_elements_z,2) == 0:
                origin[2] += 1/2
        centers = np.zeros((n_particles,3))
        counter = 0
        for i in range(num_particles_along_x):
            for j in range(num_particles_along_y):
                for h in range(num_particles_along_z):
                    centers[counter] = origin + np.array([i*separation,j*separation,h*separation])
                    counter += 1
    elif particle_placement == 'regular_noisy':
        # cubic crystal structure placement. need to find "origin" and then translate properly according to number of particles in each direction, including some limited "noise" in the placement
        ss = rand.SeedSequence()
        seed = ss.entropy
        print(f'seed = {seed}')
        rng = rand.default_rng(seed)
        if num_particles_along_axes == None:
            num_particles_along_x = int(np.power(n_particles,(1/3)))
            num_particles_along_y = num_particles_along_x
            num_particles_along_z = num_particles_along_x
            edge_elements = Nel_x - separation*(num_particles_along_x - 1)
            origin = np.ones((3,))*np.round(edge_elements/2,decimals=1)
            if np.mod(edge_elements,2) == 0:
                origin += 1/2
        else:
            num_particles_along_x = num_particles_along_axes[0]
            num_particles_along_y = num_particles_along_axes[1]
            num_particles_along_z = num_particles_along_axes[2]
            edge_elements_x = Nel_x - separation*(num_particles_along_x - 1)
            edge_elements_y = Nel_y - separation*(num_particles_along_y - 1)
            edge_elements_z = Nel_z - separation*(num_particles_along_z - 1)
            origin = np.ones((3,))
            origin[0] *= np.round(edge_elements_x/2,decimals=1)
            origin[1] *= np.round(edge_elements_y/2,decimals=1)
            origin[2] *= np.round(edge_elements_z/2,decimals=1)
            if np.mod(edge_elements_x,2) == 0:
                origin[0] += 1/2
            if np.mod(edge_elements_y,2) == 0:
                origin[1] += 1/2
            if np.mod(edge_elements_z,2) == 0:
                origin[2] += 1/2
        centers = np.zeros((n_particles,3))
        counter = 0
        for i in range(num_particles_along_x):
            for j in range(num_particles_along_y):
                for h in range(num_particles_along_z):
                    #rng.integers returns random integers from a discrete unifrom distribution from low(inclusive) to high (exclusive), the interval [low,high)
                    centers[counter] = origin + np.array([i*separation,j*separation,h*separation]) + rng.integers(low=-1,high=2,size=3)
                    counter += 1
    elif 'regular_anisotropic' in particle_placement:
        if 'noisy' in particle_placement:
            ss = rand.SeedSequence()
            seed = ss.entropy
            print(f'seed = {seed}')
            rng = rand.default_rng(seed)
        anisotropic_separation = (separation*anisotropy_factor).astype(np.int64)
        if np.any(np.less_equal(anisotropic_separation,2*radius_voxels+2)):
            raise ValueError(f'anisotropy factor is such that particles would be placed too close to one another for simulation to function properly.',f'particle separation:{anisotropic_separation*l_e}')
        # cubic crystal structure placement, transversely isotropic?. need to find "origin" and then translate properly according to number of particles in each direction, also taking into account anisotropy factor in each direction (how much closer/further apart particles are separated)
        num_particles_along_x = num_particles_along_axes[0]
        num_particles_along_y = num_particles_along_axes[1]
        num_particles_along_z = num_particles_along_axes[2]
        edge_elements_x = Nel_x - anisotropic_separation[0]*(num_particles_along_x - 1)
        edge_elements_y = Nel_y - anisotropic_separation[1]*(num_particles_along_y - 1)
        edge_elements_z = Nel_z - anisotropic_separation[2]*(num_particles_along_z - 1)
        origin = np.ones((3,))
        origin[0] *= np.round(edge_elements_x/2,decimals=1)
        origin[1] *= np.round(edge_elements_y/2,decimals=1)
        origin[2] *= np.round(edge_elements_z/2,decimals=1)
        if np.mod(edge_elements_x,2) == 0:
            origin[0] += 1/2
        if np.mod(edge_elements_y,2) == 0:
            origin[1] += 1/2
        if np.mod(edge_elements_z,2) == 0:
            origin[2] += 1/2
        centers = np.zeros((n_particles,3))
        counter = 0
        for i in range(num_particles_along_x):
            for j in range(num_particles_along_y):
                for h in range(num_particles_along_z):
                    centers[counter] = origin + np.array([i*anisotropic_separation[0],j*anisotropic_separation[1],h*anisotropic_separation[2]])
                    if 'noisy' in particle_placement:
                        #rng.integers returns random integers from a discrete unifrom distribution from low(inclusive) to high (exclusive), the interval [low,high)
                        centers[counter] += rng.integers(low=-1,high=2,size=3)
                    counter += 1
    else:
        raise NotImplementedError(f'{particle_placement} type of placement of particles not implemented')
    particles = mre.sphere_rasterization.place_spheres_normalized(radius_voxels,centers,dimensions)
    # particle_nodes = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center-np.array([shift_l,0,0]),dimensions)
    # particle_nodes2 = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center+np.array([shift_r,0,0]),dimensions)
    # particles = np.vstack((particle_nodes,particle_nodes2))
    if 'noisy' in particle_placement:
        return particles, seed
    else:
        return particles
    
def place_particles_normalized_by_hand(radius,l_e,dimensions,particle_posns):
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    num_particles = particle_posns.shape[0]
    normalized_particle_posns = np.floor_divide(particle_posns,l_e) + np.array([0.5,0.5,0.5])
    #make sure the particles are at least 1+2*radius_voxels away from each other in distance. May have to be further than that (add an extra 1 to 3 voxels of space). if they aren't far enough apart some nodes could belong to more than one particle, which won't work out well.

    for i in range(num_particles):
        for j in range(i+1,num_particles):
            if np.linalg.norm(normalized_particle_posns[i]-normalized_particle_posns[j]) < (1+2*radius_voxels):
                raise ParticlePlacementError

    particles = mre.sphere_rasterization.place_spheres_normalized(radius_voxels,normalized_particle_posns,dimensions)
    return particles

class ParticlePlacementError(Exception):
    pass

def placeholder_particle_placement(radius,l_e,dimensions,placement_type,volume_fraction=None,n_particles=1):
    Nel_x, Nel_y, Nel_z = dimensions
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    #if i want to think about the grid of possible "points" where a particle would be entirely within the internal volume (no voxel of the particle existing at the boundaries of the simulated system), then i need to think about the number of potential voxels in each dimension that could be the center voxel of a particle. if i am placing them "randomly" then i need to ensure that the particles neither overlap, or have immediately adjacent voxels. 
    allowed_elements_x = (Nel_x - 2*radius_voxels - 1).astype(np.int32)
    allowed_elements_y = (Nel_y - 2*radius_voxels - 1).astype(np.int32)
    allowed_elements_z = (Nel_z - 2*radius_voxels - 1).astype(np.int32)
    #if i want a particular volume fraction, i should estimate the number of particles
    if volume_fraction != None:
        total_volume = Nel_x*Nel_y*Nel_z*np.power(l_e,3)
        total_particle_volume = total_volume*volume_fraction
        single_particle_volume = (4/3)*np.pi*np.power(radius,3)
        num_particles = int(np.floor(total_particle_volume/single_particle_volume))
    else:
        num_particles = n_particles
    particles = []
    particle_posns = []
    placement_allowed_array = np.ones((Nel_x,Nel_y,Nel_z),dtype=np.bool_)
    for i in range(np.ceil(radius_voxels).astype(np.int32)):
        placement_allowed_array[i,:,:] = False
        placement_allowed_array[:,i,:] = False
        placement_allowed_array[:,:,i] = False
        placement_allowed_array[(-1*i)-1,:,:] = False
        placement_allowed_array[:,(-1*i)-1,:] = False
        placement_allowed_array[:,:,(-1*i)-1] = False
    # print(placement_allowed_array)
    if 'regular' in placement_type:
        pass
    elif 'random' in placement_type:
        failed_attempt_count = 0
        particles_placed = 0
        while particles_placed < num_particles:
            if failed_attempt_count > 10:
                print(f'failed to place particle {particles_placed+1}, after {failed_attempt_count} attempts. exiting particle placement loop')
            potential_x_posn = random.randint(0,allowed_elements_x) + int((2*radius_voxels + 1)/2)
            potential_y_posn = random.randint(0,allowed_elements_y) + int((2*radius_voxels + 1)/2)
            potential_z_posn = random.randint(0,allowed_elements_z) + int((2*radius_voxels + 1)/2)
            if placement_allowed_array[potential_x_posn,potential_y_posn,potential_z_posn]:
                # placement_allowed_array[potential_x_posn,potential_y_posn,potential_z_posn] = False
                exclusion_range = int(radius_voxels*2)
                for i in range(potential_x_posn-exclusion_range,potential_x_posn+exclusion_range):
                    for j in range(potential_y_posn-exclusion_range,potential_y_posn+exclusion_range):
                        for k in range(potential_z_posn-exclusion_range,potential_z_posn+exclusion_range):
                            if i < 0 or j < 0 or k < 0 or i >= Nel_x or j >= Nel_y or k >= Nel_z:
                                pass
                            else:
                                grid_distance = np.abs(potential_x_posn-i) + np.abs(potential_y_posn-j) + np.abs(potential_z_posn-k)
                                if grid_distance <= exclusion_range:
                                    placement_allowed_array[i,j,k] = False
                particles.append(mre.sphere_rasterization.place_sphere_normalized(radius_voxels,np.array([potential_x_posn,potential_y_posn,potential_z_posn]),dimensions))
                particles_placed +=1
                failed_attempt_count = 0
                # print(placement_allowed_array)
            else:
                failed_attempt_count += 1
    return np.array(particles,dtype=np.int64)

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
        print('deprecated function, do not use, will fail')
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

def reinforce_particle_particle_spring(springs,particles):
    """go through the springs variable and greatly increase the spring stiffness of intra-particle nodes"""
    particles_set = set(np.ravel(particles))
    for i in range(springs.shape[0]):
        spring_node_indices = set(springs[i,0:2].astype(np.int32))
        if spring_node_indices.issubset(particles_set):
            springs[i,2] *= 100
    return springs

def batch_job_runner():
    """Wrapper function. Future implementation should take in a config file describing the set of simulations, and could produce config files for each simulation that is passed to the function actually running the simulations"""
    youngs_modulus = [9e3]#[9e3,2*9e3]#[1e-2]#[1e6]#
    discretizations = [7]#[3,4,5,6]#[0,1,2,3,4,5]
    poisson_ratios = [0.47]#[0.47]
    volume_fractions = np.concatenate((np.array([5.5e-2,6.5e-2,7e-2,7.5e-2]),np.linspace(0.02,0.2,10)))#np.concatenate((np.array([1e-2,1.5e-2,2.5e-2,3e-2,3.5e-2,4.5e-2,5e-2,5.5e-2,6.5e-2,7e-2,7.5e-2]),np.linspace(0.02,0.2,10)))#np.array([3e-2])#np.array([3e-2,6e-2,9e-2])#
    bc_directions = ((('z','x'),),)#((('z','z'),),(('z','z'),),)#((('z','x'),),(('z','z'),),(('z','z'),),)#((('x','x'),),)#((('x','x'),('y','y'),('z','z'),),)#((('x','y'),),)#((('z','z'),),(('x','x'),('z','z')))#((('x','y'),),(('x','x'),('z','z')),(('x','x'),('z','z')),)
    Hext_angles = ((0,0),)#((np.pi/2,0),(0,0))#((np.pi/2,np.pi/2),)#((np.pi/2,0),(np.pi/2,np.pi/2))#((0,0),(np.pi/2,0),)#
    sim_types = ('strain_simple_shearing',)#('hysteresis',)#('simple_stress_shearing',)#('strain_shearing',)#('strain_tension','strain_compression')#('strain_shearing','strain_tension','strain_compression')#('hysteresis',)#('strain_compression',)#('simple_stress_tension',)#('test_simple_stress_tension',)#('strain_tension','simple_stress_tension')#('simple_stress_shearing',)#('simple_stress_compression','simple_stress_tension',)
    bc_type = sim_types#('hysteresis',)#('simple_stress_shearing',)#('simple_stress_compression','simple_stress_tension')#('simple_stress_shearing','simple_stress_compression','simple_stress_tension')
    
    total_sim_num = 0
    step_sizes = [np.float32(5e-3)]#[np.float32(0.01/2)]#[np.float32(0.01/4),np.float32(0.01/8)]#[np.float32(0.01)]#[np.float32(0.01/2),np.float32(0.01/4),np.float32(0.01/8)]
    max_integration_steps = [5000]#[10000, 20000]#[2500]#[5000, 10000, 20000]
    num_particles = 2
    particle_arrangements = [[1,1,2]]#[[1,1,4],[1,1,6],[2,2,4],[3,3,4]]#[[3,3,4]]#
    particle_posns = np.zeros((num_particles,3))
    Lx, Ly, Lz = (12e-6,12e-6,40e-6)#
    # total_volume = Lx*Ly*Lz
    # particle_posns[:,0] = 6e-6
    # particle_posns[:,1] = 6e-6
    # for i in range(num_particles):
    #     particle_posns[i,2] = 11e-6 + i*(6e-6)
    #asymmetric straight chain
    # particle_posns[1,2] = 15.5e-6

    #helical chain
    # Lx, Ly, Lz = (13e-6,13e-6,62e-6)#
    # particle_radius = 1.5e-6
    # vertical_spacing = 5e-6
    # horizontal_spacing = particle_radius
    # particle_posns[:,0] = 6.5e-6
    # particle_posns[:,1] = 6.5e-6
    # angle_increment = 2*np.pi/num_particles
    # for i in range(num_particles):
    #     particle_posns[i,2] = 13.5e-6 + i*vertical_spacing
    #     particle_posns[i,0] += horizontal_spacing*np.cos(i*angle_increment)
    #     particle_posns[i,1] += horizontal_spacing*np.sin(i*angle_increment)

    for particle_arrangement in particle_arrangements:
        for E in youngs_modulus:
            for poisson_ratio in poisson_ratios:
                for volume_fraction in volume_fractions:
                    for discretization_order in discretizations:
                        for i, sim_type in enumerate(sim_types):
                            for step_size,integration_steps in zip(step_sizes,max_integration_steps):
                                for bc_direction in bc_directions[i]:
                                    for Hext_angle in Hext_angles:
                                        # this may not always be correct, but for some simulations not every combination of field angle + boundary condition direction is necessary
                                        # for an isotropic  MRE, there are two configurations for a strain/stress simulation: field + boundary condition parallel, field + boundary condition perpendicular
                                        # for an anisotropic MRE, there are 5 configurations: field + anisotropy + bc parallel, field + anisotropy parallel with bc perpendicular, field + bc parallel with anisotropy perpendicular, anisotropy + bc parallel with field perpendicular, field perpendicular to anisotropy perpendicular to bc 
                                        if np.isclose(Hext_angle[0],np.pi/2) and bc_direction[0] == 'y':
                                            break
                                        parameters = dict({})
                                        parameters['max_integrations'] = 40
                                        # parameters['step_size'] = np.float32(0.01/2)
                                        parameters['max_integration_steps'] = integration_steps
                                        parameters['step_size'] = step_size
                                        parameters['gpu_flag'] = True 
                                        parameters['particle_rotation_flag'] = True
                                        parameters['persistent_checkpointing_flag'] = True
                                        parameters['plotting_flag'] = False
                                        parameters['criteria_flag'] = False
                                        parameters['particle_radius'] = 1.5e-6
                                        parameters['youngs_modulus'] = E
                                        parameters['poisson_ratio'] = poisson_ratio
                                        parameters['drag'] = 1#1#0#20
                                        parameters['discretization_order'] = discretization_order
                                        parameters['particle_placement'] = 'regular'#'by_hand'#'regular_anisotropic'#'regular_anisotropic_noisy'#'regular_noisy'#
                                        parameters['num_particles_along_axes'] = particle_arrangement#[1,1,4]#[1,1,2]#[2,2,2]#[3,1,1]#[8,8,8]#
                                        parameters['num_particles'] = parameters['num_particles_along_axes'][0]*parameters['num_particles_along_axes'][1]*parameters['num_particles_along_axes'][2]
                                        if parameters['num_particles'] == 0 or 'hand' in parameters['particle_placement']:
                                            #dimensions in normalized units, the number of elements in each direction                                            
                                            particle_diameter = 2*parameters['particle_radius']
                                            l_e = (particle_diameter/2) / (discretization_order + 1/2)                                            
                                            parameters['dimensions'] = np.array([np.floor_divide(Lx,l_e),np.floor_divide(Ly,l_e),np.floor_divide(Lz,l_e)])                    
                                            parameters['particle_posns'] = particle_posns
                                        parameters['anisotropy_factor'] = np.array([1.0,1.0,1.0])#np.array([1.0,1.0,0.9])#
                                        parameters['anisotropy_factor'][:2] = 1/np.sqrt(parameters['anisotropy_factor'][2])
                                        parameters['volume_fraction'] = volume_fraction
                                        parameters['particle_separation'] = 9e-6
                                        tmp_field_var = np.array([0.0,1.4e-1])#np.array([0.0,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,1e-1,1.2e-1,1.4e-1])#np.array([0.0,4e-2,12e-2])#np.array([0.0,1e-2,2e-2])#np.array([0.0,4e-2,8e-2,1.2e-1,1.6e-1,2e-1,2.4e-1,3e-1,3.5e-1,4e-1])#np.array([0.0])#np.array([1e-4,1e-2,5e-2,1e-1,1.5e-1])#np.array([1e-4,1e-2,1e-1,2e-1])#np.array([1e-4,1e-2,2e-2,3e-2,5e-2,8e-2,1e-1,1.2e-1,1.4e-1,1.5e-1])
                                        tmp_field_vectors = np.zeros((tmp_field_var.shape[0],3),dtype=np.float32)
                                        if np.isclose(Hext_angle[0],np.pi/2):
                                            tmp_field_vectors[:,0] = (1/mu0)*tmp_field_var
                                        else:
                                            tmp_field_vectors[:,2] = (1/mu0)*tmp_field_var
                                        parameters['Hext_series_magnitude'] = (1/mu0)*tmp_field_var
                                        if 'hysteresis' in sim_type:
                                            first_leg = np.linspace(0,6e-2,13)
                                            # downward_leg = np.concatenate((first_leg[-2::-1],first_leg[1::]*-1))
                                            hysteresis_loop_series = np.concatenate((first_leg,first_leg[-2::-1],first_leg[1::]*-1,first_leg[-2::-1]*-1,first_leg[1::]))
                                            Hext_series = np.zeros((hysteresis_loop_series.shape[0],3),dtype=np.float32)
                                            Hext_series[:,2] = hysteresis_loop_series
                                            parameters['Hext_series'] = (1/mu0)*Hext_series
                                            # parameters['Hext_series_magnitude'] = (1/mu0)*hysteresis_loop_series
                                        # parameters['Hext_series'] = tmp_field_vectors#(1/mu0)*np.array([[1e-4,0,0],[1e-2,0,0],[2e-2,0,0],[3e-2,0,0],[5e-2,0,0],[8e-2,0,0],[1e-1,0,0],[1.2e-1,0,0],[1.4e-1,0,0],[1.5e-1,0,0],],dtype=np.float32)
                                        parameters['max_field'] = 0.06
                                        parameters['field_angle_theta'] = Hext_angle[0]
                                        parameters['field_angle_phi'] = Hext_angle[1]
                                        parameters['num_field_steps'] = 15
                                        if 'stress' in sim_type:
                                            parameters['boundary_condition_value_series'] = np.linspace(0,100,3)#np.array([0,2.5,5.0,7.5,10.0,12.5,15.0])
                                        elif 'strain' in sim_type:
                                            parameters['boundary_condition_value_series'] = np.linspace(0,1e-1,11)#np.array([0.0])#np.linspace(0,5e-2,21)#np.array([0.0,1e-2,2e-2,3e-2,4e-2,5e-2])#np.array([0.0,1e-3,2e-3,5e-3,1e-2,2e-2,3e-2,4e-2])#np.concatenate((np.linspace(0,2e-4,5),np.
                                        parameters['boundary_condition_max_value'] = 0.0010
                                        parameters['num_boundary_condition_steps'] = 5
                                        parameters['boundary_condition_type'] = bc_type[i]
                                        parameters['boundary_condition_direction'] = bc_direction
                                        print(f'sim type = {sim_type}\nbc_direction = {bc_direction}\n')
                                        print(f"Young's modulus = {E} Pa\nPoisson Ratio = {poisson_ratio}\ndiscretization order = {discretization_order}\n")
                                        print(f"N particles = {parameters['num_particles']}\nField angle theta = {Hext_angle[0]}\nField angle phi = {Hext_angle[1]}\n")
                                        print(f"gpu based calculation {parameters['gpu_flag']}")
                                        # parameters could be a dict with key value pairs describing the simulation, acting like a struct, sim_type would be a string used to describe the type of simulation to run (stress based/strain based boundary conditions, magnetic hysteresis, etc.)
                                        run_sim(parameters,sim_type)
                                        total_sim_num += 1
    print(total_sim_num)

def run_sim(parameters: dict[str,Any],sim_type: str) -> Any:
    # needs to select amongst different simulation types, and needs to initialize things
    """Driver for a simulation of a particular type with a given set of parameters."""
    sim_variables_dict, sim_logger = initialize_simulation_variables(parameters,sim_type)
    output_dir = sim_variables_dict['output_dir']
    if 'test' in sim_type:
        simulation_run_time, return_status = run_test_sim(sim_variables_dict)
    elif 'hysteresis' in sim_type:
        simulation_run_time, return_status = run_hysteresis_sim(sim_variables_dict)
    elif 'stress' in sim_type:
        simulation_run_time, return_status = run_stress_sim(sim_variables_dict)
    elif 'strain' in sim_type:
        simulation_run_time, return_status = run_strain_sim(sim_variables_dict)
    else:
        print(f'simulation of type: {sim_type} is not defined\n Exiting Program')
        return -1
    print(f'Simulation took:{simulation_run_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n')
    sim_logger.append_log(f'Simulation took:{simulation_run_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',output_dir)

def reinitialize_sim(sim_dir: str):
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

    _, _, _, boundary_condition, _ = mre.initialize.read_output_file(sim_dir+f'output_{continuation_index-1}.h5')
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

def jumpstart_sim(sim_dir,jumpstart_type,sim_checkpoint_dirs=[]):
    """Restart an interrupted simulation, extend a particular simulation step (field + b.c.) from a checkpoint, or re-run a (set of) simulation steps (field + b.c.). Pass the simulation directory, the desired behavior as a string ('restart','extend','rerun'), and for extending or re-running an optional argument of the simulation steps to extend or re-run as a list of strings of absolute paths to the relevant directories"""
    sim_variables_dict, sim_logger = reinitialize_sim(sim_dir)

    sim_restart_flag = False
    sim_extend_flag = False
    sim_rerun_flag = False

    if 'restart' in jumpstart_type:
        sim_restart_flag = True
    elif 'extend' in jumpstart_type:
        sim_extend_flag = True
    elif 'rerun' in jumpstart_type:
        sim_rerun_flag = True
    else:
        print(f'jumpstart type: {jumpstart_type} is not defined\n Exiting Program')
        raise NotImplementedError 
    sim_type = sim_variables_dict['boundary_condition_type']
    #TODO deal with iterating over the checkpoint directories for simulations to extend or re-run, assigning the correct value to the sim_variables_dict key:value pair
    if sim_checkpoint_dirs == []:
        max_counter_val = 1
        sim_checkpoint_dirs = ['']
    else:
        max_counter_val = len(sim_checkpoint_dirs)
    for i in range(max_counter_val):
        sim_variables_dict['checkpoint_dir'] = sim_checkpoint_dirs[i]
        if 'hysteresis' in sim_type:
            simulation_run_time, return_status = run_hysteresis_sim(sim_variables_dict,sim_restart_flag,sim_extend_flag,sim_rerun_flag)
        elif 'stress' in sim_type:
            simulation_run_time, return_status = run_stress_sim(sim_variables_dict,sim_restart_flag,sim_extend_flag,sim_rerun_flag)
        elif 'strain' in sim_type:
            simulation_run_time, return_status = run_strain_sim(sim_variables_dict,sim_restart_flag,sim_extend_flag,sim_rerun_flag)
        else:
            print(f'simulation of type: {sim_type} is not defined\n Exiting Program')
            return -1
        sim_logger.append_log(f'Simulation took:{simulation_run_time} seconds\nReturned with status {return_status}(0 for converged, -1 for diverged, 1 for reaching maximum integrations)\n',sim_dir)

def initialize_simulation_variables(parameters: dict[str,Any],sim_type: str) -> tuple[dict[str,Any], Any]:
    start = time.time()
    mu0 = 4*np.pi*1e-7
    E = parameters['youngs_modulus']
    nu = parameters['poisson_ratio']
    youngs_modulus_particles = 2.11e11
    max_stiffness_ratio = 10000
    if youngs_modulus_particles/E > max_stiffness_ratio:
        youngs_modulus_particles = max_stiffness_ratio*E
    nu_particles = 0.29
    drag = parameters['drag']
    discretization_order = parameters['discretization_order']

    num_particles = parameters['num_particles']
    particle_placement = parameters['particle_placement']
    particle_separation = parameters['particle_separation']
    num_particles_along_axes = parameters['num_particles_along_axes']
    if 'regular' in particle_placement:
        num_particles_check = num_particles_along_axes[0]*num_particles_along_axes[1]*num_particles_along_axes[2]
        if not (num_particles_check == num_particles):
            raise ValueError('mismatch between stated number of particles in system and numbers of particles along each axis')
    if 'anisotropic' in particle_placement:
        anisotropy_factor = parameters['anisotropy_factor']
    else:
        anisotropy_factor = None
    max_magnetic_field_strength = parameters['max_field']/mu0
    field_angle_theta = parameters['field_angle_theta']
    field_angle_phi = parameters['field_angle_phi']
    num_field_steps = parameters['num_field_steps']
    max_boundary_condition_value = parameters['boundary_condition_max_value']
    num_boundary_condition_steps = parameters['num_boundary_condition_steps']
    bc_type = parameters['boundary_condition_type']
    bc_direction = parameters['boundary_condition_direction']

    particle_radius = parameters['particle_radius']
    particle_volume = (4/3)*np.pi*np.power(particle_radius,3)

    gpu_flag = parameters['gpu_flag']
    # particle_rotation_flag = parameters['particle_rotation_flag']
    # persistent_checkpointing_flag = parameters['persistent_checkpointing_flag']
    # plotting_flag = parameters['plotting_flag']
    # criteria_flag = parameters['criteria_flag']

    max_integrations = parameters['max_integrations']
    max_integration_steps = parameters['max_integration_steps']
    # tolerance = parameters['tolerance']

    sim_variables_dict = parameters.copy()
    # E = 9e3
    # nu = 0.499
    # max_integrations = 5
    # max_integration_steps = 5000
    # tolerance = 1e-6
    #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    particle_diameter = 2*particle_radius

    l_e = (particle_diameter/2) / (discretization_order + 1/2)

    # separation_volume_elements = int(np.round(particle_separation / l_e,decimals=1))
    # separation_meters = l_e*separation_volume_elements
    # if num_particles < 2:
    #     Lx = separation_meters + particle_diameter + 1.8*separation_meters
    #     Ly = particle_diameter * 7
    #     Lz = Ly
    #     # raise NotImplementedError('Determination of system size for less than two particles not implemented')
    # if num_particles == 2:
    #     Lx = separation_meters + particle_diameter + 1.8*separation_meters
    #     Ly = particle_diameter * 7
    #     Lz = Ly
    # elif num_particles == 3:
    #     Lx = 2*separation_meters + particle_diameter + 1.8*separation_meters
    #     Ly = particle_diameter * 7
    #     Lz = Ly
    # elif num_particles > 3:
    #     raise NotImplementedError('Determination of system size for more than 3 particles not implemented')
    if num_particles == 0 and 'dimensions' in parameters:
        Lx, Ly, Lz = parameters['dimensions']*l_e
        volume_fraction = 0
    elif 'hand' in particle_placement and 'dimensions' in parameters:
        Lx, Ly, Lz = parameters['dimensions']*l_e
        total_volume = Lx*Ly*Lz
        volume_fraction = np.round(num_particles*particle_volume/total_volume,decimals=5)
    elif 'regular' in particle_placement:
        if 'volume_fraction' in parameters:
            volume_fraction = parameters['volume_fraction']
        else:
            volume_fraction = 0.03 # volume_fraction = num_particles*particle_volume/total_volume
        # num_particles = 8 #27 #going for a crystalline like arrangement, needs to be a value equal to an integer raised to the 3rd power
        # particle_volume = (4/3)*np.pi*np.power(particle_radius,3)
        #volume_fraction = particle_volume/total_volume
        fictional_total_volume = num_particles*particle_volume/volume_fraction
        fictional_side_length = np.power(fictional_total_volume,(1/3))
        particle_density = num_particles/fictional_total_volume
        particle_separation_metric = 1/np.power(particle_density,(1/3))
        # minimum_dimension = particle_separation_metric*(np.power(num_particles,(1/3)) - 1) + particle_diameter
        # Lx = minimum_dimension*2
        # Ly = Lx
        # Lz = Ly
        # current_total_volume = Lx*Ly*Lz
        # current_volume_fraction = num_particles*particle_volume/current_total_volume
        # Lx = fictional_side_length
        # Ly = fictional_side_length
        # Lz = fictional_side_length
        fictional_single_particle_volume = particle_volume/volume_fraction
        single_particle_side_length = np.power(fictional_single_particle_volume,(1/3))
        if 'anisotropic' in particle_placement:
            Lx = num_particles_along_axes[0]*single_particle_side_length*anisotropy_factor[0]
            Ly = num_particles_along_axes[1]*single_particle_side_length*anisotropy_factor[1]
            Lz = num_particles_along_axes[2]*single_particle_side_length*anisotropy_factor[2]    
        elif num_particles != 0:
            Lx = num_particles_along_axes[0]*single_particle_side_length
            Ly = num_particles_along_axes[1]*single_particle_side_length
            Lz = num_particles_along_axes[2]*single_particle_side_length
        separation_volume_elements = int(np.round(particle_separation_metric / l_e,decimals=1))
        separation_meters = l_e*separation_volume_elements
    else:
        raise NotImplementedError


    t_f = 300
    
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)

    # approximately 7 significant digits in decimal representation for 32bit floating point numbers. assuming that the postions are of order of magnitude 10^1, we can take into account the time step size and approximately calculate the smallest velocity that would still increment a position value of 10, or the largest velocity small enough that it would not increment a position value of 10. that should be our tolerance on the velocity norm (technically the velocity component magnitude). For the acceleration norm (or more significantly the component magnitude), we could have a separate tolerance based on the time step size and some magnitude of velocity, intuitively it would be the largest velocity small enough that in would not increment the position. the acceleration tolerance follows a similar logic to the velocity tolerance, if the time step size and the acceleration product would result in a value too small to incrememnt the floating point representation of the relevant velocity magnitude, it is effectively zero, and a lower bound on what our tolerance should be to consider the system converged (since it will be incapable of evolving if all nodes are experiencing velocities low enough to not cause position updates and accelerations low enough to not cause velocity updates.)
    #if the position value is 10., we need a velocity*time_step value that is at least 1e-6 (ref: https://float.exposed/, addtl terms like floating-point arithmetic, dynamic range, and "unit in the last place" or "unit of least precision")
    # 1e-6 = nearly_effectively_zero_velocity*time_step 
    #if i can find the position values of nodes, or estimate them, and their counts, i can get a sense for the effective_zero_velocity and average those values... if the magnitude of the component of velocity along a direction is close to the ulp average for that direction, the system has stopped evolving in that direction

    #using numpy.nextafter and ensuring 32 bit floats, can determine the ulp for incrementing
    # ulp = np.nextafter(np.float32(1),np.float32(np.Inf))-np.float32(1)
    # effective_zero_velocity = (1e-6)/step_size
    # if the effectively zero velocity couldn't update... but this is going to be a significantly smaller tolerance than the velocity tolerance, and likely more stringent than what i am currently using
    # effective_zero_velocity*1e-6 = effective_zero_acceleration*time_step
    # effective_zero_acceleration = (effective_zero_velocity*1e-6)/step_size
    # parameters['tolerance'] = effective_zero_velocity
    #if i have the position values of nodes, or estimate them, and their counts, i can get a sense for the effective_zero_velocity and average those values... if the magnitude of the component of velocity along a direction is close to the ulp average for that direction, the system has stopped evolving in that direction

    unique_values, unique_value_counts = np.unique(np.ravel(normalized_posns),return_counts=True)
    ulp = np.zeros(unique_values.shape,dtype=np.float32)
    for index, value in enumerate(unique_values):
        #using numpy.nextafter and ensuring 32 bit floats, can determine the ulp for incrementing
        ulp[index] = np.float32(np.nextafter(np.float32(value),np.float32(np.Inf))-np.float32(value))
    step_size = parameters['step_size']
    ulp_velocity = ulp/step_size
    weighted_ulp_velocity = ulp_velocity*unique_value_counts
    tolerance = np.float32(np.sum(weighted_ulp_velocity)/np.sum(unique_value_counts))

    #2024-03-29, commenting out the below. useful for making more granular tolerance values, but not necessary. future implementations would require checking velocity and acceleration component magnitudes against tolerance values for each cartesian axis.
    #same concept, but for each coordinate separately
    # unique_values, unique_value_counts = np.unique(normalized_posns[:,0],return_counts=True)
    # ulp = np.zeros(unique_values.shape,dtype=np.float32)
    # for index, value in enumerate(unique_values):
    #     #using numpy.nextafter and ensuring 32 bit floats, can determine the ulp for incrementing
    #     ulp[index] = np.float32(np.nextafter(np.float32(value),np.float32(np.Inf))-np.float32(value))
    # step_size = parameters['step_size']
    # ulp_velocity = ulp/step_size
    # weighted_ulp_velocity = ulp_velocity*unique_value_counts
    # x_tolerance = np.float32(np.sum(weighted_ulp_velocity)/np.sum(unique_value_counts))

    # unique_values, unique_value_counts = np.unique(normalized_posns[:,1],return_counts=True)
    # ulp = np.zeros(unique_values.shape,dtype=np.float32)
    # for index, value in enumerate(unique_values):
    #     #using numpy.nextafter and ensuring 32 bit floats, can determine the ulp for incrementing
    #     ulp[index] = np.float32(np.nextafter(np.float32(value),np.float32(np.Inf))-np.float32(value))
    # step_size = parameters['step_size']
    # ulp_velocity = ulp/step_size
    # weighted_ulp_velocity = ulp_velocity*unique_value_counts
    # y_tolerance = np.float32(np.sum(weighted_ulp_velocity)/np.sum(unique_value_counts))

    # unique_values, unique_value_counts = np.unique(normalized_posns[:,2],return_counts=True)
    # ulp = np.zeros(unique_values.shape,dtype=np.float32)
    # for index, value in enumerate(unique_values):
    #     #using numpy.nextafter and ensuring 32 bit floats, can determine the ulp for incrementing
    #     ulp[index] = np.float32(np.nextafter(np.float32(value),np.float32(np.Inf))-np.float32(value))
    # step_size = parameters['step_size']
    # ulp_velocity = ulp/step_size
    # weighted_ulp_velocity = ulp_velocity*unique_value_counts
    # z_tolerance = np.float32(np.sum(weighted_ulp_velocity)/np.sum(unique_value_counts))

    sim_variables_dict['tolerance'] = tolerance
    # get the actual system dimensions now that the number of nodes and elements has been set
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    k_particles = mre.initialize.get_spring_constants(youngs_modulus_particles,l_e)
    kappa_particles = mre.initialize.get_kappa(youngs_modulus_particles,nu_particles)
    if num_particles == 0:
        particles = np.zeros((0,1),dtype=np.int64)
    elif 'hand' in particle_placement:
        particle_posns = parameters['particle_posns']
        particles = place_particles_normalized_by_hand(particle_radius,l_e,normalized_dimensions,particle_posns)
    elif particle_placement == 'regular':
        particles = place_n_particles_normalized(num_particles,particle_radius,l_e,normalized_dimensions,separation_volume_elements,particle_placement,num_particles_along_axes)
    # elif num_particles == 2:
    #     particles = place_two_particles_normalized(particle_radius,l_e,normalized_dimensions,separation_volume_elements)
    elif num_particles == 1:
        print(f'implement single particle placement')
        raise NotImplementedError(f'implement single particle placement')
    elif num_particles == 3:
        particles = place_n_particles_normalized(num_particles,particle_radius,l_e,normalized_dimensions,separation_volume_elements)
    elif particle_placement == 'regular_noisy':
        particles, seed = place_n_particles_normalized(num_particles,particle_radius,l_e,normalized_dimensions,separation_volume_elements,particle_placement,num_particles_along_axes)
    elif particle_placement == 'regular_anisotropic':
        particles = place_n_particles_normalized(num_particles,particle_radius,l_e,normalized_dimensions,separation_volume_elements,particle_placement,num_particles_along_axes,anisotropy_factor)
    elif particle_placement == 'regular_anisotropic_noisy':
        particles,seed = place_n_particles_normalized(num_particles,particle_radius,l_e,normalized_dimensions,separation_volume_elements,particle_placement,num_particles_along_axes,anisotropy_factor)
    else:
        print(f'implement multi-particle placement')
        raise NotImplementedError(f'implement multi-particle placement')
    #DMarchfield 2024-06-18
    #testing to see if i can determine the particle position from the nodes that are the first 8 entries of the particles row
    # nodes_per_particle = particles.shape[1]
    # device_particles = cp.asarray(cp.reshape(particles,(particles.shape[0]*particles.shape[1],1)),dtype=cp.int32,order='C')
    # device_posns = cp.asarray(cp.reshape(normalized_posns,(normalized_posns.shape[0]*normalized_posns.shape[1],1)),dtype=cp.float32,order='C')
    # particle_posns = simulate.get_particle_posns_gpu(device_posns,device_particles,num_particles,nodes_per_particle)
    # host_gpu_particle_posns = cp.asnumpy(particle_posns).reshape((num_particles,3))
    # for particle_count, particle in enumerate(particles):
    #     true_particle_position = simulate.get_particle_center(particle,normalized_posns)
    #     trial_particle_position = np.array([np.mean(normalized_posns[particle[:8],0]),np.mean(normalized_posns[particle[:8],1]),np.mean(normalized_posns[particle[:8],2])])
    #     print(f'True posn: {true_particle_position}')
    #     print(f'Trial posn: {trial_particle_position}')
    #     print(f'Same value position?: {np.allclose(true_particle_position,trial_particle_position)}')
    #     if not np.allclose(true_particle_position,trial_particle_position):
    #         print('New method to find particle position failed')
    #     print(f'gpu posn:{host_gpu_particle_posns[particle_count]}')
    #     print(f'Same value from gpu?:{np.allclose(true_particle_position,host_gpu_particle_posns[particle_count])}')
    #     if not np.allclose(true_particle_position,host_gpu_particle_posns[particle_count]):
    #         print('New gpu method to find particle position failed')

    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    # if num_particles == 0:
    #     node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    # else:
    node_types = springs.get_node_type_normalized_v2(normalized_posns.shape[0],boundaries,dimensions_normalized,particles)
    k = np.array(k,dtype=np.float64)
    k_particles = np.array(k_particles,dtype=np.float64)
    k_e = k[0]
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    effective_stiffness = 2*k[0] + 8*k[1]/np.sqrt(2) + 4*k[2]
    if gpu_flag:
        k *= l_e
        k_particles *= l_e
    # num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    num_springs = springs.get_springs_v2(node_types, springs_var, max_springs, k, dimensions_normalized, 1, k_particles)
    springs_var = springs_var[:num_springs,:]
    print(f'number of unique spring constant values:{np.unique(springs_var[:,2]).shape[0]}')
    print(f'unique spring constant values:{np.unique(springs_var[:,2].astype(np.float32))}')
    # print(f'unique spring values for subset of springs containing at least one particle node:{np.unique(springs_var[np.equal(springs_var[:,0].astype(np.int64),particles[0,0]),2])}')
    chi = 70#131#70
    Ms = 1.6e6#1.9e6

    field_angle_theta = parameters['field_angle_theta']
    field_angle_phi = parameters['field_angle_phi']
    num_field_steps = parameters['num_field_steps']
    near_zero_field = 1e-4
    near_zero_H_field = near_zero_field/mu0
    if 'Hext_series' in parameters:
        Hext_series = parameters['Hext_series']
    else:
        if num_field_steps == 0:
            H_step = max_magnetic_field_strength + 1
        else:
            H_step = max_magnetic_field_strength/num_field_steps
        if 'Hext_series_magnitude' in parameters:
            Hext_series_magnitude = parameters['Hext_series_magnitude']
        elif max_magnetic_field_strength == 0:
            Hext_series_magnitude = np.array([0.0])
        else:
            Hext_series_magnitude = np.arange(0.0,max_magnetic_field_strength + 1,H_step)
        # Hext_series_magnitude[0] = near_zero_H_field
        if 'hysteresis' in sim_type:
            Hext_series_magnitude = np.append(Hext_series_magnitude,Hext_series_magnitude[-2::-1])        
        Hext_series = np.zeros((len(Hext_series_magnitude),3))
        Hext_series[:,0] = Hext_series_magnitude*np.cos(field_angle_phi)*np.sin(field_angle_theta)
        Hext_series[:,1] = Hext_series_magnitude*np.sin(field_angle_phi)*np.sin(field_angle_theta)
        Hext_series[:,2] = Hext_series_magnitude*np.cos(field_angle_theta)
    Bext_angle = np.zeros((2,))
    Bext_angle[0] = field_angle_theta*180/np.pi
    Bext_angle[1] = field_angle_phi*180/np.pi
    # if num_field_steps == 0 or max_magnetic_field_strength == 0:
    #     Bext_angle = None
    # elif Hext_series.shape[0] > 1 and np.linalg.norm(Hext_series[1]) != 0:
    #     Bext_angle = np.zeros((2,))
    #     try:
    #         Bext_angle[0] = np.round(np.arctan(np.linalg.norm(Hext_series[1,:2])/np.abs(Hext_series[1,2]))*180/np.pi,decimals=0)
    #     except RuntimeWarning:
    #         Bext_angle[0] = 90
    #     try:
    #         Bext_angle[1] = np.round(np.arctan(np.abs(Hext_series[1,1])/np.abs(Hext_series[1,0]))*180/np.pi,decimals=0)
    #     except RuntimeWarning:
    #         Bext_angle[1] = 90
    # #     Bext_angle = np.arctan(Hext_series[1,1]/Hext_series[1,0])*180/np.pi
    # # else:
    # #     Bext_angle = 90

    if 'stress' in sim_type:
        if 'boundary_condition_value_series' in parameters:
            stresses = parameters['boundary_condition_value_series']
        else:
            if num_boundary_condition_steps < 1:
                raise ValueError('Number of boundary condition steps must be greater than 0')
            elif num_boundary_condition_steps == 1:
                boundary_condition_value_step = max_boundary_condition_value
            else:
                boundary_condition_value_step = max_boundary_condition_value/num_boundary_condition_steps
            if max_boundary_condition_value == 0:
                stresses = np.array([0.0],dtype=np.float64)
            else:
                stresses = np.arange(0.0,max_boundary_condition_value*1.01,boundary_condition_value_step)
        if 'compression' in sim_type:
            stresses *= -1
        elif 'tension' in sim_type:
            pass
        elif 'shearing' in sim_type:
            pass
        # raise NotImplementedError(f'Implement stress based simulation initialization specifics')
    if 'strain' in sim_type:
        if 'boundary_condition_value_series' in parameters:
            strains = parameters['boundary_condition_value_series']
        else:
            if num_boundary_condition_steps < 1:
                raise ValueError('Number of boundary condition steps must be greater than 0')
            elif num_boundary_condition_steps == 1:
                boundary_condition_value_step = max_boundary_condition_value
            else:
                boundary_condition_value_step = max_boundary_condition_value/num_boundary_condition_steps
            if max_boundary_condition_value == 0:
                strains = np.array([0.0],dtype=np.float64)
            else:
                strains = np.arange(0.0,max_boundary_condition_value*1.01,boundary_condition_value_step)
        if 'compression' in sim_type:
            strains *= -1
        elif 'tension' in sim_type:
            pass
        elif 'shearing' in sim_type:
            pass
        # raise NotImplementedError(f'Implement strain based simulation initialization specifics')

    # check if the directory for output exists, if not make the directory
    # current_dir = os.path.abspath('.')
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_radius)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    #trying this with an effective stiffness that i am intuiting to be a linear combination of the 3 spring stiffness types, with the characteristic stiffness being that which an interior volume node (not a magnetic particle node) would experience. the number of springs and angle with respect to the x axis is informing my choice of coefficients for each spring type. there are 6 edge springs, of which at least one pair if not two will not be contributing. imagine a displacement in the xy plane. the springs oriented along the z axis will not significantly contribute, and as we approximate the stiffness we can negelect their contribution. if the displacement is along the x direction, the same is true for the springs along the y axis. since there is a spring in front of and behind the node, there is the equivalent stiffness of two edge springs. for an arbitrary direction in the xy-plane the contribution from springs in the x direction depends on cosine of the angle wrt the x axis, and the contribution from the y direction springs depends on sine of the  angle wrt the x axis, so that for those spring types, the effective stiffness contribution for small displacements is approximated as twice the stiffness of a single edge spring. Arguments for the face springs require projection onto the z-axis or onto the xy-plane, but similar arguments apply. in this case there are 4 face springs in each of the xy, xz, and yz- planes. considering a displacement along the x axis the springs in the yz-plane will contribute the least to the effective stiffness, leaving 8 springs whose contribution to the effective stiffness is equal, but we can project each spring onto the x axis as sine or cosine of 45 degrees, resulting in 8/sqrt(2) k_f contribution to the effective stiffness. the center diagonal springs, of which there are 8 total connected to the node, can all be projected into the xy-plane by multiplying by cosine 45, 8/sqrt(2), followed by an additional projection onto the x-axis, resulting in a contribution of 8/2 or 4* k_c.
    #already found an issue, I am assuming these things are all additive, but maybe that isn't the case. the rules for adding springs in series vs parallel are similar to that of a capacitor/resistor. if the springs are in parallel, the stiffnesses add, if the springs were in series, they would have the reciprocal of the effective stiffness equal to the sum of the reciprocal individual stiffnesses. i do think they should all be in parallel...
    #found the actual issue, I was using the scaled stiffness values instead of the SI values, moving the commented out line further aboev, before the scaling by l_e occurs
    # effective_stiffness = 2*k[0] + 8*k[1]/np.sqrt(2) + 4*k[2]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/effective_stiffness)#2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    if (not np.isclose(k_e,0)):
        beta = np.power(characteristic_time,2)/l_e
        #beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    else:
        beta = 1e-9
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    
    #2024-04-09 DBM: debugging applied stress boundary conditions to determine a lower bound on applied stress that should actually result in a deformation
    # stressed_nodes = boundaries['right']
    # print(f'ulp starting at zero {np.float32(np.nextafter(np.float32(0.0),np.float32(np.Inf))-np.float32(0.0))}')
    # print(f'beta_i values on the boundary: {np.unique(beta_i[stressed_nodes])}')
    # surface_area = dimensions[2]*dimensions[1]
    # for stress_val in stresses:
    #     net_force_mag = stress_val*surface_area
    #     single_node_accel_values = np.unique(np.squeeze((net_force_mag/stressed_nodes.shape[0])*beta_i[stressed_nodes]))
    #     single_node_accel_max = np.max(single_node_accel_values)
    #     print(f'single node acceleration values for the boundaries: {single_node_accel_values}')
    #     if single_node_accel_max < tolerance:
    #         print(f'stress {stress_val} results in single node acceleration lower than convergence criteria')
    #     else:
    #         print(f'stress {stress_val} results in single node acceleration exceeding convergence criteria')
    #     print(f'with {max_integration_steps} integration steps and a step size of {step_size}, the change in velocity for the ndoes over one integration round is {single_node_accel_values*step_size*max_integration_steps}')
    #     print(f'change in velocity each step {single_node_accel_values*step_size}')

    today = date.today()
    if 'hysteresis' in sim_type:
        # output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_{num_particles}_particle_hysteresis_order_{discretization_order}_E_{E}_nu_{nu}_Bext_angles_{np.round(Bext_angle[0])}_{np.round(Bext_angle[1])}_{particle_placement}_vol_frac_{volume_fraction}/'
        sim_type_string = 'hysteresis'
        field_or_bc_series = Hext_series
        field_or_bc_type_string = 'hysteresis'
    elif 'stress' in sim_type:
        # output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_{num_particles}_particle_field_dependent_modulus_stress_{bc_type}_direction{bc_direction}_order_{discretization_order}_E_{E}_nu_{nu}_drag_{drag}_Bext_angles_{np.round(Bext_angle[0])}_{np.round(Bext_angle[1])}_{particle_placement}_vol_frac_{volume_fraction}/'
        sim_type_string = 'field_dependent_modulus' + f'_{bc_type}_direction{bc_direction}'
        field_or_bc_series = stresses
        field_or_bc_type_string = bc_type
    elif 'strain' in sim_type:
        # output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_{num_particles}_particle_field_dependent_modulus_{bc_type}_direction{bc_direction}_order_{discretization_order}_E_{E}_nu_{nu}_Bext_angles_{np.round(Bext_angle[0])}_{np.round(Bext_angle[1])}_{particle_placement}_vol_frac_{volume_fraction}/'
        sim_type_string = 'field_dependent_modulus' + f'_{bc_type}_direction{bc_direction}'
        field_or_bc_series = strains
        field_or_bc_type_string = bc_type
    else:
        raise ValueError(f'{sim_type} is not an acceptable sim_type value')
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/{today.isoformat()}_{num_particles}_particle_{sim_type_string}_order_{discretization_order}_E_{np.format_float_scientific(E,exp_digits=2)}_nu_{nu}/'
    if num_particles != 0:
        output_dir = output_dir[:-1] + f'_Bext_angles_{np.round(Bext_angle[0])}_{np.round(Bext_angle[1])}_{particle_placement}_vol_frac_{np.format_float_scientific(volume_fraction,exp_digits=1)}/'
    elif 'dimensions' in parameters:
        output_dir = output_dir[:-1] + f"_{parameters['dimensions']}/"
    if 'noisy' in particle_placement:
        output_dir = output_dir[:-1] + f'_starttime_{time.strftime("%H-%M",time.localtime())}/'
    if gpu_flag:
        step_size = parameters['step_size']
        output_dir = output_dir[:-1] + f"_stepsize_{np.format_float_scientific(parameters['step_size'],exp_digits=1)}/"
    else:
        step_size = None
    sim_variables_dict['output_dir'] = output_dir
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    mre.analyze.plot_particle_nodes(normalized_posns,normalized_posns,l_e,particles,output_dir,tag=f"initial_particle_placement")

    my_sim = mre.initialize.Simulation(E,nu,kappa,k,drag,l_e,Lx,Ly,Lz,particle_radius,particle_mass,Ms,chi,beta,characteristic_mass,characteristic_time,max_integrations,max_integration_steps,step_size,tolerance,anisotropy_factor)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    if particle_placement == 'regular_noisy':
        my_sim.append_log(f'particle placement seed: {seed}',output_dir)
    mre.initialize.write_init_file(normalized_posns,m,springs_var,elements,particles,boundaries,my_sim,Hext_series,field_or_bc_series,sim_type,output_dir)

    if gpu_flag:
        x0 = cp.array(normalized_posns.astype(np.float32)).reshape((normalized_posns.shape[0]*normalized_posns.shape[1],1),order='C')
        beta_i = cp.array(beta_i.astype(np.float32)).reshape((beta_i.shape[0],1),order='C')
        beta = np.float32(beta)
        drag = np.float32(drag)
        if 'stress' in sim_type:
            stresses = np.float32(stresses)
        if 'strain' in sim_type:
            strains = np.float32(strains)
        Hext_series = np.float32(Hext_series)
        for key in boundaries:
            boundaries[key] = np.int32(boundaries[key])
        dimensions = np.float32(dimensions)
        kappa = cp.float32(kappa*(l_e**2))
        l_e = np.float32(l_e)
        particles = np.int32(particles)
        particle_radius = np.float32(particle_radius)
        particle_volume = np.float32(particle_volume)
        particle_mass = np.float32(particle_mass)
        chi = np.float32(chi)
        Ms = np.float32(Ms)
        elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
        springs_var = cp.array(springs_var.astype(np.float32)).reshape((springs_var.shape[0]*springs_var.shape[1],1),order='C')
        step_size = cp.float32(parameters['step_size'])#cp.float32(0.01)
    
    #packing the necessary arguments into the dictionary
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

    if 'stress' in sim_type:
        sim_variables_dict['stresses'] = stresses
    elif 'strain' in sim_type:
        sim_variables_dict['strains'] = strains
    # particle_rotation_flag = sim_variables_dict['particle_rotation_flag']
    # persistent_checkpointing_flag = sim_variables_dict['persistent_checkpointing_flag']
    # plotting_flag = sim_variables_dict['plotting_flag']
    # criteria_flag = sim_variables_dict['criteria_flag']
    
    if gpu_flag:
        sim_variables_dict['step_size'] = step_size
    else:
        sim_variables_dict['t_f'] = t_f
    return sim_variables_dict, my_sim

def unpack_sim_variables(sim_variables_dict):
    """Does what it says on the label. For internal use only."""
    output_dir = sim_variables_dict['output_dir']
    x0 = sim_variables_dict['initial_posns']

    bc_type = sim_variables_dict['boundary_condition_type']
    bc_direction = sim_variables_dict['boundary_condition_direction']

    Hext_series = sim_variables_dict['Hext_series']

    springs_var = sim_variables_dict['springs']
    elements = sim_variables_dict['elements']
    dimensions = sim_variables_dict['dimensions']
    boundaries = sim_variables_dict['boundaries']
    kappa = sim_variables_dict['kappa']
    l_e = sim_variables_dict['element_length']
    beta = sim_variables_dict['beta']
    beta_i = sim_variables_dict['beta_i']
    drag = sim_variables_dict['drag']

    particles = sim_variables_dict['particles']
    host_particles = sim_variables_dict['particles']
    particles = cp.asarray(cp.reshape(host_particles,(host_particles.shape[0]*host_particles.shape[1],1)),dtype=cp.int32,order='C')
    particle_radius = sim_variables_dict['particle_radius']
    particle_volume = sim_variables_dict['particle_volume']
    particle_mass = sim_variables_dict['particle_mass']
    chi = sim_variables_dict['chi']
    Ms = sim_variables_dict['Ms']

    max_integrations = sim_variables_dict['max_integrations']
    max_integration_steps = sim_variables_dict['max_integration_steps']
    tolerance = sim_variables_dict['tolerance']

    gpu_flag = sim_variables_dict['gpu_flag']
    particle_rotation_flag = sim_variables_dict['particle_rotation_flag']
    persistent_checkpointing_flag = sim_variables_dict['persistent_checkpointing_flag']
    plotting_flag = sim_variables_dict['plotting_flag']
    criteria_flag = sim_variables_dict['criteria_flag']

    step_size = sim_variables_dict['step_size']

    return output_dir, x0, bc_type, bc_direction, Hext_series, springs_var, elements, dimensions, boundaries, kappa, l_e, beta, beta_i, drag, particles, host_particles, particle_radius, particle_volume, particle_mass, chi, Ms, max_integrations, max_integration_steps, tolerance, gpu_flag, particle_rotation_flag, persistent_checkpointing_flag, plotting_flag, criteria_flag, step_size

def run_hysteresis_sim(sim_variables_dict,sim_restart_flag=False,sim_extend_flag=False,sim_rerun_flag=False):#(output_dir,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,particle_rotation_flag=False,gpu_flag=False,step_size=np.float32(0.005)):

    output_dir, x0, bc_type, bc_direction, Hext_series, springs_var, elements, dimensions, boundaries, kappa, l_e, beta, beta_i, drag, particles, host_particles, particle_radius, particle_volume, particle_mass, chi, Ms, max_integrations, max_integration_steps, tolerance, gpu_flag, particle_rotation_flag, persistent_checkpointing_flag, plotting_flag, criteria_flag, step_size = unpack_sim_variables(sim_variables_dict)

    if gpu_flag:
        step_size = sim_variables_dict['step_size']
    else:
        t_f = sim_variables_dict['t_f']

    if sim_restart_flag:
        # TODO Modify this code block from restarting an interrupted strain simulation to handle restarting an interrupted hysteresis simulation.
        # this will also require modifying the iteration over applied fields line 'for count, Hext in enumerate(Hext_series):""
        continuation_index = sim_variables_dict['continuation_index']
        #the number of output files tells me how many field steps I've completed, and which step i am on
        field_start_index = continuation_index

        #need to track total simulation time, so read in how long the other sims took
        total_delta = 0
        for i in range(continuation_index):
            _, _, _, _, sim_time = mre.initialize.read_output_file(output_dir+f'output_{i}.h5')
            total_delta += sim_time

        # find the correct subdirectory to load in the checkpoint file
        Hext = Hext_series[field_start_index]
        if output_dir[-1] != '/':
            current_output_dir = output_dir + f'/field_{field_start_index}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        elif output_dir[-1] == '/':
            current_output_dir = output_dir + f'field_{field_start_index}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        if not (os.path.isdir(current_output_dir)):
            os.mkdir(current_output_dir)
        # read in the checkpoint file and continue the simulation
        num_checkpoint_files = mre.analyze.get_num_named_files(current_output_dir,'checkpoint')
        #if there are checkpoint files, use them, if not...
        if num_checkpoint_files != 0:
            checkpoint_offset = num_checkpoint_files
            solution, normalized_magnetization, _, boundary_conditions, _ = mre.initialize.read_checkpoint_file(current_output_dir+f'checkpoint{num_checkpoint_files-1}.h5')
            boundary_conditions = format_boundary_conditions(boundary_conditions)
            N_nodes = int(solution.shape[0]/6)
            posn_soln = solution[:3*N_nodes]
            x0 = cp.array(posn_soln.astype(np.float32)).reshape((posn_soln.shape[0]*posn_soln.shape[1],1),order='C')
            starting_velocities = solution[3*N_nodes:]
            starting_velocities = cp.array(starting_velocities.astype(np.float32)).reshape((starting_velocities.shape[0]*starting_velocities.shape[1],1),order='C')
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
        else:
            checkpoint_offset = 0
            posns, normalized_magnetization, _, _, _ = mre.initialize.read_output_file(output_dir+f'output_{continuation_index-1}.h5')
            x0 = posns 
            if gpu_flag:
                x0 = cp.array(x0.astype(np.float32)).reshape((x0.shape[0]*x0.shape[1],1),order='C')
                normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
            starting_velocities = None
        print(f'Restarting interrupted simulation at step {continuation_index+1} out of {Hext_series.shape[0]}.')
    else:
        eq_posns = x0.copy()
        field_start_index = 0
        total_delta = 0
        starting_velocities = None
        checkpoint_offset = 0
        normalized_magnetization = None
    for count in range(field_start_index,Hext_series.shape[0]):# for count, Hext in enumerate(Hext_series):
        Hext = Hext_series[count]
        boundary_conditions = ('hysteresis',('free','free'),0) 
        if output_dir[-1] != '/':
            current_output_dir = output_dir + f'/field_{count}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        elif output_dir[-1] == '/':
            current_output_dir = output_dir + f'field_{count}_Bext_{np.round(np.linalg.norm(Hext)*mu0,decimals=3)}/'
        if not (os.path.isdir(current_output_dir)):
            os.mkdir(current_output_dir)
        try:
            print(f'Running simulation with external magnetic field: ({np.round(Hext[0]*mu0,decimals=3)}, {np.round(Hext[1]*mu0,decimals=3)}, {np.round(Hext[2]*mu0,decimals=3)}) T\n')
            start = time.time()
            if not particle_rotation_flag:
                sol, return_status = simulate.simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
            elif particle_rotation_flag and (not gpu_flag):
                sol, return_status = simulate.simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,eq_posns,current_output_dir,max_integrations,max_integration_steps,tolerance,criteria_flag,plotting_flag,persistent_checkpointing_flag)
            elif gpu_flag:
                sol, normalized_magnetization, return_status = simulate.simulate_scaled_gpu_leapfrog_v3(x0,elements,host_particles,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag,starting_velocities,checkpoint_offset,normalized_magnetization=normalized_magnetization)
                #if there was a continuation, after that first simulation I don't want to seed starting velocities erroneously or have an offset the checkpoint numbering
                starting_velocities = None
                checkpoint_offset = 0
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
        except simulate.FixedPointMethodError as e:
                print(f'Fixed point method failed at {np.round(mu0*Hext*1000,decimals=1)} mT')
                raise
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
            raise
        end = time.time()
        delta = end - start
        print('took %.2f seconds to simulate' % delta)
        total_delta += delta
        if not gpu_flag:
            print('this conditional path is deprecated and should be removed, gpu implementation only')
            raise
            end_result = sol
            x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
            mre.initialize.write_output_file(count,x0,Hext,boundary_conditions,delta,output_dir)
        else:
            final_posns = sol[:int(sol.shape[0]/2)]
            N_nodes = int(sol.shape[0]/6)
            final_posns = np.reshape(final_posns,(N_nodes,3))
            mre.initialize.write_output_file(count,final_posns,normalized_magnetization,Hext,boundary_conditions,delta,output_dir)
            x0 = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
    return total_delta, return_status

def run_stress_sim(sim_variables_dict,sim_restart_flag=False,sim_extend_flag=False,sim_rerun_flag=False):#(output_dir,bc_type,bc_direction,stresses,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation applying a series of a particular type of stress to the volume with a series of applied external magnetic fields, by passing the boundary condition type as a string (one of the following: simple_stress_* where * is one of tension/compression/shearing), the stress direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the stress/strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing strain the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""

    output_dir, x0, bc_type, bc_direction, Hext_series, springs_var, elements, dimensions, boundaries, kappa, l_e, beta, beta_i, drag, particles, host_particles, particle_radius, particle_volume, particle_mass, chi, Ms, max_integrations, max_integration_steps, tolerance, gpu_flag, particle_rotation_flag, persistent_checkpointing_flag, plotting_flag, criteria_flag, step_size = unpack_sim_variables(sim_variables_dict)
    stresses = sim_variables_dict['stresses']
    normalized_magnetization = None
    total_delta = 0
    for count, stress in enumerate(stresses):
        boundary_conditions = (bc_type,(bc_direction[0],bc_direction[1]),stress)
        for i, Hext in enumerate(Hext_series):
            if output_dir[-1] != '/':
                current_output_dir = output_dir + f'/{bc_type}_{count}_{np.format_float_scientific(stress,precision=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            elif output_dir[-1] == '/':
                current_output_dir = output_dir + f'{bc_type}_{count}_{np.format_float_scientific(stress,precision=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            if not (os.path.isdir(current_output_dir)):
                os.mkdir(current_output_dir)
            print(f'Running simulation with external magnetic field: ({np.round(Hext[0]*mu0,decimals=3)}, {np.round(Hext[1]*mu0,decimals=3)}, {np.round(Hext[2]*mu0,decimals=3)}) T\nApplied stress {boundary_conditions[1]} {np.format_float_scientific(stress,precision=5)}\n')
            start = time.time()
            try:
                sol, normalized_magnetization, return_status = simulate.simulate_scaled_gpu_leapfrog_v3(x0,elements,host_particles,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag,normalized_magnetization=normalized_magnetization)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
            except simulate.FixedPointMethodError as e:
                print(f'Fixed point method failed at {np.round(mu0*Hext*1000,decimals=1)} mT with strain {np.round(stress,decimals=5)}')
                raise
            except Exception as inst:
                print('Exception raised during simulation')
                print(type(inst))
                print(inst)
                raise
            end = time.time()
            delta = end - start
            total_delta += delta
            final_posns = sol[:int(sol.shape[0]/2)]
            N_nodes = int(sol.shape[0]/6)
            final_posns = np.reshape(final_posns,(N_nodes,3))
            print('took %.2f seconds to simulate' % delta)
            output_file_number = count*Hext_series.shape[0]+i
            mre.initialize.write_output_file(output_file_number,final_posns,normalized_magnetization,Hext,boundary_conditions,delta,output_dir)
            #if we have already run a particular simulation with zero stress/strain and at some field, use that as the starting point for the solution
            if Hext_series.shape[0] > 1 and (output_file_number >= (Hext_series.shape[0]-1)) and (output_file_number < Hext_series.shape[0]*len(stresses)-1):
                output_file_num_to_reuse = output_file_number-(Hext_series.shape[0]-1)
                x0, normalized_magnetization, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
                print(f'reusing previously calculated solution with B_ext = {np.round(mu0*output_file_Hext,decimals=3)}')
                x0 = cp.array(x0.astype(np.float32)).reshape((x0.shape[0]*x0.shape[1],1),order='C')
                normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
            else:#use the last solution vector of positions as the starting set of positions for the next step
                x0 = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
                normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
    return total_delta, return_status

def run_strain_sim(sim_variables_dict,sim_restart_flag=False,sim_extend_flag=False,sim_rerun_flag=False):#(output_dir,bc_type,bc_direction,stresses,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation applying a series of a particular type of stress/strain to the volume with a series of applied external magnetic fields, by passing the boundary condition type as a string (one of the following: tension, compression, shearing, torsion, or simple_stress_* where * is one of tension/compression/shearing), the stress/strain direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the stress/strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing strain the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""
    output_dir, x0, bc_type, bc_direction, Hext_series, springs_var, elements, dimensions, boundaries, kappa, l_e, beta, beta_i, drag, particles, host_particles, particle_radius, particle_volume, particle_mass, chi, Ms, max_integrations, max_integration_steps, tolerance, gpu_flag, particle_rotation_flag, persistent_checkpointing_flag, plotting_flag, criteria_flag, step_size = unpack_sim_variables(sim_variables_dict)

    strains = sim_variables_dict['strains']
    normalized_magnetization = None
    if gpu_flag:
        x0 = cp.asnumpy(x0)
        N_nodes = int(x0.shape[0]/3)
        x0 = np.reshape(x0,(N_nodes,3))
    else:
        t_f = sim_variables_dict['t_f']

    if sim_restart_flag:
        continuation_index = sim_variables_dict['continuation_index']
        #if i've got as many output files as unique applied fields, I've completed the first set of rounds with a strain of 0, etc.
        strain_start_index = int(np.floor_divide(continuation_index,Hext_series.shape[0]))
        #by subtracting I end up with the number of output files for simulations at the current strain value, so i know what field step i am on
        field_start_index = int(continuation_index-strain_start_index*Hext_series.shape[0])

        #if we are past the zero-strain simulations, we need to know the reference configuration we should be using
        if strain_start_index != 0:
            output_file_num_for_reference_configuration = field_start_index
            eq_posns, _, _, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_for_reference_configuration}.h5')
        else:
            output_file_num_for_reference_configuration = 0
            eq_posns = x0.copy()
        total_delta = 0
        #need to track total simulation time, so read in how long the other sims took
        for i in range(continuation_index):
            _, _, _, _, sim_time = mre.initialize.read_output_file(output_dir+f'output_{i}.h5')
            total_delta += sim_time
        # find the correct subdirectory to load in the checkpoint file
        strain = strains[strain_start_index]
        Hext = Hext_series[field_start_index]
        if output_dir[-1] != '/':
            current_output_dir = output_dir + f'/{bc_type}_{strain_start_index}_{np.format_float_scientific(strain,precision=5)}_field_{field_start_index}_Bext_{np.round(Hext*mu0,decimals=3)}/'
        elif output_dir[-1] == '/':
            current_output_dir = output_dir + f'{bc_type}_{strain_start_index}_{np.format_float_scientific(strain,precision=5)}_field_{field_start_index}_Bext_{np.round(Hext*mu0,decimals=3)}/'
        if not (os.path.isdir(current_output_dir)):
            os.mkdir(current_output_dir)
        # read in the checkpoint file and continue the simulation
        num_checkpoint_files = mre.analyze.get_num_named_files(current_output_dir,'checkpoint')
        #if there are checkpoint files, use them, if not...
        if num_checkpoint_files != 0:
            checkpoint_offset = num_checkpoint_files
            solution, normalized_magnetization, _, boundary_conditions, _ = mre.initialize.read_checkpoint_file(current_output_dir+f'checkpoint{num_checkpoint_files-1}.h5')
            boundary_conditions = format_boundary_conditions(boundary_conditions)
            posn_soln = solution[:3*N_nodes]
            x0 = cp.array(posn_soln.astype(np.float32)).reshape((posn_soln.shape[0]*posn_soln.shape[1],1),order='C')
            reuse_solution_flag = False
            starting_velocities = solution[3*N_nodes:]
            starting_velocities = cp.array(starting_velocities.astype(np.float32)).reshape((starting_velocities.shape[0]*starting_velocities.shape[1],1),order='C')
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
        else:
            checkpoint_offset = 0
            posns, normalized_magnetization, _, _, _ = mre.initialize.read_output_file(output_dir+f'output_{continuation_index-1}.h5')
            x0 = posns 
            reuse_solution_flag = True
            starting_velocities = None
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
        print(f'Restarting interrupted simulation at step {continuation_index+1} out of {Hext_series.shape[0]*strains.shape[0]}.')
    elif sim_extend_flag:
        current_output_dir = sim_variables_dict['checkpoint_dir']
        # read in the checkpoint file and continue the simulation
        num_checkpoint_files = mre.analyze.get_num_named_files(current_output_dir,'checkpoint')
        #if there are checkpoint files, use them, if not...
        if num_checkpoint_files != 0:
            checkpoint_offset = num_checkpoint_files
            solution, normalized_magnetization, applied_field, boundary_conditions, _ = mre.initialize.read_checkpoint_file(current_output_dir+f'checkpoint{num_checkpoint_files-1}.h5')
            boundary_conditions = format_boundary_conditions(boundary_conditions)
            strain_start_index = np.nonzero(np.isclose(boundary_conditions[2],strains))[0][0]
            field_start_index = np.nonzero(np.isclose(np.linalg.norm(applied_field),np.linalg.norm(Hext_series,axis=1)))[0][0]
            posn_soln = solution[:3*N_nodes]
            x0 = cp.array(posn_soln.astype(np.float32)).reshape((posn_soln.shape[0]*posn_soln.shape[1],1),order='C')
            reuse_solution_flag = False
            starting_velocities = solution[3*N_nodes:]
            starting_velocities = cp.array(starting_velocities.astype(np.float32)).reshape((starting_velocities.shape[0]*starting_velocities.shape[1],1),order='C')
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
        else:
            raise NotImplementedError
    elif sim_rerun_flag:
        #if i am rerunning, i need to read in the appropriate output file. trying to enumerate the cases... assume that the entire simulation has "completed."
        # 1) first sim step, which means using the initial system, and so not using any output files at all, just the init file, or the init file contents, except that would be, in most cases, 0 strain/stress and zero field. i suppose for future proofing, I might as well do this
        # 2) zero strain sim step, which means reading in the output from the prior magnetic field, but still at zero strain.
        # 3) non-zero strain sim step, which means reading in the output from the same magnetic field, but the prior strain, and setting the correct eq_posns based on the correct reference configuration
        # these three cases should constitute all possibilities, but i will think on this to be sure
        current_output_dir = sim_variables_dict['checkpoint_dir']
        checkpoint_offset = 0
        starting_velocities = None
        solution, normalized_magnetization, applied_field, boundary_conditions, _ = mre.initialize.read_checkpoint_file(current_output_dir+f'checkpoint0.h5')
        boundary_conditions = format_boundary_conditions(boundary_conditions)
        strain_start_index = np.nonzero(np.isclose(boundary_conditions[2],strains))[0][0]
        field_start_index = np.nonzero(np.isclose(np.linalg.norm(applied_field),np.linalg.norm(Hext_series,axis=1)))[0][0]
        output_file_number = strain_start_index*Hext_series.shape[0]+field_start_index
        if output_file_number == 0:
            eq_posns = x0.copy()
            output_file_num_for_reference_configuration = 0
            total_delta = 0
            reuse_solution_flag = False
        elif boundary_conditions[2] == 0:
            eq_posns = x0.copy()
            posns, normalized_magnetization, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_number-1}.h5')
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
            x0 = posns 
            reuse_solution_flag = True
        else:
            output_file_num_for_reference_configuration = field_start_index
            eq_posns, _, _, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_for_reference_configuration}.h5')
            output_file_num_to_reuse = output_file_number-Hext_series.shape[0]
            posns, normalized_magnetization, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
            print(f'Rerunning simulation step {output_file_number} reusing previously calculated solution with B_ext = {np.round(mu0*output_file_Hext,decimals=4)}')
            x0 = posns 
            normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
            reuse_solution_flag = True
    else:
        strain_start_index = 0
        field_start_index = 0
        checkpoint_offset = 0
        eq_posns = x0.copy()
        output_file_num_for_reference_configuration = 0
        total_delta = 0
        reuse_solution_flag = False
        starting_velocities = None
    if sim_extend_flag or sim_rerun_flag:
        total_delta = 0
        num_output_files = mre.analyze.get_num_named_files(output_dir,'output')
        #need to track total simulation time, so read in how long the other sims took
        for i in range(num_output_files):
            _, _, _, _, sim_time = mre.initialize.read_output_file(output_dir+f'output_{i}.h5')
            try:
                total_delta += sim_time
            except:
                if sim_time.shape == (1,):
                    total_delta += sim_time[0]
                elif sim_time.shape == (1,1):
                    total_delta += sim_time[0][0]
                elif sim_time.shape == (1,1,1):
                    total_delta += sim_time[0][0][0]
            if i == strain_start_index*Hext_series.shape[0]+field_start_index and sim_extend_flag:
                extension_time_offset = sim_time
    #2024-05-03 D Marchfield:if i am allowing the boundary to move as a whole when the strain is zero, how do i ensure i set the boundary position properly for non-zero strain, using the correct reference configuration?
    # output_file_num_for_reference_configuration = 0
    for count in range(strain_start_index,strains.shape[0]):#for count, strain in enumerate(strains):
        strain = strains[count]
        #if we are applying our first strain, we aren't reusing a previously found solution. when we are reusing a previous solution we need to apply the new strain after we read that solution further down
        if count == 0 and checkpoint_offset == 0:#if the checkpoint offset is non-zero, we already have x0 and boundary conditions set up, having read them in from a checkpoint file
            #if the first strain is zero, and it should be in all cases, this won't actually do anything to the position values, but it does return the boundary conditions variable and the x0 variable as a cupy array on device memory
            x0, boundary_conditions = apply_strain_to_boundary(x0,eq_posns,boundaries,bc_type,bc_direction,strain,dimensions,l_e,gpu_flag)
        #if we are restarting a simulation, when we have completed a set of sims for a given strain value, we need to start counting the fields from 0 again
        if sim_restart_flag and count == strain_start_index + 1:
            field_start_index = 0
        for i in range(field_start_index,Hext_series.shape[0]):#for i, Hext in enumerate(Hext_series):
            Hext = Hext_series[i]
            if output_dir[-1] != '/':
                current_output_dir = output_dir + f'/{bc_type}_{count}_{np.format_float_scientific(strain,precision=5)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            elif output_dir[-1] == '/':
                current_output_dir = output_dir + f'{bc_type}_{count}_{np.format_float_scientific(strain,precision=5)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            # if output_dir[-1] != '/':
            #     current_output_dir = output_dir + f'/strain_{count}_{bc_type}_{np.round(strain,decimals=5)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            # elif output_dir[-1] == '/':
            #     current_output_dir = output_dir + f'strain_{count}_{bc_type}_{np.round(strain,decimals=5)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            if sim_extend_flag:
                current_output_dir = current_output_dir[:-1] + '_extension/'
            if not (os.path.isdir(current_output_dir)):
                os.mkdir(current_output_dir)
            if reuse_solution_flag and (type(x0) == type(np.array([0]))):
                #here i need to ensure what the "eq_posns" variable is more carefully now that I am trying to use simulation results as reference configurations for non-zero applied strains
                x0, boundary_conditions = apply_strain_to_boundary(x0,eq_posns,boundaries,bc_type,bc_direction,strain,dimensions,l_e,gpu_flag)
            print(f'Running simulation with external magnetic field: {np.round(Hext*mu0,decimals=4)} T\nApplied strain {boundary_conditions[1]} {np.format_float_scientific(strain,precision=5)}\n')
            start = time.time()
            try:
                sol, normalized_magnetization, return_status = simulate.simulate_scaled_gpu_leapfrog_v3(x0,elements,host_particles,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag,starting_velocities,checkpoint_offset,normalized_magnetization=normalized_magnetization)
                #if there was a continuation, after that first simulation I don't want to seed starting velocities erroneously or have an offset the checkpoint numbering
                starting_velocities = None
                checkpoint_offset = 0
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
            except simulate.FixedPointMethodError as e:
                print(f'Fixed point method failed at {np.round(mu0*Hext*1000,decimals=1)} mT with strain {np.round(strain,decimals=5)}')
                raise
            except Exception as inst:
                print('Exception raised during simulation')
                print(type(inst))
                print(inst)
                raise
            end = time.time()
            delta = end - start
            total_delta += delta
            if sim_extend_flag:
                delta += extension_time_offset
            final_posns = sol[:int(sol.shape[0]/2)]
            N_nodes = int(sol.shape[0]/6)
            final_posns = np.reshape(final_posns,(N_nodes,3))
            print('took %.2f seconds to simulate' % delta)
            output_file_number = count*Hext_series.shape[0]+i
            mre.initialize.write_output_file(output_file_number,final_posns,normalized_magnetization,Hext,boundary_conditions,delta,output_dir)
            #if there was an extension, want to stop after the one step
            if sim_extend_flag or sim_rerun_flag:
                return total_delta, return_status
            #if we have already run a particular simulation with zero (or non-zero) strain and at some field we have a configuration solution for, use that as the starting point for the next simulation
            if Hext_series.shape[0] > 1 and (output_file_number >= (Hext_series.shape[0]-1)) and (output_file_number < Hext_series.shape[0]*strains.shape[0]-1):
                output_file_num_to_reuse = output_file_number-(Hext_series.shape[0]-1)
                x0, normalized_magnetization, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
                print(f'reusing previously calculated solution with B_ext = {np.round(mu0*output_file_Hext,decimals=4)}')
                reuse_solution_flag = True
                normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
                print(f'using output_{output_file_num_for_reference_configuration}.h5 for reference configuration')
                eq_posns, _, _, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_for_reference_configuration}.h5')
                output_file_num_for_reference_configuration += 1
                if output_file_num_for_reference_configuration > Hext_series.shape[0] - 1:
                    output_file_num_for_reference_configuration = 0
                #the eq_posns variable is used to set the new, fixed positions of the strained boundary are, and i'm using the reference configuration of the system at the same external field to do so.
                # x0 = cp.array(x0.astype(np.float32)).reshape((x0.shape[0]*x0.shape[1],1),order='C')
            #2024-11-15 D Marchfield
            #i commented this out, because i think it is unnecessary. If i'm running a single applied field value, it is probably zero, but whether it is or not, I would want to use the previously found set of nodal positions to start off the next simulation step, which would be a different applied strain value, taking into account the initialized node configuration.
            # elif Hext_series.shape[0] == 1:
            #     output_file_num_to_reuse = 0
            #     x0, normalized_magnetization, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
            #     normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
            #     print(f'reusing previously calculated solution with B_ext = {np.round(mu0*output_file_Hext,decimals=4)}')
            #     reuse_solution_flag = True
            else:#use the last solution vector of positions as the starting set of positions for the next step
                x0 = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
                normalized_magnetization = cp.array(normalized_magnetization.astype(np.float32),order='C')
    return total_delta, return_status

def apply_strain_to_boundary(x0,eq_posns,boundaries,bc_type,bc_direction,strain,dimensions,l_e,gpu_flag=True):
    boundary_conditions = (bc_type,(bc_direction[0],bc_direction[1]),strain)
    if 'tension' in bc_type:
        if strain < 0:
            strain *= -1
        boundary_conditions = (bc_type,(bc_direction[0],bc_direction[0]),strain)
        if bc_direction[0] == 'x':
            x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
        elif bc_direction[0] == 'y':
            x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
        elif bc_direction[0] == 'z':
            x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
    elif 'compression' in bc_type:
        if strain >= 1 or strain <= -1:
            raise ValueError('For compressive strains, cannot exceed 100"%" strain')
        if strain > 0:
            strain *= -1
        boundary_conditions = (bc_type,(bc_direction[0],bc_direction[0]),strain)
        if bc_direction[0] == 'x':
            x0[boundaries['right'],0] = eq_posns[boundaries['right'],0] * (1 + strain)
        elif bc_direction[0] == 'y':
            x0[boundaries['back'],1] = eq_posns[boundaries['back'],1] * (1 + strain)
        elif bc_direction[0] == 'z':
            x0[boundaries['top'],2] = eq_posns[boundaries['top'],2] * (1 + strain)
        else:
            raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
    elif 'simple_shear' in bc_type:
        surface_nodes = np.concatenate((boundaries['left'],boundaries['right'],boundaries['front'],boundaries['back'],boundaries['top'],boundaries['bot']))
        if strain >= np.abs(np.pi/2):
            raise ValueError('For shearing strains, cannot use or exceed an angle of pi/2')
        if bc_direction[0] == 'x':
            if bc_direction[1] == 'x':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in x direction and force in x direction')
            adjacent_length = eq_posns[surface_nodes,0]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'y':
                x0[surface_nodes,1] = eq_posns[surface_nodes,1] + opposite_length
            elif bc_direction[1] == 'z':
                x0[surface_nodes,2] = eq_posns[surface_nodes,2] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif bc_direction[0] == 'y':
            if bc_direction[1] == 'y':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in y direction and force in y direction')
            adjacent_length = eq_posns[surface_nodes,1]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'x':
                x0[surface_nodes,0] = eq_posns[surface_nodes,0] + opposite_length
            elif bc_direction[1] == 'z':
                x0[surface_nodes,2] = eq_posns[surface_nodes,2] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif bc_direction[0] == 'z':
            if bc_direction[1] == 'z':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in z direction and force in z direction')
            adjacent_length = eq_posns[surface_nodes,2]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'x':
                x0[surface_nodes,0] = eq_posns[surface_nodes,0] + opposite_length
            elif bc_direction[1] == 'y':
                x0[surface_nodes,1] = eq_posns[surface_nodes,1] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
    elif 'shearing' in bc_type:
        if strain >= np.abs(np.pi/2):
            raise ValueError('For shearing strains, cannot use or exceed an angle of pi/2')
        if bc_direction[0] == 'x':
            if bc_direction[1] == 'x':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in x direction and force in x direction')
            # adjacent_length = dimensions[0]/l_e
            adjacent_length = eq_posns[boundaries['right'],0]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'y':
                x0[boundaries['right'],1] = eq_posns[boundaries['right'],1] + opposite_length
            elif bc_direction[1] == 'z':
                x0[boundaries['right'],2] = eq_posns[boundaries['right'],2] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif bc_direction[0] == 'y':
            if bc_direction[1] == 'y':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in y direction and force in y direction')
            # adjacent_length = dimensions[1]/l_e
            adjacent_length = eq_posns[boundaries['back'],1]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'x':
                x0[boundaries['back'],0] = eq_posns[boundaries['back'],0] + opposite_length
            elif bc_direction[1] == 'z':
                x0[boundaries['back'],2] = eq_posns[boundaries['back'],2] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
        elif bc_direction[0] == 'z':
            if bc_direction[1] == 'z':
                raise ValueError('Cannot have a shear strain applied to a surface with surface normal in z direction and force in z direction')
            # adjacent_length = dimensions[2]/l_e
            adjacent_length = eq_posns[boundaries['top'],2]
            opposite_length = np.tan(strain)*adjacent_length
            if bc_direction[1] == 'x':
                x0[boundaries['top'],0] = eq_posns[boundaries['top'],0] + opposite_length
            elif bc_direction[1] == 'y':
                x0[boundaries['top'],1] = eq_posns[boundaries['top'],1] + opposite_length
            else:
                raise ValueError('strain direction not one of acceptable directions ("x","y","z")')
    elif 'torsion' in bc_type:
        #torsion will only be applied to the top surface
        #because the center of the system isn't at the origin, I need to translate the positions so that they are in a coordinate system where the center of the system in the 2D plane 'xy' is at (0,0), then rotate, then translate back to the original coordinate system
        if bc_direction[1] == 'CW':
            starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
            x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + starting_positions[:,1]*np.sin(strain)
            x0[boundaries['top'],1] = -1*starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
            x0[boundaries['top']] += np.array([dimensions[0]/l_e,dimensions[1]/l_e,0])
        elif bc_direction[1] == 'CCW':
            starting_positions = eq_posns[boundaries['top']].copy() - np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
            x0[boundaries['top'],0] = starting_positions[:,0]*np.cos(strain) + -1*starting_positions[:,1]*np.sin(strain)
            x0[boundaries['top'],1] = starting_positions[:,0]*np.sin(strain) + starting_positions[:,1]*np.cos(strain)
            x0[boundaries['top']] += np.array([dimensions[0]/l_e/2,dimensions[1]/l_e/2,0])
        else:
            raise ValueError('strain direction for torsion must be one of ("CW", "CCW") for clockwise or counterclockwise rotation of the top surface of the simulated volume')
    else:
        raise ValueError('Strain type not one of the following accepted types ("tension", "compression", "shearing", "torsion")')
    if gpu_flag:
        x0 = cp.array(x0.astype(np.float32)).reshape((x0.shape[0]*x0.shape[1],1),order='C')
    return x0, boundary_conditions

def run_test_sim(sim_variables_dict):#(output_dir,bc_type,bc_direction,stresses,Hext_series,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,particle_radius,particle_mass,chi,Ms,drag=10,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation applying a series of a particular type of stress/strain to the volume with a series of applied external magnetic fields for the purpose of testing new implementations of functions against old, accepted implementations. Works by passing the boundary condition type as a string (one of the following: tension, compression, shearing, torsion, or simple_stress_* where * is one of tension/compression/shearing), the stress/strain direction as a tuple of strings e.g. ('x','x') from the choice of ('x','y','z') for any non-torsion strains and ('CW','CCW') for torsion, the stress/strains as a list of floating point values (for compression, strain must not exceed 1.0 (100%), for torsion the value is an angle in radians, for shearing strain the value is an angle in radians and should not be equal to or exceed pi/2), the external magnetic field vector, initialized node positions, list of elements, particles, boundary nodes stored in a dictionary, scaled dimensions of the system, the list of springs, the additional bulk modulus kappa, the volume element edge length, the scaling coefficient beta, the node specific scaling coefficients beta_i, the total time to integrate in a single integration step, the particle radius in meters, the particle mass, the particle magnetic suscpetibility chi, the particle magnetization saturation Ms, the drag coefficient, the maximum number of integration runs per strain value, and the maximum number of integration steps within an integration run."""
    output_dir, x0, bc_type, bc_direction, Hext_series, springs_var, elements, dimensions, boundaries, kappa, l_e, beta, beta_i, drag, particles, host_particles, particle_radius, particle_volume, particle_mass, chi, Ms, max_integrations, max_integration_steps, tolerance, gpu_flag, particle_rotation_flag, persistent_checkpointing_flag, plotting_flag, criteria_flag, step_size = unpack_sim_variables(sim_variables_dict)

    stresses = sim_variables_dict['stresses']

    if gpu_flag:
        step_size = sim_variables_dict['step_size']
    else:
        t_f = sim_variables_dict['t_f']

    total_delta = 0
    for count, stress in enumerate(stresses):
        boundary_conditions = (bc_type,(bc_direction[0],bc_direction[1]),stress)
        for i, Hext in enumerate(Hext_series):
            if output_dir[-1] != '/':
                current_output_dir = output_dir + f'/stress_{count}_{bc_type}_{np.round(stress,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            elif output_dir[-1] == '/':
                current_output_dir = output_dir + f'stress_{count}_{bc_type}_{np.round(stress,decimals=3)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/'
            if not (os.path.isdir(current_output_dir)):
                os.mkdir(current_output_dir)
            print(f'Running simulation with external magnetic field: ({np.round(Hext[0]*mu0,decimals=3)}, {np.round(Hext[1]*mu0,decimals=3)}, {np.round(Hext[2]*mu0,decimals=3)}) T\nApplied stress {boundary_conditions[1]} {np.round(stress,decimals=5)}\n')
            start = time.time()
            try:
                sol, return_status = simulate.simulate_scaled_gpu_leapfrog_test(x0,elements,host_particles,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
                # sol, return_status = simulate.simulate_scaled_gpu_leapfrog(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,current_output_dir,max_integrations,max_integration_steps,tolerance,step_size,persistent_checkpointing_flag)
            except Exception as inst:
                print('Exception raised during simulation')
                print(type(inst))
                print(inst)
                raise
            end = time.time()
            delta = end - start
            total_delta += delta
            final_posns = sol[:int(sol.shape[0]/2)]
            N_nodes = int(sol.shape[0]/6)
            final_posns = np.reshape(final_posns,(N_nodes,3))
            print('took %.2f seconds to simulate' % delta)
            output_file_number = count*Hext_series.shape[0]+i
            mre.initialize.write_output_file(output_file_number,final_posns,None,Hext,boundary_conditions,np.array([delta]),output_dir)
            #if we have already run a particular simulation with zero stress/strain and at some field, use that as the starting point for the solution
            if Hext_series.shape[0] > 1 and (output_file_number >= (Hext_series.shape[0]-1)) and (output_file_number < Hext_series.shape[0]*len(stresses)-1):
                output_file_num_to_reuse = output_file_number-(Hext_series.shape[0]-1)
                x0, output_file_Hext, _, _ = mre.initialize.read_output_file(output_dir+f'output_{output_file_num_to_reuse}.h5')
                print(f'reusing previously calculated solution with B_ext = {np.round(mu0*output_file_Hext,decimals=3)}')
                x0 = cp.array(x0.astype(np.float32)).reshape((x0.shape[0]*x0.shape[1],1),order='C')
            else:#use the last solution vector of positions as the starting set of positions for the next step
                x0 = cp.array(final_posns.astype(np.float32)).reshape((final_posns.shape[0]*final_posns.shape[1],1),order='C')
    return total_delta, return_status

def format_boundary_conditions(boundary_conditions):
    boundary_conditions = (str(boundary_conditions[0][0])[1:],(str(boundary_conditions[0][1])[1:],str(boundary_conditions[0][2])[1:]),boundary_conditions[0][3])
    boundary_conditions = (boundary_conditions[0][1:-1],(boundary_conditions[1][0][1:-1],boundary_conditions[1][1][1:-1]),boundary_conditions[2])
    return boundary_conditions

if __name__ == "__main__":
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-05-07_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_stepsize_5.e-3/"
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-05-10_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_90_gpu_True_starttime_11-33_stepsize_5.e-3/"
    # continue_interrupted_sim(sim_dir)
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-05-20_512_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_17-40_stepsize_5.e-3/"
    # # sim_checkpoint_dir = [sim_dir+"strain_0_strain_tension_0.0_field_2_Bext_[0.05 0.   0.  ]/",sim_dir+"strain_0_strain_tension_0.0_field_3_Bext_[0.1 0.  0. ]/"]
    # sim_checkpoint_dir = []
    # sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-05-09_125_particle_field_dependent_modulus_strain_strain_tension_direction('x', 'x')_order_3_E_9000.0_nu_0.47_Bext_angle_0.0_gpu_True_starttime_16-55_stepsize_5.e-3/"
    # node_posns, mass, springs_var, elements, boundaries, particles, parameters, field_series, boundary_condition_series, sim_type = mre.initialize.read_init_file(sim_dir+'init.h5')
    # max_boundary_condition_value = 1e-3
    # boundary_condition_value_step = max_boundary_condition_value/5
    # strains = np.float32(boundary_condition_series)
    # # strains = np.arange(0.0,max_boundary_condition_value*1.01,boundary_condition_value_step)
    # strain_start_index = 1
    # field_start_index = 7
    # Bext_series = np.zeros((16,3),dtype=np.float32)
    # for i in range(16):
    #     Bext_series[i,0] = i*1e-2
    # Hext_series = Bext_series/mu0
    # for count in range(strain_start_index,strains.shape[0]):#for count, strain in enumerate(strains):
    #     strain = strains[count]
    #     for i in range(field_start_index,field_start_index+1):#Hext_series.shape[0]):#for i, Hext in enumerate(Hext_series):
    #         Hext = Hext_series[i]
    #         sim_checkpoint_dir.append(sim_dir + f'strain_{count}_strain_tension_{np.round(strain,decimals=5)}_field_{i}_Bext_{np.round(Hext*mu0,decimals=3)}/')

    jumpstart_type = 'restart'#'rerun'#'extend'#
    sim_dir = "/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2024-11-20_2_particle_hysteresis_order_5_E_9.e+03_nu_0.47_Bext_angles_0.0_0.0_regular_vol_frac_3.e-2_stepsize_5.e-3/"
    # jumpstart_sim(sim_dir,jumpstart_type,sim_checkpoint_dirs=[sim_dir+'strain_0_strain_shearing_0.0_field_2_Bext_[0.   0.   0.14]/'])
    # jumpstart_sim(sim_dir,jumpstart_type)

    # extend_sim(sim_dir,sim_checkpoint_dir)
    batch_job_runner()

    # radius = 1.5e-6
    # l_e = 1e-6
    # dimensions = [10,10,10]
    # placement_type = 'random'
    # volume_fraction = 0.03
    # placeholder_particle_placement(radius,l_e,dimensions,placement_type,volume_fraction)
    # experimental_stress_simulation_tests()
    # experimental_simulation_tests()
    # main_series_simulations()
    # main_strain()
    # main2()