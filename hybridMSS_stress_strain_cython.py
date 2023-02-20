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

# simulate(node_posns,connectivity,separations,boundary_conditions) -> at each time step, output the nodal positions, velocities, and accelerations, or after each succesful energy minimization output the nodal positions

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sci
import scipy.sparse
import time
import os
import lib_programname
import tables as tb#pytables, for HDF5 interface
import get_volume_correction_force_cy_nogil

#remember, purpose, signature, stub

#given a spring network and boundary conditions, determine the equilibrium displacements/configuration of the spring network
#if using numerical integration, at each time step output the nodal positions, velocities, and accelerations, or if using energy minimization, after each succesful energy minimization output the nodal positions

# !!! might want to wrap this method into the Simulation class, so that the actual interface is... cleaner. i can let the user (me) set the parameters, then run some initialization method (or have the constructor run the other methods for setting things up, like the connectivity, elements, and stiffnesses). then a method for running the simulation
def simulate(node_posns,elements,boundaries,dimensions,connectivity,separations,kappa,l_e,boundary_conditions,time_steps,dt):
    """Run a simulation of a hybrid mass spring system using the Verlet algorithm. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. Connectivity is the N_vertices by N_vertices array whose elements connectivity[i,j] are zero if there is no spring/elastic coupling between vertex i and j, and the numeric stiffness constant value otherwise. Separations is the N_vertices by N_vertices numpy array whose elements are the magnitude of separation of each vertex pair at elastic equilibrium. kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. time_steps is the number of iterations to calculate, and dt is the time step for each iteration."""
    
    epsilon = np.spacing(1)
    tol = epsilon*1e5
    # fig = plt.figure()
    # ax = fig.add_subplot(projection= '3d')
    x0,v0,m = node_posns.copy(), np.zeros(node_posns.shape), np.ones(node_posns.shape[0])*1e-2
    if boundary_conditions[0] == 'strain':
        # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
        for surface in boundary_conditions[1]:
            if surface == 'right' or surface == 'left':
                pinned_axis = 0
            elif surface == 'top' or surface == 'bottom':
                pinned_axis = 2
            else:
                pinned_axis = 1
            x0[:,pinned_axis] *= (1+ boundary_conditions[2])
            # x0[boundaries[surface],pinned_axis] *= (1 + boundary_conditions[2])#!!! i have altered how the strain is applied, but this needds to be handled differently for the different methods... i need to deal with the applied boundary conditions more effectively. the single material case is simpler than the mre case. for the single material case i can try stretching each eleemnts x, y, or z postion by the same amount fora  simple axial strain
    for s in range(time_steps):
        x1, v1, a = timestep(x0,v0,m,elements,connectivity,separations,kappa,l_e,boundary_conditions,boundaries,dimensions,dt)
        # below, setting a convergence criteria check on the sum of norms of the forces on each vertex, and the max norm of force on any vertex
        if (np.max(np.linalg.norm(a,axis=1)) <=tol*1e3 or np.sum(np.linalg.norm(a,axis=1)) <= tol):
            break
        else:
            x0, v0 = x1, v1
    print('sum of the norms of the accelerations was '+ str(np.sum(np.linalg.norm(a,axis=1))))
    return x1, v1, a#return positions, and velocities, and accelerations/forces, want to get an idea of how close to equilibrium i am. may want to set things up so the simulation runs until a keyboard interrupt (i think there's a way to include an object that listens for keyboard input), and checks if equilibrium has been reached (largest force being less than some threshold)

# placeholder functions for doing purpose, signature, stub when doing planning/design and wishlisting
def do_stuff():
    return 0

def do_other_stuff():
    return 0

def timestep(x0,v0,m,elements,connectivity,eq_separations,kappa,l_e,bc,boundaries,dimensions,dt):
    """computes the next position and velocity for the given masses, initial conditions, and timestep"""
    N = len(x0)
    separations = np.empty((N,N))
    drag = 1
    x1 = np.empty(x0.shape,dtype=float)
    v1 = np.empty(v0.shape,dtype=float)
    a = np.empty(x0.shape,dtype=float)
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
        for surface in bc[1]:
            do_stuff()
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = get_spring_force_magnitude(x0,connectivity,eq_separations)
    update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,boundaries,bc)
    return x1, v1, a

#how can i intelligently calculate the spring forces and update the accelerations? I mean that in the sense of reducing the number of loops over anything, or completely avoiding loops. i need to review the method i am using now and identify places where i can perform additional computations in a single loop, even if it increases code complexity. i have a version that is... still code complex but follows a human logic of looping over nodes to calculate the spring forces on each node one at a time, which later loops again and updates the positions one node at a time after getting the force vectors calculated
def get_spring_force_magnitude(x0,connectivity,eq_separations):
    """calculate the magnitude of the force acting on each node due to a spring force, where entry ij is the negative magnitude on i in the rij_hat direction"""
    N = x0.shape[0]
    separations = np.empty((N,N))
    for i, posn in enumerate(x0):
        rij = posn - x0
        rij_mag = np.sqrt(np.sum(rij**2,1))
        separations[:,i] = rij_mag
    displacement = separations - eq_separations
    force = -1*connectivity * displacement
    return force

def get_spring_force_vector(i,posn,x0,spring_force):
    """given the negative magnitude of the force on node i at position due to every node j, calculate the rij_hat vectors for the node i at posn, and return the vector sum of the forces acting on node i"""
    rij = posn - x0
    rij_mag = np.sqrt(np.sum(rij**2,1))
    rij_mag[rij_mag == 0] = 1#this shouldn't cause issues, it is here to prevent a load of divide by zero errors occuring. if rij is zero length, it is the vector pointing from the vertex to itself, and so rij/rij_mag will cause a divide by zero warning. by setting the magnitude to 1 in that case we avoid that error, and that value should only occur for the vector pointing to itself, which shouldn't contirbute to the force
    # while i understand at the moment why i calculated the elastic forces that way i did, it is unintuitive. I am trying to use numpy's broadcasting and matrix manipulation to improve speeds, but the transformations aren't obviously useful. maybe there is a clearer way to do this that is still fast enough. or maybe this is the best i can do (though i will need to use cython to create compiled code literally everywhere i have for loops over anything, which means getting more comfortable with cython and cythonic code)
    force_vector = np.transpose(np.tile(spring_force[i,:],(3,1)))*(rij/np.tile(rij_mag[:,np.newaxis],(1,3)))
    return force_vector

def update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,boundaries,bc):
    """taking into account boundary conditions, drag, velocity, volume correction and spring forces, calculate the particle accelerations and update the particle positions and velocities"""
    for i, posn in enumerate(x0):
        if not (bc[0] == 'strain' and (np.any(i==boundaries[bc[1][0]]) or np.any(i==boundaries[bc[1][1]]))):
            force_vector = get_spring_force_vector(i,posn,x0,spring_force)
            a[i] = np.sum(force_vector,0)/m[i] - drag * v0[i] + volume_correction_force[i] + bc_forces[i]
        else:
            a[i] = 0
    for i in range(x0.shape[0]):
        v1[i] = a[i] * dt + v0[i]
        x1[i] = a[i] * dt**2 + v0[i] * dt + x0[i]

#Given the dimensions of a rectilinear space describing the system of interest, and the side length of the unit cell that will be used to discretize the space, return list of vectors that point to the nodal positions at stress free equilibrium
def discretize_space(Lx,Ly,Lz,cube_side_length):
    """Given the side lengths of a rectilinear space and the side length of the cubic unit cell to discretize the space, return arrays of (respectively) the node positions as an N_vertices x 3 array, N_cells x 8 array, and N_vertices x 8 array"""#??? should it be N_vertices by 8? i can't mix types in the python numpy arrays. if it is N by 8 I can store index values for the unit cells that each node belongs to, and maybe negative values or NaN for the extra entries if the vertex/node doesn't belong to 8 unit cells
    #check the side length compared to the dimensions of the space of interest to determine if the side length is appropriate for the space?
    [x,y,z] = np.meshgrid(np.r_[0:Lx+cube_side_length*0.1:cube_side_length],
                          np.r_[0:Ly+cube_side_length*0.1:cube_side_length],
                          np.r_[0:Lz+cube_side_length*0.1:cube_side_length])
    #one of my ideas for implementing this was to create a single unit cell and tile it to fill the space, which could allow me to create the unit_cell_def array and maybe the node_sharing array more easily
    node_posns = np.concatenate((np.reshape(x,np.size(x))[:,np.newaxis],
                                np.reshape(y,np.size(y))[:,np.newaxis],
                                np.reshape(z,np.size(z))[:,np.newaxis]),1)
    #need to keep track of which nodes belong to a unit cell (at some point)
    N_el_x = np.int32(round(Lx/cube_side_length))
    N_el_y = np.int32(round(Ly/cube_side_length))
    N_el_z = np.int32(round(Lz/cube_side_length))
    N_el = N_el_x * N_el_y * N_el_z
    #finding the indices for the nodes/vertices belonging to each element
    #!!! need to check if there is any ordering to the vertices right now that I can use. I need to have each vertex for each element assigned an identity relative to the element for calculating average edge vectors to estimate the volume after deformation
    elements = np.empty((N_el,8))
    counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                elements[counter,:] = np.nonzero((node_posns[:,0] <= cube_side_length*(k+1)) & (node_posns[:,0] >= cube_side_length*k) & (node_posns[:,1] >= cube_side_length*j) & (node_posns[:,1] <= cube_side_length*(j+1)) & (node_posns[:,2] >= cube_side_length*i) & (node_posns[:,2] <= cube_side_length*(i+1)))[0]
                counter += 1
    top_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].max())[0]
    bot_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].min())[0]
    left_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].min())[0]
    right_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].max())[0]
    front_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].min())[0]
    back_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].max())[0]
    boundaries = {'top': top_bdry, 'bot': bot_bdry, 'left': left_bdry, 'right': right_bdry, 'front': front_bdry, 'back': back_bdry}
    return node_posns, np.int32(elements), boundaries

    #given the node positions and stiffness constants for the different types of springs, calculate and return the connectivity matrix, and equilibrium separation matrix
def create_connectivity_v3(node_posns,stiffness_constants,cube_side_length,dimensions):
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #besides the node positions and stiffness constants (which is more complicated than just 3 numbers for the MRE case where there are two isotropic phases present with different material properties), we need to determine the connectivity matrix by summing up the stiffness contribution from each cell that share the vertices whose element is being calculated. A vertex in the inner part of the body will be shared by 8 cells, while one on a surface boundary may be shared by 4, or at a corner, only belonging to one unit cell. Similarly, if those unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    #since i have a unit cell, and since the connectivity matrix looks the same for each unit cell of a material type, can I construct the connectivity matrix for a single unit cell of each type, and combine them into the overall connectivity matrix? that way I can avoid calculating separations for all nodes when I know that there's only so many connections per node possible (26 total for edge, face diagonal, and center diagonal springs in a cubic cell)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    connectivity = np.zeros((N,N))
    separations = np.empty((N,N))
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    i = 0
    for posn in node_posns:
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        set_stiffness_shared_elements_v3(i,node_posns,rij_mag,dimensions,connectivity,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        set_stiffness_shared_elements_v3(i,node_posns,rij_mag,dimensions,connectivity,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        connectivity[np.abs(rij_mag - center_diagonal_length) < epsilon,i] = stiffness_constants[2]
        separations[:,i] = rij_mag
        i += 1
    return (connectivity,separations)

def create_connectivity_sparse(node_posns,elements,stiffness_constants,cube_side_length):
    """Returns a scipy.sparse CSR array of the connectivity and separation matrices."""
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #besides the node positions and stiffness constants (which is more complicated than just 3 numbers for the MRE case where there are two isotropic phases present with different material properties), we need to determine the connectivity matrix by summing up the stiffness contribution from each cell that share the vertices whose element is being calculated. A vertex in the inner part of the body will be shared by 8 cells, while one on a surface boundary may be shared by 4, or at a corner, only belonging to one unit cell. Similarly, if those unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    #since i have a unit cell, and since the connectivity matrix looks the same for each unit cell of a material type, can I construct the connectivity matrix for a single unit cell of each type, and combine them into the overall connectivity matrix? that way I can avoid calculating separations for all nodes when I know that there's only so many connections per node possible (26 total for edge, face diagonal, and center diagonal springs in a cubic cell)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    data = []
    separation_data = []
    row_ind = []
    col_ind = []
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    i = 0
    for posn in node_posns:
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        result = set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(cube_side_length)
        result = set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(face_diagonal_length)
        center_diagonal_indices = np.where(np.abs(rij_mag - center_diagonal_length) < epsilon)[0]
        result = (center_diagonal_indices, np.ones(center_diagonal_indices.size,)*stiffness_constants[2])
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(center_diagonal_length)
        i += 1
    sparse_connectivity = scipy.sparse.csr_array((data, (row_ind, col_ind)), shape=(N, N))
    sparse_separations = scipy.sparse.csr_array((separation_data, (row_ind, col_ind)), shape=(N, N))
    return (sparse_connectivity,sparse_separations)

#functionalizing the setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
def set_stiffness_shared_elements_v3(i,node_posns,rij_mag,dimensions,connectivity,stiffness_constant,comparison_length,max_shared_elements):
    """setting the stiffness of a particular element based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    connected_vertices = np.where(np.abs(rij_mag - comparison_length) < epsilon)[0]
    node_type_i = identify_node_type(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
    for v in connected_vertices:
        if connectivity[i,v] == 0:
            node_type_v = identify_node_type(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
            if node_type_i == 'interior' and node_type_v == 'interior':
                if max_shared_elements == 4:
                    connectivity[i,v] = stiffness_constant*4
                    connectivity[v,i] = connectivity[i,v]
                else:
                    connectivity[i,v] = stiffness_constant*2
                    connectivity[v,i] = connectivity[i,v]
            elif (node_type_i == 'interior' and node_type_v == 'surface') or (node_type_i == 'surface' and node_type_v == 'interior'):
                if max_shared_elements == 4:
                    connectivity[i,v] = stiffness_constant*4
                    connectivity[v,i] = connectivity[i,v]
                else:
                    connectivity[i,v] = stiffness_constant*2
                    connectivity[v,i] = connectivity[i,v]
            elif (node_type_i == 'interior' and node_type_v == 'edge') or (node_type_i == 'edge' and node_type_v == 'interior'):
                connectivity[i,v] = stiffness_constant*2
                connectivity[v,i] = connectivity[i,v]
            elif node_type_i == 'surface' and node_type_v == 'surface':
                if max_shared_elements == 4:#two shared elements for a cube edge spring in this case if they are both on the same surface, so check for shared surfaces. otherwise the answer is 4.
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        connectivity[i,v] = stiffness_constant*2
                        connectivity[v,i] = connectivity[i,v]
                    else:
                        connectivity[i,v] = stiffness_constant*4
                        connectivity[v,i] = connectivity[i,v]
                else:#face spring, if the two nodes are on the same surface theres only one element, if they are on two different surfaces theyre are two shared elements
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        connectivity[i,v] = stiffness_constant
                        connectivity[v,i] = connectivity[i,v]
                    else:#on different surfaces, two shared elements
                        connectivity[i,v] = stiffness_constant*2
                        connectivity[v,i] = connectivity[i,v]
            elif (node_type_i == 'surface' and node_type_v == 'edge') or (node_type_i == 'edge' and node_type_v == 'surface'):
                if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
                    connectivity[i,v] = stiffness_constant*2
                    connectivity[v,i] = connectivity[i,v]
                else:#this is a face spring with only a single element if the edge and surface node have a shared surface
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        connectivity[i,v] = stiffness_constant
                        connectivity[v,i] = connectivity[i,v]
                    else:#they don't share a surface, and so they share two elements
                        connectivity[i,v] = stiffness_constant*2
                        connectivity[v,i] = connectivity[i,v]
            elif node_type_i == 'edge' and node_type_v == 'edge':
                #both nodes belong to two surfaces (if they are edge nodes). if the surfaces are the same, then it is a shared edge, if they are not, they are separate edges of the simulated volume. there aer 6 surfaces
                node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                if ((node_i_surf[0] == node_v_surf[0] and node_i_surf[1] == node_v_surf[1] and (node_i_surf[0] != 0 and node_i_surf[1] != 0)) or (node_i_surf[0] == node_v_surf[0] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[0] != 0 and node_i_surf[2] != 0)) or(node_i_surf[1] == node_v_surf[1] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[1] != 0 and node_i_surf[2] != 0))):#if both nodes belong to the same two surfaces, they are on the same edge
                    connectivity[i,v] = stiffness_constant
                    connectivity[v,i] = connectivity[i,v]
                elif max_shared_elements == 4:#if they don't share two surfaces and it's a cube edge spring, they share two elements
                    connectivity[i,v] = stiffness_constant*2
                    connectivity[v,i] = connectivity[i,v]
                else:#if it's a face spring
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):#if they do share a surface, then the face spring has as single element
                        connectivity[i,v] = stiffness_constant
                        connectivity[v,i] = connectivity[i,v]
                    else:#they don't share a single surface, then they diagonally across one another and have two shared elements
                        connectivity[i,v] = stiffness_constant*2
                        connectivity[v,i] = connectivity[i,v]
            elif node_type_i == 'corner' or node_type_v == 'corner':#any spring involving a corner node covered
                connectivity[i,v] = stiffness_constant
                connectivity[v,i] = connectivity[i,v]

def identify_node_type(node_posn,Lx,Ly,Lz):
    """based on the node position and the dimensions of the simulation, identify if the node is a corner, edge, surface, or interior point
    """
    eps = np.spacing(1)
    if ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        #if at extremes in 3 of 3 position components
        return 'corner'
    elif (((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps)) or ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)) or ((node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps))):
        #if at an edge (at extremes in two of the 3 position components)
        return 'edge'
    elif ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) or (node_posn[1] == 0 or node_posn[1] == Ly) or (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        return 'surface'
    else:
        return 'interior'

def get_node_surf(node_posn,Lx,Ly,Lz):
    eps = np.spacing(1)
    surfaces = [0, 0, 0]
    if np.abs(node_posn[0]- Lx) < eps:
        surfaces[0] = 1
    elif node_posn[0] == 0:
        surfaces[0] = -1
    if np.abs(node_posn[1] -Ly) < eps:
        surfaces[1] = 1
    elif node_posn[1] == 0:
        surfaces[1] = -1
    if np.abs(node_posn[2] -Lz) < eps:
        surfaces[2] = 1
    elif node_posn[2] == 0:
        surfaces[2] = -1
    return surfaces

def set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constant,comparison_length,max_shared_elements):
    """Return list of tuples of index of connected node and stiffness constant, setting the stiffness of a particular connection based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    shared_elements = 0
    result = []
    indices = []
    effective_stiffness =[]
    connected_vertices = np.nonzero(np.abs(rij_mag - comparison_length) < epsilon)[0]
    for v in connected_vertices:
        for el in elements:
            if np.any(el == i) & np.any(el == v):
                shared_elements += 1
                if shared_elements == max_shared_elements:
                    break
        tmp_stiffness_constant = stiffness_constant * shared_elements
        indices.append(v)
        effective_stiffness.append(tmp_stiffness_constant)
        shared_elements = 0
    result = (indices,effective_stiffness)
    return result

def set_stiffness_shared_elements_sparse_v2(i,rij_mag,elements,stiffness_constant,comparison_length,max_shared_elements):
    """Return list of tuples of index of connected node and stiffness constant, setting the stiffness of a particular connection based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    shared_elements = 0
    result = []
    indices = []
    effective_stiffness =[]
    connected_vertices = np.nonzero(np.abs(rij_mag - comparison_length) < epsilon)[0]
    for v in connected_vertices:
        # if node_posns[i,0] == 0 || node_posns[i,0] == 0 || node_posns[i,0] == 0 || node_posns[i,0] == 0 || node_posns[i,0] == 0 || 
        for el in elements:
            if np.any(el == i) & np.any(el == v):
                shared_elements += 1
                if shared_elements == max_shared_elements:
                    break
        tmp_stiffness_constant = stiffness_constant * shared_elements
        indices.append(v)
        effective_stiffness.append(tmp_stiffness_constant)
        shared_elements = 0
    result = (indices,effective_stiffness)
    return result
            
#first goal is to calculate the distance between one node and the other nodes in the cell. then you can use a for loop to iterate over each node. when you know the distances for a single node, can do a conditional check on the magnitude of separation to determine which stiffness constant belongs to which entry in the 8x8 matrix representing connectivity within that unit cell. then can think about how to do that process simultaneously using numpy and scipy functionality
#after that process works, the next step is figuring out how to combine connectivity matrix entries. the more i think about it the less convinced i am that i can combine unit cell connectivity matrices together in a reasonable matter, as the order of the nodes in the connectivity matrix for one unit cell doesn't have particular meaning related to their spacing, and the order of nodes in the overall connectivity matrix involving more than one unit cell will also not be related to the relative positions in space
#i don't need to be clever for this bit, at least not right now. first write something that works, even if it is brute force. try to be clever when it works.


#given the material properties (Young's modulus, shear modulus, and poisson's ratio) of an isotropic material, calculate the spring stiffness constants for edge springs, center diagonal springs, and face diagonal springs for a cubic unit cell
def get_spring_constants(E,nu,l_e):
    """given the Young's modulus, poisson's ratio, and the length of the edge springs, calculate the edge, central diagonal, and face diagonal stiffness constants of the system"""
    A = 1 #ratio of the stiffness constants of the center diagonal to face diagonal springs
    k_e = 0.4 * (E * l_e) * (8 + 3 * A) / (4 + 3 * A)
    k_c = 1.2 * (E * l_e) / (4 + 3 * A)
    k_f = A * k_c
    k = [k_e, k_f, k_c]
    return k

def get_kappa(E,nu):
    """Given the Young's modulus and Poissons's ratio, return the value of the additional bulk modulus, kappa, for the volume correction forces"""
    kappa = E * (4 * nu - 1) / (2 * (1 + nu) * (1 - 2 * nu))
    return kappa

#given the nodal positions defining the ends of a spring, the equilibrium spring length, and the stiffness constant, calculated the energy of the spring and  the forces (force as a vector) acting on each node
def spring_response(node1,node2,k,r_eq):
    energy = 1/2 * k * (sci.distance.pdist(np.array([node1,node2]),'euclidean') - r_eq)**2
    force = -k*(node2-node1)/sci.distance.pdist(np.array([node1,node2]),'euclidean') * (sci.distance.pdist(np.array([node1,node2]),'euclidean') - r_eq)#calculating force on node 2
    return -force,force,energy #Newton's second law, force on node 1 is negative force on node 2

#!!! generate traction forces or displacements based on some other criteria (choice of experimental setup with a switch statement? stress applied on boundary and then appropriately split onto the correct nodes in the correct directions in the correct amounts based on surface area?)

#function which plots with a 3D scatter and lines, the connectivity of the unit cell
def plot_unit_cell(node_posns,connectivity):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(node_posns[:,0],node_posns[:,1],node_posns[:,2],'o')
    for i in range(len(node_posns)-1):
        for j in range(i+1,len(node_posns)):
            if connectivity[i,j] != 0:#add conditional to check equilibrium separation and only show edge springs
                x,y,z = (np.array((node_posns[i,0],node_posns[j,0])),
                         np.array((node_posns[i,1],node_posns[j,1])),
                         np.array((node_posns[i,2],node_posns[j,2])))
                ax.plot(x,y,z)
        #others = remove_i(node_posns,i)
        #feels like this should be recursive. I have a listof points, I want to draw lines from each pair of points but without redrawing lines. I have one point, I remove it from the list, if there's nothing left in the list I have nothing to draw, if there's one thing left in the list, I draw the line connecting this point to that point, if I have more than one point left in the list, I 

#calculating the volume of the unit cell (deformed typically) by averaging edge vectors to approximate the volume. V_c^' = \vec{a} \cdot (\vec{b} \times \vec {c})
def get_unit_cell_volume(avg_vectors):
    #"""Return an approximation of the unit cell's deformed volume by passing the 8 vectors that define the vertices of the cell"""
    N_el = avg_vectors.shape[2]
    V = np.zeros((N_el,))
    a_vec = np.transpose(avg_vectors[0,:,:])
    b_vec = np.transpose(avg_vectors[1,:,:])
    c_vec = np.transpose(avg_vectors[2,:,:])
    for i in range(N_el):
        #need to look into functions do handle the dot products properly... 
        V[i] = np.dot(a_vec[i],np.cross(b_vec[i],c_vec[i]))
    return V

#helper function for getting the unit cell volume. I need the averaged edge vectors used in the volume calculation for other calculations later (the derivative of the deformed volume with respect to the position of each vertex is used to calculate the volume correction force). However, the deformed volume is also used in that expression. Really these are two helper functions for the volume correction force
def get_average_edge_vectors(node_posns,elements):
    avg_vectors = np.empty((3,3,elements.shape[0]))
    counter = 0
    for el in elements:
        vectors = node_posns[el]
        avg_vectors[0,:,counter] = vectors[2] - vectors[0] + vectors[3] - vectors[1] + vectors[6] - vectors[4] + vectors[7] - vectors[5]
        avg_vectors[1,:,counter] = vectors[4] - vectors[0] + vectors[6] - vectors[2] + vectors[5] - vectors[1] + vectors[7] - vectors[3]
        avg_vectors[2,:,counter] = vectors[1] - vectors[0] + vectors[3] - vectors[2] + vectors[5] - vectors[4] + vectors[7] - vectors[6]
        counter += 1
    avg_vectors *= 0.25
    return avg_vectors

#!!! construct the boundary conditions data structure
def get_boundary_conditions(boundary_condition_type,):
    #given a few experimental setups (plus fixed displacement type boundary conditions...)
    #experimental setups: shear, compression, tension, torsion, bending
    if boundary_condition_type == 'shear':
        return 0
    elif boundary_condition_type == 'compression':
        return 0
    elif boundary_condition_type == 'tension':
        return 0
    elif boundary_condition_type == 'torsion':
        return 0
    elif boundary_condition_type == 'bending':
        return 0
    elif boundary_condition_type == 'displacement':
        return 0
    elif boundary_condition_type == 'mixed':
        return 0
    
def get_accelerations_post_simulation(x0,boundaries,connectivity,eq_separations,elements,kappa,l_e,bc):
    N = len(x0)
    m = np.ones(x0.shape[0])*1e-2
    separations = np.empty((N,N))
    a = np.empty(x0.shape,dtype=float)
    avg_vectors = get_average_edge_vectors(x0,elements)
    i = 0
    for posn in x0:
        rij = posn - x0
        rij_mag = np.sqrt(np.sum(rij**2,1))
        separations[:,i] = rij_mag
        i += 1
    displacement = separations - eq_separations
    force = -1*connectivity * displacement
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    correction_force_cy_nogil = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, correction_force_cy_nogil)

    i = 0
    for posn in x0:
        if (np.any(i==boundaries[bc[1][0]]) or np.any(i==boundaries[bc[1][1]])):
            rij = posn - x0
            rij_mag = np.sqrt(np.sum(rij**2,1))
            rij_mag[rij_mag == 0] = 1#this is here to prevent divide by zero errors occuring. if rij is zero length, it is the vector pointing from the vertex to itself, and so rij/rij_mag will cause a divide by zero warning. by setting the magnitude to 1 in that case we avoid that error, and that value should only occur for the vector pointing to itself, which shouldn't contribute to the force
            # while i understand at the moment why i calculated the elastic forces that way i did, it is unintuitive. I am trying to use numpy's broadcasting and matrix manipulation to improve speeds, but the transformations aren't obviously useful. maybe there is a clearer way to do this that is still fast enough. or maybe this is the best i can do (though i will need to use cython to create compiled code literally everywhere i have for loops over anything, which means getting more comfortable with cython and cythonic code)
            force_vector = np.transpose(np.tile(force[i,:],(3,1)))*(rij/np.tile(rij_mag[:,np.newaxis],(1,3)))
            force_vector = np.nan_to_num(force_vector,posinf=0)
            a[i] = np.sum(force_vector,0)/m[i] + correction_force_cy_nogil[i]
        else:
            a[i] = 0
        i +=1
    return a

def remove_i(x,i):
    """remove the ith entry from an array"""
    shape = (x.shape[0]-1,) + x.shape[1:]
    y = np.empty(shape,dtype=float)
    y[:i] = x[:i]
    y[i:] = x[i+1:]
    return y

def post_plot(node_posns,connectivity,stiffness_constants):
    x0 = node_posns
    epsilon = np.spacing(1)
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(x0[:,0],x0[:,1],x0[:,2],'o')
    plt.cla()
    ax.scatter(x0[:,0],x0[:,1],x0[:,2],'o')
    ax.set_xlim((-0.3,1.2*node_posns[:,0].max()))
    ax.set_ylim((0,1.2*node_posns[:,1].max()))
    ax.set_zlim((0,1.2*node_posns[:,2].max()))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for i in range(len(x0)-1):
        for j in range(i+1,len(x0)):
            if np.abs(connectivity[i,j] - stiffness_constants[0]) <= epsilon or np.abs(connectivity[i,j] - stiffness_constants[0]/2) <= epsilon or np.abs(connectivity[i,j] - stiffness_constants[0]/4) <= epsilon:#connectivity[i,j] != 0:
                x,y,z = (np.array((x0[i,0],x0[j,0])),
                          np.array((x0[i,1],x0[j,1])),
                          np.array((x0[i,2],x0[j,2])))
                ax.plot(x,y,z)
                
class Simulation(object):
    """A simulation has properties which define the simulation. These include the Modulus, Poisson's ratio, cubic element side length, simulation dimensions.
    
    Attributes
    ----------
    E : Young's modulus [Pa]
    nu : Poisson's ratio []
    l_e : side length of an element [m]
    Lx : length in x direction of the object [m]
    Ly : length in y direction of the object [m]
    Lz : length in z direction of the object [m]
    """
    def __init__(self,E=1,nu=0.49,l_e=0.1,Lx=0.4,Ly=0.4,Lz=0.4):
        """Initializes simulation with default values if they are not passed"""
        self.E = E
        self.nu = nu
        self.l_e = l_e
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.t = 0
        self.N_iter = 0
        
    def set_time(self,time):
        self.t = time
        
    def set_iterations(self,N_iter):
        self.N_iter = N_iter
        
    def report(self):
        """Using using hand written string of the instance variables of the object to create a descriptuion of the simulation parameters (useful for writing a log file)"""
        report_string = 'E = ' + str(self.E) + ' m\n'+ 'nu = ' + str(self.nu) + '\n'+'l_e = ' + str(self.l_e) + ' m\n'+'Lx = ' + str(self.Lx) + ' m\n'+'Ly = ' + str(self.Ly) + ' m\n'+'Lz = ' + str(self.Lz) + ' m\n'+ 'total_time = ' + str(self.t) + ' s\n'+ 'N_iterations = ' + str(self.N_iter) + '  iterations\n'
        return report_string
    
    def report2(self):
        """Using built-in python features to iterate over the instance variables of the object to create a set of strings describing the simulation parameters (useful for writing a log file)"""
        my_keys = list(vars(self).keys())
        # my_vals = list(vars(self).values())
        report_string = ''
        for key in my_keys:
            report_string += key + ' = ' + str(vars(self).get(key)) + ' \n'
        return report_string
                
def write_log(simulation,output_dir):
    timestamp = time.ctime()
    script_name = lib_programname.get_path_executed_script()
    with open(output_dir+'logfile.txt','a') as f_obj:
        f_obj.writelines([simulation.report2(),str(script_name)+'\n',timestamp+'\n'])

    
def write_init_file(posns,connectivity,k,elements,output_dir):
    """Write out the vertex positions, connectivity matrix defined by equilibrium separation, connectivity matrix defined by stiffness constant, and the nodes that make up each cubic element as .csv files (or HDF5 files). To be modified in the future, to handle large systems (which will require sparse matrix representations due to memory limits)"""
    f = tb.open_file(output_dir+'temp.h5','w')
    f.create_array('/','vertex_posns',posns)
    # posn_dt = np.dtype([('x',np.float64),('y',np.float64),('z',np.float64)])
    # f.create_table('/','vertex_posns',posn_dt)
    # f.root.vertex_posns.append(posns)
    f.close()
    
def read_init_file(fn):
    f = tb.open_file(fn)
    return f
    

def main():
    E = 1
    nu = 0.49
    l_e = .1#cubic element side length
    Lx = 0.5
    Ly = 0.5
    Lz = 0.5
    dt = 1e-3
    N_iter = 1000
    dimensions = [Lx,Ly,Lz]
    k = get_spring_constants(E, nu, l_e)
    kappa = get_kappa(E, nu)


    node_posns,elements,boundaries = discretize_space(Lx,Ly,Lz,l_e)
    # (sparse_connectivity,sparse_separation) = create_connectivity_sparse(node_posns,elements,k,l_e)
    (c,s) = create_connectivity_v3(node_posns,k,l_e,dimensions)

    boundary_conditions = ('strain',('left','right'),.05)
    # strains = np.array([0.01])
    strains = np.arange(0.001,-0.81,-0.05)
    
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    count = 0
    for strain in strains:
        boundary_conditions = ('strain',('left','right'),strain)
        try:
            start = time.time()
            posns, v, a = simulate(node_posns,elements,boundaries,dimensions,c,s,kappa,l_e,boundary_conditions,N_iter,dt)
        except:
            print('Exception')
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        max_accel = np.max(np.linalg.norm(a,axis=1))
        print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        a_var = get_accelerations_post_simulation(posns,boundaries,c,s,elements,kappa,l_e,boundary_conditions)
        m = 1e-2
        end_boundary_forces = a_var[boundaries['right']]*m
        boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces))/(Ly*Lz)
        effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        count +=1
        
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,boundary_stress_xx_magnitude)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('stress_xx')
    plt.savefig('/home/leshy/MRE_Golec/stress-strain.png')
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,effective_modulus)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('effective modulus')
    plt.savefig('/home/leshy/MRE_Golec/strain-modulus.png')
    plt.close()

    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')+'\\'
    output_dir = current_dir+'golec_output\\'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    script_name = lib_programname.get_path_executed_script()

    my_sim = Simulation()
    write_log(my_sim,output_dir)
    write_init_file(node_posns,c,s,elements,output_dir)
# f = tb.open_file(output_dir+'temp.h5')
# calculate approximate volumes of each element after simulation
#deformed_avg_vecs = get_average_edge_vectors(posns, elements)
#deformed_vol = get_unit_cell_volume(deformed_avg_vecs)


if __name__ == "__main__":
    main()

#I need to adjust the method to check for some convergence criteria based on the accelerations each particle is experiencing (or some other convergence criteria)
#I need to somehow record the particle positions at equilibrium for the initial configuration and under user defined strain/stress. stress may be the most appropriate initial choice, since strain can be computed more directly than the stress. but both methods should eventually be used.


#Goals:
    #1 implement a convergence criteria
    #2 record initial configuration and equilibrium configuration in appropriately named directories and with appropriate file names
    #3 update functionality to allow for iteration over a list of stresses, which should include outputting the relevant simulation information (date and time, script name and location, parameters (stiffness, poisson ratio, element size, length of system, time step, viscous drag coefficient?, time to complete, number of iterations))
    #4 write a separate post-simulation script for analyzing the strain response as a function of stress, and plot the visual
    #5 update functionality to allow iteration over a list of strains