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
import scipy.integrate as sci
import time
import os
import lib_programname
import tables as tb#pytables, for HDF5 interface
import get_volume_correction_force_cy_nogil
import get_spring_force_cy
import update_positions_cy_nogil
import mre.initialize

#remember, purpose, signature, stub

#given a spring network and boundary conditions, determine the equilibrium displacements/configuration of the spring network
#if using numerical integration, at each time step output the nodal positions, velocities, and accelerations, or if using energy minimization, after each succesful energy minimization output the nodal positions

# !!! might want to wrap this method into the Simulation class, so that the actual interface is... cleaner. i can let the user (me) set the parameters, then run some initialization method (or have the constructor run the other methods for setting things up, like the connectivity, elements, and stiffnesses). then a method for running the simulation
def simulate(node_posns,elements,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,time_steps,dt):
    """Run a simulation of a hybrid mass spring system using the Verlet algorithm. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. Connectivity is the N_vertices by N_vertices array whose elements connectivity[i,j] are zero if there is no spring/elastic coupling between vertex i and j, and the numeric stiffness constant value otherwise. Separations is the N_vertices by N_vertices numpy array whose elements are the magnitude of separation of each vertex pair at elastic equilibrium. kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. time_steps is the number of iterations to calculate, and dt is the time step for each iteration."""
    
    epsilon = np.spacing(1)
    tol = epsilon*1e5
    # fig = plt.figure()
    # ax = fig.add_subplot(projection= '3d')
    x0,v0,m = node_posns.copy(), np.zeros(node_posns.shape), np.ones(node_posns.shape[0])*1e-2
    if boundary_conditions[0] == 'strain':
        # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
        surface = boundary_conditions[1][0]
        if surface == 'right' or surface == 'left':
            pinned_axis = 0
        elif surface == 'top' or surface == 'bottom':
            pinned_axis = 2
        else:
            pinned_axis = 1
        x0[:,pinned_axis] *= (1+ boundary_conditions[2])
        # x0[boundaries[surface],pinned_axis] *= (1 + boundary_conditions[2])#!!! i have altered how the strain is applied, but this needds to be handled differently for the different methods... i need to deal with the applied boundary conditions more effectively. the single material case is simpler than the mre case. for the single material case i can try stretching each eleemnts x, y, or z postion by the same amount fora  simple axial strain
    for s in range(time_steps):
        x1, v1, a = timestep(x0,v0,m,elements,springs,kappa,l_e,boundary_conditions,boundaries,dimensions,dt)
        # below, setting a convergence criteria check on the sum of norms of the forces on each vertex, and the max norm of force on any vertex
        try:
            if (np.max(np.linalg.norm(a,axis=1)) <=tol*1e3 or np.sum(np.linalg.norm(a,axis=1)) <= tol):
                break
            else:
                x0, v0 = x1, v1
        except RuntimeWarning:
            print('bar')
    print('sum of the norms of the accelerations was '+ str(np.sum(np.linalg.norm(a,axis=1))))
    return x1, v1, a#return positions, and velocities, and accelerations/forces, want to get an idea of how close to equilibrium i am. may want to set things up so the simulation runs until a keyboard interrupt (i think there's a way to include an object that listens for keyboard input), and checks if equilibrium has been reached (largest force being less than some threshold)

def simulate_v2(node_posns,elements,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,t_f):
    """Run a simulation of a hybrid mass spring system using the Verlet algorithm. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    
    epsilon = np.spacing(1)
    tol = epsilon*1e5
    # fig = plt.figure()
    # ax = fig.add_subplot(projection= '3d')
    x0,v0,m = node_posns.copy(), np.zeros(node_posns.shape), np.ones(node_posns.shape[0])*1e-2
    if boundary_conditions[0] == 'strain':
        # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
        surface = boundary_conditions[1][0]
        if surface == 'right' or surface == 'left':
            pinned_axis = 0
        elif surface == 'top' or surface == 'bottom':
            pinned_axis = 2
        else:
            pinned_axis = 1
        x0[:,pinned_axis] *= (1+ boundary_conditions[2])
        # x0[boundaries[surface],pinned_axis] *= (1 + boundary_conditions[2])#!!! i have altered how the strain is applied, but this needds to be handled differently for the different methods... i need to deal with the applied boundary conditions more effectively. the single material case is simpler 
        # than the mre case. for the single material case i can try stretching each eleemnts x, y, or z postion by the same amount fora  simple axial strain
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    sol = sci.solve_ivp(fun,[0,t_f],y_0,args=(m,elements,springs,kappa,l_e,boundary_conditions,boundaries,dimensions))
    return sol#returning a solution object, that can then have it's attributes inspected

# placeholder functions for doing purpose, signature, stub when doing planning/design and wishlisting
def do_stuff():
    return 0

def do_other_stuff():
    return 0

def timestep(x0,v0,m,elements,springs,kappa,l_e,bc,boundaries,dimensions,dt):
    """computes the next position and velocity for the given masses, initial conditions, and timestep"""
    N = len(x0)
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
    spring_force = np.empty(x0.shape,dtype=np.float64)
    # try:
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force)
    # except ZeroDivisionError:
    #     print('foo')
    fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
    update_positions_cy_nogil.update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,fixed_nodes)
    # if np.any(np.isnan(volume_correction_force)):
    #     print('volumecf')
    # if np.any(np.isnan(spring_force)):
    #     print('springforce')
    # if np.any(np.isnan(x1)):
    #     print('position')
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
    force_vector = np.sum(force_vector,0)
    return force_vector

def update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,boundaries,bc):
    """taking into account boundary conditions, drag, velocity, volume correction and spring forces, calculate the particle accelerations and update the particle positions and velocities"""
    for i, posn in enumerate(x0):
        if not (bc[0] == 'strain' and (np.any(i==boundaries[bc[1][0]]) or np.any(i==boundaries[bc[1][1]]))):
            a[i] = spring_force[i]/m[i] - drag * v0[i] + volume_correction_force[i] + bc_forces[i]
        else:
            a[i] = 0
        v1[i] = a[i] * dt + v0[i]
        x1[i] = a[i] * dt**2 + v0[i] * dt + x0[i]

#!!! generate traction forces or displacements based on some other criteria (choice of experimental setup with a switch statement? stress applied on boundary and then appropriately split onto the correct nodes in the correct directions in the correct amounts based on surface area?)

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

def get_accelerations_post_simulation_v2(x0,boundaries,springs,elements,kappa,l_e,bc):
    N = len(x0)
    m = np.ones(x0.shape[0])*1e-2
    a = np.empty(x0.shape,dtype=float)
    avg_vectors = get_average_edge_vectors(x0,elements)
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    correction_force_cy_nogil = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, correction_force_cy_nogil)
    spring_force_cy = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force_cy)
    for i, posn in enumerate(x0):
        if (np.any(i==boundaries[bc[1][0]]) or np.any(i==boundaries[bc[1][1]])):
            a[i] = (spring_force_cy[i] + correction_force_cy_nogil[i])/m[i]
        else:
            a[i] = 0
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

def post_plot_v2(node_posns,springs,boundary_conditions,output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(node_posns[:,0],node_posns[:,1],node_posns[:,2],'o')
    ax.set_xlim((-0.3,1.2*node_posns[:,0].max()))
    ax.set_ylim((0,1.2*node_posns[:,1].max()))
    ax.set_zlim((0,1.2*node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    for spring in springs:
        x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                          np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                          np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
        ax.plot(x,y,z)
    savename = output_dir + 'post_plotv2' + str(boundary_conditions[2]) +'.png'
    plt.savefig(savename)
    plt.close()

def post_plot_v3(node_posns,springs,boundary_conditions,boundaries,output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    boundary_nodes = np.zeros((1,),dtype=int)
    for key, val in boundaries.items():
        boundary_nodes = np.concatenate((boundary_nodes,val))
    ax.scatter(node_posns[boundary_nodes,0],node_posns[boundary_nodes,1],node_posns[boundary_nodes,2],'o')    
    ax.set_xlim((-0.3,1.2*node_posns[:,0].max()))
    ax.set_ylim((0,1.2*node_posns[:,1].max()))
    ax.set_zlim((0,1.2*node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    boundary_nodes = set(np.unique(boundary_nodes))
    for spring in springs:
        subset = set(spring[:2])
        if subset < boundary_nodes:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
            x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                            np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                            np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
            ax.plot(x,y,z)
    savename = output_dir + 'post_plotv3' + str(boundary_conditions[2]) +'.png'
    plt.savefig(savename)
    plt.close()

def update_positions_testing(x0,v0,m,elements,springs,kappa,l_e,bc,boundaries,dimensions,dt):
    """computes the next position and velocity for the given masses, initial conditions, and timestep"""
    N = len(x0)
    drag = 1
    x1 = np.empty(x0.shape,dtype=float)
    v1 = np.empty(v0.shape,dtype=float)
    a = np.empty(x0.shape,dtype=float)
    x1_cy = np.empty(x0.shape,dtype=float)
    v1_cy = np.empty(v0.shape,dtype=float)
    a_cy = np.empty(x0.shape,dtype=float)
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
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force)
    update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,boundaries,bc)
    fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
    update_positions_cy_nogil.update_positions(x0,v0,a_cy,x1_cy,v1_cy,dt,m,spring_force,volume_correction_force,drag,bc_forces,fixed_nodes)
    try:
        assert(np.allclose(x1_cy,x1))
        print('positions match between methods for update_positions()')
    except:
        print('positions do not match between methods for update_positions()')
        print(str(x1-x1_cy))
        print('the square root of the sum of the square of the differences in position is ' + str(np.sqrt(np.sum((x1-x1_cy)**2))))
        x1_diff = x1-x1_cy
        max_pct_error = 0
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x1_cy[i,j] == 0:
                    pct_error = -1
                else:
                    pct_error = x1_diff[i,j]/x1_cy[i,j]
                if pct_error > max_pct_error:
                    max_pct_error = pct_error
        print('max percent error is ' + str(max_pct_error*100) + '%')
        if max_pct_error*100 > 0.01:
            print('large error')
    try:
        assert(np.allclose(v1_cy,v1))
        print('velocities match between methods for update_positions()')
    except:
        print('velocities do not match between methods for update_positions()')
    try:
        assert(np.allclose(a_cy,a))
        print('accelerations match between methods for update_positions()')
    except:
        print('accelerations do not match between methods for update_positions()')

def update_positions_testing_suite():
    E = 1
    nu = 0.49
    l_e = 1e-8#cubic element side length
    Lx = np.arange(1e-8,5e-8,1e-8)
    Ly, Lz = Lx, Lx
    dt = 1e-3
    k = get_spring_constants(E, nu, l_e)
    kappa = get_kappa(E, nu)
    strain = 0.05
    boundary_conditions = ('strain',('left','right'),strain)
    for lx in Lx:
        for ly in Ly:
            for lz in Lz:
                dimensions = [lx,ly,lz]
                node_posns,elements,boundaries = discretize_space(lx,ly,lz,l_e)
                springs = create_springs(node_posns,k,l_e,dimensions)
                x0 = node_posns
                v0 = np.zeros(x0.shape)
                m = np.ones(x0.shape[0])*1e-2
                if boundary_conditions[0] == 'strain':
                    # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
                    surface = boundary_conditions[1][0]
                    if surface == 'right' or surface == 'left':
                        pinned_axis = 0
                    elif surface == 'top' or surface == 'bottom':
                        pinned_axis = 2
                    else:
                        pinned_axis = 1
                    x0[:,pinned_axis] *= (1+ boundary_conditions[2])
                update_positions_testing(x0,v0,m,elements,springs,kappa,l_e,boundary_conditions,boundaries,dimensions,dt)

def update_positions_performance_testing_suite():
    N_iter = 1000
    E = 1
    nu = 0.49
    l_e = 1e-8#cubic element side length
    Lx = np.arange(1e-8,5e-8,1e-8)
    Ly, Lz = Lx, Lx
    dt = 1e-3
    k = get_spring_constants(E, nu, l_e)
    kappa = get_kappa(E, nu)
    strain = 0.05
    boundary_conditions = ('strain',('left','right'),strain)
    py_tot_time = 0
    cy_tot_time = 0
    for lx in Lx:
        for ly in Ly:
            for lz in Lz:
                dimensions = [lx,ly,lz]
                node_posns,elements,boundaries = discretize_space(lx,ly,lz,l_e)
                springs = create_springs(node_posns,k,l_e,dimensions)
                x0 = node_posns
                v0 = np.zeros(x0.shape)
                m = np.ones(x0.shape[0])*1e-2
                if boundary_conditions[0] == 'strain':
                    # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
                    surface = boundary_conditions[1][0]
                    if surface == 'right' or surface == 'left':
                        pinned_axis = 0
                    elif surface == 'top' or surface == 'bottom':
                        pinned_axis = 2
                    else:
                        pinned_axis = 1
                    x0[:,pinned_axis] *= (1+ boundary_conditions[2])
                py_time, cy_time = update_positions_perf_testing(x0,v0,m,elements,springs,kappa,l_e,boundary_conditions,boundaries,dimensions,dt,N_iter)
                py_tot_time += py_time
                cy_tot_time += cy_time
    print('total py_time = {}, total cy_time = {}'.format(py_tot_time,cy_tot_time))
    print('Cython is {}x faster'.format(py_tot_time/cy_tot_time))

def update_positions_perf_testing(x0,v0,m,elements,springs,kappa,l_e,bc,boundaries,dimensions,dt,N_iter):
    """computes the next position and velocity for the given masses, initial conditions, and timestep"""
    N = len(x0)
    drag = 1
    x1 = np.empty(x0.shape,dtype=float)
    v1 = np.empty(v0.shape,dtype=float)
    a = np.empty(x0.shape,dtype=float)
    x1_cy = np.empty(x0.shape,dtype=float)
    v1_cy = np.empty(v0.shape,dtype=float)
    a_cy = np.empty(x0.shape,dtype=float)
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
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force)
    start = time.perf_counter()
    for i in range(N_iter):
        update_positions(x0,v0,a,x1,v1,dt,m,spring_force,volume_correction_force,drag,bc_forces,boundaries,bc)
    end = time.perf_counter()
    py_time = end-start
    start = time.perf_counter()
    for i in range(N_iter):
       fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
       update_positions_cy_nogil.update_positions(x0,v0,a_cy,x1_cy,v1_cy,dt,m,spring_force,volume_correction_force,drag,bc_forces,fixed_nodes)
    end = time.perf_counter()
    cy_time = end-start
    print('Lx = {}, Ly = {}, Lz = {}'.format(dimensions[0],dimensions[1],dimensions[2]))
    print('pytime = {}, cytime = {}'.format(py_time,cy_time))
    print('Cython is {}x faster'.format(py_time/cy_time))
    try:
        assert(np.allclose(x1_cy,x1))
        print('positions match between methods for update_positions()')
    except:
        print('positions do not match between methods for update_positions()')
        print(str(x1-x1_cy))
        print('the square root of the sum of the square of the differences in position is ' + str(np.sqrt(np.sum((x1-x1_cy)**2))))
        x1_diff = x1-x1_cy
        max_pct_error = 0
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x1_cy[i,j] == 0:
                    pct_error = -1
                else:
                    pct_error = x1_diff[i,j]/x1_cy[i,j]
                if pct_error > max_pct_error:
                    max_pct_error = pct_error
        print('max percent error is ' + str(max_pct_error*100) + '%')
        if max_pct_error*100 > 0.01:
            print('large error')
    return py_time,cy_time

#function to pass to scipy.integrate.solve_ivp()
#must be of the form fun(t,y)
#can be more than fun(t,y,additionalargs), and then the additional args are passed to solve_ivp via keyword argument args=(a,b,c,...) where a,b,c are the additional arguments to fun in order of apperance in the function definition
def fun(t,y,m,elements,springs,kappa,l_e,bc,boundaries,dimensions):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    x0 = np.reshape(y[:N],(int(np.round(N/3)),3))
    v0 = np.reshape(y[N:],(int(np.round(N/3)),3))
    N = x0.shape[0]
    drag = 1
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
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force)
    fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
    accel = (spring_force + volume_correction_force - drag * v0 + 
             bc_forces)/m[:,np.newaxis]
    for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
        for j in range(3):
            accel[fixed_nodes[i],j] = 0
            # v0[fixed_nodes[i],j] = 0#this shouldn't be necessary, since the initial conditions have the velocities set to zero, the accelerations being set to zero means they should never change (and there was overhead associated with this additional random access write)
    accel = np.reshape(accel,(3*N,1))
    result = np.concatenate((v0.reshape((3*N,1)),accel))
    result = np.reshape(result,(result.shape[0],))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def scipy_style():
    E = 1
    nu = 0.499
    l_e = .1#cubic element side length
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    dimensions = [Lx,Ly,Lz]
    k = mre.initialize.get_spring_constants(E, nu, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    t_f = 15

    node_posns,elements,boundaries = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    springs = mre.initialize.create_springs(node_posns,k,l_e,dimensions)
    boundary_conditions = ('strain',('left','right'),.05)

    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name.stem + '/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    mre.initialize.write_log(my_sim,output_dir)
    mre.initialize.write_init_file(node_posns,springs,boundaries,output_dir)
    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.21,-0.1)
    
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    for count, strain in enumerate(strains):
        boundary_conditions = ('strain',('left','right'),strain)
        try:
            start = time.time()
            sol = simulate_v2(node_posns,elements,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,t_f)
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        end_result = sol.y[:,-1]
        posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        a_var = get_accelerations_post_simulation_v2(posns,boundaries,springs,elements,kappa,l_e,boundary_conditions)
        m = 1e-2
        end_boundary_forces = a_var[boundaries['right']]*m
        boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,posns,boundary_conditions,output_dir)
        post_plot_v2(posns,springs,boundary_conditions,output_dir)
        post_plot_v3(posns,springs,boundary_conditions,boundaries,output_dir)
    
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,boundary_stress_xx_magnitude)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('stress_xx')
    savename = output_dir + 'stress-strain.png'
    plt.savefig(savename)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,effective_modulus)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('effective modulus')
    savename = output_dir + 'strain-modulus.png'
    plt.savefig(savename)
    plt.close()

def main():
    E = 1
    nu = 0.499
    l_e = .1#cubic element side length
    Lx = .5
    Ly = .5
    Lz = .5
    dt = 1e-3
    N_iter = 15000
    dimensions = [Lx,Ly,Lz]
    k = mre.initialize.get_spring_constants(E, nu, l_e)
    kappa = mre.initialize.get_kappa(E, nu)


    node_posns,elements,boundaries = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    springs = mre.initialize.create_springs(node_posns,k,l_e,dimensions)
    boundary_conditions = ('strain',('left','right'),.05)
    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name + '/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    mre.initialize.write_log(my_sim,output_dir)
    mre.initialize.write_init_file(posns,springs,boundaries,output_dir)

    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.81,-0.1)
    
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    for count, strain in enumerate(strains):
        boundary_conditions = ('strain',('left','right'),strain)
        try:
            start = time.time()
            posns, v, a = simulate(node_posns,elements,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,N_iter,dt)
        except:
            print('Exception raised during simulation')
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        max_accel = np.max(np.linalg.norm(a,axis=1))
        print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        a_var = get_accelerations_post_simulation_v2(posns,boundaries,springs,elements,kappa,l_e,boundary_conditions)
        m = 1e-2
        end_boundary_forces = a_var[boundaries['right']]*m
        boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        post_plot_v2(posns,springs,boundary_conditions,output_dir)
        post_plot_v3(posns,springs,boundary_conditions,boundaries,output_dir)
    
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,boundary_stress_xx_magnitude)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('stress_xx')
    savename = output_dir + 'stress-strain.png'
    plt.savefig(savename)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(strains,effective_modulus)
    ax.set_xlabel('strain_xx')
    ax.set_ylabel('effective modulus')
    savename = output_dir + 'strain-modulus.png'
    plt.savefig(savename)
    plt.close() 

    
    
# f = tb.open_file(output_dir+'temp.h5')
# calculate approximate volumes of each element after simulation
#deformed_avg_vecs = get_average_edge_vectors(posns, elements)
#deformed_vol = get_unit_cell_volume(deformed_avg_vecs)


if __name__ == "__main__":
    scipy_style()
    #main()

#I need to adjust the method to check for some convergence criteria based on the accelerations each particle is experiencing (or some other convergence criteria)
#I need to somehow record the particle positions at equilibrium for the initial configuration and under user defined strain/stress. stress may be the most appropriate initial choice, since strain can be computed more directly than the stress. but both methods should eventually be used.


#Goals:
    #1 implement a convergence criteria
    #2 record initial configuration and equilibrium configuration in appropriately named directories and with appropriate file names
    #3 update functionality to allow for iteration over a list of stresses, which should include outputting the relevant simulation information (date and time, script name and location, parameters (stiffness, poisson ratio, element size, length of system, time step, viscous drag coefficient?, time to complete, number of iterations))
    #4 write a separate post-simulation script for analyzing the strain response as a function of stress, and plot the visual
    #5 update functionality to allow iteration over a list of strains