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
# alternative gpu calculation of spring forces to avoid the use of atomic functions (see notes on laptop from april 7th, 2023):DONE, slower than use of atomic functions
# gpu implementation of the volume correction force, which will require the use of atomic functions, unless i can be clever with the use of multiple kernel calls to different subsets of the elements to avoid calculations for elements with shared nodes. doable, but i have to review how the elements are constructed and think through the implementation details more carefully. also requires syncrhonization commands to ensure that the kernel calls happen sequentially anyway, which might be slower than the use of atomic functions anyway
# use density values for PDMS blends and carbonyl iron to set the mass values for each node properly.
# use realistic values for the young's modulus
# consider and implement options for particle magnetization (mumax3 results are only useful for the two particle case with particle aligned field): options include froehlich-kennely, hyperbolic tangent (anhysteretic saturable models)

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
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import springs
import magnetism

#remember, purpose, signature, stub

#given a spring network and boundary conditions, determine the equilibrium displacements/configuration of the spring network
#if using numerical integration, at each time step output the nodal positions, velocities, and accelerations, or if using energy minimization, after each succesful energy minimization output the nodal positions
def simulate_v2(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,t_f,Hext,particle_size,chi,Ms,m):
    """Run a simulation of a hybrid mass spring system using the Verlet algorithm. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    epsilon = np.spacing(1)
    tolerance = 1e-4
    max_iters = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(projection= '3d')
    # x0,v0,m = node_posns.copy(), np.zeros(node_posns.shape), np.ones(node_posns.shape[0])*1e-2
    v0 = np.zeros(x0.shape)
    # if boundary_conditions[0] == 'strain':
    #     # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
    #     surface = boundary_conditions[1][1]
    #     if surface == 'right' or surface == 'left':
    #         pinned_axis = 0
    #     elif surface == 'top' or surface == 'bottom':
    #         pinned_axis = 2
    #     else:
    #         pinned_axis = 1
    #     x0[boundaries[surface],pinned_axis] *= (1 + boundary_conditions[2])   
        # x0[:,pinned_axis] *= (1+ boundary_conditions[2])
        # x0[boundaries[surface],pinned_axis] *= (1 + boundary_conditions[2])#!!! i have altered how the strain is applied, but this needds to be handled differently for the different methods... i need to deal with the applied boundary conditions more effectively. the single material case is simpler 
        # than the mre case. for the single material case i can try stretching each eleemnts x, y, or z postion by the same amount fora  simple axial strain
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(fun).set_integrator('dopri5',nsteps=100,verbosity=1)
    r.set_solout(solout)
    for i in range(max_iters):
        r.set_initial_value(y_0).set_f_params(m,elements,springs,particles,kappa,l_e,boundary_conditions,boundaries,dimensions,Hext,particle_size,chi,Ms)
        sol = r.integrate(t_f)
        plot_criteria_v_iteration(solutions,m,elements,springs,particles,kappa,l_e,boundary_conditions,boundaries,dimensions,Hext,particle_size,chi,Ms)
        a_var = get_accel_post_sim(sol,m,elements,springs,particles,kappa,l_e,boundary_conditions,boundaries,dimensions,Hext,particle_size,chi,Ms)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        if a_norm_avg < tolerance:
            break
        else:
            y_0 = sol
    
    # sol = sci.solve_ivp(fun,[0,t_f],y_0,args=(m,elements,springs,particles,kappa,l_e,boundary_conditions,boundaries,dimensions,Hext,particle_size,chi,Ms))
    return sol#returning a solution object, that can then have it's attributes inspected

def simulate_scaled(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    tolerance = 1e-4
    max_iters = 12
    v0 = np.zeros(x0.shape)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=1000,verbosity=1)
    r.set_solout(solout)
    for i in range(max_iters):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        sol = r.integrate(t_f)
        a_var = get_accel_post_sim_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        if a_norm_avg < tolerance:
            break
        else:
            y_0 = sol
    plot_criteria_v_iteration_scaled(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
    N_nodes = int(x0.shape[0])
    final_posns = np.reshape(sol[:N_nodes*3],(N_nodes,3))
    x1 = get_particle_center(particles[0],final_posns)
    x2 = get_particle_center(particles[1],final_posns)
    particle_separation = np.sqrt(np.sum(np.power(x1-x2,2)))
    return sol#returning a solution object, that can then have it's attributes inspected

#function for checking out convergence criteria vs iteration, currently showing the mean acceleration vector norm for the system
def plot_criteria_v_iteration(solutions,*args):
    iterations = np.array(solutions).shape[0]
    a_norm_avg = np.zeros((iterations,))
    for count, row in enumerate(solutions):
        a_var = get_accel_post_sim(np.array(row[1:]),*args)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
    fig = plt.figure()
    plt.plot(np.arange(iterations),a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('average acceleration norm')
    fig2 = plt.figure()
    plt.plot(np.arange(iterations-1),a_norm_avg[1:]-a_norm_avg[:-1])
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('change in average acceleraton norm')
    plt.show()

def plot_criteria_v_iteration_scaled(solutions,*args):
    iterations = np.array(solutions).shape[0]
    a_norm_avg = np.zeros((iterations,))
    a_norm_max = np.zeros((iterations,))
    a_particle_norm = np.zeros((iterations,))
    for count, row in enumerate(solutions):
        a_var = get_accel_post_sim_scaled(np.array(row[1:]),*args)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_max[count] = np.max(a_norms)
        a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
        a_particles = a_var[args[2][0],:]
        a_particle_norm[count] = np.linalg.norm(a_particles[0,:]) 
    delta_a_norm_avg = a_norm_avg[1:]-a_norm_avg[:-1]
    fig = plt.figure()
    plt.plot(np.arange(iterations),a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('average acceleration norm')
    fig2 = plt.figure()
    plt.plot(np.arange(iterations),a_norm_max)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('acceleration norm max')
    fig3 = plt.figure()
    plt.plot(np.arange(iterations-1),delta_a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('change in average acceleraton norm')
    fig4 = plt.figure()
    percent_change_a_norm_avg = 100*delta_a_norm_avg/a_norm_avg[:-1]
    plt.plot(np.arange(iterations-1),percent_change_a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('percent change in average acceleraton norm')
    fig5 = plt.figure()
    plt.plot(np.arange(iterations),a_particle_norm)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('particle acceleration norm')
    plt.show()

# placeholder functions for doing purpose, signature, stub when doing planning/design and wishlisting
def do_stuff():
    return 0

def do_other_stuff():
    return 0

#!!! generate traction forces or displacements based on some other criteria (choice of experimental setup with a switch statement? stress applied on boundary and then appropriately split onto the correct nodes in the correct directions in the correct amounts based on surface area?)

def remove_i(x,i):
    """remove the ith entry from an array"""
    shape = (x.shape[0]-1,) + x.shape[1:]
    y = np.empty(shape,dtype=float)
    y[:i] = x[:i]
    y[i:] = x[i+1:]
    return y

#function to pass to scipy.integrate.solve_ivp()
#must be of the form fun(t,y)
#can be more than fun(t,y,additionalargs), and then the additional args are passed to solve_ivp via keyword argument args=(a,b,c,...) where a,b,c are the additional arguments to fun in order of apperance in the function definition
def fun(t,y,m,elements,springs,particles,kappa,l_e,bc,boundaries,dimensions,Hext,particle_size,chi,Ms):
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
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
        for surface in bc[1]:
            do_stuff()
    else:
        fixed_nodes = np.ndarray([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    
    accel = (spring_force + volume_correction_force - drag * v0 + 
             bc_forces)/m[:,np.newaxis]
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative(Hext,particle_centers,particle_size,chi,Ms)
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]/particle.shape[0]/m[particle,np.newaxis]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    # for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
    #     for j in range(3):
    #         accel[fixed_nodes[i],j] = 0
            # v0[fixed_nodes[i],j] = 0#this shouldn't be necessary, since the initial conditions have the velocities set to zero, the accelerations being set to zero means they should never change (and there was overhead associated with this additional random access write)
    #TODO instead of reshaping as a 3N by 1, do (3*N,), and try concatenating. ideally should work and remove an additional and unnecessary reshape call
    accel = np.reshape(accel,(3*N,1))
    result = np.concatenate((v0.reshape((3*N,1)),accel))
    # alternative to concatenate is to create an empty array and then assign the values, this should in theory be faster
    result = np.reshape(result,(result.shape[0],))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def scaled_fun(t,y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    x0 = np.reshape(y[:N],(int(np.round(N/3)),3))
    v0 = np.reshape(y[N:],(int(np.round(N/3)),3))
    N = x0.shape[0]
    drag = 2
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
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
        for surface in bc[1]:
            do_stuff()
    else:
        fixed_nodes = np.ndarray([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    # accel = (spring_force + volume_correction_force - drag * v0 + 
    #          bc_forces)/m[:,np.newaxis]
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative(Hext,particle_centers,particle_size,chi,Ms)
    M_normalized = M/Ms
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    mag_forces *= beta/(particle_mass*(l_e**4))
    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    # for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
    #     for j in range(3):
    #         accel[fixed_nodes[i],j] = 0
            # v0[fixed_nodes[i],j] = 0#this shouldn't be necessary, since the initial conditions have the velocities set to zero, the accelerations being set to zero means they should never change (and there was overhead associated with this additional random access write)
    #TODO instead of reshaping as a 3N by 1, do (3*N,), and try concatenating. ideally should work and remove an additional and unnecessary reshape call
    accel = np.reshape(accel,(3*N,1))
    result = np.concatenate((v0.reshape((3*N,1)),accel))
    # alternative to concatenate is to create an empty array and then assign the values, this should in theory be faster
    result = np.reshape(result,(result.shape[0],))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def get_accel_post_sim(y,m,elements,springs,particles,kappa,l_e,bc,boundaries,dimensions,Hext,particle_size,chi,Ms):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
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
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
        for surface in bc[1]:
            do_stuff()
    else:
        fixed_nodes = np.ndarray([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    
    accel = (spring_force + volume_correction_force - drag * v0 + 
             bc_forces)/m[:,np.newaxis]
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative(Hext,particle_centers,particle_size,chi,Ms)
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]/particle.shape[0]/m[particle,np.newaxis]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    return accel

def get_accel_post_sim_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    x0 = np.reshape(y[:N],(int(np.round(N/3)),3))
    v0 = np.reshape(y[N:],(int(np.round(N/3)),3))
    N = x0.shape[0]
    drag = 2
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
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        fixed_nodes = np.concatenate((boundaries[bc[1][0]],boundaries[bc[1][1]]))
        for surface in bc[1]:
            do_stuff()
    else:
        fixed_nodes = np.ndarray([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    # accel = (spring_force + volume_correction_force - drag * v0 + 
    #          bc_forces)/m[:,np.newaxis]
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative(Hext,particle_centers,particle_size,chi,Ms)
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    mag_forces *= beta/(particle_mass*(l_e**4))
    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    return accel

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

def set_fixed_nodes(accel,fixed_nodes):
    for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
        #TODO almost certainly faster to remove the inner loop and just set each value to 0 in order, or using python semantics, just set the row to zero?
        for j in range(3):
            accel[fixed_nodes[i],j] = 0
    return accel

def place_two_particles(radius,l_e,dimensions,separation):
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Lx, Ly, Lz = dimensions
    # radius = 0.5*l_e# radius = l_e*(4.5)
    assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    Nel_x = np.round(Lx/l_e).astype(np.int32)
    Nel_y = np.round(Ly/l_e).astype(np.int32)
    Nel_z = np.round(Lz/l_e).astype(np.int32)
    #find the center of the simulated system
    center = (np.array([Nel_x,Nel_y,Nel_z])/2) * l_e
    #if there are an even number of elements in a direction, need to increment the central position by half an edge length so the particle centers match up with the centers of cubic elements
    if np.mod(Nel_x,2) == 0:
        center[0] += l_e/2
    if np.mod(Nel_y,2) == 0:
        center[1] += l_e/2
    if np.mod(Nel_z,2) == 0:
        center[2] += l_e/2
    #check particle separation to see if it is acceptable or not for the shift in particle placement from the simulation "center" to align with the cubic element centers
    if np.mod(separation,2) == 1:
        shift_l = (separation-1)*l_e/2
        shift_r = (separation+1)*l_e/2
    else:
        shift_l = separation*l_e/2
        shift_r = shift_l
    particle_nodes = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center-np.array([shift_l,0,0]),dimensions)
    particle_nodes2 = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center+np.array([shift_r,0,0]),dimensions)
    particles = np.vstack((particle_nodes,particle_nodes2))
    return particles

def place_two_particles_normalized(radius,l_e,dimensions,separation):
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Nel_x, Nel_y, Nel_z = dimensions
    # radius = 0.5*l_e# radius = l_e*(4.5)
    assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
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
    if np.mod(separation,2) == 1:
        shift_l = (separation-1)*1/2
        shift_r = (separation+1)*1/2
    else:
        shift_l = separation*1/2
        shift_r = shift_l
    particle_nodes = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center-np.array([shift_l,0,0]),dimensions)
    particle_nodes2 = mre.sphere_rasterization.place_sphere_normalized(radius_voxels,center+np.array([shift_r,0,0]),dimensions)
    particles = np.vstack((particle_nodes,particle_nodes2))
    return particles

def run_strain_sim(output_dir,strains,eq_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms):
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
            x0[boundaries[surface],pinned_axis] = eq_posns[boundaries[surface],pinned_axis] * (1 + boundary_conditions[2])   
        try:
            start = time.time()
            # sol = simulate_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,boundary_conditions,t_f,Hext,particle_size,chi,Ms,m)
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms)
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
        x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
        # posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        # a_var = mre.analyze.get_accelerations_post_simulation_v3(x0,boundaries,springs_var,elements,kappa,l_e,boundary_conditions)
        # end_boundary_forces = a_var[boundaries['right']]*m[boundaries['right'],np.newaxis]
        # boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        # effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,x0,boundary_conditions,output_dir)
        mre.analyze.post_plot_cut_normalized(eq_posns,x0,springs_var,particles,boundary_conditions,output_dir)

def run_hysteresis_sim(output_dir,Hext_series,eq_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_size,particle_mass,chi,Ms):
    for count, Hext in enumerate(Hext_series):
        #TODO better implementation of boundary conditions
        boundary_conditions = ('free',('free','free'),0) 
        try:
            start = time.time()
            # sol = simulate_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,boundary_conditions,t_f,Hext,particle_size,chi,Ms,m)
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms)
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
        x0 = np.reshape(end_result[:eq_posns.shape[0]*eq_posns.shape[1]],eq_posns.shape)
        # posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        # a_var = mre.analyze.get_accelerations_post_simulation_v3(x0,boundaries,springs_var,elements,kappa,l_e,boundary_conditions)
        # end_boundary_forces = a_var[boundaries['right']]*m[boundaries['right'],np.newaxis]
        # boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        # effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,x0,boundary_conditions,output_dir)
        # mre.analyze.post_plot_cut_normalized_hyst(eq_posns,x0,springs_var,particles,Hext,output_dir)

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
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    # normalized_posns = node_posns/l_e
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    # springs_var2 = np.empty((max_springs,4),dtype=np.float64)
    # num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, l_e)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    # for i in range(num_springs):
    #     correctness = np.allclose(springs_var[i,:3],springs_var2[i,:3])
    #     if not correctness:
    #         print(f'incompatible spring variable values at row {i}')
    springs_var = springs_var[:num_springs,:]
    separation = 5
    radius = 0.5*l_e# radius = l_e*(4.5)
    #2023-07-05 implementing two particle placement with the normalized length scheme in place. test the function outputs by comparing against the original implementations to ensure correctness
    # particles = place_two_particles(radius,l_e,dimensions,separation)
    particles = place_two_particles_normalized(radius,l_e,normalized_dimensions,separation)
    # particles_sorted = np.sort(particles)
    # particles2_sorted = np.sort(particles2)
    # correctness = np.allclose(particles_sorted,particles2_sorted)
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

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    mre.initialize.write_log(my_sim,output_dir)
    

    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.05,-0.02)
    Hext = np.array([0,0,0],dtype=np.float64)
    particle_size = radius
    chi = 131
    Ms = 1.9e6
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    # x0 = node_posns.copy()
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_size)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    #TODO properly motivated average acceleration l2 norm tolerance to consider system converged to a solution
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
            # sol = simulate_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,boundary_conditions,t_f,Hext,particle_size,chi,Ms,m)
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms)
            #below, a very incorrect attempt at implementing convergence criteria, where i have multiple issues, including not using the solution returned as the next starting point. i'll be implementing the convergence criteria inside the simulate_v2 function (which needs renaming)
            # a_norm_avg = 10
            # while(a_norm_avg > tolerance):
            #     sol = simulate_v2(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,boundary_conditions,t_f,Hext,particle_size,chi,Ms,m)
            #     a_var = get_accel_post_sim(sol,m,elements,springs_var,particles,kappa,l_e,boundary_conditions,boundaries,dimensions,Hext,particle_size,chi,Ms)
            #     a_norms = np.linalg.norm(a_var,axis=1)
            #     a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
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
        mre.initialize.write_output_file(count,x0,boundary_conditions,output_dir)
        # mre.analyze.post_plot_v2(x0,springs,boundary_conditions,output_dir)
        # mre.analyze.post_plot_v3(node_posns,x0,springs,boundary_conditions,boundaries,output_dir)
        # mre.analyze.post_plot_cut(normalized_posns,x0,springs_var,particles,dimensions,l_e,boundary_conditions,output_dir)
        mre.analyze.post_plot_cut_normalized(normalized_posns,x0,springs_var,particles,boundary_conditions,output_dir)
        # mre.analyze.post_plot_particle(node_posns,x0,particle_nodes,springs,boundary_conditions,output_dir)
    
def main2():
    E = 1e3
    nu = 0.499
    l_e = 3e-6#cubic element side length
    Lx = 15*l_e
    Ly = 11*l_e
    Lz = 11*l_e
    t_f = 30
    dimensions = np.array([Lx,Ly,Lz])
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    # normalized_posns = node_posns/l_e
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    # springs_var2 = np.empty((max_springs,4),dtype=np.float64)
    # num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, l_e)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    # for i in range(num_springs):
    #     correctness = np.allclose(springs_var[i,:3],springs_var2[i,:3])
    #     if not correctness:
    #         print(f'incompatible spring variable values at row {i}')
    springs_var = springs_var[:num_springs,:]
    separation = 4
    radius = 0.5*l_e# radius = l_e*(4.5)
    #2023-07-05 implementing two particle placement with the normalized length scheme in place. test the function outputs by comparing against the original implementations to ensure correctness
    # particles = place_two_particles(radius,l_e,dimensions,separation)
    particles = place_two_particles_normalized(radius,l_e,normalized_dimensions,separation)
    # particles_sorted = np.sort(particles)
    # particles2_sorted = np.sort(particles2)
    # correctness = np.allclose(particles_sorted,particles2_sorted)
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
    # boundary_conditions = ('strain',('left','right'),.05)

    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name.stem + '/tests/2_dip/'
    output_dir = '/mnt/c/Users/bagaw/Desktop/normalization_testing/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    mre.initialize.write_log(my_sim,output_dir)
    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.01,-0.02)
    mu0 = 4*np.pi*1e-7
    H_mag = 1/mu0
    Hext_series_magnitude = np.arange(1/mu0,1/mu0 + 1,2000)
    Hext_series = np.zeros((len(Hext_series_magnitude),3))
    Hext_series[:,0] = Hext_series_magnitude
    # Hext = np.array([10000,0,0],dtype=np.float64)
    particle_size = radius
    chi = 131
    Ms = 1.9e6
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
    # x0 = node_posns.copy()
    x0 = normalized_posns.copy()
    #mass assignment per node according to density of PDMS-527 (or matrix material) and carbonyl iron
    #if using periodic boundary conditions, the system is inside the bulk of an MRE, and each node should be assumed to be sharing 8 volume elements, and have the same mass. if we do not use periodic boundary conditions, the nodes on surfaces, edges, and corners need to have their mass adjusted based on the number of shared elements. periodic boundary conditions imply force accumulation at boundaries due to effective wrap around, magnetic interactions of particles are more complicated, but symmetries likely to reduce complexity of the calculations. Unlikely to attempt to deal with peridoic boudnary conditions in this work.
    N_nodes = (N_nodes_x*N_nodes_y*N_nodes_z).astype(np.int64)
    m, characteristic_mass, particle_mass = mre.initialize.get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_size)
    #calculating the characteristic time, t_c, as part of the process of calculating the scaling coefficients for the forces/accelerations
    k_e = k[0]
    characteristic_time = 2*np.pi*np.sqrt(characteristic_mass/k_e)
    #we will call the scaling coefficient beta
    beta = 4*(np.pi**2)*characteristic_mass/(k_e*l_e)
    #and if we want to have the scaling factor include the node mass we can calculate the suite of beta_i values, (or we could use the node_types variable and recognize that the masses of the non-particle nodes are all 2**n multiples of characteristic_mass/8 where n is an integer from 0 to 3)
    beta_i = beta/m
    #TODO properly motivated average acceleration l2 norm tolerance to consider system converged to a solution
    run_hysteresis_sim(output_dir,Hext_series,normalized_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_size,particle_mass,chi,Ms)

if __name__ == "__main__":
    main2()


 