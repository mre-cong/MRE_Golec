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
import mre.initialize
import mre.analyze
import mre.sphere_rasterization

#remember, purpose, signature, stub

#given a spring network and boundary conditions, determine the equilibrium displacements/configuration of the spring network
#if using numerical integration, at each time step output the nodal positions, velocities, and accelerations, or if using energy minimization, after each succesful energy minimization output the nodal positions
def simulate_v2(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,t_f):
    """Run a simulation of a hybrid mass spring system using the Verlet algorithm. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    
    epsilon = np.spacing(1)
    tol = epsilon*1e5
    # fig = plt.figure()
    # ax = fig.add_subplot(projection= '3d')
    # x0,v0,m = node_posns.copy(), np.zeros(node_posns.shape), np.ones(node_posns.shape[0])*1e-2
    v0,m = np.zeros(x0.shape), np.ones(x0.shape[0])*1e-2
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
    sol = sci.solve_ivp(fun,[0,t_f],y_0,args=(m,elements,springs,particles,kappa,l_e,boundary_conditions,boundaries,dimensions))
    return sol#returning a solution object, that can then have it's attributes inspected

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
def fun(t,y,m,elements,springs,particles,kappa,l_e,bc,boundaries,dimensions):
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
    accel = set_fixed_nodes(accel,fixed_nodes)
    vecsum = np.sum(accel[particles],axis=0)
    accel[particles] = vecsum/particles.shape[0]
    # for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
    #     for j in range(3):
    #         accel[fixed_nodes[i],j] = 0
            # v0[fixed_nodes[i],j] = 0#this shouldn't be necessary, since the initial conditions have the velocities set to zero, the accelerations being set to zero means they should never change (and there was overhead associated with this additional random access write)
    accel = np.reshape(accel,(3*N,1))
    result = np.concatenate((v0.reshape((3*N,1)),accel))
    # alternative to concatenate is to create an empty array and then assign the values, this should in theory be faster
    result = np.reshape(result,(result.shape[0],))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def set_fixed_nodes(accel,fixed_nodes):
    for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
        for j in range(3):
            accel[fixed_nodes[i],j] = 0
    return accel

def main():
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

    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    elements = mre.initialize.get_elements(node_posns,Lx,Ly,Lz,l_e)
    boundaries = mre.initialize.get_boundaries(node_posns)
    springs = mre.initialize.create_springs(node_posns,k,l_e,dimensions)
    radius = l_e*2.5
    assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
    radius_voxels = np.round(radius/l_e).astype(np.int32)
    center = (np.round(np.array([Lx/l_e,Ly/l_e,Lz/l_e]))/2) * l_e
    particle_nodes = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center+l_e/2,dimensions)
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
    x0 = node_posns.copy()
    for count, strain in enumerate(strains):
        boundary_conditions = ('strain',('left','right'),strain)
        if boundary_conditions[0] == 'strain':
        # !!! there has to be a better way to enforce the strain conditions, but for now this will do. the issue is that the two surfaces involved, if the surface does not sit on a constant value of 0 for the relevant axis the overall strain will be greater than that assigned, if the corner of the cubic volume always sits at the origin then this shouldn't be an issue, as only the relevant surface will be strained
            surface = boundary_conditions[1][1]
            if surface == 'right' or surface == 'left':
                pinned_axis = 0
            elif surface == 'top' or surface == 'bottom':
                pinned_axis = 2
            else:
                pinned_axis = 1
            x0[boundaries[surface],pinned_axis] = node_posns[boundaries[surface],pinned_axis] * (1 + boundary_conditions[2])   
        try:
            start = time.time()
            sol = simulate_v2(x0,elements,particle_nodes,boundaries,dimensions,springs,kappa,l_e,boundary_conditions,t_f)
        except Exception as inst:
            print('Exception raised during simulation')
            print(type(inst))
            print(inst)
        # post_plot(posns,c,k)
        end = time.time()
        delta = end - start
        #below, getting the solution at the final time, since the solution at all times is recorded (there has to be some way for me to alter the behavior of the function in my own separate version so that i'm not storing intermediate states i don't want or need (memory optimization))
        end_result = sol.y[:,-1]
        x0 = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # posns = np.reshape(end_result[:node_posns.shape[0]*node_posns.shape[1]],node_posns.shape)
        # max_accel = np.max(np.linalg.norm(a,axis=1))
        # print('max acceleration was %.4f' % max_accel)
        print('took %.2f seconds to simulate' % delta)
        a_var = mre.analyze.get_accelerations_post_simulation_v2(x0,boundaries,springs,elements,kappa,l_e,boundary_conditions)
        # a_var = mre.analyze.get_accelerations_post_simulation_v2(posns,boundaries,springs,elements,kappa,l_e,boundary_conditions)
        m = 1e-2
        end_boundary_forces = a_var[boundaries['right']]*m
        boundary_stress_xx_magnitude[count] = np.abs(np.sum(end_boundary_forces,0)[0])/(Ly*Lz)
        effective_modulus[count] = boundary_stress_xx_magnitude[count]/boundary_conditions[2]
        mre.initialize.write_output_file(count,x0,boundary_conditions,output_dir)
        mre.analyze.post_plot_v2(x0,springs,boundary_conditions,output_dir)
        mre.analyze.post_plot_v3(x0,springs,boundary_conditions,boundaries,output_dir)
    
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
    main()

#I need to adjust the method to check for some convergence criteria based on the accelerations each particle is experiencing (or some other convergence criteria)
#I need to somehow record the particle positions at equilibrium for the initial configuration and under user defined strain/stress. stress may be the most appropriate initial choice, since strain can be computed more directly than the stress. but both methods should eventually be used.


#Goals:
    #1 implement a convergence criteria
    #2 record initial configuration and equilibrium configuration in appropriately named directories and with appropriate file names
    #3 update functionality to allow for iteration over a list of stresses, which should include outputting the relevant simulation information (date and time, script name and location, parameters (stiffness, poisson ratio, element size, length of system, time step, viscous drag coefficient?, time to complete, number of iterations))
    #4 write a separate post-simulation script for analyzing the strain response as a function of stress, and plot the visual
    #5 update functionality to allow iteration over a list of strains