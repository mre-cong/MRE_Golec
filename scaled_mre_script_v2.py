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
def simulate_scaled(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,initialized_posns,output_dir):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    tolerance = 1e-4
    max_iters = 8
    v0 = np.zeros(x0.shape)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    N_nodes = int(x0.shape[0])
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=1000,verbosity=1)
    r.set_solout(solout)
    for i in range(max_iters):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        sol = r.integrate(t_f)
        a_var = get_accel_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        if i == 0:
            criteria = SimCriteria(solutions,initialized_posns,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        else:
            other_criteria = SimCriteria(solutions,initialized_posns,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
            criteria.append_criteria(other_criteria)
        # plot_criteria_v_iteration_scaled(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        final_posns = np.reshape(solutions[-1][1:N_nodes*3+1],(N_nodes,3))
        # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir)
        solutions = []
        r.set_solout(solout)
        if a_norm_avg < tolerance:
            break
        else:
            y_0 = sol
    # plot_criteria_v_time(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
    criteria.plot_criteria(output_dir)
    criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    # criteria.plot_criteria_v_time(output_dir)
    return sol#returning a solution object, that can then have it's attributes inspected

def simulate_scaled_alt(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,initialized_posns,output_dir,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    tolerance = 1e-4
    max_iters = 8
    v0 = np.zeros(x0.shape)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    N_nodes = int(x0.shape[0])
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=1000,verbosity=1)
    r.set_solout(solout)
    for i in range(max_iters):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        sol = r.integrate(t_f)
        a_var = get_accel_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        a_var_alt = get_accel_scaled_alt(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        if i == 0:
            criteria = SimCriteria(solutions,initialized_posns,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        else:
            other_criteria = SimCriteria(solutions,initialized_posns,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
            criteria.append_criteria(other_criteria)
        # plot_criteria_v_iteration_scaled(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
        final_posns = np.reshape(solutions[-1][1:N_nodes*3+1],(N_nodes,3))
        # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir)
        solutions = []
        r.set_solout(solout)
        if a_norm_avg < tolerance:
            break
        else:
            y_0 = sol
    # plot_criteria_v_time(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
    criteria.plot_criteria(output_dir)
    criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    # criteria.plot_criteria_v_time(output_dir)
    return sol#returning a solution object, that can then have it's attributes inspected

class SimCriteria:
    def __init__(self,solutions,initialized_posns,*args):
        self.get_criteria_per_iteration(solutions,initialized_posns,*args)
        self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]

    def get_criteria_per_iteration(self,solutions,initialized_posns,*args):
        iterations = np.array(solutions).shape[0]
        self.iter_number = np.arange(iterations)
        N_nodes = int((np.array(solutions).shape[1] - 1)/2/3)
        self.a_norm_avg = np.zeros((iterations,))
        self.a_norm_max = np.zeros((iterations,))
        self.particle_a_norm = np.zeros((iterations,))
        self.particle_separation = np.zeros((iterations,))
        self.v_norm_avg = np.zeros((iterations,))
        self.v_norm_max = np.zeros((iterations,))
        self.particle_v_norm = np.zeros((iterations,))
        self.time = np.array(solutions)[:,0]
        self.max_x = np.zeros((iterations,))
        self.max_y = np.zeros((iterations,))
        self.max_z = np.zeros((iterations,))
        self.min_x = np.zeros((iterations,))
        self.min_y = np.zeros((iterations,))
        self.min_z = np.zeros((iterations,))
        self.length_x = np.zeros((iterations,))
        self.length_y = np.zeros((iterations,))
        self.length_z = np.zeros((iterations,))
        for count, row in enumerate(solutions):
            a_var = get_accel_scaled(np.array(row[1:]),*args)
            a_norms = np.linalg.norm(a_var,axis=1)
            self.a_norm_max[count] = np.max(a_norms)
            self.a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
            a_particles = a_var[args[2][0],:]
            self.particle_a_norm[count] = np.linalg.norm(a_particles[0,:])
            final_posns = np.reshape(row[1:N_nodes*3+1],(N_nodes,3))
            final_v = np.reshape(row[N_nodes*3+1:],(N_nodes,3))
            v_norms = np.linalg.norm(final_v,axis=1)
            self.v_norm_max[count] = np.max(v_norms)
            self.v_norm_avg[count] = np.sum(v_norms)/np.shape(v_norms)[0]
            v_particles = final_v[args[2][0],:]
            self.particle_v_norm[count] = np.linalg.norm(v_particles[0,:])
            x1 = get_particle_center(args[2][0],final_posns)
            x2 = get_particle_center(args[2][1],final_posns)
            self.particle_separation[count] = np.sqrt(np.sum(np.power(x1-x2,2)))
            self.get_system_extent(final_posns,count)
    
    def get_system_extent(self,posns,count):
        """Assign values to the objects instance variables that give some sense of the physical extent/dimensions/size of the simulated system as the simulation progresses"""
        self.max_x[count] = np.max(posns[:,0])
        self.max_y[count] = np.max(posns[:,1])
        self.max_z[count] = np.max(posns[:,2])
        self.min_x[count] = np.min(posns[:,0])
        self.min_y[count] = np.min(posns[:,1])
        self.min_z[count] = np.min(posns[:,2])
        self.length_x[count] = self.max_x[count] - self.min_x[count]
        self.length_y[count] = self.max_y[count] - self.min_y[count]
        self.length_z[count] = self.max_z[count] - self.min_z[count]

    def plot_displacement_hist(self,final_posns,initialized_posns,output_dir):
        displacement = final_posns-initialized_posns
        displacement_norms = np.linalg.norm(displacement,axis=1)
        max_displacement = np.max(displacement_norms)
        mean_displacement = np.mean(displacement_norms)
        rms_displacement = np.sqrt(np.sum(np.power(displacement_norms,2))/np.shape(displacement_norms)[0])
        counts, bins = np.histogram(displacement_norms, bins=20)
        fig,ax = plt.subplots()
        ax.hist(bins[:-1], bins, weights=counts)
        sigma = np.std(displacement_norms)
        mu = mean_displacement
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)))**2)
        ax.plot(bins,y,'--')
        ax.set_title(f'Displacement Histogram\nMaximum {max_displacement}\nMean {mean_displacement}\n$\sigma={sigma}$\nRMS {rms_displacement}')
        ax.set_xlabel('displacement (units of l_e)')
        ax.set_ylabel('counts')
        savename = output_dir +'node_displacement_hist.png'
        plt.savefig(savename)
        plt.show()
        plt.close()

    def old_append_criteria(self,other):
        self.a_norm_avg = np.append(self.a_norm_avg,other.a_norm_avg)
        self.delta_a_norm = np.append(self.delta_a_norm,other.delta_a_norm)
        self.a_norm_max = np.append(self.a_norm_max,other.a_norm_max)
        self.particle_a_norm = np.append(self.particle_a_norm,other.particle_a_norm)
        self.particle_separation = np.append(self.particle_separation,other.particle_separation)
        self.iter_number = np.append(self.iter_number,np.max(self.iter_number)+other.iter_number + 1)
        self.time = np.append(self.time,np.max(self.time)+other.time)
    
    def append_criteria(self,other):
        """Append data from one SimCriteria object to another, by appending each member variable. Special cases (like time, iteration number) are appended in a manner to reflect the existence of prior integration iterations of the simulation"""
        #vars(self) returns a dictionary containing the member variables names and values as key-value pairs, allowing for this dynamic sort of access, meaning that extendingh the class with more member variables will allow this method to be used without changes (unless a new special case arises)
        my_keys = list(vars(self).keys())
        for key in my_keys:
            if key != 'iter_number' and key != 'time':
                vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],vars(other)[f'{key}'])
            elif key == 'iter_number':
                vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],np.max(vars(self)[f'{key}'])+vars(other)[f'{key}']+1)
            elif key == 'time':
                vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],np.max(vars(self)[f'{key}'])+vars(other)[f'{key}'])

    def plot_criteria(self,output_dir):
        #TODO Unfinished. use the member variable names to generate save names and figure labels automatically. have a separate variable passed to choose which variable to plot against (time, iteration, anything else?)
        """Generating plots of simulation criteria using matplotlib and using built-in python features to iterate over the instance variables of the object"""
        if not (os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
        my_keys = list(vars(self).keys())
        for key in my_keys:
            if key != 'iter_number' and key != 'time':
                fig = plt.figure()
                if key != 'delta_a_norm':
                    plt.plot(self.time,vars(self)[f'{key}'])
                else:
                    plt.plot(self.time[:self.delta_a_norm.shape[0]],vars(self)[f'{key}'])
                ax = plt.gca()
                ax.set_xlabel('scaled time')
                ax.set_ylabel(f'{key}')
                savename = output_dir + f'{key}_v_time.png'
                plt.savefig(savename)
        plt.show()
        plt.close('all')

    def plot_criteria_v_time(self,output_dir):
        if not (os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        fig1 = plt.figure()
        plt.plot(self.time,self.a_norm_avg)
        ax = plt.gca()
        ax.set_xlabel('scaled time')
        ax.set_ylabel('average acceleration norm')
        savename = output_dir + f'avg_accel_norm_v_time.png'
        plt.savefig(savename)

        fig2 = plt.figure()
        plt.plot(self.time,self.a_norm_max)
        ax = plt.gca()
        ax.set_xlabel('scaled time')
        ax.set_ylabel('acceleration norm max')
        savename = output_dir + f'accel_norm_max_v_time.png'
        plt.savefig(savename)

        fig3 = plt.figure()
        plt.plot(self.time[:self.delta_a_norm.shape[0]],self.delta_a_norm)
        ax = plt.gca()
        ax.set_xlabel('scaled time')
        ax.set_ylabel('change in average acceleraton norm')
        savename = output_dir + f'delta_avg_accel_norm_v_time.png'
        plt.savefig(savename)

        fig5 = plt.figure()
        plt.plot(self.time,self.particle_a_norm)
        ax = plt.gca()
        ax.set_xlabel('scaled time')
        ax.set_ylabel('particle acceleration norm')
        savename = output_dir + f'particle_accel_norm_v_time.png'
        plt.savefig(savename)

        fig6 = plt.figure()
        plt.plot(self.time,self.particle_separation)
        ax = plt.gca()
        ax.set_xlabel('scaled time')
        ax.set_ylabel('particle separation (in units of l_e)')
        savename = output_dir + f'particle_separation_v_time.png'
        plt.savefig(savename)
        plt.show()
        plt.close('all')
    
#function for checking out convergence criteria vs iteration, currently showing the mean acceleration vector norm for the system
def plot_criteria_v_iteration_scaled(solutions,N_nodes,integration_iteration,*args):
    iterations = np.array(solutions).shape[0]
    a_norm_avg = np.zeros((iterations,))
    a_norm_max = np.zeros((iterations,))
    a_particle_norm = np.zeros((iterations,))
    particle_separation = np.zeros((iterations,))
    output_dir = '/mnt/c/Users/bagaw/Desktop/scaled_mre_system_magnetic_particle_debugging/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    for count, row in enumerate(solutions):
        a_var = get_accel_scaled(np.array(row[1:]),*args)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_max[count] = np.max(a_norms)
        max_accel_node_ids = np.argmax(a_norms)
        check = max_accel_node_ids in args[2]
        a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
        a_particles = a_var[args[2][0],:]
        a_particle_norm[count] = np.linalg.norm(a_particles[0,:])
        final_posns = np.reshape(row[1:N_nodes*3+1],(N_nodes,3))
        x1 = get_particle_center(args[2][0],final_posns)
        x2 = get_particle_center(args[2][1],final_posns)
        particle_separation[count] = np.sqrt(np.sum(np.power(x1-x2,2)))
        # if count > 0:
        #     delta_a_particle = a_particle_norm[count] - a_particle_norm[count-1]
        #     if delta_a_particle > 10:
        #         get_accel_scaled(np.array(row[1:]),*args,debug_flag=True)
    delta_a_norm_avg = a_norm_avg[1:]-a_norm_avg[:-1]
    fig1 = plt.figure()
    plt.plot(np.arange(iterations),a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('average acceleration norm')
    savename = output_dir + f'avg_accel_norm{integration_iteration}.png'
    plt.savefig(savename)

    fig2 = plt.figure()
    plt.plot(np.arange(iterations),a_norm_max)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('acceleration norm max')
    savename = output_dir + f'accel_norm_max{integration_iteration}.png'
    plt.savefig(savename)

    fig3 = plt.figure()
    plt.plot(np.arange(iterations-1),delta_a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('change in average acceleraton norm')
    savename = output_dir + f'delta_avg_accel_norm{integration_iteration}.png'
    plt.savefig(savename)

    fig4 = plt.figure()
    percent_change_a_norm_avg = 100*delta_a_norm_avg/a_norm_avg[:-1]
    plt.plot(np.arange(iterations-1),percent_change_a_norm_avg)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('percent change in average acceleraton norm')
    savename = output_dir + f'percent_dekta_avg_accel_norm{integration_iteration}.png'
    plt.savefig(savename)

    fig5 = plt.figure()
    plt.plot(np.arange(iterations),a_particle_norm)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('particle acceleration norm')
    savename = output_dir + f'particle_accel_norm{integration_iteration}.png'
    plt.savefig(savename)

    fig6 = plt.figure()
    plt.plot(np.arange(iterations),particle_separation)
    ax = plt.gca()
    ax.set_xlabel('iteration number')
    ax.set_ylabel('particle separation (in units of l_e)')
    savename = output_dir + f'particle_separation{integration_iteration}.png'
    plt.savefig(savename)

    fig7 = plt.figure()
    times_array = np.array(solutions)[:,0]
    plt.plot(times_array,particle_separation)
    ax = plt.gca()
    ax.set_xlabel('scaled time')
    ax.set_ylabel('particle separation (in units of l_e)')
    savename = output_dir + f'particle_separation{integration_iteration}_vs_time.png'
    plt.savefig(savename)

    fig8 = plt.figure()
    plt.plot(times_array,a_particle_norm)
    ax = plt.gca()
    ax.set_xlabel('scaled time')
    ax.set_ylabel('particle acceleration norm')
    savename = output_dir + f'particle_accel_norm{integration_iteration}_vs_time.png'
    plt.savefig(savename)
    plt.show()
    plt.close('all')

# placeholder functions for doing purpose, signature, stub when doing planning/design and wishlisting
def do_stuff():
    return 0

def do_other_stuff():
    return 0

#!!! generate traction forces or displacements based on some other criteria (choice of experimental setup with a switch statement? stress applied on boundary and then appropriately split onto the correct nodes in the correct directions in the correct amounts based on surface area?)

#function to pass to scipy.integrate.solve_ivp()
#must be of the form fun(t,y)
#can be more than fun(t,y,additionalargs), and then the additional args are passed to solve_ivp via keyword argument args=(a,b,c,...) where a,b,c are the additional arguments to fun in order of apperance in the function definition
def scaled_fun(t,y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    accel = get_accel_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms)
    #TODO instead of reshaping as a 3N by 1, do (3*N,), and try concatenating. ideally should work and remove an additional and unnecessary reshape call
    N_nodes = int(np.round(N/3))
    accel = np.reshape(accel,(3*N_nodes,1))
    v0 = np.reshape(y[N:],(N_nodes,3))
    result = np.concatenate((v0.reshape((3*N_nodes,1)),accel))
    # alternative to concatenate is to create an empty array and then assign the values, this should in theory be faster
    result = np.reshape(result,(result.shape[0],))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def get_accel_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
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
        fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_size,chi,Ms,l_e)
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    mag_forces *= beta/(particle_mass*(l_e**4))
    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    if debug_flag:
        inspect_vcf = volume_correction_force[particles[0],:]
        inspect_springWCA = spring_force[particles[0],:]
        inspect_particle = accel[particles[0],:]
    return accel

def get_accel_scaled_alt(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_size,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
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
        fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)

    volume_correction_force_alt = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,scaled_kappa,correction_force_el,vectors,avg_vectors, volume_correction_force_alt)
    spring_force_alt = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, scaled_springs_var, spring_force_alt)

    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]

    volume_correction_force_alt *= m_ratio[:,np.newaxis]
    spring_force_alt *= m_ratio[:,np.newaxis]

    vcf_correctness = np.allclose(volume_correction_force,volume_correction_force_alt)
    springs_correctness = np.allclose(spring_force,spring_force_alt)
    print(f'Volume Correction Forces agreeing:{vcf_correctness}')
    print(f'Spring Forces Agreeing:{springs_correctness}')

    accel = spring_force + volume_correction_force - drag * v0 + bc_forces
    accel = set_fixed_nodes(accel,fixed_nodes)
    #for each particle, find the position of the center
    particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
    for i, particle in enumerate(particles):
        particle_centers[i,:] = get_particle_center(particle,x0)
    M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_size,chi,Ms,l_e)
    mag_forces = magnetism.get_dip_dip_forces(M,particle_centers,particle_size)
    mag_forces_alt = mag_forces * scaled_magnetic_force_coefficient
    mag_forces *= beta/(particle_mass*(l_e**4))

    mag_correctness = np.allclose(mag_forces,mag_forces_alt)
    print(f'magnetic forces agreeing:{mag_correctness}')

    for i, particle in enumerate(particles):
        accel[particle] += mag_forces[i]
    #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
    for particle in particles:
        vecsum = np.sum(accel[particle],axis=0)
        accel[particle] = vecsum/particle.shape[0]
    if debug_flag:
        inspect_vcf = volume_correction_force[particles[0],:]
        inspect_springWCA = spring_force[particles[0],:]
        inspect_particle = accel[particles[0],:]
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
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,eq_posns,output_dir)
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
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,eq_posns,output_dir)
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

def run_hysteresis_sim_testing_scaling_alt(output_dir,Hext_series,eq_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_size,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio):
    for count, Hext in enumerate(Hext_series):
        #TODO better implementation of boundary conditions
        boundary_conditions = ('free',('free','free'),0) 
        try:
            start = time.time()
            sol = simulate_scaled_alt(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,eq_posns,output_dir,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio)
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

def initialize(*args):
    """Given the parameters defining the system to be simulated, check if the variables have previously been initialized and saved out. If they have been, read them in, if they have not, initialize the variables and save them out an an init file."""
    #TODO: update and improve implementation of saving out/checking/reading in initialization files
    #need functionality to check some central directory containing initialization files
    system_string = f'E_{E}_le_{l_e}_Lx_{Lx}_Ly_{Ly}_Lz_{Lz}'
    current_dir = os.path.abspath('.')
    input_dir = current_dir + f'/init_files/{system_string}/'
    if not (os.path.isdir(input_dir)):#TODO add and statement that checks if the init file also exists?
        os.mkdir(input_dir)
        node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
        normalized_posns = mre.initialize.discretize_space(Lx/l_e,Ly/l_e,Lz/l_e,1)
        elements = springs.get_elements(normalized_posns, dimensions, 1)
        boundaries = mre.initialize.get_boundaries(normalized_posns)
        k = mre.initialize.get_spring_constants(E, nu, l_e)
        node_types = springs.get_node_type(normalized_posns.shape[0],boundaries,dimensions,1)
        k = np.array(k,dtype=np.float64)
        max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
        springs_var = np.empty((max_springs,4),dtype=np.float64)
        num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, 1)
        springs_var = springs_var[:num_springs,:]
        separation = 5
        radius = 0.5*l_e# radius = l_e*(4.5)
        particles = place_two_particles(radius,l_e,dimensions,separation)
        mre.initialize.write_init_file(node_posns,springs_var,elements,particles,boundaries,input_dir)
    elif os.path.isfile(input_dir+'init.h5'):
        node_posns, springs_var, elements, boundaries = mre.initialize.read_init_file(input_dir+'init.h5')
        #TODO implement support functions for particle placement to ensure matching to existing grid of points and avoid unnecessary repetition
        #radius = l_e*0.5
        separation = 5
        radius = 0.5*l_e# radius = l_e*(4.5)
        particles = place_two_particles(radius,l_e,dimensions,separation)
        # #TODO do better at placing multiple particles, make the helper functionality to ensure placement makes sense
    else:
        node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
        normalized_posns = mre.initialize.discretize_space(Lx,Ly,Lz,1)
        # normalized_posns = node_posns/l_e
        elements = springs.get_elements(normalized_posns, dimensions, 1)
        boundaries = mre.initialize.get_boundaries(normalized_posns)
        k = mre.initialize.get_spring_constants(E, nu, l_e)
        node_types = springs.get_node_type(normalized_posns.shape[0],boundaries,dimensions,1)
        k = np.array(k,dtype=np.float64)
        max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
        springs_var = np.empty((max_springs,4),dtype=np.float64)
        num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, 1)
        springs_var = springs_var[:num_springs,:]
        separation = 5
        radius = 0.5*l_e# radius = l_e*(4.5)
        particles = place_two_particles(radius,l_e,dimensions,separation)

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

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    

    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.05,-0.02)
    Hext = np.array([0,0,0],dtype=np.float64)
    particle_size = radius
    chi = 131
    Ms = 1.9e6
    effective_modulus = np.zeros(strains.shape)
    boundary_stress_xx_magnitude = np.zeros(strains.shape)
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
            sol = simulate_scaled(x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_size,particle_mass,chi,Ms,normalized_posns,output_dir)
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
    separation = 4
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
    # boundary_conditions = ('strain',('left','right'),.05)

    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name.stem + '/tests/2_dip/'
    output_dir = '/mnt/c/Users/bagaw/Desktop/MRE/two_particle/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.01,-0.02)
    mu0 = 4*np.pi*1e-7
    H_mag = 1/mu0
    Hext_series_magnitude = np.arange(H_mag,H_mag + 1,2000)
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
    #new beta coefficient without characteristic mass
    beta_new = 4*(np.pi**2)/(k_e*l_e)
    m_ratio = characteristic_mass/m
    scaled_kappa = (l_e**2)*beta_new
    example_scaled_k = k[0]*beta_new*l_e
    scaled_magnetic_force_coefficient = beta/(particle_mass*(l_e**4))
    run_hysteresis_sim(output_dir,Hext_series,normalized_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_size,particle_mass,chi,Ms)

def main3():
    """Testing scaled coefficients for forces and use of mass ratio variable in lieu of previous implementations of scaling"""
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
    separation = 4
    radius = 0.5*l_e# radius = l_e*(4.5)
    particles = place_two_particles_normalized(radius,l_e,normalized_dimensions,separation)
    kappa = mre.initialize.get_kappa(E, nu)

    script_name = lib_programname.get_path_executed_script()
    # check if the directory for output exists, if not make the directory
    current_dir = os.path.abspath('.')
    output_dir = current_dir + '/results/' + script_name.stem + '/tests/2_dip/'
    output_dir = '/mnt/c/Users/bagaw/Desktop/MRE/two_particle/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    my_sim = mre.initialize.Simulation(E,nu,l_e,Lx,Ly,Lz)
    my_sim.set_time(t_f)
    my_sim.write_log(output_dir)
    # strains = np.array([0.01])
    strains = np.arange(-.001,-0.01,-0.02)
    mu0 = 4*np.pi*1e-7
    H_mag = 1/mu0
    Hext_series_magnitude = np.arange(H_mag,H_mag + 1,2000)
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
    #new beta coefficient without characteristic mass
    beta_new = 4*(np.pi**2)/(k_e*l_e)
    m_ratio = characteristic_mass/m
    scaled_kappa = (l_e**2)*beta_new*kappa
    scaled_k = k*beta_new*l_e
    scaled_magnetic_force_coefficient = beta/(particle_mass*(l_e**4))
    scaled_springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, scaled_springs_var, max_springs, scaled_k, dimensions_normalized, 1)
    scaled_springs_var = scaled_springs_var[:num_springs,:]
    run_hysteresis_sim_testing_scaling_alt(output_dir,Hext_series,normalized_posns,x0,elements,particles,boundaries,dimensions,springs_var,kappa,l_e,beta,beta_i,t_f,particle_size,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio)

if __name__ == "__main__":
    main3()