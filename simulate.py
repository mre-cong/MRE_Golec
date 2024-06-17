"""
Created on Tues October 3 10:20:19 2023

@author: David Marchfield
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy as cp
from cupyx.profiler import benchmark
import cupyx
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib
import os
import get_volume_correction_force_cy_nogil
import get_spring_force_cy
import magnetism
import mre.initialize
import mre.analyze
mu0 = 4*np.pi*1e-7
# plt.switch_backend('TkAgg')

#given a spring network and boundary conditions, determine the equilibrium displacements/configuration of the spring network
#if using numerical integration, at each time step output the nodal positions, velocities, and accelerations, or if using energy minimization, after each succesful energy minimization output the nodal positions
def simulate_scaled(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,initialized_posns,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    v0 = np.zeros(x0.shape)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    if plotting_flag:
        mre.analyze.plot_center_cuts(initialized_posns,x0,springs,particles,boundary_conditions,output_dir,tag='starting_configuration')
    #TODO decide if you want to bother with doing a backtracking if the system diverges. there is a significant memory overhead associated with this approach.
    backstop_solution = y_0.copy()
    N_nodes = int(x0.shape[0])
    my_nsteps = max_integration_steps
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=max_integration_steps,verbosity=1)
    if criteria_flag:
        r.set_solout(solout)
    max_displacement = np.zeros((max_integrations,))
    mean_displacement = np.zeros((max_integrations,))
    return_status = 1
    for i in range(max_integrations):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        print(f'starting integration run {i+1}')
        sol = r.integrate(t_f)
        a_var = get_accel_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        max_accel_norm_avg = 0
        if criteria_flag:
            if i == 0:
                criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(criteria.a_norm_avg)
            else:
                other_criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(other_criteria.a_norm_avg)
        final_posns = np.reshape(sol[:N_nodes*3],(N_nodes,3))
        final_v = np.reshape(sol[N_nodes*3:],(N_nodes,3))
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        #below is a 2D scatter plot of center cuts with the markers colored to give depth information
        # mre.analyze.center_cut_visualization(initialized_posns,final_posns,springs,particles,output_dir,tag=f'{i}th_configuration')
        #below is the original 3D scatter plot of center cuts which colored polymer nodes and springs differently than particle nodes and springs
        # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag=f'{i}th_configuration')
        #below is an attempt to change the original 3D scatter plot approach to a 2D approach, with no color information to provide sense of depth
        if i == 0:
            tag = '1st_configuration'
        elif i == 1:
            tag = '2nd_configuration'
        elif i == 2:
            tag = '3rd_configuration'
        else:
            tag = f'{i+1}th_configuration'
        if plotting_flag:
            mre.analyze.plot_center_cuts(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag)
        if a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            return_status = 0
            break
        elif max_accel_norm_avg > 1e3:
            print(f'strong accelerations detected during integration run {i+1}')
            my_nsteps = int(my_nsteps/2)
            if my_nsteps < 10:
                print(f'total steps allowed down to: {my_nsteps}\n breaking out with last acceptable solution')
                sol = backstop_solution.copy()
                del backstop_solution
                del y_0
                del solutions
                return_status = -1
                break
            print(f'restarting from last acceptable solution with acceleration norm mean of {criteria.a_norm_avg[-1]}')
            print(f'running with halved maximum number of steps: {my_nsteps}')
            r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=my_nsteps,verbosity=1)
            print(f'Increasing drag coefficient from {drag} to {10*drag}')
            drag *= 10
            y_0 = backstop_solution.copy()
        else:
            print(f'Post-Integration norms\nacceleration norm average = {a_norm_avg}\nvelocity norm average = {v_norm_avg}')
            y_0 = sol.copy()
            last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            # if i == 0:
            #     #TODO, update to handle hysteresis/strain sims, where the starting position to compare the final positions to may not be the initiailized positions of the system
            #     mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,initialized_posns)
            # else:
            #     last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            #     mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            #     criteria.append_criteria(other_criteria)
            backstop_solution = sol.copy()
        if criteria_flag:
            solutions = []
            r.set_solout(solout)
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
    plot_displacement_v_integration(max_integrations,mean_displacement,max_displacement,output_dir)
    if criteria_flag:
        mre.initialize.write_criteria_file(criteria,output_dir)
        criteria.plot_criteria_subplot(output_dir)
        criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag='end_configuration')
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def simulate_scaled_rotation(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,initialized_posns,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False,get_time_flag=False,get_norms_flag=False):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a ... dictionary(?) where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    sim_time = []
    solution_norms = []
    derivative_norms = []
    def solout_norms(t,y):
        solution_norms.append(np.linalg.norm(y))
        accel = get_accel_scaled_rotation(y,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
        N = int(np.round(y.shape[0]/2))
        accel = np.reshape(accel,(3*N_nodes,))
        derivatives = np.concatenate((y[N:],accel))
        derivative_norms.append(np.linalg.norm(derivatives))
    def solout_timestep(t,y):
        if sim_time == []:
            sim_time.append(t)
        else:
            new_time = t + np.max(np.array(sim_time))
            sim_time.append(new_time)
        # solutions.append([*y])
    def solout(t,y):
        solutions.append([t,*y])
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    v0 = np.zeros(x0.shape)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    if plotting_flag:
        mre.analyze.plot_center_cuts(initialized_posns,x0,springs,particles,boundary_conditions,output_dir,tag='starting_configuration')
    #TODO decide if you want to bother with doing a backtracking if the system diverges. there is a significant memory overhead associated with this approach.
    backstop_solution = y_0.copy()
    N_nodes = int(x0.shape[0])
    my_nsteps = max_integration_steps
    particle_moment_of_inertia = (2/5)*particle_mass*np.power(particle_radius,2)
    #needs to be scaled, but the scaling here includes (inside of beta) the characteristic time, squared, which is necessary for scaling the angular acceleration. the angular acceleration is scaling handled internally in the function get_accel_scaled_rotation(), but we need to account for the term to scale the moment of inertia properly, so we remove the time scaling that was previously involved here. the particle mass and the number of nodes making up the particle is used to go from beta to beta_i
    # characteristic_time_squared = beta*l_e
    # scaled_moment_of_inertia = particle_moment_of_inertia*beta/(particle_mass/particles.shape[1])/l_e/characteristic_time_squared
    # using the fact that we have an analytical expression, I will rewrite the above initialization of the scaled moment of inertia in a way that uses less operations
    scaled_moment_of_inertia = particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2))
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun_rotation).set_integrator('dopri5',nsteps=max_integration_steps,verbosity=1)
    if criteria_flag:
        r.set_solout(solout)
    elif get_time_flag:
        r.set_solout(solout_timestep)
    elif get_norms_flag:
        r.set_solout(solout_norms)
    max_displacement = np.zeros((max_integrations,))
    mean_displacement = np.zeros((max_integrations,))
    return_status = 1
    for i in range(max_integrations):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        print(f'starting integration run {i+1}')
        sol = r.integrate(t_f)
        a_var = get_accel_scaled_rotation(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        max_accel_norm_avg = 0
        if criteria_flag:
            if i == 0:
                criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(criteria.a_norm_avg)
            else:
                other_criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(other_criteria.a_norm_avg)
        final_posns = np.reshape(sol[:N_nodes*3],(N_nodes,3))
        final_v = np.reshape(sol[N_nodes*3:],(N_nodes,3))
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        #below is a 2D scatter plot of center cuts with the markers colored to give depth information
        # mre.analyze.center_cut_visualization(initialized_posns,final_posns,springs,particles,output_dir,tag=f'{i}th_configuration')
        #below is the original 3D scatter plot of center cuts which colored polymer nodes and springs differently than particle nodes and springs
        # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag=f'{i}th_configuration')
        #below is an attempt to change the original 3D scatter plot approach to a 2D approach, with no color information to provide sense of depth
        if i == 0:
            tag = '1st_configuration'
        elif i == 1:
            tag = '2nd_configuration'
        elif i == 2:
            tag = '3rd_configuration'
        else:
            tag = f'{i+1}th_configuration'
        if plotting_flag:
            mre.analyze.plot_center_cuts(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag)
        if a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            return_status = 0
            break
        elif max_accel_norm_avg > 1e3:
            print(f'strong accelerations detected during integration run {i+1}')
            my_nsteps = int(my_nsteps/2)
            if my_nsteps < 10:
                print(f'total steps allowed down to: {my_nsteps}\n breaking out with last acceptable solution')
                sol = backstop_solution.copy()
                del backstop_solution
                del y_0
                del solutions
                return_status = -1
                break
            print(f'restarting from last acceptable solution with acceleration norm mean of {criteria.a_norm_avg[-1]}')
            print(f'running with halved maximum number of steps: {my_nsteps}')
            r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=my_nsteps,verbosity=1)
            print(f'Increasing drag coefficient from {drag} to {10*drag}')
            drag *= 10
            y_0 = backstop_solution.copy()
        else:
            print(f'Post-Integration norms\nacceleration norm average = {a_norm_avg}\nvelocity norm average = {v_norm_avg}')
            y_0 = sol.copy()
            last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            # if i == 0:
            #     #TODO, update to handle hysteresis/strain sims, where the starting position to compare the final positions to may not be the initiailized positions of the system
            #     mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,initialized_posns)
            # else:
            #     last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            #     mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            #     criteria.append_criteria(other_criteria)
            backstop_solution = sol.copy()
        if criteria_flag:
            solutions = []
            r.set_solout(solout)
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
    plot_displacement_v_integration(max_integrations,mean_displacement,max_displacement,output_dir)
    if criteria_flag:
        mre.initialize.write_criteria_file(criteria,output_dir)
        criteria.plot_criteria_subplot(output_dir)
        criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    if get_time_flag:
        plot_simulation_time_versus_integration_step(sim_time,output_dir)
        # np.save(output_dir+'solutions.npy',solutions,allow_pickle=False)
    if get_norms_flag:
        plot_solution_and_derivative_norms_versus_integration_step(solution_norms,derivative_norms,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag='end_configuration')
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def plot_simulation_time_versus_integration_step(sim_time,output_dir):
    """Plot the simulation time versus integration step. Shows how the adaptive time step is adjusting through the course of numerical integration"""
    fig, axs = plt.subplots(1,3)
    default_width, default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(100)
    integration_number = np.arange(len(sim_time))
    delta_t = np.array(sim_time[1:]) - np.array(sim_time[:-1])
    axs[0].plot(sim_time[1:],delta_t,'.')
    axs[0].set_title('Time Step Taken')
    axs[0].set_xlabel('scaled time')
    axs[0].set_ylabel('time step')
    axs[1].plot(np.arange(len(delta_t)),delta_t,'.')
    axs[1].set_title('Time Step Taken')
    axs[1].set_xlabel('integration number')
    axs[1].set_ylabel('time step')
    axs[2].plot(integration_number,sim_time,'.')
    axs[2].set_title('Total Time')
    axs[2].set_xlabel('integration number')
    axs[2].set_ylabel('total scaled time')
    savename = output_dir + 'timestep_per_iteration_and_time.png'
    plt.savefig(savename)
    plt.close()
    np.save(output_dir+'timesteps.npy',delta_t,allow_pickle=False)

def plot_solution_and_derivative_norms_versus_integration_step(solution_norms,derivative_norms,output_dir):
    """Plot the simulation time versus integration step. Shows how the adaptive time step is adjusting through the course of numerical integration"""
    fig, axs = plt.subplots(1,3)
    default_width, default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    fig.set_dpi(100)
    integration_number = np.arange(len(solution_norms))
    axs[0].plot(integration_number,solution_norms,'.')
    axs[0].set_title('Sol Vector Norms')
    axs[0].set_xlabel('integration number')
    axs[0].set_ylabel('Sol Vector Norm')
    axs[1].plot(integration_number,derivative_norms,'.')
    axs[1].set_title('Derivative Vector Norms')
    axs[1].set_xlabel('integration number')
    axs[1].set_ylabel('Derivative Vector Norm')
    axs[2].plot(integration_number,np.array(solution_norms)/np.array(derivative_norms),'.')
    axs[2].set_title('Ratio of Norms')
    axs[2].set_xlabel('integration number')
    axs[2].set_ylabel('Sol Norm/Derivative Norm')
    savename = output_dir + 'solution_and_derivative_norms_vs_integration_number.png'
    plt.savefig(savename)
    plt.close()

def extend_simulate_scaled(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,initialized_posns,output_dir,starting_checkpoint_count=0,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=True,plotting_flag=True,persistent_checkpointing_flag=False):
    """Extend a simulation of a hybrid mass spring system from a checkpoint file using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])

    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    y_0 = x0
    # y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    backstop_solution = y_0.copy()
    N_nodes = int(initialized_posns.shape[0])
    my_nsteps = max_integration_steps
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=max_integration_steps,verbosity=1)
    if criteria_flag:
        r.set_solout(solout)
    max_displacement = np.zeros((max_integrations,))
    mean_displacement = np.zeros((max_integrations,))
    return_status = 1
    for i in range(starting_checkpoint_count+1,max_integrations+starting_checkpoint_count+1):
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        print(f'starting integration run {i+1}')
        sol = r.integrate(t_f)
        a_var = get_accel_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        max_accel_norm_avg = 0
        if criteria_flag:
            if i == starting_checkpoint_count+1:#TODO if extending simulation and deciding i want to calculate criteria (Which maybe i shouldn't even allow at all, maybe that has to happen post simulation) I need to reflect the fact that i am starting from a non-zero i, but won't have an initial SimCriteria object until after the first extension run
                criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(criteria.a_norm_avg)
            else:
                other_criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(other_criteria.a_norm_avg)
        final_posns = np.reshape(sol[:N_nodes*3],(N_nodes,3))
        final_v = np.reshape(sol[N_nodes*3:],(N_nodes,3))
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if i == 0:
            tag = '1st_configuration'
        elif i == 1:
            tag = '2nd_configuration'
        elif i == 2:
            tag = '3rd_configuration'
        else:
            tag = f'{i+1}th_configuration'
        if plotting_flag:
            mre.analyze.plot_center_cuts(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag)
        if a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if i != starting_checkpoint_count+1 and criteria_flag:
                criteria.append_criteria(other_criteria)
            return_status = 0
            break
        elif max_accel_norm_avg > 1e3:
            print(f'strong accelerations detected during integration run {i+1}')
            my_nsteps = int(my_nsteps/2)
            if my_nsteps < 10:
                print(f'total steps allowed down to: {my_nsteps}\n breaking out with last acceptable solution')
                sol = backstop_solution.copy()
                del backstop_solution
                del y_0
                del solutions
                return_status = -1
                break
            print(f'restarting from last acceptable solution with acceleration norm mean of {criteria.a_norm_avg[-1]}')
            print(f'running with halved maximum number of steps: {my_nsteps}')
            r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=my_nsteps,verbosity=1)
            print(f'Increasing drag coefficient from {drag} to {10*drag}')
            drag *= 10
            y_0 = backstop_solution.copy()
        else:
            print(f'Post-Integration norms\nacceleration norm average = {a_norm_avg}\nvelocity norm average = {v_norm_avg}')
            y_0 = sol.copy()
            last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            mean_displacement[i-(starting_checkpoint_count+1)], max_displacement[i-(starting_checkpoint_count+1)] = get_displacement_norms(final_posns,last_posns)
            if i != starting_checkpoint_count+1 and criteria_flag:
                criteria.append_criteria(other_criteria)
            backstop_solution = sol.copy()
        if criteria_flag:
            solutions = []
            r.set_solout(solout)
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        if not persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
    plot_displacement_v_integration(max_integrations,mean_displacement,max_displacement,output_dir)
    if criteria_flag:
        criteria.plot_criteria_subplot(output_dir+'extended_folder/')
        criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir+'extended_folder/')
        mre.initialize.write_criteria_file(criteria,output_dir+'extended_folder/') 
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def plot_residual_vector_norms_hist(a_norms,output_dir,tag=""):
    """Plot a histogram of the acceleration of the nodes. Intended for analyzing the behavior at the end of simulations that are ended before convergence criteria are met."""
    max_accel = np.max(a_norms)
    mean_accel = np.mean(a_norms)
    rms_accel = np.sqrt(np.sum(np.power(a_norms,2))/np.shape(a_norms)[0])
    counts, bins = np.histogram(a_norms, bins=30)
    fig,ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    ax.hist(a_norms,bins=30)
    # ax.hist(bins[:-1], bins, weights=counts)
    sigma = np.std(a_norms)
    mu = mean_accel
    ax.set_title(f'Residual '+tag+f' Histogram\nMaximum {max_accel}\nMean {mean_accel}\n$\sigma={sigma}$\nRMS {rms_accel}')
    ax.set_xlabel(tag + ' norm')
    ax.set_ylabel('counts')
    savename = output_dir +'node_residual_'+tag+'_hist.png'
    mre.analyze.format_figure(ax)
    plt.savefig(savename)
    plt.close()
    np.save(output_dir+'node_residual_'+tag+'_hist',a_norms)

def get_displacement_norms(final_posns,start_posns):
    """Given the final and starting positions, return the mean and maximum node displacement."""
    displacement = final_posns-start_posns
    displacement_norms = np.linalg.norm(displacement,axis=1)
    max_displacement = np.max(displacement_norms)
    mean_displacement = np.mean(displacement_norms)
    return mean_displacement,max_displacement

def plot_displacement_v_integration(num_integration_rounds,mean_displacement,max_displacement,output_dir):
    """Given the number of integrations,array of mean and maximum displacements, and a directory to save to, generate and save a figure with two plots of mean and maximum node displacement versus integration."""
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(num_integration_rounds),mean_displacement[:num_integration_rounds],'r--',label='mean')
    axs[1].plot(np.arange(num_integration_rounds),max_displacement[:num_integration_rounds],'k-',label='max')
    axs[0].set_title('displacement between integration iterations')
    axs[0].set_ylabel('displacement mean (units of l_e)')
    axs[1].set_ylabel('displacement max (units of l_e)')
    axs[1].set_xlabel('iteration number')
    # Hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig(output_dir+'displacement.png')
    plt.close()
    np.save(output_dir+'mean_displacement',mean_displacement)
    np.save(output_dir+'max_displacement',max_displacement)
    
def plot_snapshots(snapshot_stepsize,step_size,total_entries,snapshot_values,output_dir,tag=""):
    """Generate and save a figure showing the evolution of some value at particular steps throughout integration."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    simulation_time = snapshot_stepsize*snapshot_values.shape[0]*step_size
    snapshot_times = np.arange(0,simulation_time,step_size*snapshot_stepsize)
    axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries],'o-')
    axs[0].set_xlabel('simulation time')
    axs[0].set_ylabel(f'{tag}')
    mre.analyze.format_figure(axs[0])
    midpoint = int(total_entries/2)
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries],'o-')
    axs[1].set_xlabel('simulation time')
    axs[1].set_ylabel(f'{tag}')
    mre.analyze.format_figure(axs[1])
    plt.savefig(output_dir+f'{tag}_snapshots.png')
    plt.close()
    np.save(output_dir+'simulation_time',simulation_time)
    np.save(output_dir+f'{tag}_snapshots',snapshot_values)

def plot_snapshots_vector_components(snapshot_stepsize,step_size,total_entries,snapshot_values,output_dir,tag=""):
    """Generate and save a figure showing the evolution of some value at particular steps throughout integration."""
    simulation_time = snapshot_stepsize*snapshot_values.shape[0]*step_size
    snapshot_times = np.arange(0,simulation_time,step_size*snapshot_stepsize)
    if 'boundary' in tag:
        fig, axs = plt.subplots(2,3)
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(3*default_width,3*default_height)
        fig.set_dpi(200)
        axs[0,0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,0],'o-',label=f'{tag} x')
        axs[0,1].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,1],'o-',label=f'{tag} y')
        axs[0,2].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,2],'o-',label=f'{tag} z')
        axs[0,0].set_xlabel('simulation time')
        axs[0,0].set_ylabel(f'{tag}')
        axs[0,1].set_xlabel('simulation time')
        axs[0,1].set_ylabel(f'{tag}')
        axs[0,2].set_xlabel('simulation time')
        axs[0,2].set_ylabel(f'{tag}')
        mre.analyze.format_figure(axs[0,0])
        mre.analyze.format_figure(axs[0,1])
        mre.analyze.format_figure(axs[0,2])
        midpoint = int(total_entries/2)
        axs[1,0].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,0],'o-')
        axs[1,1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,1],'o-')
        axs[1,2].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,2],'o-')
        axs[1,0].set_xlabel('simulation time')
        axs[1,0].set_ylabel(f'{tag}')
        axs[1,1].set_xlabel('simulation time')
        axs[1,1].set_ylabel(f'{tag}')
        axs[1,2].set_xlabel('simulation time')
        axs[1,2].set_ylabel(f'{tag}')
        fig.legend(fontsize=20)
        mre.analyze.format_figure(axs[1,0])
        mre.analyze.format_figure(axs[1,1])
        mre.analyze.format_figure(axs[1,2])
    else:
        fig, axs = plt.subplots(2)
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(3*default_width,3*default_height)
        fig.set_dpi(200)
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,0],'o-',label=f'{tag} x')
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,1],'o-',label=f'{tag} y')
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,2],'o-',label=f'{tag} z')
        axs[0].set_xlabel('simulation time')
        axs[0].set_ylabel(f'{tag}')
        mre.analyze.format_figure(axs[0])
        midpoint = int(total_entries/2)
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,0],'o-')
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,1],'o-')
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,2],'o-')
        axs[1].set_xlabel('simulation time')
        axs[1].set_ylabel(f'{tag}')
        fig.legend(fontsize=20)
        mre.analyze.format_figure(axs[1])
    plt.savefig(output_dir+f'{tag}_snapshots.png')
    plt.close()
    np.save(output_dir+f'{tag}_snapshots',snapshot_values)

def plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,total_entries,snapshot_values_one,snapshot_values_two,output_dir,tag=""):
    """Generate and save a figure showing the evolution of some value at particular steps throughout integration."""
    fig, axs = plt.subplots(2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    simulation_time = snapshot_stepsize*snapshot_values_one.shape[0]*step_size
    snapshot_times = np.arange(0,simulation_time,step_size*snapshot_stepsize)
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_one[:total_entries,0],'o-',label=f"{tag+'one'} x")
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_one[:total_entries,1],'o-',label=f"{tag+'one'} y")
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_one[:total_entries,2],'o-',label=f"{tag+'one'} z")
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_two[:total_entries,0],'x--',label=f"{tag+'two'} x")
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_two[:total_entries,1],'x--',label=f"{tag+'two'} y")
    axs[0].plot(snapshot_times[:total_entries],snapshot_values_two[:total_entries,2],'x--',label=f"{tag+'two'} z")
    axs[0].set_xlabel('simulation time')
    axs[0].set_ylabel(f'{tag}')
    mre.analyze.format_figure(axs[0])
    midpoint = int(total_entries/2)
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,0],'o-')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,1],'o-')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,2],'o-')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,0],'x--')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,1],'x--')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,2],'x--')
    axs[1].set_xlabel('simulation time')
    axs[1].set_ylabel(f'{tag}')
    fig.legend(fontsize=20)
    mre.analyze.format_figure(axs[1])
    plt.savefig(output_dir+f'{tag}_snapshots.png')
    plt.close()

class SimCriteria:
    """Class for calculating criteria for a simulation of up to two particles from the solution vector generated at each integration step. Criteria include:acceleration norm mean, acceleration norm max, particle acceleration norm, particle separation, mean and maximum velocity norms, maximum and minimum node positions in each cartesian direction, mean cartesian coordinate of nodes belonging to each surface of the simulated volume"""
    def __init__(self,solutions,*args):
        self.get_criteria_per_iteration(solutions,*args)
        self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
        self.timestep = self.time[1:] - self.time[:-1]

    def get_criteria_per_iteration(self,solutions,*args):
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
        boundaries = args[8]
        self.left = np.zeros((iterations,))
        self.right = np.zeros((iterations,))
        self.top = np.zeros((iterations,))
        self.bottom = np.zeros((iterations,))
        self.front = np.zeros((iterations,))
        self.back = np.zeros((iterations,))
        particles = args[2]
        for count, row in enumerate(solutions):
            a_var = get_accel_scaled(np.array(row[1:]),*args)
            a_norms = np.linalg.norm(a_var,axis=1)
            self.a_norm_max[count] = np.max(a_norms)
            # if self.a_norm_max[count] > 10000:
            #     a_var = get_accel_scaled(np.array(row[1:]),*args)
            self.a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
            final_posns = np.reshape(row[1:N_nodes*3+1],(N_nodes,3))
            final_v = np.reshape(row[N_nodes*3+1:],(N_nodes,3))
            v_norms = np.linalg.norm(final_v,axis=1)
            self.v_norm_max[count] = np.max(v_norms)
            self.v_norm_avg[count] = np.sum(v_norms)/np.shape(v_norms)[0]
            if particles.shape[0] != 0:
                a_particles = a_var[particles[0],:]
                self.particle_a_norm[count] = np.linalg.norm(a_particles[0,:])
                v_particles = final_v[particles[0],:]
                self.particle_v_norm[count] = np.linalg.norm(v_particles[0,:])
                x1 = get_particle_center(particles[0],final_posns)
                x2 = get_particle_center(particles[1],final_posns)
                self.particle_separation[count] = np.sqrt(np.sum(np.power(x1-x2,2)))
            self.get_system_extent(final_posns,boundaries,count)
    
    def get_system_extent(self,posns,boundaries,count):
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
        self.left[count] = np.mean(posns[boundaries['left'],0])
        self.right[count] = np.mean(posns[boundaries['right'],0])
        self.top[count] = np.mean(posns[boundaries['top'],2])
        self.bottom[count] = np.mean(posns[boundaries['bot'],2])
        self.front[count] = np.mean(posns[boundaries['front'],1])
        self.back[count] = np.mean(posns[boundaries['back'],1])

    def plot_displacement_hist(self,final_posns,initialized_posns,output_dir):
        displacement = final_posns-initialized_posns
        displacement_norms = np.linalg.norm(displacement,axis=1)
        max_displacement = np.max(displacement_norms)
        mean_displacement = np.mean(displacement_norms)
        rms_displacement = np.sqrt(np.sum(np.power(displacement_norms,2))/np.shape(displacement_norms)[0])
        counts, bins = np.histogram(displacement_norms, bins=20)
        fig,ax = plt.subplots()
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(2*default_width,2*default_height)
        ax.hist(bins[:-1], bins, weights=counts)
        sigma = np.std(displacement_norms)
        mu = mean_displacement
        # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)))**2)
        # ax.plot(bins,y,'--')
        ax.set_title(f'Displacement Histogram\nMaximum {max_displacement}\nMean {mean_displacement}\n$\sigma={sigma}$\nRMS {rms_displacement}')
        ax.set_xlabel('displacement (units of l_e)')
        ax.set_ylabel('counts')
        savename = output_dir +'node_displacement_hist.png'
        plt.savefig(savename)
        # plt.show()
        plt.close()
    
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
                plt.close()
        # plt.show()
        # plt.close('all')
    
    def plot_criteria_subplot(self,output_dir):
        """Generate subplots of simulation criteria using matplotlib"""
        if not (os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
        # first plotting the system extent (to get a sense of the change in the system size
        fig, axs = plt.subplots(3,3)
        default_width, default_height = fig.get_size_inches()
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(100)
        axs[0,0].plot(self.time,self.min_x)
        axs[0,0].set_title('x')
        axs[0,0].set_ylabel('minimum x position (l_e)')
        axs[1,0].plot(self.time,self.max_x)
        axs[1,0].set_ylabel('maximum x position (l_e)')
        axs[2,0].plot(self.time,self.length_x)
        axs[2,0].set_ylabel('length (max - min) in x direction (l_e)')
        axs[2,0].set_xlabel('scaled time')

        axs[0,1].plot(self.time,self.min_y)
        axs[0,1].set_title('y')
        axs[0,1].set_ylabel('minimum y position (l_e)')
        axs[1,1].plot(self.time,self.max_y)
        axs[1,1].set_ylabel('maximum y position (l_e)')
        axs[2,1].plot(self.time,self.length_y)
        axs[2,1].set_ylabel('length (max - min) in y direction (l_e)')
        axs[2,1].set_xlabel('scaled time')

        axs[0,2].plot(self.time,self.min_z)
        axs[0,2].set_title('z')
        axs[0,2].set_ylabel('minimum z position (l_e)')
        axs[1,2].plot(self.time,self.max_z)
        axs[1,2].set_ylabel('maximum z position (l_e)')
        axs[2,2].plot(self.time,self.length_z)
        axs[2,2].set_ylabel('length (max - min) in z direction (l_e)')
        axs[2,2].set_xlabel('scaled time')

        savename = output_dir + 'systemextent_v_time.png'
        plt.savefig(savename)
        plt.close()
        
        # plot the acceleration and velocity norms
        fig, axs = plt.subplots(2,2)
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(100)
        axs[0,0].plot(self.time,self.a_norm_avg)
        axs[0,0].set_title('node acceleration')
        axs[0,0].set_ylabel('node acceleration norm mean (unitless)')
        axs[1,0].plot(self.time,self.a_norm_max)
        axs[1,0].set_ylabel('node acceleration norm max (unitless)')
        axs[1,0].set_xlabel('scaled time')
        axs[0,1].plot(self.time,self.v_norm_avg)
        axs[0,1].set_title('node velocity')
        axs[0,1].set_ylabel('node velocity norm mean (unitless)')    
        axs[1,1].plot(self.time,self.v_norm_max)
        axs[1,1].set_ylabel('node velocity norm max (unitless)')
        axs[1,0].set_xlabel('scaled time')

        savename = output_dir + 'node_behavior_v_time.png'
        plt.savefig(savename)
        plt.close()

        # plot the particle acceleration, velocity, and separation
        fig, axs = plt.subplots(3)
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(100)
        axs[0].plot(self.time,self.particle_separation)
        axs[0].set_ylabel('particle separation (l_e)')
        axs[0].set_title('particle position, velocity, and acceleration')
        axs[1].plot(self.time,self.particle_v_norm)
        axs[1].set_ylabel('particle velocity norm (unitless)')
        axs[2].plot(self.time,self.particle_a_norm)
        axs[2].set_ylabel('particle acceleration norm (unitless)')
        axs[2].set_xlabel('scaled time')

        savename = output_dir + 'particle_behavior_v_time.png'
        plt.savefig(savename)
        plt.close()

        fig, axs = plt.subplots(3,2)
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(100)
        axs[0,0].plot(self.time,self.left)
        axs[0,0].set_ylabel('mean node position: left bdry (l_e)')
        axs[0,0].set_title('Mean Boundary Node Positions')
        axs[0,1].plot(self.time,self.right)
        axs[0,1].set_ylabel('mean node position: right bdry (l_e)')
        axs[1,0].plot(self.time,self.front)
        axs[1,0].set_ylabel('mean node position: front bdry (l_e)')
        axs[1,1].plot(self.time,self.back)
        axs[1,1].set_ylabel('mean node position: back bdry (l_e)')
        axs[2,0].plot(self.time,self.top)
        axs[2,0].set_ylabel('mean node position: top bdry (l_e)')
        axs[2,1].plot(self.time,self.bottom)
        axs[2,1].set_ylabel('mean node position: bottom bdry (l_e)')
        axs[2,0].set_xlabel('scaled time')
        savename = output_dir + 'mean_boundaries_v_time.png'
        plt.savefig(savename)
        plt.close()
        # plt.show()
        # plt.close('all')

        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(2*default_width,2*default_height)
        fig.set_dpi(100)
        axs[0].plot(self.time[:self.timestep.shape[0]],self.timestep,'.')
        axs[0].set_title('Time Step Taken')
        axs[0].set_xlabel('scaled time')
        axs[0].set_ylabel('time step')
        axs[1].plot(self.iter_number[:self.timestep.shape[0]],self.timestep,'.')
        axs[1].set_title('Time Step Taken')
        axs[1].set_xlabel('integration number')
        axs[1].set_ylabel('time step')
        axs[2].plot(self.iter_number,self.time,'.')
        axs[2].set_title('Total Time')
        axs[2].set_xlabel('integration number')
        axs[2].set_ylabel('total scaled time')
        savename = output_dir + 'timestep_per_iteration_and_time.png'
        plt.savefig(savename)
        plt.close()
    
#!!! generate traction forces or displacements based on some other criteria (choice of experimental setup with a switch statement? stress applied on boundary and then appropriately split onto the correct nodes in the correct directions in the correct amounts based on surface area?)

#function to pass to scipy.integrate.solve_ivp()
#must be of the form fun(t,y)
#can be more than fun(t,y,additionalargs), and then the additional args are passed to solve_ivp via keyword argument args=(a,b,c,...) where a,b,c are the additional arguments to fun in order of apperance in the function definition
def scaled_fun(t,y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    accel = get_accel_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
    N_nodes = int(np.round(N/3))
    accel = np.reshape(accel,(3*N_nodes,))
    v0 = y[N:]
    result = np.concatenate((v0,accel))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def scaled_fun_rotation(t,y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    accel = get_accel_scaled_rotation(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
    N_nodes = int(np.round(N/3))
    accel = np.reshape(accel,(3*N_nodes,))
    v0 = y[N:]
    result = np.concatenate((v0,accel))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result#np.transpose(np.column_stack((v0.reshape((3*N,1)),accel)))

def get_accel_scaled(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag=10,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    # drag = 20
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
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
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
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
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

def get_accel_scaled_rotation(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    # drag = 20
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0# + bc_forces
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        single_node_accel = (net_force_mag/stressed_nodes.shape[0])*beta_i[stressed_nodes]
        accel[stressed_nodes,force_index] += single_node_accel
    elif bc[0] == 'stress_compression':
        plate_orientation = bc[1][0]
        stress_direction = bc[1][1]
        stress = bc[2]
        global_index_interacting_nodes, plate_force = distribute_plate_stress(x0,stress,stress_direction,dimensions,l_e,plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
    elif bc[0] == 'plate_compression':
        plate_orientation = bc[1][0]
        global_index_interacting_nodes, plate_force = get_plate_force(x0,bc[2],plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
        #opposing surface to the probe surface needs to be held fixed
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        fixed_nodes = np.array([0])
        accel = set_fixed_nodes(accel,fixed_nodes)
    #characteristic time necessary to get the scaled angular acceleration correctly.
    characteristic_time_squared = beta*l_e
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
        mag_forces *= beta/(particle_mass*(l_e**4))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            #determine the net torque acting on the particle before the net force acting on the particle is calculated and distributed to the particle nodes
            total_torque[i,:] = get_torque_on_particle(particle,accel,x0)
            #get the net force acting on the particle and distribute to the particle nodes
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
            #find the magnitude of the net torque, and if it is not 0, calculate and distribute the forces necessary to match rigid particle rotation behavior
            torque_magnitude = np.linalg.norm(total_torque[i,:])
            if not np.isclose(torque_magnitude,0):
                angular_acceleration = total_torque[i,:]/particle_moment_of_inertia
                angular_acceleration_magnitude = np.linalg.norm(angular_acceleration)
                torque_unit_vector = total_torque[i,:]/torque_magnitude
                #we are dealing with a coordinate system relative to the center of the particle, so we need to translate all the particle nodes in such a way that the center of the central voxel is at (0,0,0). This is necessary for calculating the vectors pointing from the axis of rotation to the particle nodes, so that we can calculate the correct forces involved
                translated_particle_nodes = x0[particle,:] - particle_centers[i,:]
                # print(f'Vectors to particle nodes with particle center at origin:{translated_particle_nodes}\n')
                # r_magnitude = np.sqrt(np.sum(translated_particle_nodes*translated_particle_nodes,axis=1))
                # print(f'Magnitude of vectors to particle nodes with particle center at origin:{r_magnitude}\n')
                r_parallel_to_axis_of_rotation = np.sum(translated_particle_nodes*torque_unit_vector[np.newaxis,:],axis=1)[:,np.newaxis]*torque_unit_vector[np.newaxis,:]
                # print(f'r_parallel_to_axis_of_rotation:{r_parallel_to_axis_of_rotation}\n')
                # magnitude_r_parallel_to_axis_of_rotation = np.linalg.norm(r_parallel_to_axis_of_rotation,axis=1)
                # print(f'Magnitude of r_parallel_to_axis_of_rotation:{magnitude_r_parallel_to_axis_of_rotation}\n')
                # print(f'Torque unit vector:{torque_unit_vector}\n')
                # print(f'r_parallel_to_torque_unit_vector:{r_parallel_to_axis_of_rotation/magnitude_r_parallel_to_axis_of_rotation[:,np.newaxis]}\n')
                r_perpendicular_to_axis_of_rotation = translated_particle_nodes - r_parallel_to_axis_of_rotation
                # print(f'r_perpendiular to axis of rotation:{r_perpendicular_to_axis_of_rotation}\n')
                r_perp_magnitude = np.linalg.norm(r_perpendicular_to_axis_of_rotation,axis=1)
                # print(f'magnitude of component perpendicular to axis of rotation:{r_perp_magnitude}\n')
                rotational_acceleration_magnitude = r_perp_magnitude*angular_acceleration_magnitude
                rotational_acceleration_nonunit_vector = np.cross(torque_unit_vector,r_perpendicular_to_axis_of_rotation)
                rotational_acceleration_unit_vector = rotational_acceleration_nonunit_vector/np.linalg.norm(rotational_acceleration_nonunit_vector,axis=1)[:,np.newaxis]
                # print(f'rotational acceleration unit vectors:{rotational_acceleration_unit_vector}\n')
                # print(f'magnitude of rotational acceleration unit vectors:{np.linalg.norm(rotational_acceleration_unit_vector,axis=1)}\n')
                rotational_acceleration = rotational_acceleration_unit_vector*rotational_acceleration_magnitude[:,np.newaxis]
                # print(f'magnitude of rotational acceleration vectors:{np.linalg.norm(rotational_acceleration,axis=1)}\n')
                accel[particle] += rotational_acceleration
                # post_torque = get_torque_on_particle(particle,accel,x0)
                # print(f'Torque prior to maniuplation:{total_torque[i,:]}\n')
                # print(f'Torque after manipulations:{post_torque}\n')
                # print(f'{post_torque/total_torque[i,:]}')
                # print(f'pre-operation torque unit vector:\n{torque_unit_vector}\npost-operation torque unit vector:\n{post_torque/np.linalg.norm(post_torque)}')
                #calculate the angular acceleration vector/magnitude for each node to see if the manipulations resulted in the expected (and fixed in magnitude for each node) angular acceleration for rigid body rotation
                # rotational_acceleration_only_torque = np.cross(translated_particle_nodes,rotational_acceleration)
                # rotational_accel_only_angular_accel = rotational_acceleration_only_torque/particle_moment_of_inertia
                # print(f'considering rotational acceleration only, torque:\n{rotational_acceleration_only_torque}\n')
                # rot_accel_only_net_torque = np.sum(rotational_acceleration_only_torque,axis=0)
                # print(f'considering rotational acceleration only, net torque:\n{rot_accel_only_net_torque}\n')
                # print(f'considering rotational acceleration only, angular acceleration:\n{rotational_accel_only_angular_accel}')
                # print(f'considering rotational acceleration only, net angular acceleration:\n{rot_accel_only_net_torque/particle_moment_of_inertia}')
                # print(f'angular acceleration magnitude:\n{np.linalg.norm(rot_accel_only_net_torque/particle_moment_of_inertia)}\n')
                # #make plots showing the vector quantities for the rotational acceleration
                # ax = plt.figure().add_subplot(projection='3d')
                # x = translated_particle_nodes[:,0]
                # y = translated_particle_nodes[:,1]
                # z = translated_particle_nodes[:,2]
                # u = rotational_acceleration[:,0]
                # v = rotational_acceleration[:,1]
                # w = rotational_acceleration[:,2]
                # ax.quiver(x,y,z,u,v,w,length=0.1,normalize=True)
                # X = 0
                # Y = 0
                # Z = 0
                # U = torque_unit_vector[0]
                # V = torque_unit_vector[1]
                # W = torque_unit_vector[2]
                # ax.quiver(X,Y,Z,U,V,W,length=0.5,normalize=True)
                # u = accel[particle,0]
                # v = accel[particle,1]
                # w = accel[particle,2]
                # ax.quiver(x,y,z,u,v,w,length=0.1,normalize=True,color='r')
                # plt.show()
                # post_angular_acceleration = rotational_acceleration/r_perp_magnitude[:,np.newaxis]
                # print(f'angular acceleration vectors:\n{post_angular_acceleration}\n')
                # post_angular_acceleration_magnitude = np.linalg.norm(post_angular_acceleration,axis=1)
                # print(f'angular acceleration vector magnitudes:\n{post_angular_acceleration_magnitude}\n')
    return accel

def get_particle_center(particle_nodes,node_posns):
    """Given the indices of the nodes making up the particle, and the node positions variable describing the system, find the center of the spherical particle."""
    particle_node_posns = node_posns[particle_nodes,:]
    x_max = np.max(particle_node_posns[:,0])
    y_max = np.max(particle_node_posns[:,1])
    z_max = np.max(particle_node_posns[:,2])
    x_min = np.min(particle_node_posns[:,0])
    y_min = np.min(particle_node_posns[:,1])
    z_min = np.min(particle_node_posns[:,2])
    particle_center = np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2],dtype=np.float64)
    return particle_center

def get_particle_posns(particles,node_posns):
    """Given the particle node indices and the node positions, find the positions of the particle centers"""
    particle_posns = np.zeros((particles.shape[0],3))
    for i, particle in enumerate(particles):
        particle_posns[i] = get_particle_center(particle,node_posns)
    return particle_posns

def set_fixed_nodes(accel,fixed_nodes):
    """Given the acceleration variable and the indices of the nodes that should be held fixed in the simulation, set the acceleration variable entries to zero for all components of acceleration for the passed nodes"""
    accel[fixed_nodes] = 0
    # for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
    #     #TODO almost certainly faster to remove the inner loop and just set each value to 0 in order, or using python semantics, just set the row to zero?
    #     for j in range(3):
    #         accel[fixed_nodes[i],j] = 0
    return accel

def set_plate_fixed_nodes(accel,fixed_nodes,plate_orientation):
    """Given the acceleration variable and the indices of the nodes that should be held fixed (in the plane of and) by the probe plate, return a modified acceleration variable preventing that in-plane motion"""
    if plate_orientation == 'x':
        relevant_indices = [1,2]
    elif plate_orientation == 'y':
        relevant_indices = [0,2]
    elif plate_orientation == 'z':
        relevant_indices = [0,1]
    accel[fixed_nodes,relevant_indices[0]] = 0
    accel[fixed_nodes,relevant_indices[1]] = 0
    # for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
    #     #TODO almost certainly faster to remove the inner loop and just set each value to 0 in order, or using python semantics, just set the row to zero?
    #     for j in range(3):
    #         accel[fixed_nodes[i],j] = 0
    return accel

def get_accel_scaled_no_fixed_nodes(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    if bc[0] == 'simple_stress_compression':
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        single_node_accel = (net_force_mag/stressed_nodes.shape[0])*beta_i[stressed_nodes]
        accel[stressed_nodes,force_index] += single_node_accel
    elif bc[0] == 'stress_compression':
        plate_orientation = bc[1][0]
        stress_direction = bc[1][1]
        stress = bc[2]
        global_index_interacting_nodes, plate_force = distribute_plate_stress(x0,stress,stress_direction,dimensions,l_e,plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
    elif bc[0] == 'plate_compression':
        plate_orientation = bc[1][0]
        global_index_interacting_nodes, plate_force = get_plate_force(x0,bc[2],plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
        #opposing surface to the probe surface needs to be held fixed
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    # if bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
    #     #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
    #     if bc[1][0] == 'x':
    #         fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
    #     elif bc[1][0] == 'y':
    #         fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
    #     elif bc[1][0] == 'z':
    #         fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
    # else:
    #     fixed_nodes = np.array([0])
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force_normalized(x0,elements,kappa,correction_force_el,vectors,avg_vectors, volume_correction_force)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces_WCA(x0, springs, spring_force)
    volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    spring_force *= l_e*beta_i[:,np.newaxis]
    accel = spring_force + volume_correction_force - drag * v0
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float64)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
        mag_forces *= beta/(particle_mass*(l_e**4))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        #TODO remove loops as much as possible within python. this function has to be cythonized anyway, but there is serious overhead with any looping, even just dealing with the rigid particles
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            total_torque[i,:] = get_torque_on_particle(particle,accel,x0)
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
    else:
        total_torque = None
    return accel, total_torque

def get_torque_on_particle(particle,accel,node_posns):
    """Given the list of particle nodes, the accelerations and positions of the nodes, calculate the torque acting on the particle relative to the center of the particle"""
    particle_center = get_particle_center(particle,node_posns)
    nodal_torques = np.zeros((particle.shape[0],3))
    nodal_torques = np.cross(node_posns[particle,:] - particle_center,accel[particle,:])
    total_torque = np.sum(nodal_torques,axis=0)
    return total_torque

def get_plate_force(node_posns,plate_posn,plate_orientation,boundaries):
    """Given the position and orientation of the plate (experimental probe with force reading), and the position of the nodes on the relevant surface, calculate the forces acting on those nodes due to the interaction of the material surface with the plate surface"""
    # based on the plate orientation, decide which boundary is relevant, which coordinate of position is relevant, and the direction of the normal force exerted on the material surface by the plate
    if plate_orientation == 'x':
        relevant_index = 0
        relevant_boundary = 'right'
    elif plate_orientation == 'y':
        relevant_index = 1
        relevant_boundary = 'back'
    elif plate_orientation == 'z':
        relevant_index = 2
        relevant_boundary = 'top'
    # based on the distance to the plate, decide if the nodes are interacting with the plate, and how much force they experience
    distance_to_plate = plate_posn - node_posns[boundaries[relevant_boundary],relevant_index]
    eps_constant = 0.1
    sigma = 0.01
    distance_to_plate[distance_to_plate<0] = sigma
    cutoff_distance = sigma * np.power(2,(1/6))
    interacting_nodes = distance_to_plate<cutoff_distance
    sigma_over_separation = sigma/distance_to_plate[interacting_nodes]
    np.nan_to_num(sigma_over_separation,copy=False,nan=0.0,posinf=1,neginf=-1)
    force_mag = 4*eps_constant*(12*np.power(sigma_over_separation,13)/sigma - 6* np.power(sigma_over_separation,7)/sigma)
    force_mag[force_mag>100] = 100
    force = np.zeros((force_mag.shape[0],3))
    #pushing the material boundary away from the plate
    force[:,relevant_index] -= force_mag
    #if there is a normal force exerted by the plate on the surface, there should be some frictional forces exerted in a direction spanning the plane of the plate
    # coefficient_of_friction = 100
    # frictional_force_magnitude = coefficient_of_friction*force_mag
    global_index_interacting_nodes = boundaries[relevant_boundary][interacting_nodes]
    return global_index_interacting_nodes, force#, frictional_force_magnitude

def get_frictional_force_vector(node_velocities,node_accel,frictional_force_magnitude,plate_orientation):
    # I technically don't need to do this. i can just say that if the nodes are interacting with the plate, that is, have some normal force due to interaction with the plate, then the acceleration in the plane of the plate has to be zero. when i want to deal with shearing forces, instead of considering forces i can just consider some plate velocity and force the nodes interacting with the plate to move at that velocity until the plate has come to a rest
    """Given the velocities and accelerations of the boundary nodes that are interacting with the probe plate, the maximum frictional force for each node, and the orientation of the force probe plate, return the frictional force vector acting on the boundary nodes"""
    #i started doing things this way...
    # velocity_unit_vectors = node_velocities/np.linalg.norm(node_velocities,axis=1)
    # dot_product_velocity_unit_vec_and_accel_vec = np.tensordot(velocity_unit_vectors,node_accel,(1,1))
    # acceleration_parallel_to_velocity = dot_product_velocity_unit_vec_and_accel_vec * velocity_unit_vectors
    # parallel_accel_magnitude = np.linalg.norm(acceleration_parallel_to_velocity,axis=1)
    # (dot_product_velocity_unit_vec_and_accel_vec > 0) and (parallel_accel_magnitude < frictional_force_magnitude)
    #but i could also do this
    if plate_orientation == 'x':
        relevant_index = 0
    elif plate_orientation == 'y':
        relevant_index = 1
    elif plate_orientation == 'z':
        relevant_index = 2
    in_plane_accel = node_accel.copy()
    in_plane_accel[:,relevant_index] = 0
    in_plane_accel_magnitude = np.linalg.norm(in_plane_accel,axis=1)
    #if the nodes are moving, we need friction to be in the direction opposing the motion, if the nodes aren't moving, we need the friction to be in the direction of any acceleration in the plane of the probe plate

    
def distribute_plate_stress(node_posns,stress,stress_direction,dimensions,beta_i,plate_orientation,boundaries):
    """Given the boundary node positions, desired stress, stress direction, material boundary dimensions, discretization length, orientation of the plate (experimental probe with force reading), and the position of the nodes on the relevant surface, calculate the forces acting on those nodes due to the interaction of the material surface with the plate surface"""
    # based on the plate orientation, decide which boundary is relevant, which coordinate of position is relevant, and the direction of the normal force exerted on the material surface by the plate
    if plate_orientation == 'x':
        relevant_index = 0
        relevant_boundary = 'right'
        relevant_dimension_indices = [1,2]
    elif plate_orientation == 'y':
        relevant_index = 1
        relevant_boundary = 'back'
        relevant_dimension_indices = [0,2]
    elif plate_orientation == 'z':
        relevant_index = 2
        relevant_boundary = 'top'
        relevant_dimension_indices = [0,1]
    # based on the maximum node position of the relevant coordinate, decide which nodes are being acted upon by the fictitious plate, and how much force they experience
    plate_posn = np.max(node_posns[boundaries[relevant_boundary],relevant_index])
    distance_to_plate = plate_posn - node_posns[boundaries[relevant_boundary],relevant_index]
    cutoff_distance = 0.01
    interacting_nodes = distance_to_plate<cutoff_distance
    global_index_interacting_nodes = boundaries[relevant_boundary][interacting_nodes]
    num_interacting_nodes = np.nonzero(interacting_nodes)
    net_force_mag = stress*dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]#/np.power(l_e,2)
    force_mag = (net_force_mag/num_interacting_nodes)*beta_i[global_index_interacting_nodes]
    force = np.zeros((force_mag.shape[0],3))
    #pushing the interacting nodes of the material boundary in the desired direction
    if stress_direction == 'x':
        force[:,0] += force_mag
    elif stress_direction == 'y':
        force[:,1] += force_mag
    elif stress_direction == 'z':
        force[:,2] += force_mag
    return global_index_interacting_nodes, force

###### GPU Kernels and Functions

### ENERGY CALCULATIONS

scaled_element_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void element_energy(const int* elements, const float* node_posns, const float kappa, float* energies, const int size_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_elements)
    {
        int index0 = elements[8*tid+0];
        int index1 = elements[8*tid+1];
        int index2 = elements[8*tid+2];
        int index3 = elements[8*tid+3];
        int index4 = elements[8*tid+4];
        int index5 = elements[8*tid+5];
        int index6 = elements[8*tid+6];
        int index7 = elements[8*tid+7];
        //printf("element tid = %i, size_elements = %i\n",tid,size_elements);                   
        //printf("tid = %i, index0 = %i, index1 = %i, index2 = %i, index3 = %i, index4 = %i, index5 = %i, index6 = %i, index7 = %i\n",tid,index0,index1,index2,index3,index4,index5,index6,index7);
        //printf("tid = %i, node_posns[3*index2] = %f, node_posns[3*index0]= %f\n",tid,node_posns[3*index2],node_posns[3*index0]);                     
        //for(int i = 0; i < 24; i++){
        //    printf("node_posns[%i] = %f\n",i,node_posns[i]);
        //}
        //for the element, get the average edge vectors, then using the average edge vectors, get the volume correction force
        float avg_vector_i[3];
        float avg_vector_j[3];
        float avg_vector_k[3];
                                
        avg_vector_i[0] = (node_posns[3*index2] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index1] + node_posns[3*index6] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index5])/4;
        avg_vector_i[1] = (node_posns[1+3*index2] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index1] + node_posns[1+3*index6] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index5])/4;
        avg_vector_i[2] = (node_posns[2+3*index2] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index1] + node_posns[2+3*index6] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index5])/4;

        //printf("tid = %i, avg_vector_i[0] = %f, avg_vector_i[1] = %f, avg_vector_i[2] = %f\n",tid,avg_vector_i[0],avg_vector_i[1],avg_vector_i[2]);
                                
        avg_vector_j[0] = (node_posns[3*index4] - node_posns[3*index0] + node_posns[3*index6] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index1] + node_posns[3*index7] - node_posns[3*index3])/4;
        avg_vector_j[1] = (node_posns[1+3*index4] - node_posns[1+3*index0] + node_posns[1+3*index6] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index1] + node_posns[1+3*index7] - node_posns[1+3*index3])/4;
        avg_vector_j[2] = (node_posns[2+3*index4] - node_posns[2+3*index0] + node_posns[2+3*index6] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index1] + node_posns[2+3*index7] - node_posns[2+3*index3])/4;
                                
        //printf("tid = %i, avg_vector_j[0] = %f, avg_vector_j[1] = %f, avg_vector_j[2] = %f\n",tid,avg_vector_j[0],avg_vector_j[1],avg_vector_j[2]);

        avg_vector_k[0] = (node_posns[3*index1] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index6])/4;
        avg_vector_k[1] = (node_posns[1+3*index1] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index6])/4;
        avg_vector_k[2] = (node_posns[2+3*index1] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index6])/4;                    
        
        //float acrossb[3];
        float bcrossc[3];
        //float ccrossa[3];
        float adotbcrossc;
        /*                        
        acrossb[0] = avg_vector_i[1]*avg_vector_j[2] - avg_vector_i[2]*avg_vector_j[1];
        acrossb[1] = avg_vector_i[2]*avg_vector_j[0] - avg_vector_i[0]*avg_vector_j[2];
        acrossb[2] = avg_vector_i[0]*avg_vector_j[1] - avg_vector_i[1]*avg_vector_j[0];
        */                        
        bcrossc[0] = avg_vector_j[1]*avg_vector_k[2] - avg_vector_j[2]*avg_vector_k[1];
        bcrossc[1] = avg_vector_j[2]*avg_vector_k[0] - avg_vector_j[0]*avg_vector_k[2];
        bcrossc[2] = avg_vector_j[0]*avg_vector_k[1] - avg_vector_j[1]*avg_vector_k[0];
        /*                        
        ccrossa[0] = avg_vector_k[1]*avg_vector_i[2] - avg_vector_k[2]*avg_vector_i[1];
        ccrossa[1] = avg_vector_k[2]*avg_vector_i[0] - avg_vector_k[0]*avg_vector_i[2];
        ccrossa[2] = avg_vector_k[0]*avg_vector_i[1] - avg_vector_k[1]*avg_vector_i[0];
        */                        
        adotbcrossc = avg_vector_i[0]*bcrossc[0] + avg_vector_i[1]*bcrossc[1] + avg_vector_i[2]*bcrossc[2];

        float energy = (1./2.)*kappa*powf(adotbcrossc-1,2);
        energies[tid] = energy;
    }
    }
    ''', 'element_energy')

scaled_spring_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void spring_energy(const float* edges, const float* node_posns, float* energies, const int size_edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("entered the kernel");
    if (tid < size_edges)
    {
        //printf("entered the if block");
        //printf("spring tid = %i, size_springs = %i\n",tid,size_edges);  
        int iid = edges[4*tid];
        int jid = edges[4*tid+1];
        float rij[3];
        rij[0] = node_posns[3*iid]-node_posns[3*jid];
        rij[1] = node_posns[3*iid+1]-node_posns[3*jid+1];
        rij[2] = node_posns[3*iid+2]-node_posns[3*jid+2];

        float mag = sqrtf(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
        float energy = (1./2.)*edges[4*tid+2]*powf(mag-edges[4*tid+3],2);
        energies[tid] = energy;
    }
    }
    ''', 'spring_energy')

dipole_energy_kernel = cp.RawKernel(r'''
float PI = 3.141592654f;
extern "C" __global__
void get_dipole_energy(const float* magnetic_moment, const float* Htot, const float* Hext, float* energies, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float mi[3];
        mi[0] = magnetic_moment[3*tid];
        mi[1] = magnetic_moment[3*tid+1];
        mi[2] = magnetic_moment[3*tid+2];
        float Bdip[3];
        float mu0 = 4.*PI*1e-7;
        Bdip[0] = mu0*(Htot[3*tid] - Hext[0]);
        Bdip[1] = mu0*(Htot[3*tid+1] - Hext[1]);
        Bdip[2] = mu0*(Htot[3*tid+2] - Hext[2]);
        float energy = 0;

        energy += -1.*mi[0]*Bdip[0];
        energy += -1.*mi[1]*Bdip[1];
        energy += -1.*mi[2]*Bdip[2];
        
        energies[tid] = energy;
    }
    }
    ''', 'get_dipole_energy')


wca_energy_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
float SURFACE_TO_SURFACE_SPACING = 1e-7;
extern "C" __global__
void get_wca_energy(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float particle_radius, const float l_e, const float inv_l_e, float* energies, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float eps_constant = (1e-7)*4*powf(3.141592654f,2)*powf(1.9e6,2)*powf(1.5e-6,3)/72;
        //printf("eps_constant = %f e-13\n",eps_constant*1e13);
        float sigma = (2*particle_radius+SURFACE_TO_SURFACE_SPACING);
        float cutoff_length = powf(2.f,(1.f/6.f))*sigma;
        //printf("2^(1/6) = %f\n",powf(2.f,(1.f/6.f)));
        int separation_vector_idx;
        int inv_magnitude_idx;
        float mu0_over_four_pi = 1e-7;
        float rijmag;
        float sigma_over_separation;
        for (int j = 0; j < size_particles; j++)
        {
            if (tid != j)
            {
                separation_vector_idx = tid*3*size_particles+3*j;
                inv_magnitude_idx = tid*size_particles + j;
                //printf("tid = %i, j = %i, r_hat = %f, %f, %f\n",tid,j,r_hat[0],r_hat[1],r_hat[2]);
                rijmag = norm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2])*l_e;
                //printf("rij magnitude = %f e-6, cutoff_length = %f e-6\n",rijmag*1e6,cutoff_length*1e6);
                if (rijmag <= cutoff_length)
                {
                    sigma_over_separation = sigma*separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e;
                    energies[tid] += 4*eps_constant*(powf(sigma_over_separation,12) - powf(sigma_over_separation,6)) + eps_constant;
                    //printf("inside WCA force calculation bit\ntid = %i, j = %i, wca_force = %f,%f,%f e-6\n",tid,j,force_temp_var*r_hat[0]*1e6,force_temp_var*r_hat[1]*1e6,force_temp_var*r_hat[2]*1e6);
                }
            }
        }
    }
    }
    ''', 'get_wca_energy')

### Force calculation and distribution

boundary_stress_kernel = cp.RawKernel(r'''
extern "C" __global__
void apply_boundary_stress(const int* boundary_node_idx, const int boundary_force_direction, const float force_magnitude, float* force, const int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes)
    {
        int node_idx = boundary_node_idx[tid];
        force[3*node_idx+boundary_force_direction] += force_magnitude;
    }
    }
    ''', 'apply_boundary_stress')

fixed_boundary_kernel = cp.RawKernel(r'''
extern "C" __global__
void fix_boundary_node(const int* boundary_node_idx, float* force, const int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes)
    {
        int node_idx = boundary_node_idx[tid];
        //printf("node index held fixed: %i\n",node_idx);
        force[3*node_idx] = 0;
        force[3*node_idx+1] = 0;
        force[3*node_idx+2] = 0;
    }
    }
    ''', 'fix_boundary_node')

strained_boundary_kernel = cp.RawKernel(r'''
extern "C" __global__
void set_boundary_force(const int* boundary_nodes, float* node_force, const float* boundary_force, const int num_boundary_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_boundary_nodes)
    {                
        int node_force_idx = boundary_nodes[tid];

        node_force[3*node_force_idx] = boundary_force[0];
        node_force[3*node_force_idx+1] = boundary_force[1];
        node_force[3*node_force_idx+2] = boundary_force[2];
    }
    }
    ''', 'set_boundary_force')

scaled_element_kernel = cp.RawKernel(r'''
extern "C" __global__
void element_force(const int* elements, const float* node_posns, const float kappa, float* forces, const int size_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_elements)
    {
        int index0 = elements[8*tid+0];
        int index1 = elements[8*tid+1];
        int index2 = elements[8*tid+2];
        int index3 = elements[8*tid+3];
        int index4 = elements[8*tid+4];
        int index5 = elements[8*tid+5];
        int index6 = elements[8*tid+6];
        int index7 = elements[8*tid+7];
        //printf("element tid = %i, size_elements = %i\n",tid,size_elements);                   
        //printf("tid = %i, index0 = %i, index1 = %i, index2 = %i, index3 = %i, index4 = %i, index5 = %i, index6 = %i, index7 = %i\n",tid,index0,index1,index2,index3,index4,index5,index6,index7);
        //printf("tid = %i, node_posns[3*index2] = %f, node_posns[3*index0]= %f\n",tid,node_posns[3*index2],node_posns[3*index0]);                     
        //for(int i = 0; i < 24; i++){
        //    printf("node_posns[%i] = %f\n",i,node_posns[i]);
        //}
        //for the element, get the average edge vectors, then using the average edge vectors, get the volume correction force
        float avg_vector_i[3];
        float avg_vector_j[3];
        float avg_vector_k[3];
                                
        avg_vector_i[0] = (node_posns[3*index2] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index1] + node_posns[3*index6] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index5])/4;
        avg_vector_i[1] = (node_posns[1+3*index2] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index1] + node_posns[1+3*index6] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index5])/4;
        avg_vector_i[2] = (node_posns[2+3*index2] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index1] + node_posns[2+3*index6] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index5])/4;

        //printf("tid = %i, avg_vector_i[0] = %f, avg_vector_i[1] = %f, avg_vector_i[2] = %f\n",tid,avg_vector_i[0],avg_vector_i[1],avg_vector_i[2]);
                                
        avg_vector_j[0] = (node_posns[3*index4] - node_posns[3*index0] + node_posns[3*index6] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index1] + node_posns[3*index7] - node_posns[3*index3])/4;
        avg_vector_j[1] = (node_posns[1+3*index4] - node_posns[1+3*index0] + node_posns[1+3*index6] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index1] + node_posns[1+3*index7] - node_posns[1+3*index3])/4;
        avg_vector_j[2] = (node_posns[2+3*index4] - node_posns[2+3*index0] + node_posns[2+3*index6] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index1] + node_posns[2+3*index7] - node_posns[2+3*index3])/4;
                                
        //printf("tid = %i, avg_vector_j[0] = %f, avg_vector_j[1] = %f, avg_vector_j[2] = %f\n",tid,avg_vector_j[0],avg_vector_j[1],avg_vector_j[2]);

        avg_vector_k[0] = (node_posns[3*index1] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index6])/4;
        avg_vector_k[1] = (node_posns[1+3*index1] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index6])/4;
        avg_vector_k[2] = (node_posns[2+3*index1] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index6])/4;
        /*
        if (isnan(avg_vector_i[0]) || isinf(avg_vector_i[0])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_i[1]) || isinf(avg_vector_i[1])){
            printf("nan value in average vector component");
        }                            
        if (isnan(avg_vector_i[2]) || isinf(avg_vector_i[2])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_j[0]) || isinf(avg_vector_j[0])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_j[1]) || isinf(avg_vector_j[1])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_j[2]) || isinf(avg_vector_j[2])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_k[0]) || isinf(avg_vector_k[0])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_k[1]) || isinf(avg_vector_k[1])){
            printf("nan value in average vector component");
        }
        if (isnan(avg_vector_k[2]) || isinf(avg_vector_k[2])){
            printf("nan value in average vector component");
        }*/
        //printf("tid = %i, avg_vector_k[0] = %f, avg_vector_k[1] = %f, avg_vector_k[2] = %f\n",tid,avg_vector_k[0],avg_vector_k[1],avg_vector_k[2]);                      
        
        //need to get cross products of average vectors, stored as variables for gradient vectors, prefactor, then atomicAdd for assignment to forces
        float acrossb[3];
        float bcrossc[3];
        float ccrossa[3];
        float adotbcrossc;
                                
        acrossb[0] = avg_vector_i[1]*avg_vector_j[2] - avg_vector_i[2]*avg_vector_j[1];
        acrossb[1] = avg_vector_i[2]*avg_vector_j[0] - avg_vector_i[0]*avg_vector_j[2];
        acrossb[2] = avg_vector_i[0]*avg_vector_j[1] - avg_vector_i[1]*avg_vector_j[0];
                                
        bcrossc[0] = avg_vector_j[1]*avg_vector_k[2] - avg_vector_j[2]*avg_vector_k[1];
        bcrossc[1] = avg_vector_j[2]*avg_vector_k[0] - avg_vector_j[0]*avg_vector_k[2];
        bcrossc[2] = avg_vector_j[0]*avg_vector_k[1] - avg_vector_j[1]*avg_vector_k[0];
                                
        ccrossa[0] = avg_vector_k[1]*avg_vector_i[2] - avg_vector_k[2]*avg_vector_i[1];
        ccrossa[1] = avg_vector_k[2]*avg_vector_i[0] - avg_vector_k[0]*avg_vector_i[2];
        ccrossa[2] = avg_vector_k[0]*avg_vector_i[1] - avg_vector_k[1]*avg_vector_i[0];
                                
        adotbcrossc = avg_vector_i[0]*bcrossc[0] + avg_vector_i[1]*bcrossc[1] + avg_vector_i[2]*bcrossc[2];
        /*
        if (isnan(acrossb[0]) || isinf(acrossb[0])){
            printf("nan value in average vector cross product\n");
            printf("acrossb[0] = %f\n",acrossb[0]);
        }
        if (isnan(acrossb[1]) || isinf(acrossb[1])){
            printf("nan value in average vector cross product\n");
            printf("acrossb[1] = %f\n",acrossb[1]);
        }                            
        if (isnan(acrossb[2]) || isinf(acrossb[2])){
            printf("nan value in average vector cross product\n");
            printf("acrossb[2] = %f\n",acrossb[2]);
        }
        if (isnan(bcrossc[0]) || isinf(bcrossc[0])){
            printf("nan value in average vector cross product\n");
            printf("bcrossc[0] = %f\n",bcrossc[0]);
        }
        if (isnan(bcrossc[1]) || isinf(bcrossc[1])){
            printf("nan value in average vector cross product\n");
            printf("bcrossc[1] = %f\n",bcrossc[1]);
        }
        if (isnan(bcrossc[2]) || isinf(bcrossc[2])){
            printf("nan value in average vector cross product\n");
            printf("bcrossc[2] = %f\n",bcrossc[2]);
        }
        if (isnan(ccrossa[0]) || isinf(ccrossa[0])){
            printf("nan value in average vector cross product\n");
            printf("ccrossa[0] = %f\n",ccrossa[0]);
        }
        if (isnan(ccrossa[1]) || isinf(ccrossa[1])){
            printf("nan value in average vector cross product\n");
            printf("ccrossa[1] = %f\n",ccrossa[1]);
        }
        if (isnan(ccrossa[2]) || isinf(ccrossa[2])){
            printf("nan value in average vector cross product\n");
            printf("ccrossa[2] = %f\n",ccrossa[2]);
        }                                                   
        if (isnan(adotbcrossc)){
            printf("nan value in adotbcrossc\n");
            //printf("avg_vector_i[0]*bcrossc[0] = %f,\navg_vector_i[1]*bcrossc[1] = %f,\navg_vector_i[2]*bcrossc[2] = %f\n",avg_vector_i[0]*bcrossc[0],avg_vector_i[1]*bcrossc[1],avg_vector_i[2]*bcrossc[2]);
            printf("avg_vector_i[0] = %f,\navg_vector_i[1] = %f,\navg_vector_i[2] = %f\n",avg_vector_i[0],avg_vector_i[1],avg_vector_i[2]);
            printf("(node_posns[3*index2]= %f, node_posns[3*index0] = %f,\n node_posns[3*index3] = %f node_posns[3*index1] = %f,\n node_posns[3*index6] = %f, node_posns[3*index4] = %f,\n node_posns[3*index7] = %f, node_posns[3*index5] = %f\n",node_posns[3*index2],node_posns[3*index0],node_posns[3*index3],node_posns[3*index1],node_posns[3*index6],node_posns[3*index4],node_posns[3*index7],node_posns[3*index5]);
            //printf("bcrossc[0] = %f,\nbcrossc[1] = %f,\nbcrossc[2] = %f\n",bcrossc[0],bcrossc[1],bcrossc[2]);
        }*/ 
        float gradV1[3];
        float gradV8[3];
        float gradV3[3];
        float gradV6[3];
        float gradV7[3];
        float gradV2[3];
        float gradV5[3];
        float gradV4[3];
                                
        gradV1[0] = -1*bcrossc[0] -1*ccrossa[0] -1*acrossb[0];
        gradV8[0] = -1*gradV1[0];
        gradV3[0] = bcrossc[0] -1*ccrossa[0] -1*acrossb[0];
        gradV6[0] = -1*gradV3[0];
        gradV7[0] = bcrossc[0] + ccrossa[0] -1*acrossb[0];
        gradV2[0] = -1*gradV7[0];
        gradV5[0] = -1*bcrossc[0] + ccrossa[0] -1*acrossb[0];
        gradV4[0] = -1*gradV5[0];
        
        gradV1[1] = -1*bcrossc[1] -1*ccrossa[1] -1*acrossb[1];
        gradV8[1] = -1*gradV1[1];
        gradV3[1] = bcrossc[1] -1*ccrossa[1] -1*acrossb[1];
        gradV6[1] = -1*gradV3[1];
        gradV7[1] = bcrossc[1] + ccrossa[1] -1*acrossb[1];
        gradV2[1] = -1*gradV7[1];
        gradV5[1] = -1*bcrossc[1] + ccrossa[1] -1*acrossb[1];
        gradV4[1] = -1*gradV5[1];
                                
        gradV1[2] = -1*bcrossc[2] -1*ccrossa[2] -1*acrossb[2];
        gradV8[2] = -1*gradV1[2];
        gradV3[2] = bcrossc[2] -1*ccrossa[2] -1*acrossb[2];
        gradV6[2] = -1*gradV3[2];
        gradV7[2] = bcrossc[2] + ccrossa[2] -1*acrossb[2];
        gradV2[2] = -1*gradV7[2];
        gradV5[2] = -1*bcrossc[2] + ccrossa[2] -1*acrossb[2];
        gradV4[2] = -1*gradV5[2];

        float prefactor = -1*kappa * (adotbcrossc - 1);
        atomicAdd(&forces[3*index0],prefactor*gradV1[0]);
        atomicAdd(&forces[3*index0+1],prefactor*gradV1[1]);
        atomicAdd(&forces[3*index0+2],prefactor*gradV1[2]);
        atomicAdd(&forces[3*index1],prefactor*gradV2[0]);
        atomicAdd(&forces[3*index1+1],prefactor*gradV2[1]);
        atomicAdd(&forces[3*index1+2],prefactor*gradV2[2]);
        atomicAdd(&forces[3*index2],prefactor*gradV3[0]);
        atomicAdd(&forces[3*index2+1],prefactor*gradV3[1]);
        atomicAdd(&forces[3*index2+2],prefactor*gradV3[2]);
        atomicAdd(&forces[3*index3],prefactor*gradV4[0]);
        atomicAdd(&forces[3*index3+1],prefactor*gradV4[1]);
        atomicAdd(&forces[3*index3+2],prefactor*gradV4[2]);
        atomicAdd(&forces[3*index4],prefactor*gradV5[0]);
        atomicAdd(&forces[3*index4+1],prefactor*gradV5[1]);
        atomicAdd(&forces[3*index4+2],prefactor*gradV5[2]);
        atomicAdd(&forces[3*index5],prefactor*gradV6[0]);
        atomicAdd(&forces[3*index5+1],prefactor*gradV6[1]);
        atomicAdd(&forces[3*index5+2],prefactor*gradV6[2]);
        atomicAdd(&forces[3*index6],prefactor*gradV7[0]);
        atomicAdd(&forces[3*index6+1],prefactor*gradV7[1]);
        atomicAdd(&forces[3*index6+2],prefactor*gradV7[2]);
        atomicAdd(&forces[3*index7],prefactor*gradV8[0]);
        atomicAdd(&forces[3*index7+1],prefactor*gradV8[1]);
        atomicAdd(&forces[3*index7+2],prefactor*gradV8[2]);
    }
    }
    ''', 'element_force')

scaled_spring_kernel = cp.RawKernel(r'''
extern "C" __global__
void spring_force(const float* edges, const float* node_posns, float* forces, const int size_edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("entered the kernel");
    if (tid < size_edges)
    {
        //printf("entered the if block");
        //printf("spring tid = %i, size_springs = %i\n",tid,size_edges);  
        int iid = edges[4*tid];
        int jid = edges[4*tid+1];
        float rij[3];
        rij[0] = node_posns[3*iid]-node_posns[3*jid];
        rij[1] = node_posns[3*iid+1]-node_posns[3*jid+1];
        rij[2] = node_posns[3*iid+2]-node_posns[3*jid+2];
        float inv_mag = rsqrtf(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
        float rij_hat[3];
        rij_hat[0] = rij[0]*inv_mag;
        rij_hat[1] = rij[1]*inv_mag; 
        rij_hat[2] = rij[2]*inv_mag;
        float mag = sqrtf(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
        float force_mag = -1*edges[4*tid+2]*(mag-edges[4*tid+3]);
        float force[3];
        force[0] = rij_hat[0]*force_mag;
        force[1] = rij_hat[1]*force_mag;
        force[2] = rij_hat[2]*force_mag;
        atomicAdd(&forces[3*iid],force[0]);
        atomicAdd(&forces[3*iid+1],force[1]);
        atomicAdd(&forces[3*iid+2],force[2]);
        atomicAdd(&forces[3*jid],-1*force[0]);
        atomicAdd(&forces[3*jid+1],-1*force[1]);
        atomicAdd(&forces[3*jid+2],-1*force[2]);
    }
    }
    ''', 'spring_force')

spring_length_kernel = cp.RawKernel(r'''
extern "C" __global__
void spring_length(const float* edges, const float* node_posns, float* lengths, const int size_edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("entered the kernel");
    if (tid < size_edges)
    {
        //printf("entered the if block");
        //printf("spring tid = %i, size_springs = %i\n",tid,size_edges);  
        int iid = edges[4*tid];
        int jid = edges[4*tid+1];
        float rij[3];
        rij[0] = node_posns[3*iid]-node_posns[3*jid];
        rij[1] = node_posns[3*iid+1]-node_posns[3*jid+1];
        rij[2] = node_posns[3*iid+2]-node_posns[3*jid+2];
        lengths[tid] = sqrtf(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
    }
    }
    ''', 'spring_length')

drag_kernel = cp.RawKernel(r'''
extern "C" __global__
void drag_force(float* acceleration, float* velocity, float drag, const int num_entries) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_entries)
    {
        acceleration[tid] -= velocity[tid] * drag;
    }
    }
    ''', 'drag_force')

beta_scaling_kernel = cp.RawKernel(r'''
extern "C" __global__
void beta_scaling(const float* beta_i, float* forces, const int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes)
    {
        forces[3*tid] *= beta_i[tid];
        forces[3*tid+1] *= beta_i[tid];
        forces[3*tid+2] *= beta_i[tid];
    }
    }
    ''', 'beta_scaling')

### Integration Kernels

leapfrog_velocity_kernel = cp.RawKernel(r'''
extern "C" __global__
void velocity_update(float* velocity, float* acceleration, float step_size, const int num_entries) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_entries)
    {
        velocity[tid] = velocity[tid] + step_size * acceleration[tid];
    }
    }
    ''', 'velocity_update')

leapfrog_position_kernel = cp.RawKernel(r'''
extern "C" __global__
void position_update(float* position, float* velocity, float step_size, const int num_entries) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_entries)
    {
        position[tid] = position[tid] + step_size * velocity[tid];
    }
    }
    ''', 'position_update')

leapfrog_kernel = cp.RawKernel(r'''
extern "C" __global__
void y_update(float* y, float* dy, const float step_size, const int num_entries) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_entries)
    {
        //if (dy[tid] < -10)
        //{
        //    printf("step_size %f * dy %f = %f\n",step_size,dy[tid],step_size*dy[tid]);                       
        //}
        y[tid] = y[tid] + step_size * dy[tid];
    }
    }
    ''', 'y_update')

### Magnetization and magnetic field kernels

magnetization_kernel = cp.RawKernel(r'''
extern "C" __global__                                    
void get_magnetization(const float Ms, const float chi, const float particle_volume, const float* H, float* magnetic_moment, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        //get magnetization based on total Hfield at each particle center
        float H_mag = sqrtf(H[3*tid]*H[3*tid] + H[3*tid+1]*H[3*tid+1] + H[3*tid+2]*H[3*tid+2]);
        //printf("H_mag: %f\n",H_mag);
        //printf("particle volume: %f\n", particle_volume);
        //printf("Ms + chi*H_mag: %f\n",Ms + chi*H_mag);
        //printf("M/Ms: %f\n",chi*H_mag/(Ms + chi*H_mag));
        float m_mag = particle_volume*(Ms*chi*H_mag/(Ms + chi*H_mag));
        //printf("m_mag: %f\n",m_mag);
        float H_hat[3];
        H_hat[0] = H[3*tid]/H_mag;
        H_hat[1] = H[3*tid+1]/H_mag;
        H_hat[2] = H[3*tid+2]/H_mag;
        //printf("H_hat: %f, %f, %f\n",H_hat[0],H_hat[1],H_hat[2]);
        magnetic_moment[3*tid] = m_mag*H_hat[0];
        magnetic_moment[3*tid+1] = m_mag*H_hat[1];
        magnetic_moment[3*tid+2] = m_mag*H_hat[2];
    }
    }
    ''', 'get_magnetization')

separation_vectors_kernel = cp.RawKernel(r'''
extern "C" __global__                                         
void get_separation_vectors(const float* particle_posns, float* separation_vectors, float* separation_vectors_inv_magnitude, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        float ri[3];
        int separation_vector_idx;
        int posn_idx;
        int inv_magnitude_idx;
        ri[0] = particle_posns[3*tid];
        ri[1] = particle_posns[3*tid+1];
        ri[2] = particle_posns[3*tid+2];
        for (int i = 0; i < size_particles; i++)
        {
            if (i != tid)
            {
            separation_vector_idx = tid*3*size_particles+3*i;
            inv_magnitude_idx = tid*size_particles + i;
            posn_idx = 3*i;
            separation_vectors[separation_vector_idx] = ri[0] - particle_posns[posn_idx];                                         
            separation_vectors[separation_vector_idx+1] = ri[1] - particle_posns[posn_idx+1];                                                                
            separation_vectors[separation_vector_idx+2] = ri[2] - particle_posns[posn_idx+2];                                                                
            separation_vectors_inv_magnitude[inv_magnitude_idx] = rnorm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2]);
            }
        }
    }
    }
    ''', 'get_separation_vectors')

dipole_field_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
extern "C" __global__
void get_dipole_field(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float* magnetic_moment, const float inv_l_e, float* Htot, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float prefactor;
        float r_hat[3];
        float m_dot_r_hat;
        int separation_vector_idx;
        int inv_magnitude_idx;
        //float INV_4PI = 1/(4*3.141592654f);
        for (int i = 0; i < size_particles; i++)
        {
            separation_vector_idx = tid*3*size_particles+3*i;
            inv_magnitude_idx = tid*size_particles + i;
            r_hat[0] = separation_vectors[separation_vector_idx]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[1] = separation_vectors[separation_vector_idx+1]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            r_hat[2] = separation_vectors[separation_vector_idx+2]*separation_vectors_inv_magnitude[inv_magnitude_idx];
            m_dot_r_hat = magnetic_moment[3*i]*r_hat[0] + magnetic_moment[3*i+1]*r_hat[1] + magnetic_moment[3*i+2]*r_hat[2];
            prefactor = INV_4PI*powf(separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e,3);
            Htot[3*tid] += prefactor*(3*m_dot_r_hat*r_hat[0] - magnetic_moment[3*i]);
            Htot[3*tid+1] += prefactor*(3*m_dot_r_hat*r_hat[1] - magnetic_moment[3*i+1]);
            Htot[3*tid+2] += prefactor*(3*m_dot_r_hat*r_hat[2] - magnetic_moment[3*i+2]);
        }
    }
    }
    ''', 'get_dipole_field')

def get_particle_center_gpu(particle_nodes,node_posns):
    """With data in device memory, find the particle center using cupy functionality"""
    xindices = 3*particle_nodes
    yindices = xindices + 1
    zindices = yindices + 1
    x_max = cp.amax(cp.take(node_posns,xindices))
    x_min = cp.amin(cp.take(node_posns,xindices))
    y_max = cp.amax(cp.take(node_posns,yindices))
    y_min = cp.amin(cp.take(node_posns,yindices))
    z_max = cp.amax(cp.take(node_posns,zindices))
    z_min = cp.amin(cp.take(node_posns,zindices))
    particle_center = cp.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2],dtype=cp.float32)
    return particle_center

def get_particle_centers_gpu(node_posns,particles,num_particles):
    """Horribly slow method of getting particle centers while keeping the arrays in device memory"""
    particle_centers = cp.zeros((3*num_particles),dtype=cp.float32,order='C')
    particle_size = cp.int32(particles.shape[0]/num_particles)
    for i in range(num_particles):
        particle = cp.take(particles,cp.arange(i*particle_size,(i+1)*particle_size))
        particle_centers[3*i:3*(i+1)] = get_particle_center_gpu(particle,node_posns)
    return particle_centers

dipole_force_kernel = cp.RawKernel(r'''
float INV_4PI = 1/(4*3.141592654f);
float SURFACE_TO_SURFACE_SPACING = 1e-7;
extern "C" __global__
void get_dipole_force(const float* separation_vectors, const float* separation_vectors_inv_magnitude, const float* magnetic_moment, const float particle_radius, const float l_e, const float inv_l_e, float* force, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                
        float eps_constant = (1e-7)*4*powf(3.141592654f,2)*powf(1.9e6,2)*powf(1.5e-6,3)/72;
        //printf("eps_constant = %f e-13\n",eps_constant*1e13);
        float sigma = (2*particle_radius+SURFACE_TO_SURFACE_SPACING);
        float cutoff_length = powf(2.f,(1.f/6.f))*sigma;
        //printf("2^(1/6) = %f\n",powf(2.f,(1.f/6.f)));
        float prefactor;
        float r_hat[3];
        float mi_dot_r_hat;
        float mj_dot_r_hat;
        float m_dot_m;
        float mi[3];
        mi[0] = magnetic_moment[3*tid];
        mi[1] = magnetic_moment[3*tid+1];
        mi[2] = magnetic_moment[3*tid+2];
        float mj[3];
        int separation_vector_idx;
        int inv_magnitude_idx;
        float mu0_over_four_pi = 1e-7;
        float force_temp_var;
        float rijmag;
        float sigma_over_separation;
        for (int j = 0; j < size_particles; j++)
        {
            if (tid != j)
            {
                mj[0] = magnetic_moment[3*j];
                mj[1] = magnetic_moment[3*j+1];
                mj[2] = magnetic_moment[3*j+2];
                separation_vector_idx = tid*3*size_particles+3*j;
                inv_magnitude_idx = tid*size_particles + j;
                r_hat[0] = separation_vectors[separation_vector_idx]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                r_hat[1] = separation_vectors[separation_vector_idx+1]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                r_hat[2] = separation_vectors[separation_vector_idx+2]*separation_vectors_inv_magnitude[inv_magnitude_idx];
                //printf("tid = %i, j = %i, r_hat = %f, %f, %f\n",tid,j,r_hat[0],r_hat[1],r_hat[2]);
                mi_dot_r_hat = mi[0]*r_hat[0] + mi[1]*r_hat[1] + mi[2]*r_hat[2];
                mj_dot_r_hat = mj[0]*r_hat[0] + mj[1]*r_hat[1] + mj[2]*r_hat[2];
                m_dot_m = mi[0]*mj[0] + mi[1]*mj[1] + mi[2]*mj[2];
                prefactor = 3*mu0_over_four_pi*powf(separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e,4);
                force_temp_var = prefactor*(mj_dot_r_hat*mi[0] + mi_dot_r_hat*mj[0] + m_dot_m*r_hat[0] - 5*r_hat[0]*mi_dot_r_hat*mj_dot_r_hat);
                force[3*tid] += force_temp_var;
                force_temp_var = prefactor*(mj_dot_r_hat*mi[1] + mi_dot_r_hat*mj[1] + m_dot_m*r_hat[1] - 5*r_hat[1]*mi_dot_r_hat*mj_dot_r_hat);
                force[3*tid+1] += force_temp_var;
                force_temp_var = prefactor*(mj_dot_r_hat*mi[2] + mi_dot_r_hat*mj[2] + m_dot_m*r_hat[2] - 5*r_hat[2]*mi_dot_r_hat*mj_dot_r_hat);
                force[3*tid+2] += force_temp_var;
                rijmag = norm3df(separation_vectors[separation_vector_idx],separation_vectors[separation_vector_idx+1],separation_vectors[separation_vector_idx+2])*l_e;
                //printf("rij magnitude = %f e-6, cutoff_length = %f e-6\n",rijmag*1e6,cutoff_length*1e6);
                if (rijmag <= cutoff_length)
                {
                    sigma_over_separation = sigma*separation_vectors_inv_magnitude[inv_magnitude_idx]*inv_l_e;
                    force_temp_var = 4*eps_constant*(12*powf(sigma_over_separation,13) - 6*powf(sigma_over_separation,7))/sigma;
                    //printf("inside WCA force calculation bit\ntid = %i, j = %i, wca_force = %f,%f,%f e-6\n",tid,j,force_temp_var*r_hat[0]*1e6,force_temp_var*r_hat[1]*1e6,force_temp_var*r_hat[2]*1e6);
                    force[3*tid] += force_temp_var*r_hat[0];
                    force[3*tid+1] += force_temp_var*r_hat[1];
                    force[3*tid+2] += force_temp_var*r_hat[2];
                }
            }
        }
    }
    }
    ''', 'get_dipole_force')

distribute_magnetic_force_kernel = cp.RawKernel(r'''
extern "C" __global__
void set_dipole_force(const int* particle_nodes, const float* magnetic_force, float* node_force, const int nodes_per_particle, const int num_particle_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_particle_nodes)
    {                
        int mag_force_idx = tid/nodes_per_particle;
        /*
        printf("tid = %i\n",tid);
        printf("nodes_per_particle = %i\n", nodes_per_particle);
        printf("magnetic force index %i\n",mag_force_idx);
        */
        int node_force_idx = particle_nodes[tid];
        //printf("node force index %i\n",node_force_idx);
        /*
        printf("magnetic_force[3*mag_force_idx] = %.8f e-6\n",magnetic_force[3*mag_force_idx]*1e6);
        printf("magnetic_force[3*mag_force_idx+1] = %.8f e-6\n",magnetic_force[3*mag_force_idx+1]*1e6);
        printf("magnetic_force[3*mag_force_idx+2] = %.8f e-6\n",magnetic_force[3*mag_force_idx+2]*1e6);
        */
        node_force[3*node_force_idx] += magnetic_force[3*mag_force_idx];
        node_force[3*node_force_idx+1] += magnetic_force[3*mag_force_idx+1];
        node_force[3*node_force_idx+2] += magnetic_force[3*mag_force_idx+2];
    }
    }
    ''', 'set_dipole_force')

def get_magnetization_iterative(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    Htot_initial = cp.tile(Hext,particles.shape[0])
    for i in range(max_iters):
        Htot = cp.copy(Htot_initial)
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
    host_magnetic_moments = cp.asnumpy(magnetic_moment).reshape((particles.shape[0],3))
    return host_magnetic_moments

def get_magnetization_iterative_and_total_field(Hext,particles,particle_posns,Ms,chi,particle_volume,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moment = cp.zeros((particles.shape[0]*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,particles.shape[0])
    size_particles = particles.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moment,size_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((particles.shape[0]*particles.shape[0]*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((particles.shape[0]*particles.shape[0],1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,size_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    Htot_initial = cp.tile(Hext,particles.shape[0])
    for i in range(max_iters):
        Htot = cp.copy(Htot_initial)
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moment,size_particles))
        cupy_stream.synchronize()
    Htot = cp.copy(Htot_initial)
    dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moment,inv_l_e,Htot,size_particles))
    cupy_stream.synchronize()
    return magnetic_moment, Htot

def get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetic_moments = cp.zeros((num_particles*3,1),dtype=cp.float32)
    Hext_vector = cp.tile(Hext,num_particles)
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # normalized_magnetization_kernel((grid_size,),(block_size,),(Ms,chi,Hext_vector,magnetic_moment,size_particles))
    # magnetization_kernel used the particle volume to return the magnetic moment. the normalized approach normalizes by the saturation magnetization. the issue with using the particle volume to convert from magnetization to magnetic moment was that the result for low field values evaluated to zero.
    #the issue may have actually been that i didn't use 32 bit floats for chi, Ms, and l_e
    magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moments,num_particles))

    cupy_stream.synchronize()
    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,num_particles))
    cupy_stream.synchronize()

    inv_l_e = np.float32(1/l_e)
    max_iters = 5
    Htot_initial = cp.tile(Hext,num_particles)
    for i in range(max_iters):
        Htot = cp.copy(Htot_initial)#cp.tile(Hext,num_particles)
        dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,inv_l_e,Htot,num_particles))
        cupy_stream.synchronize()
        magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moments,num_particles))
        cupy_stream.synchronize()
    inv_l_e = np.float32(1/l_e)
    force = cp.zeros((3*num_particles,1),dtype=cp.float32,order='C')
    dipole_force_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,particle_radius,l_e,inv_l_e,force,num_particles))
    cupy_stream.synchronize()
    force *= np.float32(beta/particle_mass)
    return force

def composite_gpu_force_calc(normalized_posns,N_nodes,cupy_elements,kappa,cupy_springs):
    """Combining the two kernels so that the positions only need to be transferred from host to device memory once per calculation"""
    # mempool = cp.get_default_memory_pool()
    # print(f'default memory limit is {mempool.get_limit()}')
    # pinned_mempool = cp.get_default_pinned_memory_pool()
    # print(f'Bytes used before instantiating force arrays and moving node posn variable in GB: {mempool.used_bytes()/1024/1024/1024}')
    # print(f'Total bytes before instantiating force arrays and moving node posn variable in GB: {mempool.total_bytes()/1024/1024/1024}')
    # cupy_element_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    # cupy_spring_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    cupy_composite_element_spring_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    cupy_node_posns = cp.array(np.float32(normalized_posns)).reshape((normalized_posns.shape[0]*normalized_posns.shape[1],1),order='C')
    #print(f'Bytes used after instantiation and moving node posns from host to device in GB: {mempool.used_bytes()/1024/1024/1024}')
    #print(f'Total bytes after instantiation and moving node posns from host to device in GB: {mempool.total_bytes()/1024/1024/1024}')
    # print(f'current memory limit is {mempool.get_limit()}')
    # mempool.set_limit(size=1024**3)
    # print(f'current memory limit is {mempool.get_limit()}')
    size_elements = int(cupy_elements.shape[0]/8)
    block_size = 128
    element_grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/14)*14))
    scaled_element_kernel((element_grid_size,),(block_size,),(cupy_elements,cupy_node_posns,kappa,cupy_composite_element_spring_forces,size_elements))
    # cupy_stream = cp.cuda.get_current_stream()
    # cupy_stream.synchronize()
    size_springs = int(cupy_springs.shape[0]/4)
    spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/14)*14))
    scaled_spring_kernel((spring_grid_size,),(block_size,),(cupy_springs,cupy_node_posns,cupy_composite_element_spring_forces,size_springs))
    # cp.cuda.Device().synchronize()
    # host_element_forces = cp.asnumpy(cupy_element_forces)
    # host_spring_forces = cp.asnumpy(cupy_spring_forces)
    host_composite_element_spring_forces = cp.asnumpy(cupy_composite_element_spring_forces)
    return host_composite_element_spring_forces# host_element_forces, host_spring_forces

def get_accel_scaled_GPU(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    composite_element_spring_forces = composite_gpu_force_calc(x0,N_nodes,elements,kappa,springs)
    composite_element_spring_forces = np.reshape(composite_element_spring_forces,(N_nodes,3))
    composite_element_spring_forces *= beta_i[:,np.newaxis]
    accel = composite_element_spring_forces - drag * v0
    # volume_correction_force, spring_force = composite_gpu_force_calc(x0,N_nodes,elements,kappa,springs)
    # volume_correction_force = np.reshape(volume_correction_force,(N_nodes,3))
    # spring_force = np.reshape(spring_force,(N_nodes,3))
    # volume_correction_force *= (l_e**2)*beta_i[:,np.newaxis]
    # spring_force *= l_e*beta_i[:,np.newaxis]
    # accel = spring_force + volume_correction_force - drag * v0# + bc_forces
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        single_node_accel = (net_force_mag/stressed_nodes.shape[0])*beta_i[stressed_nodes]
        accel[stressed_nodes,force_index] += single_node_accel
    elif bc[0] == 'stress_compression':
        plate_orientation = bc[1][0]
        stress_direction = bc[1][1]
        stress = bc[2]
        global_index_interacting_nodes, plate_force = distribute_plate_stress(x0,stress,stress_direction,dimensions,l_e,plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
    elif bc[0] == 'plate_compression':
        plate_orientation = bc[1][0]
        global_index_interacting_nodes, plate_force = get_plate_force(x0,bc[2],plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        fixed_nodes = np.array([0])
        accel = set_fixed_nodes(accel,fixed_nodes)
    # #characteristic time necessary to get the scaled angular acceleration correctly. (in a previous implementation)
    # characteristic_time_squared = beta*l_e
    if particles.shape[0] != 0:
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,x0)
        M = magnetism.get_magnetization_iterative_normalized_32bit(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized_32bit(M,particle_centers,particle_radius,l_e,Ms)
        magnetic_scaling_factor = beta/(particle_mass*(np.power(l_e,4)))
        type_cast_scaling_factor = np.float32(magnetic_scaling_factor)
        mag_forces *= np.float32(beta/(particle_mass*(np.power(l_e,4))))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        #TODO remove loops as much as possible within python. this function may be cythonized anyway, there is serious overhead with any looping, even just dealing with the rigid particles
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            #determine the net torque acting on the particle before the net force acting on the particle is calculated and distributed to the particle nodes
            total_torque[i,:] = get_torque_on_particle(particle,accel,x0)
            #get the net force acting on the particle and distribute to the particle nodes
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
            #find the magnitude of the net torque, and if it is not 0, calculate and distribute the forces necessary to match rigid particle rotation behavior
            torque_magnitude = np.linalg.norm(total_torque[i,:])
            if not np.isclose(torque_magnitude,0):
                angular_acceleration = total_torque[i,:]/particle_moment_of_inertia
                angular_acceleration_magnitude = np.linalg.norm(angular_acceleration)
                torque_unit_vector = total_torque[i,:]/torque_magnitude
                #we are dealing with a coordinate system relative to the center of the particle, so we need to translate all the particle nodes in such a way that the center of the central voxel is at (0,0,0). This is necessary for calculating the vectors pointing from the axis of rotation to the particle nodes, so that we can calculate the correct forces involved
                translated_particle_nodes = x0[particle,:] - particle_centers[i,:]
                r_parallel_to_axis_of_rotation = np.sum(translated_particle_nodes*torque_unit_vector[np.newaxis,:],axis=1)[:,np.newaxis]*torque_unit_vector[np.newaxis,:]
                r_perpendicular_to_axis_of_rotation = translated_particle_nodes - r_parallel_to_axis_of_rotation
                r_perp_magnitude = np.linalg.norm(r_perpendicular_to_axis_of_rotation,axis=1)
                rotational_acceleration_magnitude = r_perp_magnitude*angular_acceleration_magnitude
                rotational_acceleration_nonunit_vector = np.cross(torque_unit_vector,r_perpendicular_to_axis_of_rotation)
                rotational_acceleration_unit_vector = rotational_acceleration_nonunit_vector/np.linalg.norm(rotational_acceleration_nonunit_vector,axis=1)[:,np.newaxis]
                rotational_acceleration = rotational_acceleration_unit_vector*rotational_acceleration_magnitude[:,np.newaxis]
                accel[particle] += rotational_acceleration
    # if np.max(np.linalg.norm(accel,axis=1)) > 1e4:
    #     print('acceleration spike')
    #     print(f'maximum norm of acceleration due to VCF: {np.max(np.linalg.norm(volume_correction_force,axis=1))}')
    #     print(f'average norm of accel due to VCF: {np.mean(np.linalg.norm(volume_correction_force,axis=1))}')
    #     print(f'maximum norm of acceleration due to springs: {np.max(np.linalg.norm(spring_force,axis=1))}')
    #     print(f'average norm of acceleration due to springs: {np.mean(np.linalg.norm(spring_force,axis=1))}')
    #     print(f'maximum norm of acceleration due to dipole-dipole: {np.max(np.linalg.norm(mag_forces,axis=1))}')
    #     print(f'particle separation: {np.linalg.norm(particle_centers[0]-particle_centers[1])}')
    return accel

def scaled_fun_gpu(t,y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting forces on each vertex/node and their velocities"""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    # y = np.float32(y)
    accel = get_accel_scaled_GPU(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
    N_nodes = int(np.round(N/3))
    accel = np.reshape(accel,(3*N_nodes,))
    v0 = y[N:]
    result = np.concatenate((v0,accel))
    #we have to reshape our results as fun() has to return something in the shape (n,) (has to return dy/dt = f(t,y,y')). because the ODE is second order we break it into a system of first order ODEs by substituting y1 = y, y2 = dy/dt. so that dy1/dt = y2, dy2/dt = f(t,y,y') (Which is the acceleration)
    return result

def simulate_scaled_gpu(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,drag,initialized_posns,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,criteria_flag=False,plotting_flag=False,persistent_checkpointing_flag=False,get_time_flag=False,get_norms_flag=False):
    """Run a simulation of a hybrid mass spring system using a Dormand-Prince adaptive step size numerical integration. Node_posns is an N_vertices by 3 numpy array of the positions of the vertices, elements is an N_elements by 8 numpy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined. t_f is the upper time integration bound, from t_i = 0 to t_f, over which the numerical integration will be performed with adaptive time steps ."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    sim_time = []
    solution_norms = []
    derivative_norms = []
    def solout_norms(t,y):
        solution_norms.append(np.linalg.norm(y))
        accel = get_accel_scaled_GPU(y,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag)
        N = int(np.round(y.shape[0]/2))
        accel = np.reshape(accel,(3*N_nodes,))
        derivatives = np.concatenate((y[N:],accel))
        derivative_norms.append(np.linalg.norm(derivatives))
    def solout_timestep(t,y):
        if sim_time == []:
            sim_time.append(t)
        else:
            new_time = t + np.max(np.array(sim_time))
            sim_time.append(new_time)
        solutions.append([*y])
    def solout(t,y):
        solutions.append([t,*y])
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    v0 = np.zeros(x0.shape,dtype=np.float32)
    y_0 = np.concatenate((x0.reshape((3*x0.shape[0],)),v0.reshape((3*v0.shape[0],))))
    if plotting_flag:
        mre.analyze.plot_center_cuts(initialized_posns,x0,springs,particles,boundary_conditions,output_dir,tag='starting_configuration')
    #TODO decide if you want to bother with doing a backtracking if the system diverges. there is a significant memory overhead associated with this approach.
    backstop_solution = y_0.copy()
    N_nodes = int(x0.shape[0])
    my_nsteps = max_integration_steps
    particle_moment_of_inertia = np.float32((2/5)*particle_mass*np.power(particle_radius,2))
    #needs to be scaled, but the scaling here includes (inside of beta) the characteristic time, squared, which is necessary for scaling the angular acceleration. the angular acceleration is scaling handled internally in the function get_accel_scaled_rotation(), but we need to account for the term to scale the moment of inertia properly, so we remove the time scaling that was previously involved here. the particle mass and the number of nodes making up the particle is used to go from beta to beta_i
    # characteristic_time_squared = beta*l_e
    # scaled_moment_of_inertia = particle_moment_of_inertia*beta/(particle_mass/particles.shape[1])/l_e/characteristic_time_squared
    # using the fact that we have an analytical expression, I will rewrite the above initialization of the scaled moment of inertia in a way that uses less operations
    scaled_moment_of_inertia = np.float32(particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2)))
    #scipy.integrate.solve_ivp() requires the solution y to have shape (n,)
    r = sci.ode(scaled_fun_gpu).set_integrator('dopri5',nsteps=max_integration_steps,verbosity=1)
    if criteria_flag:
        r.set_solout(solout)
    elif get_time_flag:
        r.set_solout(solout_timestep)
    elif get_norms_flag:
        r.set_solout(solout_norms)
    max_displacement = np.zeros((max_integrations,))
    mean_displacement = np.zeros((max_integrations,))
    return_status = 1
    for i in range(max_integrations):
        # if len(sim_time) != 0:
        #     delta_t = np.array(sim_time[1:]) - np.array(sim_time[:-1])
        #     first_step = delta_t[-1]
        #     r = sci.ode(scaled_fun_gpu).set_integrator('dopri5',nsteps=max_integration_steps,verbosity=1,first_step=first_step)
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        print(f'starting integration run {i+1}')
        sol = r.integrate(t_f)
        a_var = get_accel_scaled_GPU(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        max_accel_norm_avg = 0
        if criteria_flag:
            if i == 0:
                criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(criteria.a_norm_avg)
            else:
                other_criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
                max_accel_norm_avg = np.max(other_criteria.a_norm_avg)
        final_posns = np.reshape(sol[:N_nodes*3],(N_nodes,3))
        final_v = np.reshape(sol[N_nodes*3:],(N_nodes,3))
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if i == 0:
            tag = '1st_configuration'
        elif i == 1:
            tag = '2nd_configuration'
        elif i == 2:
            tag = '3rd_configuration'
        else:
            tag = f'{i+1}th_configuration'
        if plotting_flag:
            mre.analyze.plot_center_cuts(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir,tag)
        if a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            return_status = 0
            break
        elif max_accel_norm_avg > 1e3:
            print(f'strong accelerations detected during integration run {i+1}')
            my_nsteps = int(my_nsteps/2)
            if my_nsteps < 10:
                print(f'total steps allowed down to: {my_nsteps}\n breaking out with last acceptable solution')
                sol = backstop_solution.copy()
                del backstop_solution
                del y_0
                del solutions
                return_status = -1
                break
            print(f'restarting from last acceptable solution with acceleration norm mean of {criteria.a_norm_avg[-1]}')
            print(f'running with halved maximum number of steps: {my_nsteps}')
            r = sci.ode(scaled_fun).set_integrator('dopri5',nsteps=my_nsteps,verbosity=1)
            print(f'Increasing drag coefficient from {drag} to {10*drag}')
            drag *= 10
            y_0 = backstop_solution.copy()
        else:
            print(f'Post-Integration norms\nacceleration norm average = {a_norm_avg}\nvelocity norm average = {v_norm_avg}')
            y_0 = sol.copy()
            last_posns = np.reshape(backstop_solution[:N_nodes*3],(N_nodes,3))
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            if i != 0 and criteria_flag:
                criteria.append_criteria(other_criteria)
            backstop_solution = sol.copy()
        if criteria_flag:
            solutions = []
            r.set_solout(solout)
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
    plot_displacement_v_integration(max_integrations,mean_displacement,max_displacement,output_dir)
    if criteria_flag:
        mre.initialize.write_criteria_file(criteria,output_dir)
        criteria.plot_criteria_subplot(output_dir)
        criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    if get_time_flag:
        plot_simulation_time_versus_integration_step(sim_time,output_dir)
        np.save(output_dir+'solutions.npy',solutions,allow_pickle=False)
    if get_norms_flag:
        plot_solution_and_derivative_norms_versus_integration_step(solution_norms,derivative_norms,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def composite_gpu_force_calc_v2(posns,velocities,N_nodes,cupy_elements,kappa,cupy_springs,beta_i,drag):
    """Combining the kernels so that memory transfers from host to device and device to host are limited and as much is calculated on gpu as possible"""
    # mempool = cp.get_default_memory_pool()
    # print(f'Bytes used before instantiating force arrays and moving node posn variable in GB: {mempool.used_bytes()/1024/1024/1024}')
    # print(f'Total bytes before instantiating force arrays and moving node posn variable in GB: {mempool.total_bytes()/1024/1024/1024}')
    cupy_composite_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    # cupy_node_posns = cp.array(np.float32(normalized_posns)).reshape((normalized_posns.shape[0]*normalized_posns.shape[1],1),order='C')
    #print(f'Bytes used after instantiation and moving node posns from host to device in GB: {mempool.used_bytes()/1024/1024/1024}')
    #print(f'Total bytes after instantiation and moving node posns from host to device in GB: {mempool.total_bytes()/1024/1024/1024}')
    size_elements = int(cupy_elements.shape[0]/8)
    block_size = 128
    element_grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/14)*14))
    scaled_element_kernel((element_grid_size,),(block_size,),(cupy_elements,posns,kappa,cupy_composite_forces,size_elements))
    # host_forces = cp.asnumpy(cupy_composite_forces)
    # if np.any(np.isnan(host_forces)):
    #     print(f'NaN entries from VCF force')
    #     print(f'Num NaN Entries{np.count_nonzero(np.isnan(host_forces))}')
    #     print(f'max posns value: {cp.max(posns)}')
    #     print(f'min posns value: {cp.min(posns)}')
    size_springs = int(cupy_springs.shape[0]/4)
    spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/14)*14))
    scaled_spring_kernel((spring_grid_size,),(block_size,),(cupy_springs,posns,cupy_composite_forces,size_springs))
    # host_forces = cp.asnumpy(cupy_composite_forces)
    # if np.any(np.isnan(host_forces)):
    #     print(f'NaN entries after spring force')
    #     print(f'Num NaN Entries{np.count_nonzero(np.isnan(host_forces))}')
    beta_scaling_grid_size = (int (np.ceil((int (np.ceil(N_nodes/block_size)))/14)*14))
    beta_scaling_kernel((beta_scaling_grid_size,),(block_size,),(beta_i,cupy_composite_forces,N_nodes))
    # host_forces = cp.asnumpy(cupy_composite_forces)
    # if np.any(np.isnan(host_forces)):
    #     print(f'NaN entries after scaling force')
    #     print(f'Num NaN Entries{np.count_nonzero(np.isnan(host_forces))}')
    drag_grid_size = (int (np.ceil((int (np.ceil(3*N_nodes/block_size)))/14)*14))
    drag_kernel((drag_grid_size,),(block_size,),(cupy_composite_forces,velocities,drag,int(3*N_nodes)))
    host_composite_forces = cp.asnumpy(cupy_composite_forces)
    # if np.any(np.isnan(host_composite_forces)):
    #     print(f'NaN entries after drag force')
    #     print(f'Num NaN Entries{np.count_nonzero(np.isnan(host_composite_forces))}')
    return host_composite_forces

def composite_gpu_force_calc_v3(posns,velocities,N_nodes,cupy_elements,kappa,cupy_springs,stressed_boundary,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e):
    """Combining gpu kernels to calculate different forces and perform scaling"""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    cupy_composite_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    size_elements = int(cupy_elements.shape[0]/8)
    block_size = 128
    element_grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    scaled_element_kernel((element_grid_size,),(block_size,),(cupy_elements,posns,kappa,cupy_composite_forces,size_elements))
    cupy_stream.synchronize()

    size_springs = int(cupy_springs.shape[0]/4)
    spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    scaled_spring_kernel((spring_grid_size,),(block_size,),(cupy_springs,posns,cupy_composite_forces,size_springs))
    cupy_stream.synchronize()

    #need to apply stress if appropriate before scaling, or else i need to scale inside the stress kernel
    size_boundary = stressed_boundary.shape[0]
    if size_boundary != 0:
        boundary_stress_grid_size = (int (np.ceil((int (np.ceil(size_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary,stress_direction,stress_node_force,cupy_composite_forces,size_boundary))
        cupy_stream.synchronize()

    beta_scaling_grid_size = (int (np.ceil((int (np.ceil(N_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    beta_scaling_kernel((beta_scaling_grid_size,),(block_size,),(beta_i,cupy_composite_forces,N_nodes))
    cupy_stream.synchronize()

    drag_grid_size = (int (np.ceil((int (np.ceil(3*N_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    drag_kernel((drag_grid_size,),(block_size,),(cupy_composite_forces,velocities,drag,int(3*N_nodes)))
    cupy_stream.synchronize()

    fixed_nodes_size = fixed_nodes.shape[0]
    if fixed_nodes_size != 0:
        fixed_node_grid_size = (int (np.ceil((int (np.ceil(fixed_nodes_size/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        fixed_boundary_kernel((fixed_node_grid_size,),(block_size,),(fixed_nodes,cupy_composite_forces,fixed_nodes_size))
        cupy_stream.synchronize()

    particle_posns = get_particle_centers_gpu(posns,particles,num_particles)
    magnetic_forces = get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
    #need to assign magnetic forces. every node belonging to a given particle has the same force as returned for the particle
    nodes_per_particle = cp.int32(particles.shape[0]/num_particles)
    num_particle_nodes = cp.int32(nodes_per_particle*num_particles)
    mag_force_distribution_grid_size = (int (np.ceil((int (np.ceil(num_particle_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    distribute_magnetic_force_kernel((mag_force_distribution_grid_size,),(block_size,),(particles,magnetic_forces,cupy_composite_forces,nodes_per_particle,num_particle_nodes))
    cupy_stream.synchronize()
    return cupy_composite_forces

def composite_gpu_force_calc_v3b(posns,velocities,N_nodes,cupy_elements,kappa,cupy_springs,stressed_boundary,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary,host_moving_boundary):
    """Combining gpu kernels to calculate different forces and perform scaling"""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    cupy_composite_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    size_elements = int(cupy_elements.shape[0]/8)
    block_size = 128
    element_grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    scaled_element_kernel((element_grid_size,),(block_size,),(cupy_elements,posns,kappa,cupy_composite_forces,size_elements))
    cupy_stream.synchronize()

    size_springs = int(cupy_springs.shape[0]/4)
    spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    scaled_spring_kernel((spring_grid_size,),(block_size,),(cupy_springs,posns,cupy_composite_forces,size_springs))
    cupy_stream.synchronize()

    #need to apply stress if appropriate before scaling, or else i need to scale inside the stress kernel
    size_boundary = stressed_boundary.shape[0]
    if size_boundary != 0:
        boundary_stress_grid_size = (int (np.ceil((int (np.ceil(size_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary,stress_direction,stress_node_force,cupy_composite_forces,size_boundary))
        cupy_stream.synchronize()

    beta_scaling_grid_size = (int (np.ceil((int (np.ceil(N_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    beta_scaling_kernel((beta_scaling_grid_size,),(block_size,),(beta_i,cupy_composite_forces,N_nodes))
    cupy_stream.synchronize()

    drag_grid_size = (int (np.ceil((int (np.ceil(3*N_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    drag_kernel((drag_grid_size,),(block_size,),(cupy_composite_forces,velocities,drag,int(3*N_nodes)))
    cupy_stream.synchronize()

    fixed_nodes_size = fixed_nodes.shape[0]
    if fixed_nodes_size != 0:
        fixed_node_grid_size = (int (np.ceil((int (np.ceil(fixed_nodes_size/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        fixed_boundary_kernel((fixed_node_grid_size,),(block_size,),(fixed_nodes,cupy_composite_forces,fixed_nodes_size))
        cupy_stream.synchronize()

    size_strained_boundary = moving_boundary.shape[0]
    #if the strain is zero, allow the boundary to move as a unit
    if size_strained_boundary != 0:
        host_forces = cp.asnumpy(cupy_composite_forces).reshape((int(N_nodes),3))
        net_force = np.sum(host_forces[host_moving_boundary],axis=0)
        individual_force = cp.asarray(net_force/size_strained_boundary,dtype=cp.float32,order='C')
        strained_boundary_grid_size = (int (np.ceil((int (np.ceil(size_strained_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        strained_boundary_kernel((strained_boundary_grid_size,),(block_size,),(moving_boundary,cupy_composite_forces,individual_force,size_strained_boundary))
        cupy_stream.synchronize()

    magnetic_forces = get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
    #need to assign magnetic forces. every node belonging to a given particle has the same force as returned for the particle
    cupy_composite_forces = distribute_magnetic_forces(particles,num_particles,magnetic_forces,cupy_composite_forces,num_streaming_multiprocessors,block_size)
    # nodes_per_particle = cp.int32(particles.shape[0]/num_particles)
    # num_particle_nodes = cp.int32(nodes_per_particle*num_particles)
    # mag_force_distribution_grid_size = (int (np.ceil((int (np.ceil(num_particle_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # distribute_magnetic_force_kernel((mag_force_distribution_grid_size,),(block_size,),(particles,magnetic_forces,cupy_composite_forces,nodes_per_particle,num_particle_nodes))
    cupy_stream.synchronize()
    return cupy_composite_forces

def distribute_magnetic_forces(particles,num_particles,magnetic_forces,total_forces,num_streaming_multiprocessors,block_size=128):
    nodes_per_particle = cp.int32(particles.shape[0]/num_particles)
    num_particle_nodes = cp.int32(nodes_per_particle*num_particles)
    mag_force_distribution_grid_size = (int (np.ceil((int (np.ceil(num_particle_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    distribute_magnetic_force_kernel((mag_force_distribution_grid_size,),(block_size,),(particles,magnetic_forces,total_forces,nodes_per_particle,num_particle_nodes))
    return total_forces

def leapfrog_update(y,dy,step_size,size_entries):
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_entries/block_size)))/14)*14))
    leapfrog_kernel((grid_size,),(block_size,),(y,dy,step_size,size_entries))
    return


# #GLOBAL VARIABLE FOR MAXIMUM MAGNETIC FORCE IN SI UNITS
# mu0 = 4*np.pi*1e-7
# particle_Ms = 1.9e6
# particle_radius_si = 1.5e-6
# particle_volume_si = (4/3)*np.pi*np.power(particle_radius_si,3)
# max_magnetic_force_norm = (3*mu0)/(2*np.pi*np.power(2*particle_radius_si,4))*np.power(particle_volume_si,2)*np.power(particle_Ms,2)
def get_accel_scaled_GPU_v2(posns,velocities,elements,springs,particles,kappa,l_e,beta,device_beta_i,host_beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    N_nodes = int(posns.shape[0]/3)
    accel = composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,device_beta_i,drag)
    accel = np.reshape(accel,(N_nodes,3))
    # if np.any(np.isnan(accel)):
    #     print(f'NaN entry found in the acceleration variable')
    #     print(f'{np.count_nonzero(np.isnan(accel))} NaN entries in acceleration variable')
    #     block_size = 128
    #     size_springs = int(springs.shape[0]/4)
    #     spring_grid_size = (int (np.ceil((int (np.ceil(size_springs/block_size)))/14)*14))
    #     spring_lengths = cp.empty((size_springs,),dtype=cp.float32)
    #     spring_length_kernel((spring_grid_size,),(block_size,),(springs,posns,spring_lengths,size_springs))
    #     host_spring_lengths = cp.asnumpy(spring_lengths)
    #     print(f'Number of springs with length zero: {np.count_nonzero(np.isclose(host_spring_lengths,0.0))}')
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        # working out method for stress that takes into account node type to assign forces
        # first, find the "force per node" by considering the net force mangitude to be the sum of the forces assigned to each of the nodes, with different weights/forces dependent on node type. if the node is on the surface interior, it gets the largest force. if on the corner, 1/4 that value, on an edge, 1/2 the surface interior force value
        #\net_force_mag = \sum_{nodes} force_per_node*weight, where weight is 1, 1/2, or 1/4. 1/4 for 4 nodes. 1/2 for num_edge_nodes. need to know num_edge_nodes. then need to assign forces properly.
        # effective_num_nodes = fixed_nodes.shape[0] - 
        single_node_accel = np.squeeze((net_force_mag/stressed_nodes.shape[0])*host_beta_i[stressed_nodes])
        accel[stressed_nodes,force_index] += single_node_accel
    elif bc[0] == 'stress_compression':
        plate_orientation = bc[1][0]
        stress_direction = bc[1][1]
        stress = bc[2]
        global_index_interacting_nodes, plate_force = distribute_plate_stress(posns,stress,stress_direction,dimensions,l_e,plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
    elif bc[0] == 'plate_compression':
        plate_orientation = bc[1][0]
        global_index_interacting_nodes, plate_force = get_plate_force(posns,bc[2],plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        fixed_nodes = np.array([0])
        accel = set_fixed_nodes(accel,fixed_nodes)
    if particles.shape[0] != 0:
        #move the node positions variable from device to host memory
        host_posns = cp.asnumpy(posns)
        host_posns = np.reshape(host_posns,(N_nodes,3))
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,host_posns)
        # particle_separation = np.linalg.norm(particle_centers[0]-particle_centers[1])
        # if particle_separation*l_e < 5e-6:
        #     print(f'particle separation = {particle_separation*l_e*1e6} um')
            # if particle_separation*l_e < 4.5e-6:
            #     print(f'debugger should be used to stop here and inspect step by step to find runtimewarning: invalid value encountered in subtract and invalid value encountered in reduce')
        M = magnetism.get_magnetization_iterative_normalized_32bit(Hext,particle_centers,particle_radius,chi,Ms,l_e)
        # !!! Trying to resolve issues with float32 data types. trying to use scaled positions was resulting in the returned magnetic forces being small enough that calculating the force norm before scaling by beta and dividing by the particle mass would result in a norm of zero, even for what should result in non-zero accelerations. by using SI units for the particle positions in this function call, it is possible to avoid issues with the floating point value being so small that (what i think is an underflow) some sort of error occurs where the resulting scaled acceleration is very large (components on the order of 1e19)
        mag_forces = magnetism.get_dip_dip_forces_normalized_32bit(M,particle_centers*l_e,particle_radius,l_e,Ms)
        # if np.any(np.isnan(mag_forces)):
        #     print(f'NaN entry found in the mag_forces variable')
        # mag_forces_si = mag_forces/np.power(l_e,4)
        # mag_force_clipping_check = np.linalg.norm(mag_forces_si,axis=1)>max_magnetic_force_norm
        # if np.any(mag_force_clipping_check):
        # #     mag_forces[mag_force_clipping_check,:] *= max_magnetic_force_norm/np.power(l_e,4)/np.linalg.norm(mag_forces[mag_force_clipping_check,:],axis=1)
        # max_mag_force_norm_si = 5.3443908e-6
        # magnetic_scaling_factor = beta/(particle_mass*(np.power(l_e,4)))
        # scaled_max_force = max_mag_force_norm_si*magnetic_scaling_factor*np.power(l_e,4)
        mag_forces *= np.float32(beta/particle_mass)#np.float32(beta/(particle_mass*(np.power(l_e,4))))
        for i, particle in enumerate(particles):
            # print(f'accel[particles[{i}]]:{accel[particle]}')
            accel[particle] += mag_forces[i]
            # print('after adding magnetic forces')
            # print(f'accel[particles[{i}]]:{accel[particle]}')
        #TODO remove loops as much as possible within python. this function may be cythonized anyway, there is serious overhead with any looping, even just dealing with the rigid particles
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            #determine the net torque acting on the particle before the net force acting on the particle is calculated and distributed to the particle nodes
            total_torque[i,:] = get_torque_on_particle(particle,accel,host_posns)
            #get the net force acting on the particle and distribute to the particle nodes
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
            #find the magnitude of the net torque, and if it is not 0, calculate and distribute the forces necessary to match rigid particle rotation behavior
            # torque_magnitude = np.linalg.norm(total_torque[i,:])
            # if not np.isclose(torque_magnitude,0):
            #     angular_acceleration = total_torque[i,:]/particle_moment_of_inertia
            #     angular_acceleration_magnitude = np.linalg.norm(angular_acceleration)
            #     torque_unit_vector = total_torque[i,:]/torque_magnitude
            #     #we are dealing with a coordinate system relative to the center of the particle, so we need to translate all the particle nodes in such a way that the center of the central voxel is at (0,0,0). This is necessary for calculating the vectors pointing from the axis of rotation to the particle nodes, so that we can calculate the correct forces involved
            #     translated_particle_nodes = host_posns[particle,:] - particle_centers[i,:]
            #     r_parallel_to_axis_of_rotation = np.sum(translated_particle_nodes*torque_unit_vector[np.newaxis,:],axis=1)[:,np.newaxis]*torque_unit_vector[np.newaxis,:]
            #     r_perpendicular_to_axis_of_rotation = translated_particle_nodes - r_parallel_to_axis_of_rotation
            #     r_perp_magnitude = np.linalg.norm(r_perpendicular_to_axis_of_rotation,axis=1)
            #     rotational_acceleration_magnitude = r_perp_magnitude*angular_acceleration_magnitude
            #     rotational_acceleration_nonunit_vector = np.cross(torque_unit_vector,r_perpendicular_to_axis_of_rotation)
            #     rotational_acceleration_unit_vector = rotational_acceleration_nonunit_vector/np.linalg.norm(rotational_acceleration_nonunit_vector,axis=1)[:,np.newaxis]
            #     rotational_acceleration = rotational_acceleration_unit_vector*rotational_acceleration_magnitude[:,np.newaxis]
            #     accel[particle] += rotational_acceleration
    else:
        host_posns = None
        particle_centers = None
        total_torque = None
    return accel, total_torque, host_posns, particle_centers

def get_accel_scaled_GPU_v3(posns,velocities,elements,springs,particles,kappa,l_e,beta,device_beta_i,host_beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,particle_moment_of_inertia,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    N_nodes = int(posns.shape[0]/3)
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    accel = composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,device_beta_i,drag)
    accel = np.reshape(accel,(N_nodes,3))
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        # working out method for stress that takes into account node type to assign forces
        # first, find the "force per node" by considering the net force mangitude to be the sum of the forces assigned to each of the nodes, with different weights/forces dependent on node type. if the node is on the surface interior, it gets the largest force. if on the corner, 1/4 that value, on an edge, 1/2 the surface interior force value
        #\net_force_mag = \sum_{nodes} force_per_node*weight, where weight is 1, 1/2, or 1/4. 1/4 for 4 nodes. 1/2 for num_edge_nodes. need to know num_edge_nodes. then need to assign forces properly.
        # effective_num_nodes = fixed_nodes.shape[0] - 
        single_node_accel = np.squeeze((net_force_mag/stressed_nodes.shape[0])*host_beta_i[stressed_nodes])
        accel[stressed_nodes,force_index] += single_node_accel
    elif bc[0] == 'stress_compression':
        plate_orientation = bc[1][0]
        stress_direction = bc[1][1]
        stress = bc[2]
        global_index_interacting_nodes, plate_force = distribute_plate_stress(posns,stress,stress_direction,dimensions,l_e,plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
    elif bc[0] == 'plate_compression':
        plate_orientation = bc[1][0]
        global_index_interacting_nodes, plate_force = get_plate_force(posns,bc[2],plate_orientation,boundaries)
        accel[global_index_interacting_nodes] += plate_force
        accel = set_plate_fixed_nodes(accel,global_index_interacting_nodes,plate_orientation)
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        fixed_nodes = np.array([0])
        accel = set_fixed_nodes(accel,fixed_nodes)
    if particles.shape[0] != 0:
        #move the node positions variable from device to host memory
        host_posns = cp.asnumpy(posns)
        host_posns = np.reshape(host_posns,(N_nodes,3))
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,host_posns)
        magnetic_moments = get_magnetization_iterative(Hext,particles,cp.array(particle_centers.astype(np.float32)).reshape((particle_centers.shape[0]*particle_centers.shape[1],1),order='C'),Ms,chi,particle_volume,l_e)
        # !!! Trying to resolve issues with float32 data types. trying to use scaled positions was resulting in the returned magnetic forces being small enough that calculating the force norm before scaling by beta and dividing by the particle mass would result in a norm of zero, even for what should result in non-zero accelerations. by using SI units for the particle positions in this function call, it is possible to avoid issues with the floating point value being so small that (what i think is an underflow) some sort of error occurs where the resulting scaled acceleration is very large (components on the order of 1e19)
        mag_forces = magnetism.get_dip_dip_forces_normalized_32bit_v2(magnetic_moments,particle_centers*l_e,particle_radius,l_e)
        # if np.any(np.isnan(mag_forces)):
        #     print(f'NaN entry found in the mag_forces variable')
        # mag_forces_si = mag_forces/np.power(l_e,4)
        # mag_force_clipping_check = np.linalg.norm(mag_forces_si,axis=1)>max_magnetic_force_norm
        # if np.any(mag_force_clipping_check):
        # #     mag_forces[mag_force_clipping_check,:] *= max_magnetic_force_norm/np.power(l_e,4)/np.linalg.norm(mag_forces[mag_force_clipping_check,:],axis=1)
        # max_mag_force_norm_si = 5.3443908e-6
        # magnetic_scaling_factor = beta/(particle_mass*(np.power(l_e,4)))
        # scaled_max_force = max_mag_force_norm_si*magnetic_scaling_factor*np.power(l_e,4)
        mag_forces *= np.float32(beta/particle_mass)#np.float32(beta/(particle_mass*(np.power(l_e,4))))
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
        total_torque = np.zeros((particles.shape[0],3))
        for i, particle in enumerate(particles):
            #determine the net torque acting on the particle before the net force acting on the particle is calculated and distributed to the particle nodes
            total_torque[i,:] = get_torque_on_particle(particle,accel,host_posns)
            #get the net force acting on the particle and distribute to the particle nodes
            vecsum = np.sum(accel[particle],axis=0)
            accel[particle] = vecsum/particle.shape[0]
    else:
        host_posns = None
        particle_centers = None
        total_torque = None
    return accel, total_torque, host_posns, particle_centers

def get_accel_scaled_GPU_v4(posns,velocities,elements,springs,particles,kappa,l_e,beta,device_beta_i,host_beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    N_nodes = int(posns.shape[0]/3)
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    accel = composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,device_beta_i,drag)
    accel = np.reshape(accel,(N_nodes,3))
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        # working out method for stress that takes into account node type to assign forces
        # first, find the "force per node" by considering the net force mangitude to be the sum of the forces assigned to each of the nodes, with different weights/forces dependent on node type. if the node is on the surface interior, it gets the largest force. if on the corner, 1/4 that value, on an edge, 1/2 the surface interior force value
        #\net_force_mag = \sum_{nodes} force_per_node*weight, where weight is 1, 1/2, or 1/4. 1/4 for 4 nodes. 1/2 for num_edge_nodes. need to know num_edge_nodes. then need to assign forces properly.
        # effective_num_nodes = fixed_nodes.shape[0] - 
        single_node_accel = np.squeeze((net_force_mag/stressed_nodes.shape[0])*host_beta_i[stressed_nodes])
        # if stress != 0:
        #     print('this is for debugging the applied stress on the boundary nodes')
        accel[stressed_nodes,force_index] += single_node_accel
    elif 'strain' in bc[0]:#bc[0] == 'tension' or bc[0] == 'compression' or bc[0] == 'shearing' or bc[0] == 'torsion':
        #TODO better handling for fixed_nodes, what is fixed, and when. fleshing out the boundary conditions variable and boundary conditions handling
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'hysteresis':
        fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        pass
        #2024-04-03. DBM. setting a single node fixed did not work as anticipated... testing how things (hysteresis sims) work if we don't hold anything fixed at all
        # fixed_nodes = np.array([0])
        # accel = set_fixed_nodes(accel,fixed_nodes)
    if particles.shape[0] != 0:
        #move the node positions variable from device to host memory
        host_posns = cp.asnumpy(posns)
        host_posns = np.reshape(host_posns,(N_nodes,3))
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,host_posns)
        magnetic_moments = get_magnetization_iterative(Hext,particles,cp.array(particle_centers.astype(np.float32)).reshape((particle_centers.shape[0]*particle_centers.shape[1],1),order='C'),Ms,chi,particle_volume,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized_32bit_v2(magnetic_moments,particle_centers*l_e,particle_radius,l_e)
        mag_forces *= np.float32(beta/particle_mass)
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
    else:
        host_posns = None
    return accel, host_posns

def get_accel_scaled_GPU_test(posns,velocities,elements,springs,particles,kappa,l_e,beta,device_beta_i,host_beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    N_nodes = int(posns.shape[0]/3)
    particle_volume = np.float32((4/3)*np.pi*np.power(particle_radius,3))
    accel = composite_gpu_force_calc_v2(posns,velocities,N_nodes,elements,kappa,springs,device_beta_i,drag)
    accel = np.reshape(accel,(N_nodes,3))
    if 'simple_stress' in bc[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if bc[1][0] == 'x':
            fixed_nodes = boundaries['left']
            stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif bc[1][0] == 'y':
            fixed_nodes = boundaries['front']
            stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif bc[1][0] == 'z':
            fixed_nodes = boundaries['bot']
            stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        accel = set_fixed_nodes(accel,fixed_nodes)
        stress_direction = bc[1][1]
        if stress_direction == 'x':
            force_index = 0
        elif stress_direction == 'y':
            force_index = 1
        elif stress_direction == 'z':
            force_index = 2
        stress = bc[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        single_node_accel = np.squeeze((net_force_mag/stressed_nodes.shape[0])*host_beta_i[stressed_nodes])
        accel[stressed_nodes,force_index] += single_node_accel
    elif 'strain' in bc[0]:
        if bc[1][0] == 'x':
            fixed_nodes = np.concatenate((boundaries['left'],boundaries['right']))
        elif bc[1][0] == 'y':
            fixed_nodes = np.concatenate((boundaries['front'],boundaries['back']))
        elif bc[1][0] == 'z':
            fixed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
        accel = set_fixed_nodes(accel,fixed_nodes)
    elif bc[0] == 'hysteresis':
        fixed_nodes = boundaries['bot']
        accel = set_fixed_nodes(accel,fixed_nodes)
    else:
        pass
        #2024-04-03. DBM. setting a single node fixed did not work as anticipated... testing how things (hysteresis sims) work if we don't hold anything fixed at all
        # fixed_nodes = np.array([0])
        # accel = set_fixed_nodes(accel,fixed_nodes)
    if particles.shape[0] != 0:
        #move the node positions variable from device to host memory
        host_posns = cp.asnumpy(posns)
        host_posns = np.reshape(host_posns,(N_nodes,3))
        #for each particle, find the position of the center
        particle_centers = np.empty((particles.shape[0],3),dtype=np.float32)
        for i, particle in enumerate(particles):
            particle_centers[i,:] = get_particle_center(particle,host_posns)
        magnetic_moments = get_magnetization_iterative(Hext,particles,cp.array(particle_centers.astype(np.float32)).reshape((particle_centers.shape[0]*particle_centers.shape[1],1),order='C'),Ms,chi,particle_volume,l_e)
        mag_forces = magnetism.get_dip_dip_forces_normalized_32bit_v2(magnetic_moments,particle_centers*l_e,particle_radius,l_e)
        mag_forces *= np.float32(beta/particle_mass)
        for i, particle in enumerate(particles):
            accel[particle] += mag_forces[i]
    else:
        host_posns = None
    return accel, host_posns

def simulate_scaled_gpu_leapfrog_v2(posns,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation of a hybrid mass spring system using a leapfrog numerical integration. Node_posns is an N_vertices by 3 cupy array of the positions of the vertices, elements is an N_elements by 8 cupy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    hard_limit_max_integrations = 2*max_integrations
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    velocities = cp.zeros(posns.shape,dtype=cp.float32)
    N_nodes = int(posns.shape[0]/3)
    max_displacement = np.zeros((hard_limit_max_integrations,))
    mean_displacement = np.zeros((hard_limit_max_integrations,))
    return_status = 1
    last_posns = cp.asnumpy(posns)
    last_posns = np.reshape(last_posns,(N_nodes,3))
    last_velocity = np.zeros(last_posns.shape,dtype=np.float32)
    #first do the acceleration calculation and the first update step (the initialization step), after which point all updates will be leapfrog updates
    host_beta_i = cp.asnumpy(beta_i)
    host_accel, host_posns = get_accel_scaled_GPU_v4(posns,velocities,elements,springs,particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
    a_var = cp.array(host_accel.astype(np.float32)).reshape((host_accel.shape[0]*host_accel.shape[1],1),order='C')
    size_entries = int(N_nodes*3)
    leapfrog_update(velocities,a_var,np.float32(step_size/2),size_entries)
    #exploring issues with leapfrog. stepsize wasn't originally float32, and the division by two reset step_size to a float64 even if step_size was float32, resulting in step sizes being way off from desired values.
    # host_velocities = cp.asnumpy(velocities)
    # host_velocities = np.reshape(host_velocities,(N_nodes,3))
    # print(f'particle velocities: {host_velocities[particles[0]]}\n{host_velocities[particles[1]]}')
    # particle_center = np.zeros((2,3))
    rerun_flag = False
    snapshot_stepsize = 200
    max_snapshot_count_value = int(1 + hard_limit_max_integrations*int(np.ceil(max_integration_steps/snapshot_stepsize)))
    snapshot_count = 0
    particle_snapshot_cutoff = 8
    particle_snapshot_flag = (particles.shape[0] != 0) and (particles.shape[0] < particle_snapshot_cutoff)
    if particle_snapshot_flag:
        particle_center = np.zeros((particles.shape[0],3),dtype=np.float32)
        snapshot_particle_posn = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_velocity = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_accel = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_separation = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_accel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    snapshot_vel_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    previous_soln = np.zeros((N_nodes,3),dtype=np.float32)
    snapshot_soln_diff_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_boundary_posn = np.zeros(snapshot_accel_components_avg.shape,dtype=np.float32)
    if 'stress' in boundary_conditions[0]:
        if boundary_conditions[1][0] == 'x':
            stressed_nodes = boundaries['right']
        elif boundary_conditions[1][0] == 'y':
            stressed_nodes = boundaries['back']
        elif boundary_conditions[1][0] == 'z':
            stressed_nodes = boundaries['top']
        elif boundary_conditions[1][0] == 'free':
            stressed_nodes = np.array([],dtype=np.int64)
    elif 'strain' in boundary_conditions[0]:
        stressed_nodes = np.array([],dtype=np.int64)
    i = 0
    previously_converged = False
    while i < max_integrations:
        print(f'starting integration run {i+1}')
        for j in range(max_integration_steps):
            leapfrog_update(posns,velocities,step_size,size_entries)
            if np.mod(j+i*max_integration_steps,snapshot_stepsize) == 0:
                host_posns = cp.asnumpy(posns)
                host_posns = np.reshape(host_posns,(N_nodes,3))
                
                if stressed_nodes.shape[0] !=0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[stressed_nodes],axis=0)

                host_accel_norms = np.linalg.norm(host_accel,axis=1)
                snapshot_accel_norm[snapshot_count] = np.mean(host_accel_norms)
                snapshot_accel_norm_std[snapshot_count] = np.std(host_accel_norms)
                snapshot_accel_components_avg[snapshot_count] = np.mean(host_accel,axis=0)

                host_velocities = cp.asnumpy(velocities)
                host_velocities = np.reshape(host_velocities,(N_nodes,3))
                host_vel_norms = np.linalg.norm(host_velocities,axis=1)
                snapshot_vel_norm[snapshot_count] = np.mean(host_vel_norms)
                snapshot_vel_norm_std[snapshot_count] = np.std(host_vel_norms)
                snapshot_vel_components_avg[snapshot_count] = np.mean(host_velocities,axis=0)

                if particles.shape[0] != 0 and particles.shape[0] < particle_snapshot_cutoff:
                    for particle_count, particle in enumerate(particles):
                        particle_center[particle_count,:] = get_particle_center(particle,host_posns)
                        snapshot_particle_posn[particle_count,snapshot_count,:] = particle_center[particle_count,:]*l_e
                        snapshot_particle_velocity[particle_count,snapshot_count,:] = np.sum(host_velocities[particles[particle_count],:],axis=0)/particles[0].shape
                        snapshot_particle_accel[particle_count,snapshot_count,:] = np.sum(host_accel[particles[particle_count],:],axis=0)/particles[0].shape
                    snapshot_particle_separation[snapshot_count] = np.linalg.norm(particle_center[0]-particle_center[1])*l_e

                if snapshot_count > 0:
                    soln_diff = host_posns - previous_soln
                    snapshot_soln_diff_norm[snapshot_count-1] = np.linalg.norm(np.ravel(soln_diff))
                previous_soln = host_posns
                snapshot_count += 1
            host_accel, host_posns = get_accel_scaled_GPU_v4(posns,velocities,elements,springs,particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
            a_var = cp.array(host_accel.astype(np.float32)).reshape((host_accel.shape[0]*host_accel.shape[1],1),order='C')
            leapfrog_update(velocities,a_var,step_size,size_entries)
        host_posns = cp.asnumpy(posns)
        # host_posns = np.reshape(host_posns,(N_nodes,3))
        # host_posns = np.reshape(host_posns,(3*N_nodes,1))
        host_velocities = cp.asnumpy(velocities)
        sol = np.concatenate((host_posns,host_velocities))
        host_accel = cp.asnumpy(a_var)
        host_accel = np.reshape(host_accel,(N_nodes,3))
        #2024-04-04 trying out a thing where, despite not doing the vector summation and distribution of the forces acting on the particles for the node updates, I do it for determining the accelerations for convergence criteria, thereby ignoring internal forces for the particles
        for particle in particles:
            host_accel[particle] = np.sum(host_accel[particle],axis=0)/particle.shape[0]
        accel_comp_magnitude = np.abs(host_accel)
        accel_comp_magnitude_avg = np.mean(accel_comp_magnitude)
        vel_comp_magnitude = np.abs(host_velocities)
        vel_comp_magnitude_avg = np.mean(vel_comp_magnitude)
        a_norms = np.linalg.norm(host_accel,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        host_posns = np.reshape(host_posns,(N_nodes,3))
        final_posns = host_posns
        host_velocities = np.reshape(host_velocities,(N_nodes,3))
        final_v = host_velocities
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if stressed_nodes.shape[0] != 0:
            boundary_accel_comp_magnitude = accel_comp_magnitude[stressed_nodes]
            boundary_accel_comp_magnitude_avg = np.mean(boundary_accel_comp_magnitude)
            boundary_vel_comp_magnitude = vel_comp_magnitude[stressed_nodes]
            boundary_vel_comp_magnitude_avg = np.mean(boundary_vel_comp_magnitude)
        else:
            boundary_accel_comp_magnitude_avg = 0
            boundary_vel_comp_magnitude_avg = 0            
        if accel_comp_magnitude_avg < tolerance and vel_comp_magnitude_avg < tolerance and boundary_accel_comp_magnitude_avg < tolerance and boundary_vel_comp_magnitude_avg < tolerance:#a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(accel_comp_magnitude_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(vel_comp_magnitude_avg,decimals=6)}')
            if stressed_nodes.shape[0] != 0:
                print(f'Reached convergence criteria for stressed boundary of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}')
                print(f'Reached convergence criteria for stressed boundary of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if previously_converged:
                return_status = 0
                print('Ending integration after reaching convergence criteria')
                break
            else:
                previously_converged = True
        elif np.any(np.isnan(a_norms)):
            print(f'Integration failure mode: NaN entries in accelerations. (possible overflow error)')
            return_status = -2
            break
            # if rerun_flag:
            #     print(f'halving step size and restarting still resulted in integration failure')
            #     return_status = -2
            #     break
            # step_size = np.float32(0.5*step_size)
            # max_integration_steps = int(2*max_integration_steps)
            # posns = cp.array(last_posns.astype(np.float32)).reshape((last_posns.shape[0]*last_posns.shape[1],1),order='C')
            # velocities = cp.array(last_velocity.astype(np.float32)).reshape((last_velocity.shape[0]*last_velocity.shape[1],1),order='C')
            # i -= 1
            # snapshot_count -= (int(np.round(max_integration_steps/2/snapshot_stepsize)) + 1)
            # snapshot_stepsize *= 2
            # rerun_flag = True
        else:
            print(f'Post-Integration norms\nacceleration norm average = {np.round(a_norm_avg,decimals=6)}\nvelocity norm average = {np.round(v_norm_avg,decimals=6)}')
            print(f'Post-Integration component magnitudes\nacceleration component magnitude average = {np.round(accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(vel_comp_magnitude_avg,decimals=6)}')
            if stressed_nodes.shape[0] != 0:
                print(f'Post-Integration component magnitudes on stressed boundary\nacceleration component magnitude average = {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}')
            # print(f'last snapshot values of norms\n acceleration norm: {snapshot_accel_norm[snapshot_count-1]}\n velocity norm: {snapshot_vel_norm[snapshot_count-1]}')
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            last_posns = final_posns.copy()
            last_velocity = final_v.copy()
            if rerun_flag:
                rerun_flag = False
                step_size = np.float32(2*step_size)
                max_integration_steps = int(max_integration_steps/2)
                snapshot_stepsize *= 0.5
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
        if particles.shape[0] != 0:
            particle_velocity = np.zeros((particles.shape[0],3))
            for particle_counter, particle in enumerate(particles):
                particle_velocity[particle_counter] = np.sum(final_v[particle,:],axis=0)/particle.shape[0]#np.linalg.norm(np.sum(final_v[particles[0],:],axis=0)/particles[0].shape[0])
            print(f'particle velocity = {np.round(particle_velocity,decimals=6)}')
        i += 1
        if i == max_integrations and particles.shape[0] != 0:
            if np.any(np.abs(particle_velocity[0]) > tolerance):#if the particles are still in motion, allow the integration to continue
                max_integrations += 1
                if max_integrations > hard_limit_max_integrations:
                    break
    plot_displacement_v_integration(i,mean_displacement,max_displacement,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    if particle_snapshot_flag:
        plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_separation*1e6,output_dir,tag="particle_separation(um)")
        for particle_count in range(particles.shape[0]):
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[particle_count]*1e6,output_dir,tag=f"particle_{particle_count+1}_posn(um)")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[particle_count],output_dir,tag=f"particle_{particle_count+1}_velocity")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[particle_count],output_dir,tag=f"particle_{particle_count+1}_accel")
        if particles.shape[0] == 2:
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[0]*1e6,snapshot_particle_posn[1]*1e6,output_dir,tag=f"particle_posns(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[0],snapshot_particle_velocity[1],output_dir,tag=f"particle_velocities(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[0],snapshot_particle_accel[1],output_dir,tag=f"particle_accelerations(um)")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_components_avg,output_dir,tag="acceleration component averages")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_components_avg,output_dir,tag="velocity component averages")
    if stressed_nodes.shape[0] !=0:
        plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_boundary_posn,output_dir,tag="boundary average position")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm,output_dir,tag="acceleration norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm_std,output_dir,tag="acceleration norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm,output_dir,tag="velocity norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm_std,output_dir,tag="velocity norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count-1,snapshot_soln_diff_norm,output_dir,tag="position solution vector difference norm")
    
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def simulate_scaled_gpu_leapfrog(posns,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_mass,chi,Ms,drag,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation of a hybrid mass spring system using a leapfrog numerical integration. Node_posns is an N_vertices by 3 cupy array of the positions of the vertices, elements is an N_elements by 8 cupy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined."""
    #function to be called at every sucessful integration step to get the solution output
    solutions = []
    def solout(t,y):
        solutions.append([t,*y])
    hard_limit_max_integrations = 2*max_integrations
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    velocities = cp.zeros(posns.shape,dtype=cp.float32)
    N_nodes = int(posns.shape[0]/3)
    if particles.shape[0] != 0:
        particle_moment_of_inertia = np.float32((2/5)*particle_mass*np.power(particle_radius,2))
        #needs to be scaled, but the scaling here includes (inside of beta) the characteristic time, squared, which is necessary for scaling the angular acceleration. the angular acceleration is scaling handled internally in the function get_accel_scaled_rotation(), but we need to account for the term to scale the moment of inertia properly, so we remove the time scaling that was previously involved here. the particle mass and the number of nodes making up the particle is used to go from beta to beta_i
        # characteristic_time_squared = beta*l_e
        # scaled_moment_of_inertia = particle_moment_of_inertia*beta/(particle_mass/particles.shape[1])/l_e/characteristic_time_squared
        # using the fact that we have an analytical expression, I will rewrite the above initialization of the scaled moment of inertia in a way that uses less operations
        scaled_moment_of_inertia = np.float32(particle_moment_of_inertia/(particle_mass/particles.shape[1])/(np.power(l_e,2)))
        particle_angular_velocity = np.zeros((particles.shape[0],3),dtype=np.float32)
    else:
        scaled_moment_of_inertia = None
    max_displacement = np.zeros((hard_limit_max_integrations,))
    mean_displacement = np.zeros((hard_limit_max_integrations,))
    return_status = 1
    last_posns = cp.asnumpy(posns)
    last_posns = np.reshape(last_posns,(N_nodes,3))
    last_velocity = np.zeros(last_posns.shape,dtype=np.float32)
    #first do the acceleration calculation and the first update step (the initialization step), after which point all updates will be leapfrog updates
    host_beta_i = cp.asnumpy(beta_i)
    host_accel, particle_torques, host_posns, particle_centers = get_accel_scaled_GPU_v3(posns,velocities,elements,springs,particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
    #trying to check for deformations of the particle, which should not occur, but are currently happening in cases where the particles are close together, and more obviously for the case of smaller time step sizes, as well as stronger applied stress values (compression). this first calculation is to have a baseline to compare against as the system evolves
    relative_to_particle_center_vectors = np.zeros((particles.shape[0],particles.shape[1],3))
    particle_orientations = np.zeros((particles.shape[0],3))
    particle_orientations[:,0] = 1
    particle_initial_orientations = particle_orientations.copy()
    # current_relative_to_particle_center_vectors = np.zeros((particles.shape[0],particles.shape[1],3))
    for particle_counter, particle in enumerate(particles):
        relative_to_particle_center_vectors[particle_counter] = host_posns[particle] - particle_centers[particle_counter]
    # print(f'particle accelerations: {a_var[particles[0]]}\n{a_var[particles[1]]}')
    a_var = cp.array(host_accel.astype(np.float32)).reshape((host_accel.shape[0]*host_accel.shape[1],1),order='C')
    size_entries = int(N_nodes*3)
    leapfrog_update(velocities,a_var,np.float32(step_size/2),size_entries)
    if particles.shape[0] != 0:
        particle_angular_acceleration = particle_torques/scaled_moment_of_inertia
        particle_angular_velocity += particle_angular_acceleration*np.float32(step_size/2)
    #exploring issues with leapfrog. stepsize wasn't originally float32, and the division by two reset step_size to a float64 even if step_size was float32, resulting in step sizes being way off from desired values.
    # host_velocities = cp.asnumpy(velocities)
    # host_velocities = np.reshape(host_velocities,(N_nodes,3))
    # print(f'particle velocities: {host_velocities[particles[0]]}\n{host_velocities[particles[1]]}')
    # particle_center = np.zeros((2,3))
    rerun_flag = False
    snapshot_stepsize = 200
    max_snapshot_count_value = int(1 + hard_limit_max_integrations*int(np.ceil(max_integration_steps/snapshot_stepsize)))
    snapshot_count = 0
    if particles.shape[0] != 0:
        particle_center = np.zeros((particles.shape[0],3),dtype=np.float32)
        snapshot_particle_posn = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_velocity = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_accel = np.zeros((particles.shape[0],max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_separation = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_accel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    snapshot_vel_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    previous_soln = np.zeros((N_nodes,3),dtype=np.float32)
    snapshot_soln_diff_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_boundary_posn = np.zeros(snapshot_accel_components_avg.shape,dtype=np.float32)
    if boundary_conditions[1][0] == 'x':
        # fixed_nodes = boundaries['left']
        stressed_nodes = boundaries['right']
    elif boundary_conditions[1][0] == 'y':
        # fixed_nodes = boundaries['front']
        stressed_nodes = boundaries['back']
    elif boundary_conditions[1][0] == 'z':
        # fixed_nodes = boundaries['bot']
        stressed_nodes = boundaries['top']
    elif boundary_conditions[1][0] == 'free':
        stressed_nodes = np.array([],dtype=np.int64)
    i = 0
    previously_converged = False
    while i < max_integrations:
        print(f'starting integration run {i+1}')
        for j in range(max_integration_steps):
            #rotate particles, update the posns variable on the gpu
            if particles.shape[0] != 0:
                if not np.allclose(np.linalg.norm(particle_angular_velocity,axis=1),0):
                    for particle_counter, particle in enumerate(particles):
                        if not np.isclose(np.linalg.norm(particle_angular_velocity[particle_counter]),0):
                            axis_of_rotation = particle_angular_velocity[particle_counter]/np.linalg.norm(particle_angular_velocity[particle_counter])
                            rotation_angle = np.linalg.norm(particle_angular_velocity)*step_size
                            # print(f'rotation angle: {rotation_angle[particle_counter]*180/np.pi}')
                            # print(f'axis of rotation: {axis_of_rotation[particle_counter]}')
                            rotation_matrix_generator = R.from_rotvec(rotation_angle*axis_of_rotation)
                            # rotation_matrix = rotation_matrix_generator.as_matrix()
                            # the positions need to be translated so that the coordinate system sits at the center of the particle, then they can be rotated, before transforming/translating back to the original coordinate system
                            host_posns[particle] = rotation_matrix_generator.apply(host_posns[particle]-particle_centers[particle_counter]) + particle_centers[particle_counter]
                            particle_orientations[particle_counter] = rotation_matrix_generator.apply(particle_orientations[particle_counter])
                    posns = cp.array(host_posns.astype(np.float32)).reshape((host_posns.shape[0]*host_posns.shape[1],1),order='C')
            leapfrog_update(posns,velocities,step_size,size_entries)
            # whole_step_velocities = cp.copy(velocities)
            # leapfrog_update(whole_step_velocities,a_var,np.float32(step_size/2),size_entries)
            if np.mod(j+i*max_integration_steps,snapshot_stepsize) == 0:
                host_posns = cp.asnumpy(posns)
                host_posns = np.reshape(host_posns,(N_nodes,3))
                
                snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[stressed_nodes],axis=0)

                host_accel_norms = np.linalg.norm(host_accel,axis=1)
                snapshot_accel_norm[snapshot_count] = np.mean(host_accel_norms)
                snapshot_accel_norm_std[snapshot_count] = np.std(host_accel_norms)
                snapshot_accel_components_avg[snapshot_count] = np.mean(host_accel,axis=0)

                host_velocities = cp.asnumpy(velocities)
                host_velocities = np.reshape(host_velocities,(N_nodes,3))
                host_vel_norms = np.linalg.norm(host_velocities,axis=1)
                snapshot_vel_norm[snapshot_count] = np.mean(host_vel_norms)
                snapshot_vel_norm_std[snapshot_count] = np.std(host_vel_norms)
                snapshot_vel_components_avg[snapshot_count] = np.mean(host_velocities,axis=0)

                if particles.shape[0] != 0:
                    for particle_count, particle in enumerate(particles):
                        particle_center[particle_count,:] = get_particle_center(particle,host_posns)
                        snapshot_particle_posn[particle_count,snapshot_count,:] = particle_center[particle_count,:]*l_e
                        snapshot_particle_velocity[particle_count,snapshot_count,:] = np.sum(host_velocities[particles[particle_count],:],axis=0)/particles[0].shape
                        snapshot_particle_accel[particle_count,snapshot_count,:] = np.sum(host_accel[particles[particle_count],:],axis=0)/particles[0].shape
                        # current_relative_to_particle_center_vectors[particle_count] = host_posns[particle] - particle_center[particle_count]
                    # if not np.allclose(relative_to_particle_center_vectors,current_relative_to_particle_center_vectors):
                    #     print('relative position vectors for one or more particles do not match expectations for rigid body')
                    #     print(f'attempting to calculate the l-2 norm of the difference of the current relative vectors to the expected relative vectors\n')
                    #     relative_vector_difference_norms = np.linalg.norm(current_relative_to_particle_center_vectors-relative_to_particle_center_vectors,axis=2)
                    #     print(relative_vector_difference_norms.shape)
                    #     print(relative_vector_difference_norms)
                    #     for particle_count, particle in enumerate(particles):
                    #         print(f'particle {particle_count}')
                    #         for node in particle:
                    #             node_velocity = host_velocities[node]
                    #             print(node_velocity)
                    snapshot_particle_separation[snapshot_count] = np.linalg.norm(particle_center[0]-particle_center[1])*l_e

                if snapshot_count > 0:
                    soln_diff = host_posns - previous_soln
                    snapshot_soln_diff_norm[snapshot_count-1] = np.linalg.norm(np.ravel(soln_diff))
                previous_soln = host_posns
                snapshot_count += 1
            # if particle_separation*l_e < 5e-6:
            #     print(f'particle separation = {particle_separation*l_e*1e6} um')
            host_accel, particle_torques, host_posns, particle_centers = get_accel_scaled_GPU_v3(posns,velocities,elements,springs,particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
            a_var = cp.array(host_accel.astype(np.float32)).reshape((host_accel.shape[0]*host_accel.shape[1],1),order='C')
            leapfrog_update(velocities,a_var,step_size,size_entries)
            if particles.shape[0] != 0:
                particle_angular_acceleration = particle_torques/scaled_moment_of_inertia - drag*particle_angular_velocity
                particle_angular_velocity += particle_angular_acceleration*step_size
            # host_velocities = cp.asnumpy(velocities)
            # host_velocities = np.reshape(host_velocities,(N_nodes,3))
            # particle_velocity = host_velocities[particles[0][0],0]
            # if np.abs(particle_velocity)*step_size > 0.1:
            #     print(f'particle velocity and step size result in large change in position')
            # print(f'particle velocities : {host_velocities[particles[0]]}\n{host_velocities[particles[1]]}')
        #!!! do i need this acceleration calculation below?
        # a_var = get_accel_scaled_GPU_v2(posns,velocities,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,scaled_moment_of_inertia,chi,Ms,drag)
        host_posns = cp.asnumpy(posns)
        host_posns = np.reshape(host_posns,(N_nodes,3))
        #after each integration round, use the particle_centers, particle_orientations, and relative_to_particle_center_vectors variables to "reset" the particle nodes to match the rigid body description, while maintaining the position of the center of the particle and the orientation. it is assumed that accumulation of floating point errors leads to deformation of the particle, as well as an additional unexplained behavior where some subset of the nodes experiences displacements not matching the rigid body motion expected. the second case occurs when the particles are clustered closely together, and has been seen in the case of a compressive force on the system boundary that would tend to push the particles closer together. this reset should prevent the particle from deforming while not significantly modifying any node positions
        for particle_counter, particle in enumerate(particles):
            # we use scipy's Rotation class to find the rotation that aligns a vector describing the initial particle orientation (which will always be [1,0,0]) with the current orientation. we then use that rotation to rotate the relative to particle center vectors, and add the coordinates of the particle center to those vectors
            if not np.allclose(particle_orientations[particle_counter],particle_initial_orientations[particle_counter]):
                rigid_body_orientation_transformation, _ = R.align_vectors(particle_orientations[particle_counter][np.newaxis,:],particle_initial_orientations[particle_counter][np.newaxis,:])
                host_posns[particle] = rigid_body_orientation_transformation.apply(relative_to_particle_center_vectors[particle_counter]) + particle_centers[particle_counter]
        #after reseting the the particle nodes to match a rigid body
        host_posns = np.reshape(host_posns,(3*N_nodes,1))
        posns = cp.array(host_posns.astype(np.float32))#.reshape((host_posns.shape[0]*host_posns.shape[1],1),order='C')
        host_velocities = cp.asnumpy(velocities)
        sol = np.concatenate((host_posns,host_velocities))
        host_accel = cp.asnumpy(a_var)
        host_accel = np.reshape(host_accel,(N_nodes,3))
        accel_comp_magnitude = np.abs(host_accel)
        accel_comp_magnitude_avg = np.mean(accel_comp_magnitude)
        vel_comp_magnitude = np.abs(host_velocities)
        vel_comp_magnitude_avg = np.mean(vel_comp_magnitude)
        a_norms = np.linalg.norm(host_accel,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        host_posns = np.reshape(host_posns,(N_nodes,3))
        final_posns = host_posns
        host_velocities = np.reshape(host_velocities,(N_nodes,3))
        final_v = host_velocities
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if accel_comp_magnitude_avg < tolerance and vel_comp_magnitude_avg < tolerance:#a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(accel_comp_magnitude_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(vel_comp_magnitude_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if previously_converged:
                return_status = 0
                print('Ending integration after reaching convergence criteria')
                break
            else:
                previously_converged = True
        elif np.any(np.isnan(a_norms)):
            print(f'Integration failure mode: NaN entries in accelerations. (possible overflow error)')
            return_status = -2
            break
            # if rerun_flag:
            #     print(f'halving step size and restarting still resulted in integration failure')
            #     return_status = -2
            #     break
            # step_size = np.float32(0.5*step_size)
            # max_integration_steps = int(2*max_integration_steps)
            # posns = cp.array(last_posns.astype(np.float32)).reshape((last_posns.shape[0]*last_posns.shape[1],1),order='C')
            # velocities = cp.array(last_velocity.astype(np.float32)).reshape((last_velocity.shape[0]*last_velocity.shape[1],1),order='C')
            # i -= 1
            # snapshot_count -= (int(np.round(max_integration_steps/2/snapshot_stepsize)) + 1)
            # snapshot_stepsize *= 2
            # rerun_flag = True
        else:
            print(f'Post-Integration norms\nacceleration norm average = {np.round(a_norm_avg,decimals=6)}\nvelocity norm average = {np.round(v_norm_avg,decimals=6)}')
            print(f'Post-Integration component magnitudes\nacceleration component magnitude average = {np.round(accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(vel_comp_magnitude_avg,decimals=6)}')
            # print(f'last snapshot values of norms\n acceleration norm: {snapshot_accel_norm[snapshot_count-1]}\n velocity norm: {snapshot_vel_norm[snapshot_count-1]}')
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            last_posns = final_posns.copy()
            last_velocity = final_v.copy()
            if rerun_flag:
                rerun_flag = False
                step_size = np.float32(2*step_size)
                max_integration_steps = int(max_integration_steps/2)
                snapshot_stepsize *= 0.5
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
        if particles.shape[0] != 0:
            particle_velocity = np.zeros((particles.shape[0],3))
            for particle_counter, particle in enumerate(particles):
                particle_velocity[particle_counter] = np.sum(final_v[particle,:],axis=0)/particle.shape[0]#np.linalg.norm(np.sum(final_v[particles[0],:],axis=0)/particles[0].shape[0])
            print(f'particle velocity = {np.round(particle_velocity,decimals=6)}')
        i += 1
        if i == max_integrations and particles.shape[0] != 0:
            if np.any(np.abs(particle_velocity[0]) > tolerance):#if the particles are still in motion, allow the integration to continue
                max_integrations += 1
                if max_integrations > hard_limit_max_integrations:
                    break
    plot_displacement_v_integration(i,mean_displacement,max_displacement,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    if particles.shape[0] != 0:
        plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_separation*1e6,output_dir,tag="particle_separation(um)")
        for particle_count in range(particles.shape[0]):
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[particle_count]*1e6,output_dir,tag=f"particle_{particle_count+1}_posn(um)")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[particle_count],output_dir,tag=f"particle_{particle_count+1}_velocity")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[particle_count],output_dir,tag=f"particle_{particle_count+1}_accel")
        if particles.shape[0] == 2:
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[0]*1e6,snapshot_particle_posn[1]*1e6,output_dir,tag=f"particle_posns(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[0],snapshot_particle_velocity[1],output_dir,tag=f"particle_velocities(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[0],snapshot_particle_accel[1],output_dir,tag=f"particle_accelerations(um)")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_components_avg,output_dir,tag="acceleration component averages")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_components_avg,output_dir,tag="velocity component averages")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_boundary_posn,output_dir,tag="boundary average position")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm,output_dir,tag="acceleration norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm_std,output_dir,tag="acceleration norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm,output_dir,tag="velocity norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm_std,output_dir,tag="velocity norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count-1,snapshot_soln_diff_norm,output_dir,tag="position solution vector difference norm")
    
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def simulate_scaled_gpu_leapfrog_v3(posns,elements,host_particles,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False,starting_velocities=None,checkpoint_offset=0,sim_extend_flag=False):
    """Run a simulation of a hybrid mass spring system using a leapfrog numerical integration. Node_posns is an N_vertices by 3 cupy array of the positions of the vertices, elements is an N_elements by 8 cupy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined."""
    #function to be called at every sucessful integration step to get the solution output
    hard_limit_max_integrations = 2*max_integrations
    if max_integrations < 4:
        minimum_integration_rounds = max_integrations
    else:
        minimum_integration_rounds = 4
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    if type(starting_velocities) != type(None):
        velocities = starting_velocities
    else:
        velocities = cp.zeros(posns.shape,dtype=cp.float32)
    N_nodes = int(posns.shape[0]/3)
    max_displacement = np.zeros((hard_limit_max_integrations,))
    mean_displacement = np.zeros((hard_limit_max_integrations,))
    return_status = 1
    last_posns = cp.asnumpy(posns)
    last_posns = np.reshape(last_posns,(N_nodes,3))
    #first do the acceleration calculation and the first update step (the initialization step), after which point all updates will be leapfrog updates
    #get the fixed nodes, stressed nodes, and the (unscaled) force per node for the stress that is applied
    if 'simple_stress' in boundary_conditions[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if boundary_conditions[1][0] == 'x':
            fixed_nodes = cp.asarray(boundaries['left'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['right'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif boundary_conditions[1][0] == 'y':
            fixed_nodes = cp.asarray(boundaries['front'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['back'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif boundary_conditions[1][0] == 'z':
            fixed_nodes = cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['top'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        stress_direction_char = boundary_conditions[1][1]
        if stress_direction_char == 'x':
            stress_direction = 0
        elif stress_direction_char == 'y':
            stress_direction = 1
        elif stress_direction_char == 'z':
            stress_direction = 2
        stress = boundary_conditions[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        stress_node_force = np.squeeze(net_force_mag/stressed_nodes.shape[0])
        moving_boundary_nodes = cp.array([],dtype=np.int32)
        host_moving_boundary_nodes = np.array([],dtype=np.int64)
    elif 'strain' in boundary_conditions[0]:
        if boundary_conditions[2] == 0:
            if boundary_conditions[1][0] == 'x':
                fixed_nodes = cp.asarray(boundaries['left'],dtype=cp.int32,order='C')
                host_moving_boundary_nodes = boundaries['right']
            elif boundary_conditions[1][0] == 'y':
                fixed_nodes = cp.asarray(boundaries['front'],dtype=cp.int32,order='C')
                host_moving_boundary_nodes = boundaries['back']
            elif boundary_conditions[1][0] == 'z':
                fixed_nodes = cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')
                host_moving_boundary_nodes = boundaries['top']
            moving_boundary_nodes = cp.asarray(host_moving_boundary_nodes,dtype=cp.int32,order='C')
        else:
            if boundary_conditions[1][0] == 'x':
                fixed_nodes = cp.asarray(np.concatenate((boundaries['left'],boundaries['right'])),dtype=cp.int32,order='C')
            elif boundary_conditions[1][0] == 'y':
                fixed_nodes = cp.asarray(np.concatenate((boundaries['front'],boundaries['back'])),dtype=cp.int32,order='C')
            elif boundary_conditions[1][0] == 'z':
                fixed_nodes = cp.asarray(np.concatenate((boundaries['top'],boundaries['bot'])),dtype=cp.int32,order='C')
            moving_boundary_nodes = cp.array([],dtype=np.int32)
            host_moving_boundary_nodes = np.array([],dtype=np.int64)
        stressed_nodes = cp.array([],dtype=np.int32)
        host_stressed_nodes = np.array([],dtype=np.int64)
        stress_direction = 0
        stress_node_force = 0
    elif boundary_conditions[0] == 'hysteresis':
        fixed_nodes = cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')
        stressed_nodes = cp.array([],dtype=np.int32)
        host_stressed_nodes = np.array([],dtype=np.int64)
        moving_boundary_nodes = cp.array([],dtype=np.int32)
        host_moving_boundary_nodes = np.array([],dtype=np.int64)
        stress_direction = 0
        stress_node_force = 0
    num_particles = host_particles.shape[0]
    particle_posns = get_particle_posns(host_particles,last_posns)
    particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
    a_var = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary_nodes,host_moving_boundary_nodes)

    size_entries = int(N_nodes*3)
    leapfrog_update(velocities,a_var,np.float32(step_size/2),size_entries)
    snapshot_stepsize = 200
    max_snapshot_count_value = int(1 + hard_limit_max_integrations*int(np.ceil(max_integration_steps/snapshot_stepsize)))
    snapshot_count = 0
    particle_snapshot_cutoff = 8
    particle_snapshot_flag = (num_particles != 0) and (num_particles < particle_snapshot_cutoff)
    if particle_snapshot_flag:
        particle_center = np.zeros((num_particles,3),dtype=np.float32)
        snapshot_particle_posn = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_velocity = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_accel = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_separation = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_accel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    snapshot_vel_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    previous_soln = np.zeros((N_nodes,3),dtype=np.float32)
    snapshot_soln_diff_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_boundary_posn = np.zeros(snapshot_accel_components_avg.shape,dtype=np.float32)

    i = 0
    previously_converged = False
    while i < max_integrations:
        print(f'starting integration run {i+1}')
        for j in range(max_integration_steps):
            leapfrog_update(posns,velocities,step_size,size_entries)
            host_posns = cp.asnumpy(posns)
            host_posns = np.reshape(host_posns,(N_nodes,3))
            particle_posns = get_particle_posns(host_particles,host_posns)
            particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
            if np.mod(j+i*max_integration_steps,snapshot_stepsize) == 0:
                host_posns = cp.asnumpy(posns)
                host_posns = np.reshape(host_posns,(N_nodes,3))
                host_accel = cp.asnumpy(a_var)
                host_accel = np.reshape(host_accel,(N_nodes,3))
                
                if host_stressed_nodes.shape[0] !=0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[host_stressed_nodes],axis=0)
                elif host_moving_boundary_nodes.shape[0] != 0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[host_moving_boundary_nodes],axis=0)

                host_accel_norms = np.linalg.norm(host_accel,axis=1)
                snapshot_accel_norm[snapshot_count] = np.mean(host_accel_norms)
                snapshot_accel_norm_std[snapshot_count] = np.std(host_accel_norms)
                snapshot_accel_components_avg[snapshot_count] = np.mean(host_accel,axis=0)

                host_velocities = cp.asnumpy(velocities)
                host_velocities = np.reshape(host_velocities,(N_nodes,3))
                host_vel_norms = np.linalg.norm(host_velocities,axis=1)
                snapshot_vel_norm[snapshot_count] = np.mean(host_vel_norms)
                snapshot_vel_norm_std[snapshot_count] = np.std(host_vel_norms)
                snapshot_vel_components_avg[snapshot_count] = np.mean(host_velocities,axis=0)

                if num_particles != 0 and num_particles < particle_snapshot_cutoff:
                    for particle_count, particle in enumerate(host_particles):
                        particle_center[particle_count,:] = get_particle_center(particle,host_posns)
                        snapshot_particle_posn[particle_count,snapshot_count,:] = particle_center[particle_count,:]*l_e
                        snapshot_particle_velocity[particle_count,snapshot_count,:] = np.sum(host_velocities[host_particles[particle_count],:],axis=0)/host_particles[0].shape
                        snapshot_particle_accel[particle_count,snapshot_count,:] = np.sum(host_accel[host_particles[particle_count],:],axis=0)/host_particles[0].shape
                    snapshot_particle_separation[snapshot_count] = np.linalg.norm(particle_center[0]-particle_center[1])*l_e

                if snapshot_count > 0:
                    soln_diff = host_posns - previous_soln
                    snapshot_soln_diff_norm[snapshot_count-1] = np.linalg.norm(np.ravel(soln_diff))
                previous_soln = host_posns
                snapshot_count += 1
            a_var = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary_nodes,host_moving_boundary_nodes)
            leapfrog_update(velocities,a_var,step_size,size_entries)
        host_posns = cp.asnumpy(posns)
        host_velocities = cp.asnumpy(velocities)
        sol = np.concatenate((host_posns,host_velocities))
        host_accel = cp.asnumpy(a_var)
        host_accel = np.reshape(host_accel,(N_nodes,3))
        host_velocities = np.reshape(host_velocities,(N_nodes,3))
        #2024-04-04 trying out a thing where, despite not doing the vector summation and distribution of the forces acting on the particles for the node updates, I do it for determining the accelerations for convergence criteria, thereby ignoring internal forces for the particles
        for particle in host_particles:
            host_accel[particle] = np.sum(host_accel[particle],axis=0)/particle.shape[0]
            host_velocities[particle] = np.sum(host_velocities[particle],axis=0)/particle.shape[0]
        accel_comp_magnitude = np.abs(host_accel)
        accel_comp_magnitude_avg = np.mean(accel_comp_magnitude)
        vel_comp_magnitude = np.abs(host_velocities)
        vel_comp_magnitude_avg = np.mean(vel_comp_magnitude)
        a_norms = np.linalg.norm(host_accel,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        host_posns = np.reshape(host_posns,(N_nodes,3))
        final_posns = host_posns
        
        final_v = host_velocities
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if host_stressed_nodes.shape[0] != 0:
            boundary_accel_comp_magnitude = accel_comp_magnitude[host_stressed_nodes]
            boundary_accel_comp_magnitude_avg = np.mean(boundary_accel_comp_magnitude)
            boundary_vel_comp_magnitude = vel_comp_magnitude[host_stressed_nodes]
            boundary_vel_comp_magnitude_avg = np.mean(boundary_vel_comp_magnitude)
        elif host_moving_boundary_nodes.shape[0] != 0:
            boundary_accel_comp_magnitude = accel_comp_magnitude[host_moving_boundary_nodes]
            boundary_accel_comp_magnitude_avg = np.mean(boundary_accel_comp_magnitude)
            boundary_vel_comp_magnitude = vel_comp_magnitude[host_moving_boundary_nodes]
            boundary_vel_comp_magnitude_avg = np.mean(boundary_vel_comp_magnitude)
        else:
            boundary_accel_comp_magnitude_avg = 0
            boundary_vel_comp_magnitude_avg = 0            
        if accel_comp_magnitude_avg < tolerance and vel_comp_magnitude_avg < tolerance and boundary_accel_comp_magnitude_avg < tolerance and boundary_vel_comp_magnitude_avg < tolerance:#a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(accel_comp_magnitude_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(vel_comp_magnitude_avg,decimals=6)}\n')
            if host_stressed_nodes.shape[0] != 0 or host_moving_boundary_nodes.shape[0] != 0:
                print(f'Reached convergence criteria for stressed boundary of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}')
                print(f'Reached convergence criteria for stressed boundary of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}\n')
            # print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            #2024-05-20 additional checks: if the field is low (1 Gauss, or 0 mT) and the boundary condition value is zero, the simulation can end early. Otherwise, a minimum number of integration rounds should have occurred. The solution vector difference norm should also be decreasing
            print(f'snapshot solution vector difference norm and most recent change in value:{snapshot_soln_diff_norm[snapshot_count-2]}, {snapshot_soln_diff_norm[snapshot_count-2] - snapshot_soln_diff_norm[snapshot_count-3]}')
            if (previously_converged and (i >= minimum_integration_rounds-1) and (snapshot_soln_diff_norm[snapshot_count-2] - snapshot_soln_diff_norm[snapshot_count-3]) <= 0) or ((np.isclose(np.linalg.norm(Hext*mu0),1e-4) or np.isclose(np.linalg.norm(Hext*mu0),0)) and boundary_conditions[2] == 0):
                return_status = 0
                print('Ending integration after reaching convergence criteria')
                break
            else:
                previously_converged = True
        elif np.any(np.isnan(a_norms)):
            print(f'Integration failure mode: NaN entries in accelerations. (possible overflow error)')
            return_status = -2
            break
        else:
            previously_converged = False
            print(f'Post-Integration norms\nacceleration norm average = {np.round(a_norm_avg,decimals=6)}\nvelocity norm average = {np.round(v_norm_avg,decimals=6)}')
            print(f'Post-Integration component magnitudes\nacceleration component magnitude average = {np.round(accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(vel_comp_magnitude_avg,decimals=6)}\n')
            if host_stressed_nodes.shape[0] != 0 or host_moving_boundary_nodes.shape[0] != 0:
                print(f'Post-Integration component magnitudes on stressed boundary\nacceleration component magnitude average = {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}\n')
            # print(f'last snapshot values of norms\n acceleration norm: {snapshot_accel_norm[snapshot_count-1]}\n velocity norm: {snapshot_vel_norm[snapshot_count-1]}')
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            last_posns = final_posns.copy()
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i+checkpoint_offset}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
        if num_particles != 0 and num_particles < 8:
            particle_velocity = np.zeros((num_particles,3))
            for particle_counter, particle in enumerate(host_particles):
                particle_velocity[particle_counter] = np.sum(final_v[particle,:],axis=0)/particle.shape[0]
            print(f'particle velocity = {np.round(particle_velocity,decimals=6)}')
        i += 1
        if i == max_integrations and num_particles != 0 and num_particles < 8:
            if np.any(np.abs(particle_velocity[0]) > tolerance):#if the particles are still in motion, allow the integration to continue
                max_integrations += 1
                if max_integrations > hard_limit_max_integrations:
                    break
    plot_displacement_v_integration(i,mean_displacement,max_displacement,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    if particle_snapshot_flag:
        plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_separation*1e6,output_dir,tag="particle_separation(um)")
        for particle_count in range(num_particles):
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[particle_count]*1e6,output_dir,tag=f"particle_{particle_count+1}_posn(um)")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[particle_count],output_dir,tag=f"particle_{particle_count+1}_velocity")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[particle_count],output_dir,tag=f"particle_{particle_count+1}_accel")
        if num_particles == 2:
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[0]*1e6,snapshot_particle_posn[1]*1e6,output_dir,tag=f"particle_posns(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[0],snapshot_particle_velocity[1],output_dir,tag=f"particle_velocities(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[0],snapshot_particle_accel[1],output_dir,tag=f"particle_accelerations(um)")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_components_avg,output_dir,tag="acceleration component averages")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_components_avg,output_dir,tag="velocity component averages")
    if host_stressed_nodes.shape[0] != 0 or host_moving_boundary_nodes.shape[0] != 0:
        plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_boundary_posn,output_dir,tag="boundary average position")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm,output_dir,tag="acceleration norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm_std,output_dir,tag="acceleration norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm,output_dir,tag="velocity norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm_std,output_dir,tag="velocity norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count-1,snapshot_soln_diff_norm,output_dir,tag="position solution vector difference norm")
    
    return sol, return_status#returning a solution object, that can then have it's attributes inspected

def simulate_scaled_gpu_leapfrog_test(posns,elements,host_particles,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False):
    """Run a simulation of a hybrid mass spring system using a leapfrog numerical integration. Node_posns is an N_vertices by 3 cupy array of the positions of the vertices, elements is an N_elements by 8 cupy array whose rows contain the row indices of the vertices(in node_posns) that define each cubic element. springs is an N_springs by 4 array, first two columns are the row indices in Node_posns of nodes connected by springs, 3rd column is spring stiffness in N/m, 4th column is equilibrium separation in (m). kappa is a scalar that defines the addditional bulk modulus of the material being simulated, which is calculated using get_kappa(). l_e is the side length of the cube used to discretize the system (this is a uniform structured mesh grid). boundary_conditions is a tuple where different types of boundary conditions (displacements or stresses/external forces/tractions) and the boundary they are applied to are defined."""
    #function to be called at every sucessful integration step to get the solution output
    hard_limit_max_integrations = 2*max_integrations
    #getting the parent directory. split the output directory string by the backslash delimiter, find the length of the child directory name (the last or second to last string in the list returned by output_dir.split('/')), and use that to get a substring for the parent directory
    tmp_var = output_dir.split('/')
    if tmp_var[-1] == '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-2])-1]
    elif tmp_var[-1] != '':
        checkpoint_output_dir = output_dir[:-1*len(tmp_var[-1])-1]
    velocities = cp.zeros(posns.shape,dtype=cp.float32)
    N_nodes = int(posns.shape[0]/3)
    max_displacement = np.zeros((hard_limit_max_integrations,))
    mean_displacement = np.zeros((hard_limit_max_integrations,))
    return_status = 1
    last_posns = cp.asnumpy(posns)
    last_posns = np.reshape(last_posns,(N_nodes,3))
    #first do the acceleration calculation and the first update step (the initialization step), after which point all updates will be leapfrog updates
    #get the fixed nodes, stressed nodes, and the (unscaled) force per node for the stress that is applied
    if 'simple_stress' in boundary_conditions[0]:
        #opposing surface to the probe surface needs to be held fixed, probe surface nodes need to have additional forces applied
        if boundary_conditions[1][0] == 'x':
            fixed_nodes = cp.asarray(boundaries['left'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['right'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['right']
            relevant_dimension_indices = [1,2]
        elif boundary_conditions[1][0] == 'y':
            fixed_nodes = cp.asarray(boundaries['front'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['back'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['back']
            relevant_dimension_indices = [0,2]
        elif boundary_conditions[1][0] == 'z':
            fixed_nodes = cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')
            stressed_nodes = cp.asarray(boundaries['top'],dtype=cp.int32,order='C')
            host_stressed_nodes = boundaries['top']
            relevant_dimension_indices = [0,1]
        stress_direction_char = boundary_conditions[1][1]
        if stress_direction_char == 'x':
            stress_direction = 0
        elif stress_direction_char == 'y':
            stress_direction = 1
        elif stress_direction_char == 'z':
            stress_direction = 2
        stress = boundary_conditions[2]
        surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
        net_force_mag = stress*surface_area
        stress_node_force = np.squeeze(net_force_mag/stressed_nodes.shape[0])
    elif 'strain' in boundary_conditions[0]:
        if boundary_conditions[1][0] == 'x':
            fixed_nodes = cp.asarray(np.concatenate((boundaries['left'],boundaries['right'])),dtype=cp.int32,order='C')
        elif boundary_conditions[1][0] == 'y':
            fixed_nodes = cp.asarray(np.concatenate((boundaries['front'],boundaries['back'])),dtype=cp.int32,order='C')
        elif boundary_conditions[1][0] == 'z':
            fixed_nodes = cp.asarray(np.concatenate((boundaries['top'],boundaries['bot'])),dtype=cp.int32,order='C')
        stressed_nodes = cp.array([],dtype=np.int32)
        host_stressed_nodes = np.array([],dtype=np.int64)
        stress_direction = 0
        stress_node_force = 0
    elif boundary_conditions[0] == 'hysteresis':
        fixed_nodes = boundaries['bot']
        stressed_nodes = cp.array([],dtype=np.int32)
        host_stressed_nodes = np.array([],dtype=np.int64)
        stress_direction = 0
        stress_node_force = 0
    num_particles = host_particles.shape[0]
    particle_posns = get_particle_posns(host_particles,last_posns)
    particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
    a_var = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
    host_beta_i = cp.asnumpy(beta_i)
    host_accel, host_posns = get_accel_scaled_GPU_test(posns,velocities,elements,springs,host_particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
    host_a_var = cp.asnumpy(a_var)
    host_a_var = host_a_var.reshape((N_nodes,3))
    correctness = np.allclose(host_a_var,host_accel)
    if not correctness:
        print('acceleration calculations do not agree')
        accel_result_difference = host_a_var - host_accel
        print(f'acceleration calculations difference norm {np.linalg.norm(accel_result_difference)}\naverage acceleration calculations difference norm {np.linalg.norm(accel_result_difference)/accel_result_difference.shape[0]*accel_result_difference.shape[1]}\nmax difference {np.max(np.abs(accel_result_difference))}')

    size_entries = int(N_nodes*3)
    leapfrog_update(velocities,a_var,np.float32(step_size/2),size_entries)
    snapshot_stepsize = 200
    max_snapshot_count_value = int(1 + hard_limit_max_integrations*int(np.ceil(max_integration_steps/snapshot_stepsize)))
    snapshot_count = 0
    particle_snapshot_cutoff = 8
    particle_snapshot_flag = (num_particles != 0) and (num_particles < particle_snapshot_cutoff)
    if particle_snapshot_flag:
        particle_center = np.zeros((num_particles,3),dtype=np.float32)
        snapshot_particle_posn = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_velocity = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_accel = np.zeros((num_particles,max_snapshot_count_value,3),dtype=np.float32)
        snapshot_particle_separation = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm = np.zeros((max_snapshot_count_value,),dtype=np.float32)
    snapshot_accel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_accel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    snapshot_vel_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_norm_std = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_vel_components_avg = np.zeros((max_snapshot_count_value,3),dtype=np.float32)
    previous_soln = np.zeros((N_nodes,3),dtype=np.float32)
    snapshot_soln_diff_norm = np.zeros(snapshot_accel_norm.shape,dtype=np.float32)
    snapshot_boundary_posn = np.zeros(snapshot_accel_components_avg.shape,dtype=np.float32)

    i = 0
    previously_converged = False
    while i < max_integrations:
        print(f'starting integration run {i+1}')
        for j in range(max_integration_steps):
            leapfrog_update(posns,velocities,step_size,size_entries)
            host_posns = cp.asnumpy(posns)
            host_posns = np.reshape(host_posns,(N_nodes,3))
            particle_posns = get_particle_posns(host_particles,host_posns)
            particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
            if np.mod(j+i*max_integration_steps,snapshot_stepsize) == 0:
                host_posns = cp.asnumpy(posns)
                host_posns = np.reshape(host_posns,(N_nodes,3))
                host_accel = cp.asnumpy(a_var)
                host_accel = np.reshape(host_accel,(N_nodes,3))
                
                if host_stressed_nodes.shape[0] !=0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[host_stressed_nodes],axis=0)

                host_accel_norms = np.linalg.norm(host_accel,axis=1)
                snapshot_accel_norm[snapshot_count] = np.mean(host_accel_norms)
                snapshot_accel_norm_std[snapshot_count] = np.std(host_accel_norms)
                snapshot_accel_components_avg[snapshot_count] = np.mean(host_accel,axis=0)

                host_velocities = cp.asnumpy(velocities)
                host_velocities = np.reshape(host_velocities,(N_nodes,3))
                host_vel_norms = np.linalg.norm(host_velocities,axis=1)
                snapshot_vel_norm[snapshot_count] = np.mean(host_vel_norms)
                snapshot_vel_norm_std[snapshot_count] = np.std(host_vel_norms)
                snapshot_vel_components_avg[snapshot_count] = np.mean(host_velocities,axis=0)

                if num_particles != 0 and num_particles < particle_snapshot_cutoff:
                    for particle_count, particle in enumerate(host_particles):
                        particle_center[particle_count,:] = get_particle_center(particle,host_posns)
                        snapshot_particle_posn[particle_count,snapshot_count,:] = particle_center[particle_count,:]*l_e
                        snapshot_particle_velocity[particle_count,snapshot_count,:] = np.sum(host_velocities[host_particles[particle_count],:],axis=0)/host_particles[0].shape
                        snapshot_particle_accel[particle_count,snapshot_count,:] = np.sum(host_accel[host_particles[particle_count],:],axis=0)/host_particles[0].shape
                    snapshot_particle_separation[snapshot_count] = np.linalg.norm(particle_center[0]-particle_center[1])*l_e

                if snapshot_count > 0:
                    soln_diff = host_posns - previous_soln
                    snapshot_soln_diff_norm[snapshot_count-1] = np.linalg.norm(np.ravel(soln_diff))
                previous_soln = host_posns
                snapshot_count += 1
            a_var = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
            host_accel, host_posns = get_accel_scaled_GPU_test(posns,velocities,elements,springs,host_particles,kappa,l_e,beta,beta_i,host_beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
            host_a_var = cp.asnumpy(a_var)
            host_a_var = host_a_var.reshape((N_nodes,3))
            correctness = np.allclose(host_a_var,host_accel)
            if not correctness:
                print('acceleration calculations do not agree')
                accel_result_difference = host_a_var - host_accel
                max_diff = np.max(np.abs(accel_result_difference))
                print(f'acceleration calculations difference norm {np.linalg.norm(accel_result_difference)}\naverage acceleration calculations difference norm {np.linalg.norm(accel_result_difference)/accel_result_difference.shape[0]*accel_result_difference.shape[1]}\nmax difference {max_diff}')
                print(f'{np.count_nonzero(accel_result_difference)} node acceleration components are different')
                if max_diff > 1e-6:
                    print('max difference magnitude greater than 1e-6')
                if max_diff > 1e-4:
                    print(f'gpu calculation of accelerations has particle 1 with acceleration:{np.sum(host_a_var[host_particles[0]],axis=0)}\ncpu calculation of accelerations has particle 1 with: {np.sum(host_accel[host_particles[0]],axis=0)}')
            leapfrog_update(velocities,a_var,step_size,size_entries)
        host_posns = cp.asnumpy(posns)
        host_velocities = cp.asnumpy(velocities)
        sol = np.concatenate((host_posns,host_velocities))
        host_accel = cp.asnumpy(a_var)
        host_accel = np.reshape(host_accel,(N_nodes,3))
        host_velocities = np.reshape(host_velocities,(N_nodes,3))
        #2024-04-04 trying out a thing where, despite not doing the vector summation and distribution of the forces acting on the particles for the node updates, I do it for determining the accelerations for convergence criteria, thereby ignoring internal forces for the particles
        for particle in host_particles:
            host_accel[particle] = np.sum(host_accel[particle],axis=0)/particle.shape[0]
            host_velocities[particle] = np.sum(host_velocities[particle],axis=0)/particle.shape[0]
        accel_comp_magnitude = np.abs(host_accel)
        accel_comp_magnitude_avg = np.mean(accel_comp_magnitude)
        vel_comp_magnitude = np.abs(host_velocities)
        vel_comp_magnitude_avg = np.mean(vel_comp_magnitude)
        a_norms = np.linalg.norm(host_accel,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        host_posns = np.reshape(host_posns,(N_nodes,3))
        final_posns = host_posns
        
        final_v = host_velocities
        v_norms = np.linalg.norm(final_v,axis=1)
        v_norm_avg = np.sum(v_norms)/N_nodes
        if host_stressed_nodes.shape[0] != 0:
            boundary_accel_comp_magnitude = accel_comp_magnitude[host_stressed_nodes]
            boundary_accel_comp_magnitude_avg = np.mean(boundary_accel_comp_magnitude)
            boundary_vel_comp_magnitude = vel_comp_magnitude[host_stressed_nodes]
            boundary_vel_comp_magnitude_avg = np.mean(boundary_vel_comp_magnitude)
        else:
            boundary_accel_comp_magnitude_avg = 0
            boundary_vel_comp_magnitude_avg = 0            
        if accel_comp_magnitude_avg < tolerance and vel_comp_magnitude_avg < tolerance and boundary_accel_comp_magnitude_avg < tolerance and boundary_vel_comp_magnitude_avg < tolerance:#a_norm_avg < tolerance and v_norm_avg < tolerance:
            print(f'Reached convergence criteria of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(accel_comp_magnitude_avg,decimals=6)}')
            print(f'Reached convergence criteria of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(vel_comp_magnitude_avg,decimals=6)}')
            if host_stressed_nodes.shape[0] != 0:
                print(f'Reached convergence criteria for stressed boundary of average acceleration component magnitude < {tolerance}\n average acceleration component magnitude: {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}')
                print(f'Reached convergence criteria for stressed boundary of average velocity component magnitude < {tolerance}\n average velocity component magnitude: {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average acceleration norm < {tolerance}\n average acceleration norm: {np.round(a_norm_avg,decimals=6)}')
            # print(f'Reached convergence criteria of average velocity norm < {tolerance}\n average velocity norm: {np.round(v_norm_avg,decimals=6)}')
            if previously_converged:
                return_status = 0
                print('Ending integration after reaching convergence criteria')
                break
            else:
                previously_converged = True
        elif np.any(np.isnan(a_norms)):
            print(f'Integration failure mode: NaN entries in accelerations. (possible overflow error)')
            return_status = -2
            break
        else:
            print(f'Post-Integration norms\nacceleration norm average = {np.round(a_norm_avg,decimals=6)}\nvelocity norm average = {np.round(v_norm_avg,decimals=6)}')
            print(f'Post-Integration component magnitudes\nacceleration component magnitude average = {np.round(accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(vel_comp_magnitude_avg,decimals=6)}')
            if host_stressed_nodes.shape[0] != 0:
                print(f'Post-Integration component magnitudes on stressed boundary\nacceleration component magnitude average = {np.round(boundary_accel_comp_magnitude_avg,decimals=6)}\nvelocity component magnitude average = {np.round(boundary_vel_comp_magnitude_avg,decimals=6)}')
            # print(f'last snapshot values of norms\n acceleration norm: {snapshot_accel_norm[snapshot_count-1]}\n velocity norm: {snapshot_vel_norm[snapshot_count-1]}')
            mean_displacement[i], max_displacement[i] = get_displacement_norms(final_posns,last_posns)
            last_posns = final_posns.copy()
        if persistent_checkpointing_flag:
            mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,output_dir,tag=f'{i}')
        mre.initialize.write_checkpoint_file(i,sol,Hext,boundary_conditions,checkpoint_output_dir)
        if num_particles != 0 and num_particles < 8:
            particle_velocity = np.zeros((num_particles,3))
            for particle_counter, particle in enumerate(host_particles):
                particle_velocity[particle_counter] = np.sum(final_v[particle,:],axis=0)/particle.shape[0]
            print(f'particle velocity = {np.round(particle_velocity,decimals=6)}')
        i += 1
        if i == max_integrations and num_particles != 0:
            if np.any(np.abs(particle_velocity[0]) > tolerance):#if the particles are still in motion, allow the integration to continue
                max_integrations += 1
                if max_integrations > hard_limit_max_integrations:
                    break
    plot_displacement_v_integration(i,mean_displacement,max_displacement,output_dir)
    plot_residual_vector_norms_hist(a_norms,output_dir,tag='acceleration')
    plot_residual_vector_norms_hist(v_norms,output_dir,tag='velocity')
    if particle_snapshot_flag:
        plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_separation*1e6,output_dir,tag="particle_separation(um)")
        for particle_count in range(num_particles):
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[particle_count]*1e6,output_dir,tag=f"particle_{particle_count+1}_posn(um)")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[particle_count],output_dir,tag=f"particle_{particle_count+1}_velocity")
            plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[particle_count],output_dir,tag=f"particle_{particle_count+1}_accel")
        if num_particles == 2:
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_posn[0]*1e6,snapshot_particle_posn[1]*1e6,output_dir,tag=f"particle_posns(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_velocity[0],snapshot_particle_velocity[1],output_dir,tag=f"particle_velocities(um)")
            plot_snapshots_vector_components_comparison(snapshot_stepsize,step_size,snapshot_count,snapshot_particle_accel[0],snapshot_particle_accel[1],output_dir,tag=f"particle_accelerations(um)")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_components_avg,output_dir,tag="acceleration component averages")
    plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_components_avg,output_dir,tag="velocity component averages")
    if host_stressed_nodes.shape[0] != 0:
        plot_snapshots_vector_components(snapshot_stepsize,step_size,snapshot_count,snapshot_boundary_posn,output_dir,tag="boundary average position")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm,output_dir,tag="acceleration norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_accel_norm_std,output_dir,tag="acceleration norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm,output_dir,tag="velocity norm average")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count,snapshot_vel_norm_std,output_dir,tag="velocity norm standard deviation")
    plot_snapshots(snapshot_stepsize,step_size,snapshot_count-1,snapshot_soln_diff_norm,output_dir,tag="position solution vector difference norm")
    
    return sol, return_status#returning a solution object, that can then have it's attributes inspected