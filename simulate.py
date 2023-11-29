"""
Created on Tues October 3 10:20:19 2023

@author: David Marchfield
"""
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import os
import get_volume_correction_force_cy_nogil
import get_spring_force_cy
import magnetism
import mre.initialize

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
    fig.set_size_inches(2*default_width,2*default_height)
    ax.hist(bins[:-1], bins, weights=counts)
    sigma = np.std(a_norms)
    mu = mean_accel
    ax.set_title(f'Residual '+tag+f' Histogram\nMaximum {max_accel}\nMean {mean_accel}\n$\sigma={sigma}$\nRMS {rms_accel}')
    ax.set_xlabel(tag + ' norm')
    ax.set_ylabel('counts')
    savename = output_dir +'node_residual_'+tag+'_hist.png'
    plt.savefig(savename)
    plt.close()

def get_displacement_norms(final_posns,start_posns):
    """Given the final and starting positions, return the mean and maximum node displacement."""
    displacement = final_posns-start_posns
    displacement_norms = np.linalg.norm(displacement,axis=1)
    max_displacement = np.max(displacement_norms)
    mean_displacement = np.mean(displacement_norms)
    return mean_displacement,max_displacement

def plot_displacement_v_integration(max_iters,mean_displacement,max_displacement,output_dir):
    """Given the number of integrations,array of mean and maximum displacements, and a directory to save to, generate and save a figure with two plots of mean and maximum node displacement versus integration."""
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(max_iters),mean_displacement,'r--',label='mean')
    axs[1].plot(np.arange(max_iters),max_displacement,'k-',label='max')
    axs[0].set_title('displacement between integration iterations')
    axs[0].set_ylabel('displacement mean (units of l_e)')
    axs[1].set_ylabel('displacement max (units of l_e)')
    axs[1].set_xlabel('iteration number')
    # Hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig(output_dir+'displacement.png')
    plt.close()
    
def simulate_scaled_alt(x0,elements,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,t_f,Hext,particle_radius,particle_mass,chi,Ms,initialized_posns,output_dir,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio,drag=10):
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
        r.set_initial_value(y_0).set_f_params(elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        sol = r.integrate(t_f)
        a_var = get_accel_scaled(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
        a_var_alt = get_accel_scaled_alt(sol,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio)
        a_norms = np.linalg.norm(a_var,axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        if i == 0:
            criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms)
        else:
            other_criteria = SimCriteria(solutions,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms)
            criteria.append_criteria(other_criteria)
        # plot_criteria_v_iteration_scaled(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms)
        final_posns = np.reshape(solutions[-1][1:N_nodes*3+1],(N_nodes,3))
        # mre.analyze.post_plot_cut_normalized(initialized_posns,final_posns,springs,particles,boundary_conditions,output_dir)
        solutions = []
        r.set_solout(solout)
        if a_norm_avg < tolerance:
            break
        else:
            y_0 = sol
    # plot_criteria_v_time(solutions,N_nodes,i,elements,springs,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms)
    criteria.plot_criteria(output_dir)
    criteria.plot_displacement_hist(final_posns,initialized_posns,output_dir)
    # criteria.plot_criteria_v_time(output_dir)
    return sol#returning a solution object, that can then have it's attributes inspected

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

def set_fixed_nodes(accel,fixed_nodes):
    """Given the acceleration variable and the indices of the nodes that should be held fixed in the simulation, set the acceleration variable entries to zero for all components of acceleration for the passed nodes"""
    for i in range(fixed_nodes.shape[0]):#after the fact, can set node accelerations and velocities to zero if they are supposed to be held fixed
        #TODO almost certainly faster to remove the inner loop and just set each value to 0 in order, or using python semantics, just set the row to zero?
        for j in range(3):
            accel[fixed_nodes[i],j] = 0
    return accel

def get_accel_scaled_no_fixed_nodes(y,elements,springs,particles,kappa,l_e,beta,beta_i,Hext,particle_radius,particle_mass,chi,Ms,drag=10):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node"""
    #scipy.ode integrator requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
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
    total_torque = np.sum(nodal_torques,axis=1)
    return total_torque

def get_accel_scaled_alt(y,elements,springs,particles,kappa,l_e,beta,beta_i,bc,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,scaled_kappa,scaled_springs_var,scaled_magnetic_force_coefficient,m_ratio,debug_flag=False):
    """computes forces for the given masses, initial conditions, and can take into account boundary conditions. returns the resulting accelerations on each vertex/node. alternative method using coefficients for forces that are already scaled."""
    #scipy.integrate.solve_ivp() requires y (the initial conditions), and also the output of fun(), to be in the shape (n,). because of how the functions calculating forces expect the arguments to be shaped we have to reshape the y variable that is passed to fun()
    N = int(np.round(y.shape[0]/2))
    N_nodes = int(np.round(N/3))
    x0 = np.reshape(y[:N],(N_nodes,3))
    v0 = np.reshape(y[N:],(N_nodes,3))
    drag = 20
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
            pass
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
    M = magnetism.get_magnetization_iterative_normalized(Hext,particle_centers,particle_radius,chi,Ms,l_e)
    mag_forces = magnetism.get_dip_dip_forces_normalized(M,particle_centers,particle_radius,l_e)
    mag_forces_alt = mag_forces * scaled_magnetic_force_coefficient
    mag_forces *= beta/(particle_mass*(l_e**4))

    mag_correctness = np.allclose(mag_forces,mag_forces_alt)
    print(f'magnetic forces agreeing:{mag_correctness}')

    if not vcf_correctness or not springs_correctness or not mag_correctness:
        indices_of_interest = np.nonzero(np.logical_not(np.isclose(spring_force,spring_force_alt)))
        for i in range(len(indices_of_interest[0])):
            index = indices_of_interest[0][i]
            print(f'for index {index}')
            for j in range(3):
                print(f'original force:{spring_force[index,j]}')
            for j in range(3):
                print(f'alternative force:{spring_force_alt[index,j]}')
        possible_springs_connection = np.nonzero(np.isclose(springs[:,0],indices_of_interest[0][0]))
        possible_springs_connection2 = np.nonzero(np.isclose(springs[:,1],indices_of_interest[0][0]))
        possible_springs_connection_alt = np.nonzero(np.isclose(scaled_springs_var[:,0],indices_of_interest[0][0]))
        possible_springs_connection_alt2 = np.nonzero(np.isclose(scaled_springs_var[:,1],indices_of_interest[0][0]))
        posn1 = x0[indices_of_interest[0][0]]
        posn2 = x0[indices_of_interest[0][3]]
        inspecting_springs = springs[possible_springs_connection,:]
        inspecting_springs2 = springs[possible_springs_connection2,:]
        inspecting_springs_alt = scaled_springs_var[possible_springs_connection_alt,:]
        inspecting_springs_alt2 = scaled_springs_var[possible_springs_connection_alt2,:]
        original_vector_sum = np.zeros((3,))
        original_force_vecsum = np.zeros((3,))
        alt_vector_sum = np.zeros((3,))
        alt_force_vecsum = np.zeros((3,))
        for count in range(inspecting_springs.shape[0]):
            i = np.squeeze(inspecting_springs)[count]
            posn1 = x0[int(i[0])]
            posn2 = x0[int(i[1])]
            force_magnitude = i[2]*(np.linalg.norm(posn1 - posn2) - i[3])
            force_i = -1*force_magnitude*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i = force_i*l_e*beta_i[int(i[0])]
            original_vector_sum += accel_i
            original_force_vecsum += force_i
            i_alt = np.squeeze(inspecting_springs_alt)[count]
            force_magnitude_alt = i_alt[2]*(np.linalg.norm(posn1 - posn2) - i_alt[3])
            force_i_alt = -1*force_magnitude_alt*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i_alt = force_i_alt*m_ratio[int(i[0])]
            alt_vector_sum += accel_i_alt
            alt_force_vecsum += force_i_alt
            if not np.allclose(accel_i,accel_i_alt):
                print(f'disagreement for spring force scaled accelerations')
        for count in range(inspecting_springs2.shape[0]):
            i = np.squeeze(inspecting_springs2)[count]
            posn1 = x0[int(i[0])]
            posn2 = x0[int(i[1])]
            force_magnitude = i[2]*(np.linalg.norm(posn1 - posn2) - i[3])
            force_i = -1*force_magnitude*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i = force_i*l_e*beta_i[int(i[1])]
            original_vector_sum += -1*accel_i
            original_force_vecsum += -1*force_i
            i_alt = np.squeeze(inspecting_springs_alt2)[count]
            force_magnitude_alt = i_alt[2]*(np.linalg.norm(posn1 - posn2) - i_alt[3])
            force_i_alt = -1*force_magnitude_alt*(posn1 - posn2)/np.linalg.norm(posn1 - posn2)
            accel_i_alt = force_i_alt*m_ratio[int(i[1])]
            alt_vector_sum += -1*accel_i_alt
            alt_force_vecsum += -1*force_i_alt
            if not np.allclose(accel_i,accel_i_alt):
                print(f'disagreement for spring force scaled accelerations')
        if not np.allclose(original_vector_sum,alt_vector_sum):
            print("vector sums of scaled accelerations due to spring forces don't agree")
        if not np.allclose(original_force_vecsum*l_e*beta_i[int(i[1])],alt_force_vecsum*m_ratio[int(i[1])]):
            print("vector sums of scaled accelerations due to spring forces don't agree when summing forces before finishing scaling")
        pass

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