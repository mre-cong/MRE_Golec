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
import magnetism
import mre.initialize
import mre.analyze
import torch
import torch.optim as opt
mu0 = 4*np.pi*1e-7
# plt.switch_backend('TkAgg')

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
    fig, axs = plt.subplots(2,layout="constrained")
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    simulation_time = snapshot_stepsize*snapshot_values.shape[0]*step_size
    snapshot_times = np.arange(0,simulation_time,step_size*snapshot_stepsize)
    axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries],'o-',label='_')
    axs[0].set_xlabel('simulation time')
    axs[0].set_ylabel(f'{tag}')
    # mre.analyze.format_figure(axs[0])
    midpoint = int(total_entries/2)
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries],'o-',label='_')
    axs[1].set_xlabel('simulation time')
    axs[1].set_ylabel(f'{tag}')
    # mre.analyze.format_figure(axs[1])
    mre.analyze.format_subfigures(axs)
    plt.savefig(output_dir+f'{tag}_snapshots.png')
    plt.close()
    np.save(output_dir+'simulation_time',simulation_time)
    np.save(output_dir+f'{tag}_snapshots',snapshot_values)

def plot_snapshots_vector_components(snapshot_stepsize,step_size,total_entries,snapshot_values,output_dir,tag=""):
    """Generate and save a figure showing the evolution of some value at particular steps throughout integration."""
    simulation_time = snapshot_stepsize*snapshot_values.shape[0]*step_size
    snapshot_times = np.arange(0,simulation_time,step_size*snapshot_stepsize)
    if 'boundary' in tag:
        fig, axs = plt.subplots(2,3,layout="constrained")
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
        # mre.analyze.format_figure(axs[0,0])
        # mre.analyze.format_figure(axs[0,1])
        # mre.analyze.format_figure(axs[0,2])
        midpoint = int(total_entries/2)
        axs[1,0].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,0],'o-',label='_')
        axs[1,1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,1],'o-',label='_')
        axs[1,2].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,2],'o-',label='_')
        axs[1,0].set_xlabel('simulation time')
        axs[1,0].set_ylabel(f'{tag}')
        axs[1,1].set_xlabel('simulation time')
        axs[1,1].set_ylabel(f'{tag}')
        axs[1,2].set_xlabel('simulation time')
        axs[1,2].set_ylabel(f'{tag}')
        # fig.legend(fontsize=20)
        # mre.analyze.format_figure(axs[1,0])
        # mre.analyze.format_figure(axs[1,1])
        # mre.analyze.format_figure(axs[1,2])
    else:
        fig, axs = plt.subplots(2,layout="constrained")
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(3*default_width,3*default_height)
        fig.set_dpi(200)
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,0],'o-',label=f'{tag} x')
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,1],'o-',label=f'{tag} y')
        axs[0].plot(snapshot_times[:total_entries],snapshot_values[:total_entries,2],'o-',label=f'{tag} z')
        axs[0].set_xlabel('simulation time')
        axs[0].set_ylabel(f'{tag}')
        # mre.analyze.format_figure(axs[0])
        midpoint = int(total_entries/2)
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,0],'o-',label='_')
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,1],'o-',label='_')
        axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values[midpoint:total_entries,2],'o-',label='_')
        axs[1].set_xlabel('simulation time')
        axs[1].set_ylabel(f'{tag}')
        # fig.legend(fontsize=20)
        # mre.analyze.format_figure(axs[1])
    mre.analyze.format_subfigures(axs)
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
    # mre.analyze.format_figure(axs[0])
    midpoint = int(total_entries/2)
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,0],'o-',label='_')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,1],'o-',label='_')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_one[midpoint:total_entries,2],'o-',label='_')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,0],'x--',label='_')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,1],'x--',label='_')
    axs[1].plot(snapshot_times[midpoint:total_entries],snapshot_values_two[midpoint:total_entries,2],'x--',label='_')
    axs[1].set_xlabel('simulation time')
    axs[1].set_ylabel(f'{tag}')
    # fig.legend(fontsize=20)
    # mre.analyze.format_figure(axs[1])
    mre.analyze.format_subfigures(axs)
    plt.savefig(output_dir+f'{tag}_snapshots.png')
    plt.close()

# class SimCriteria:
#     """Class for calculating criteria for a simulation of up to two particles from the solution vector generated at each integration step. Criteria include:acceleration norm mean, acceleration norm max, particle acceleration norm, particle separation, mean and maximum velocity norms, maximum and minimum node positions in each cartesian direction, mean cartesian coordinate of nodes belonging to each surface of the simulated volume"""
#     def __init__(self,solutions,*args):
#         self.get_criteria_per_iteration(solutions,*args)
#         self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
#         self.timestep = self.time[1:] - self.time[:-1]

#     def get_criteria_per_iteration(self,solutions,*args):
#         iterations = np.array(solutions).shape[0]
#         self.iter_number = np.arange(iterations)
#         N_nodes = int((np.array(solutions).shape[1] - 1)/2/3)
#         self.a_norm_avg = np.zeros((iterations,))
#         self.a_norm_max = np.zeros((iterations,))
#         self.particle_a_norm = np.zeros((iterations,))
#         self.particle_separation = np.zeros((iterations,))
#         self.v_norm_avg = np.zeros((iterations,))
#         self.v_norm_max = np.zeros((iterations,))
#         self.particle_v_norm = np.zeros((iterations,))
#         self.time = np.array(solutions)[:,0]
#         self.max_x = np.zeros((iterations,))
#         self.max_y = np.zeros((iterations,))
#         self.max_z = np.zeros((iterations,))
#         self.min_x = np.zeros((iterations,))
#         self.min_y = np.zeros((iterations,))
#         self.min_z = np.zeros((iterations,))
#         self.length_x = np.zeros((iterations,))
#         self.length_y = np.zeros((iterations,))
#         self.length_z = np.zeros((iterations,))
#         boundaries = args[8]
#         self.left = np.zeros((iterations,))
#         self.right = np.zeros((iterations,))
#         self.top = np.zeros((iterations,))
#         self.bottom = np.zeros((iterations,))
#         self.front = np.zeros((iterations,))
#         self.back = np.zeros((iterations,))
#         particles = args[2]
#         for count, row in enumerate(solutions):
#             a_var = get_accel_scaled(np.array(row[1:]),*args)
#             a_norms = np.linalg.norm(a_var,axis=1)
#             self.a_norm_max[count] = np.max(a_norms)
#             # if self.a_norm_max[count] > 10000:
#             #     a_var = get_accel_scaled(np.array(row[1:]),*args)
#             self.a_norm_avg[count] = np.sum(a_norms)/np.shape(a_norms)[0]
#             final_posns = np.reshape(row[1:N_nodes*3+1],(N_nodes,3))
#             final_v = np.reshape(row[N_nodes*3+1:],(N_nodes,3))
#             v_norms = np.linalg.norm(final_v,axis=1)
#             self.v_norm_max[count] = np.max(v_norms)
#             self.v_norm_avg[count] = np.sum(v_norms)/np.shape(v_norms)[0]
#             if particles.shape[0] != 0:
#                 a_particles = a_var[particles[0],:]
#                 self.particle_a_norm[count] = np.linalg.norm(a_particles[0,:])
#                 v_particles = final_v[particles[0],:]
#                 self.particle_v_norm[count] = np.linalg.norm(v_particles[0,:])
#                 x1 = get_particle_center(particles[0],final_posns)
#                 x2 = get_particle_center(particles[1],final_posns)
#                 self.particle_separation[count] = np.sqrt(np.sum(np.power(x1-x2,2)))
#             self.get_system_extent(final_posns,boundaries,count)
    
#     def get_system_extent(self,posns,boundaries,count):
#         """Assign values to the objects instance variables that give some sense of the physical extent/dimensions/size of the simulated system as the simulation progresses"""
#         self.max_x[count] = np.max(posns[:,0])
#         self.max_y[count] = np.max(posns[:,1])
#         self.max_z[count] = np.max(posns[:,2])
#         self.min_x[count] = np.min(posns[:,0])
#         self.min_y[count] = np.min(posns[:,1])
#         self.min_z[count] = np.min(posns[:,2])
#         self.length_x[count] = self.max_x[count] - self.min_x[count]
#         self.length_y[count] = self.max_y[count] - self.min_y[count]
#         self.length_z[count] = self.max_z[count] - self.min_z[count]
#         self.left[count] = np.mean(posns[boundaries['left'],0])
#         self.right[count] = np.mean(posns[boundaries['right'],0])
#         self.top[count] = np.mean(posns[boundaries['top'],2])
#         self.bottom[count] = np.mean(posns[boundaries['bot'],2])
#         self.front[count] = np.mean(posns[boundaries['front'],1])
#         self.back[count] = np.mean(posns[boundaries['back'],1])

#     def plot_displacement_hist(self,final_posns,initialized_posns,output_dir):
#         displacement = final_posns-initialized_posns
#         displacement_norms = np.linalg.norm(displacement,axis=1)
#         max_displacement = np.max(displacement_norms)
#         mean_displacement = np.mean(displacement_norms)
#         rms_displacement = np.sqrt(np.sum(np.power(displacement_norms,2))/np.shape(displacement_norms)[0])
#         counts, bins = np.histogram(displacement_norms, bins=20)
#         fig,ax = plt.subplots()
#         default_width,default_height = fig.get_size_inches()
#         fig.set_size_inches(2*default_width,2*default_height)
#         ax.hist(bins[:-1], bins, weights=counts)
#         sigma = np.std(displacement_norms)
#         mu = mean_displacement
#         # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)))**2)
#         # ax.plot(bins,y,'--')
#         ax.set_title(f'Displacement Histogram\nMaximum {max_displacement}\nMean {mean_displacement}\n$\sigma={sigma}$\nRMS {rms_displacement}')
#         ax.set_xlabel('displacement (units of l_e)')
#         ax.set_ylabel('counts')
#         savename = output_dir +'node_displacement_hist.png'
#         plt.savefig(savename)
#         # plt.show()
#         plt.close()
    
#     def append_criteria(self,other):
#         """Append data from one SimCriteria object to another, by appending each member variable. Special cases (like time, iteration number) are appended in a manner to reflect the existence of prior integration iterations of the simulation"""
#         #vars(self) returns a dictionary containing the member variables names and values as key-value pairs, allowing for this dynamic sort of access, meaning that extendingh the class with more member variables will allow this method to be used without changes (unless a new special case arises)
#         my_keys = list(vars(self).keys())
#         for key in my_keys:
#             if key != 'iter_number' and key != 'time':
#                 vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],vars(other)[f'{key}'])
#             elif key == 'iter_number':
#                 vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],np.max(vars(self)[f'{key}'])+vars(other)[f'{key}']+1)
#             elif key == 'time':
#                 vars(self)[f'{key}'] = np.append(vars(self)[f'{key}'],np.max(vars(self)[f'{key}'])+vars(other)[f'{key}'])

#     def plot_criteria(self,output_dir):
#         #TODO Unfinished. use the member variable names to generate save names and figure labels automatically. have a separate variable passed to choose which variable to plot against (time, iteration, anything else?)
#         """Generating plots of simulation criteria using matplotlib and using built-in python features to iterate over the instance variables of the object"""
#         if not (os.path.isdir(output_dir)):
#             os.mkdir(output_dir)
#         self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
#         my_keys = list(vars(self).keys())
#         for key in my_keys:
#             if key != 'iter_number' and key != 'time':
#                 fig = plt.figure()
#                 if key != 'delta_a_norm':
#                     plt.plot(self.time,vars(self)[f'{key}'])
#                 else:
#                     plt.plot(self.time[:self.delta_a_norm.shape[0]],vars(self)[f'{key}'])
#                 ax = plt.gca()
#                 ax.set_xlabel('scaled time')
#                 ax.set_ylabel(f'{key}')
#                 savename = output_dir + f'{key}_v_time.png'
#                 plt.savefig(savename)
#                 plt.close()
#         # plt.show()
#         # plt.close('all')
    
#     def plot_criteria_subplot(self,output_dir):
#         """Generate subplots of simulation criteria using matplotlib"""
#         if not (os.path.isdir(output_dir)):
#             os.mkdir(output_dir)
#         self.delta_a_norm = self.a_norm_avg[1:]-self.a_norm_avg[:-1]
#         # first plotting the system extent (to get a sense of the change in the system size
#         fig, axs = plt.subplots(3,3)
#         default_width, default_height = fig.get_size_inches()
#         fig.set_size_inches(2*default_width,2*default_height)
#         fig.set_dpi(100)
#         axs[0,0].plot(self.time,self.min_x)
#         axs[0,0].set_title('x')
#         axs[0,0].set_ylabel('minimum x position (l_e)')
#         axs[1,0].plot(self.time,self.max_x)
#         axs[1,0].set_ylabel('maximum x position (l_e)')
#         axs[2,0].plot(self.time,self.length_x)
#         axs[2,0].set_ylabel('length (max - min) in x direction (l_e)')
#         axs[2,0].set_xlabel('scaled time')

#         axs[0,1].plot(self.time,self.min_y)
#         axs[0,1].set_title('y')
#         axs[0,1].set_ylabel('minimum y position (l_e)')
#         axs[1,1].plot(self.time,self.max_y)
#         axs[1,1].set_ylabel('maximum y position (l_e)')
#         axs[2,1].plot(self.time,self.length_y)
#         axs[2,1].set_ylabel('length (max - min) in y direction (l_e)')
#         axs[2,1].set_xlabel('scaled time')

#         axs[0,2].plot(self.time,self.min_z)
#         axs[0,2].set_title('z')
#         axs[0,2].set_ylabel('minimum z position (l_e)')
#         axs[1,2].plot(self.time,self.max_z)
#         axs[1,2].set_ylabel('maximum z position (l_e)')
#         axs[2,2].plot(self.time,self.length_z)
#         axs[2,2].set_ylabel('length (max - min) in z direction (l_e)')
#         axs[2,2].set_xlabel('scaled time')

#         savename = output_dir + 'systemextent_v_time.png'
#         plt.savefig(savename)
#         plt.close()
        
#         # plot the acceleration and velocity norms
#         fig, axs = plt.subplots(2,2)
#         fig.set_size_inches(2*default_width,2*default_height)
#         fig.set_dpi(100)
#         axs[0,0].plot(self.time,self.a_norm_avg)
#         axs[0,0].set_title('node acceleration')
#         axs[0,0].set_ylabel('node acceleration norm mean (unitless)')
#         axs[1,0].plot(self.time,self.a_norm_max)
#         axs[1,0].set_ylabel('node acceleration norm max (unitless)')
#         axs[1,0].set_xlabel('scaled time')
#         axs[0,1].plot(self.time,self.v_norm_avg)
#         axs[0,1].set_title('node velocity')
#         axs[0,1].set_ylabel('node velocity norm mean (unitless)')    
#         axs[1,1].plot(self.time,self.v_norm_max)
#         axs[1,1].set_ylabel('node velocity norm max (unitless)')
#         axs[1,0].set_xlabel('scaled time')

#         savename = output_dir + 'node_behavior_v_time.png'
#         plt.savefig(savename)
#         plt.close()

#         # plot the particle acceleration, velocity, and separation
#         fig, axs = plt.subplots(3)
#         fig.set_size_inches(2*default_width,2*default_height)
#         fig.set_dpi(100)
#         axs[0].plot(self.time,self.particle_separation)
#         axs[0].set_ylabel('particle separation (l_e)')
#         axs[0].set_title('particle position, velocity, and acceleration')
#         axs[1].plot(self.time,self.particle_v_norm)
#         axs[1].set_ylabel('particle velocity norm (unitless)')
#         axs[2].plot(self.time,self.particle_a_norm)
#         axs[2].set_ylabel('particle acceleration norm (unitless)')
#         axs[2].set_xlabel('scaled time')

#         savename = output_dir + 'particle_behavior_v_time.png'
#         plt.savefig(savename)
#         plt.close()

#         fig, axs = plt.subplots(3,2)
#         fig.set_size_inches(2*default_width,2*default_height)
#         fig.set_dpi(100)
#         axs[0,0].plot(self.time,self.left)
#         axs[0,0].set_ylabel('mean node position: left bdry (l_e)')
#         axs[0,0].set_title('Mean Boundary Node Positions')
#         axs[0,1].plot(self.time,self.right)
#         axs[0,1].set_ylabel('mean node position: right bdry (l_e)')
#         axs[1,0].plot(self.time,self.front)
#         axs[1,0].set_ylabel('mean node position: front bdry (l_e)')
#         axs[1,1].plot(self.time,self.back)
#         axs[1,1].set_ylabel('mean node position: back bdry (l_e)')
#         axs[2,0].plot(self.time,self.top)
#         axs[2,0].set_ylabel('mean node position: top bdry (l_e)')
#         axs[2,1].plot(self.time,self.bottom)
#         axs[2,1].set_ylabel('mean node position: bottom bdry (l_e)')
#         axs[2,0].set_xlabel('scaled time')
#         savename = output_dir + 'mean_boundaries_v_time.png'
#         plt.savefig(savename)
#         plt.close()
#         # plt.show()
#         # plt.close('all')

#         fig, axs = plt.subplots(1,3)
#         fig.set_size_inches(2*default_width,2*default_height)
#         fig.set_dpi(100)
#         axs[0].plot(self.time[:self.timestep.shape[0]],self.timestep,'.')
#         axs[0].set_title('Time Step Taken')
#         axs[0].set_xlabel('scaled time')
#         axs[0].set_ylabel('time step')
#         axs[1].plot(self.iter_number[:self.timestep.shape[0]],self.timestep,'.')
#         axs[1].set_title('Time Step Taken')
#         axs[1].set_xlabel('integration number')
#         axs[1].set_ylabel('time step')
#         axs[2].plot(self.iter_number,self.time,'.')
#         axs[2].set_title('Total Time')
#         axs[2].set_xlabel('integration number')
#         axs[2].set_ylabel('total scaled time')
#         savename = output_dir + 'timestep_per_iteration_and_time.png'
#         plt.savefig(savename)
#         plt.close()

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

scaled_element_volume_kernel = cp.RawKernel(r'''
extern "C" __global__
void element_volume(const int* elements, const float* node_posns, float* volume, const int size_elements) {
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
        //for the element, get the average edge vectors, then using the average edge vectors, get the volume correction force
        float avg_vector_i[3];
        float avg_vector_j[3];
        float avg_vector_k[3];
                                
        avg_vector_i[0] = (node_posns[3*index2] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index1] + node_posns[3*index6] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index5])/4;
        avg_vector_i[1] = (node_posns[1+3*index2] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index1] + node_posns[1+3*index6] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index5])/4;
        avg_vector_i[2] = (node_posns[2+3*index2] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index1] + node_posns[2+3*index6] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index5])/4;
                          
        avg_vector_j[0] = (node_posns[3*index4] - node_posns[3*index0] + node_posns[3*index6] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index1] + node_posns[3*index7] - node_posns[3*index3])/4;
        avg_vector_j[1] = (node_posns[1+3*index4] - node_posns[1+3*index0] + node_posns[1+3*index6] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index1] + node_posns[1+3*index7] - node_posns[1+3*index3])/4;
        avg_vector_j[2] = (node_posns[2+3*index4] - node_posns[2+3*index0] + node_posns[2+3*index6] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index1] + node_posns[2+3*index7] - node_posns[2+3*index3])/4;

        avg_vector_k[0] = (node_posns[3*index1] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index6])/4;
        avg_vector_k[1] = (node_posns[1+3*index1] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index6])/4;
        avg_vector_k[2] = (node_posns[2+3*index1] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index6])/4;                    
        
        //float acrossb[3];
        float bcrossc[3];
        //float ccrossa[3];
        float adotbcrossc;
                   
        bcrossc[0] = avg_vector_j[1]*avg_vector_k[2] - avg_vector_j[2]*avg_vector_k[1];
        bcrossc[1] = avg_vector_j[2]*avg_vector_k[0] - avg_vector_j[0]*avg_vector_k[2];
        bcrossc[2] = avg_vector_j[0]*avg_vector_k[1] - avg_vector_j[1]*avg_vector_k[0];
                 
        adotbcrossc = avg_vector_i[0]*bcrossc[0] + avg_vector_i[1]*bcrossc[1] + avg_vector_i[2]*bcrossc[2];

        volume[tid] = adotbcrossc;
    }
    }
    ''', 'element_volume')

scaled_spring_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void spring_energy(const float* edges, const float* node_posns, float* energies, const int size_edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("entered the kernel");
    if (tid < size_edges)
    {
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
        float eps_constant = (1e-7)*4*powf(3.141592654f,2)*powf(1.9e6,2)*powf(particle_radius,3)/72;
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
        float m_mag = particle_volume*(Ms*chi*H_mag/(Ms + chi*H_mag));
        float H_hat[3];
        if (fabsf(H_mag) < 1.e-4){
            H_hat[0] = 0;
            H_hat[1] = 0;
            H_hat[2] = 0;
        }
        else{
            H_hat[0] = H[3*tid]/H_mag;
            H_hat[1] = H[3*tid+1]/H_mag;
            H_hat[2] = H[3*tid+2]/H_mag;
        }
        //printf("H_hat: %f, %f, %f\n",H_hat[0],H_hat[1],H_hat[2]);
        magnetic_moment[3*tid] = m_mag*H_hat[0];
        magnetic_moment[3*tid+1] = m_mag*H_hat[1];
        magnetic_moment[3*tid+2] = m_mag*H_hat[2];
    }
    }
    ''', 'get_magnetization')

normalized_magnetization_kernel = cp.RawKernel(r'''
extern "C" __global__                                    
void get_magnetization(const float chi, const float* h, float* magnetization, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {                                           
        //get magnetization based on total Hfield at each particle center
        float hmag = sqrtf(h[3*tid]*h[3*tid] + h[3*tid+1]*h[3*tid+1] + h[3*tid+2]*h[3*tid+2]);
        float m_mag = (chi*hmag/(1 + chi*hmag));
        float h_hat[3];
        if (fabsf(hmag) < 1.e-4){
            h_hat[0] = 0;
            h_hat[1] = 0;
            h_hat[2] = 0;
        }
        else{
            h_hat[0] = h[3*tid]/hmag;
            h_hat[1] = h[3*tid+1]/hmag;
            h_hat[2] = h[3*tid+2]/hmag;
        }
        
        magnetization[3*tid] = m_mag*h_hat[0];
        magnetization[3*tid+1] = m_mag*h_hat[1];
        magnetization[3*tid+2] = m_mag*h_hat[2];
    }
    }
    ''', 'get_magnetization')

particle_positions_kernel = cp.RawKernel(r'''
extern "C" __global__                                         
void get_particle_posns(const float* node_posns, float* particle_posns, const int* particle_nodes, const int nodes_per_particle, const int size_particles) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size_particles)
    {
        float x_pos = 0;
        float y_pos = 0;
        float z_pos = 0;
        int particle_node_idx = nodes_per_particle*tid;
        for (int i = 0; i < 8; i++)
        {
            x_pos += node_posns[3*particle_nodes[particle_node_idx+i]];
            y_pos += node_posns[3*particle_nodes[particle_node_idx+i] + 1];
            z_pos += node_posns[3*particle_nodes[particle_node_idx+i] + 2];
        }
        //divide by 8 to get the average x/y/z position, taken to be the center of the particle
        x_pos *= 0.125;
        y_pos *= 0.125;
        z_pos *= 0.125;
        particle_posns[3*tid] = x_pos;
        particle_posns[3*tid+1] = y_pos;
        particle_posns[3*tid+2] = z_pos;
    }
    }
    ''', 'get_particle_posns')

def get_particle_posns_gpu(node_posns,particle_nodes,num_particles,nodes_per_particle):
    """Return the positions of the particle centers."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    particle_posns = cp.zeros((3*num_particles,1),dtype=cp.float32)
    particle_positions_kernel((grid_size,),(block_size,),(node_posns,particle_posns,particle_nodes,nodes_per_particle,num_particles))
    cupy_stream.synchronize()

    return particle_posns

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
        float eps_constant = (1e-7)*4*powf(3.141592654f,2)*powf(1.9e6,2)*powf(particle_radius,3)/72;
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
        int node_force_idx = particle_nodes[tid];

        node_force[3*node_force_idx] += magnetic_force[3*mag_force_idx];
        node_force[3*node_force_idx+1] += magnetic_force[3*mag_force_idx+1];
        node_force[3*node_force_idx+2] += magnetic_force[3*mag_force_idx+2];
    }
    }
    ''', 'set_dipole_force')

def get_normalized_dipole_field_kernel(rij,inv_rij_mag,num_particles,l_e,particle_volume):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    kernel = cp.zeros((num_particles*3,num_particles*3),dtype=cp.float32)

    nij_x = rij[0::3]*inv_rij_mag
    nij_y = rij[1::3]*inv_rij_mag
    nij_z = rij[2::3]*inv_rij_mag

    nij_x = cp.reshape(nij_x,(num_particles,num_particles),order='C')
    nij_y = cp.reshape(nij_y,(num_particles,num_particles),order='C')
    nij_z = cp.reshape(nij_z,(num_particles,num_particles),order='C')

    inv_rij_mag = cp.reshape(inv_rij_mag,(num_particles,num_particles),order='C')
    
    denom = particle_volume/(4*np.float32(np.pi))*cp.power(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[2::3,0::3] = kernel[0::3,2::3]
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom
    kernel[2::3,1::3] = kernel[1::3,2::3]
    
    return kernel

def get_normalized_dipole_field_kernel_torch(rij,inv_rij_mag,num_particles,l_e,particle_volume):
    """Return matrix containing information about relative dipole positions for calculating dipolar fields."""
    kernel = torch.zeros((num_particles*3,num_particles*3)).cuda()

    nij_x = torch.asarray(rij[0::3]*inv_rij_mag)
    nij_y = torch.asarray(rij[1::3]*inv_rij_mag)
    nij_z = torch.asarray(rij[2::3]*inv_rij_mag)

    nij_x = torch.reshape(nij_x,(num_particles,num_particles))
    nij_y = torch.reshape(nij_y,(num_particles,num_particles))
    nij_z = torch.reshape(nij_z,(num_particles,num_particles))

    inv_rij_mag = torch.reshape(torch.asarray(inv_rij_mag),(num_particles,num_particles))
    
    denom = particle_volume/(4*np.float32(np.pi))*torch.pow(inv_rij_mag/l_e,3)#

    kernel[0::3,0::3] = (3*nij_x*nij_x-1)*denom
    kernel[1::3,1::3] = (3*nij_y*nij_y-1)*denom
    kernel[2::3,2::3] = (3*nij_z*nij_z-1)*denom
    kernel[0::3,1::3] = (3*nij_y)*nij_x*denom
    kernel[1::3,0::3] = kernel[0::3,1::3]
    kernel[0::3,2::3] = (3*nij_z)*nij_x*denom
    kernel[2::3,0::3] = kernel[0::3,2::3]
    kernel[1::3,2::3] = (3*nij_z)*nij_y*denom
    kernel[2::3,1::3] = kernel[1::3,2::3]
    
    return kernel

def get_normalized_magnetization_fixed_point_iteration(hext,num_particles,particle_posns,chi,particle_volume,l_e,max_iters=20,atol=1e-3,rtol=5e-3,initial_soln=None):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetization = cp.zeros((num_particles*3,1),dtype=cp.float32)
    last_magnetization = cp.zeros((num_particles*3,1),dtype=cp.float32)
    hext_vector = cp.tile(hext,num_particles)
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    if type(initial_soln) == type(None):
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,hext_vector,magnetization,num_particles))
        cupy_stream.synchronize()
    else:
        magnetization = initial_soln.copy()

    last_magnetization = magnetization.copy()

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,num_particles))
    cupy_stream.synchronize()

    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)

    for i in range(max_iters):
        htot = cp.tile(hext,num_particles)
        dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetization))
        htot += dipolar_fields
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,htot,magnetization,num_particles))
        cupy_stream.synchronize()
        if i > 0:
            difference = magnetization - last_magnetization
            if cp.all(cp.abs(cp.ravel(difference)) < atol + cp.abs(cp.ravel(last_magnetization))*rtol):
                return (magnetization,separation_vectors,separation_vectors_inv_magnitude,0)
        last_magnetization = magnetization.copy()
    return (magnetization,separation_vectors,separation_vectors_inv_magnitude,-1)

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

def get_normalized_magnetization_and_total_field(hext,num_particles,particle_posns,chi,particle_volume,l_e,max_iters=20,atol=1e-3,rtol=5e-3,initial_soln=None):
    """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    magnetization = cp.zeros((num_particles*3,1),dtype=cp.float32)
    last_magnetization = cp.zeros((num_particles*3,1),dtype=cp.float32)
    hext_vector = cp.tile(hext,num_particles)
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    if type(initial_soln) == type(None):
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,hext_vector,magnetization,num_particles))
        cupy_stream.synchronize()
    else:
        magnetization = initial_soln.copy()

    last_magnetization = magnetization.copy()

    separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
    separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
    separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,num_particles))
    cupy_stream.synchronize()

    dipole_field_kernel = get_normalized_dipole_field_kernel(separation_vectors,separation_vectors_inv_magnitude,num_particles,l_e,particle_volume)

    for i in range(max_iters):
        htot = cp.tile(hext,num_particles)
        dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetization))
        htot += dipolar_fields
        normalized_magnetization_kernel((grid_size,),(block_size,),(chi,htot,magnetization,num_particles))
        cupy_stream.synchronize()
        if i > 0:
            difference = magnetization - last_magnetization
            if cp.all(cp.abs(cp.ravel(difference)) < atol + cp.abs(cp.ravel(last_magnetization))*rtol):
                htot = cp.tile(hext,num_particles)
                dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetization))
                htot += dipolar_fields
                return (magnetization,htot,0)
        last_magnetization = magnetization.copy()
    htot = cp.tile(hext,num_particles)
    dipolar_fields = cp.squeeze(cp.matmul(dipole_field_kernel,magnetization))
    htot += dipolar_fields
    return (magnetization,htot,-1)

class FixedPointMethodError(Exception):
    pass

def get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,starting_magnetization=None):
    cupy_stream = cp.cuda.get_current_stream()
    num_streaming_multiprocessors = 14
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))

    hext = cp.asarray(Hext/Ms)
    magnetization, separation_vectors, separation_vectors_inv_magnitude, return_code = get_normalized_magnetization_fixed_point_iteration(hext,num_particles,particle_posns,chi,particle_volume,l_e,max_iters=40,atol=1e-3,rtol=5e-3,initial_soln=starting_magnetization)

    if return_code == -1:
        print(f'fixed point method for magnetization finding failed to converge')
        raise FixedPointMethodError
    normalized_magnetization = magnetization.copy()
    magnetization *= Ms*particle_volume

    inv_l_e = np.float32(1/l_e)
    force = cp.zeros((3*num_particles,1),dtype=cp.float32,order='C')
    dipole_force_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetization,particle_radius,l_e,inv_l_e,force,num_particles))
    cupy_stream.synchronize()
    force *= np.float32(beta/particle_mass)
    return force, normalized_magnetization

# def get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e):
#     """Combining gpu kernels with forced synchronization between calls to speed up magnetization finding calculations and reuse intermediate results (separation vectors)."""
#     cupy_stream = cp.cuda.get_current_stream()
#     num_streaming_multiprocessors = 14
#     magnetic_moments = cp.zeros((num_particles*3,1),dtype=cp.float32)
#     Hext_vector = cp.tile(Hext,num_particles)
#     block_size = 128
#     grid_size = (int (np.ceil((int (np.ceil(num_particles/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
#     # normalized_magnetization_kernel((grid_size,),(block_size,),(Ms,chi,Hext_vector,magnetic_moment,size_particles))
#     # magnetization_kernel used the particle volume to return the magnetic moment. the normalized approach normalizes by the saturation magnetization. the issue with using the particle volume to convert from magnetization to magnetic moment was that the result for low field values evaluated to zero.
#     #the issue may have actually been that i didn't use 32 bit floats for chi, Ms, and l_e
#     magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Hext_vector,magnetic_moments,num_particles))

#     cupy_stream.synchronize()
#     separation_vectors = cp.zeros((num_particles*num_particles*3,1),dtype=cp.float32)
#     separation_vectors_inv_magnitude = cp.zeros((num_particles*num_particles,1),dtype=cp.float32)
#     separation_vectors_kernel((grid_size,),(block_size,),(particle_posns,separation_vectors,separation_vectors_inv_magnitude,num_particles))
#     cupy_stream.synchronize()

#     inv_l_e = np.float32(1/l_e)
#     max_iters = 5
#     Htot_initial = cp.tile(Hext,num_particles)
#     for i in range(max_iters):
#         Htot = cp.copy(Htot_initial)#cp.tile(Hext,num_particles)
#         dipole_field_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,inv_l_e,Htot,num_particles))
#         cupy_stream.synchronize()
#         magnetization_kernel((grid_size,),(block_size,),(Ms,chi,particle_volume,Htot,magnetic_moments,num_particles))
#         cupy_stream.synchronize()
#     inv_l_e = np.float32(1/l_e)
#     force = cp.zeros((3*num_particles,1),dtype=cp.float32,order='C')
#     dipole_force_kernel((grid_size,),(block_size,),(separation_vectors,separation_vectors_inv_magnitude,magnetic_moments,particle_radius,l_e,inv_l_e,force,num_particles))
#     cupy_stream.synchronize()
#     force *= np.float32(beta/particle_mass)
#     return force

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

def composite_gpu_force_calc_v3b(posns,velocities,N_nodes,cupy_elements,kappa,cupy_springs,stressed_boundary,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary,starting_magnetization=None):
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
    if type(stressed_boundary) == type(list()):
        size_boundary = stressed_boundary[0].shape[0]
        special_stress_flag = True
    else:
        size_boundary = stressed_boundary.shape[0]
        special_stress_flag = False
    if size_boundary != 0 and stress_node_force != 0:
        boundary_stress_grid_size = (int (np.ceil((int (np.ceil(size_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        if special_stress_flag:
            if len(stressed_boundary) == 2:
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[0],stress_direction,stress_node_force,cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[1],stress_direction,np.float32(-1*stress_node_force),cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()
            elif len(stressed_boundary) == 4:
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[0],stress_direction[0],stress_node_force[0],cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[1],stress_direction[0],np.float32(-1*stress_node_force[0]),cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()
                size_boundary = stressed_boundary[2].shape[0]
                boundary_stress_grid_size = (int (np.ceil((int (np.ceil(size_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[2],stress_direction[1],stress_node_force[1],cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()
                boundary_stress_kernel((boundary_stress_grid_size,),(block_size,),(stressed_boundary[3],stress_direction[1],np.float32(-1*stress_node_force[1]),cupy_composite_forces,size_boundary))
                cupy_stream.synchronize()

        else:
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
        net_force = cp.asarray([cp.sum(cp.take(cupy_composite_forces,3*moving_boundary)),cp.sum(cp.take(cupy_composite_forces,3*moving_boundary+1)),cp.sum(cp.take(cupy_composite_forces,3*moving_boundary+2))],dtype=cp.float32,order='C')
        individual_force = net_force/size_strained_boundary
        # host_forces = cp.asnumpy(cupy_composite_forces).reshape((int(N_nodes),3))
        # net_force = np.sum(host_forces[host_moving_boundary],axis=0)
        # individual_force = cp.asarray(net_force/size_strained_boundary,dtype=cp.float32,order='C')
        
        # net_force_agreement_check = np.allclose(cp.asnumpy(cp_net_force),net_force)
        # # print(f'Do the net forces agree?:{net_force_agreement_check}')
        # if not net_force_agreement_check:
        #     print(f'Forces did not agree, begin debugging!')

        strained_boundary_grid_size = (int (np.ceil((int (np.ceil(size_strained_boundary/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
        strained_boundary_kernel((strained_boundary_grid_size,),(block_size,),(moving_boundary,cupy_composite_forces,individual_force,size_strained_boundary))
        cupy_stream.synchronize()
    if num_particles != 0:
        magnetic_forces, normalized_magnetization = get_magnetic_forces_composite(Hext,num_particles,particle_posns,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,starting_magnetization)
        #need to assign magnetic forces. every node belonging to a given particle has the same force as returned for the particle
        cupy_composite_forces = distribute_magnetic_forces(particles,num_particles,magnetic_forces,cupy_composite_forces,num_streaming_multiprocessors,block_size)
    else:
        normalized_magnetization = None
    # nodes_per_particle = cp.int32(particles.shape[0]/num_particles)
    # num_particle_nodes = cp.int32(nodes_per_particle*num_particles)
    # mag_force_distribution_grid_size = (int (np.ceil((int (np.ceil(num_particle_nodes/block_size)))/num_streaming_multiprocessors)*num_streaming_multiprocessors))
    # distribute_magnetic_force_kernel((mag_force_distribution_grid_size,),(block_size,),(particles,magnetic_forces,cupy_composite_forces,nodes_per_particle,num_particle_nodes))
    cupy_stream.synchronize()
    return cupy_composite_forces, normalized_magnetization

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

def simulate_scaled_gpu_leapfrog_v3(posns,elements,host_particles,particles,boundaries,dimensions,springs,kappa,l_e,beta,beta_i,boundary_conditions,Hext,particle_radius,particle_volume,particle_mass,chi,Ms,drag,output_dir,max_integrations=10,max_integration_steps=200,tolerance=1e-4,step_size=1e-2,persistent_checkpointing_flag=False,starting_velocities=None,checkpoint_offset=0,sim_extend_flag=False,normalized_magnetization=None):
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
        stress_node_force = np.float32(np.squeeze(net_force_mag/stressed_nodes.shape[0]))
        moving_boundary_nodes = cp.array([],dtype=np.int32)
        host_moving_boundary_nodes = np.array([],dtype=np.int64)
        host_fixed_nodes = cp.asnumpy(fixed_nodes)
    if 'special_stress' in boundary_conditions[0]:
        #opposing surfaces' nodes need to have additional forces applied, no nodes are held fixed
        if boundary_conditions[1][0] == 'x':
            fixed_nodes = cp.array([],dtype=np.int32)
            stressed_nodes = [cp.asarray(boundaries['right'],dtype=cp.int32,order='C'),cp.asarray(boundaries['left'],dtype=cp.int32,order='C')]
            host_stressed_nodes = np.concatenate((boundaries['right'],boundaries['left']))
            relevant_dimension_indices = [1,2]
        elif boundary_conditions[1][0] == 'y':
            fixed_nodes = cp.array([],dtype=np.int32)
            stressed_nodes = [cp.asarray(boundaries['back'],dtype=cp.int32,order='C'),cp.asarray(boundaries['front'],dtype=cp.int32,order='C')]
            host_stressed_nodes = np.concatenate((boundaries['back'],boundaries['front']))
            relevant_dimension_indices = [0,2]
        elif boundary_conditions[1][0] == 'z':
            fixed_nodes = cp.array([],dtype=np.int32)
            stressed_nodes = [cp.asarray(boundaries['top'],dtype=cp.int32,order='C'),cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')]
            host_stressed_nodes = np.concatenate((boundaries['top'],boundaries['bot']))
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
        stress_node_force = np.float32(np.squeeze(net_force_mag/stressed_nodes[0].shape[0]))
        if 'shear' in boundary_conditions[0]:
            if boundary_conditions[1][1] == 'x':
                stressed_nodes.append(cp.asarray(boundaries['right'],dtype=cp.int32,order='C'))
                stressed_nodes.append(cp.asarray(boundaries['left'],dtype=cp.int32,order='C'))
                host_stressed_nodes = np.concatenate((host_stressed_nodes,boundaries['right'],boundaries['left']))
                relevant_dimension_indices = [1,2]
            elif boundary_conditions[1][1] == 'y':
                stressed_nodes.append(cp.asarray(boundaries['back'],dtype=cp.int32,order='C'))
                stressed_nodes.append(cp.asarray(boundaries['front'],dtype=cp.int32,order='C'))
                host_stressed_nodes = np.concatenate((host_stressed_nodes,boundaries['back'],boundaries['front']))
                relevant_dimension_indices = [0,2]
            elif boundary_conditions[1][1] == 'z':
                stressed_nodes.append(cp.asarray(boundaries['top'],dtype=cp.int32,order='C'))
                stressed_nodes.append(cp.asarray(boundaries['bot'],dtype=cp.int32,order='C'))
                host_stressed_nodes = np.concatenate((host_stressed_nodes,boundaries['top'],boundaries['bot']))
                relevant_dimension_indices = [0,1]
            stress_direction_char = boundary_conditions[1][0]
            if stress_direction_char == 'x':
                stress_direction = [stress_direction,0]
            elif stress_direction_char == 'y':
                stress_direction = [stress_direction,1]
            elif stress_direction_char == 'z':
                stress_direction = [stress_direction,2]
            stress = boundary_conditions[2]
            surface_area = dimensions[relevant_dimension_indices[0]]*dimensions[relevant_dimension_indices[1]]
            net_force_mag = stress*surface_area
            stress_node_force = [stress_node_force,np.float32(np.squeeze(net_force_mag/stressed_nodes[0].shape[0]))]
        moving_boundary_nodes = cp.array([],dtype=np.int32)
        host_moving_boundary_nodes = np.array([],dtype=np.int64)
        host_fixed_nodes = cp.asnumpy(fixed_nodes)
    elif 'strain' in boundary_conditions[0]:
        if 'special_shear' in boundary_conditions[0] and boundary_conditions[2] == 0:
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
        elif 'simple_shear' in boundary_conditions[0]:
            fixed_nodes = cp.asarray(np.concatenate((boundaries['left'],boundaries['right'],boundaries['front'],boundaries['back'],boundaries['top'],boundaries['bot'])),dtype=cp.int32,order='C')
            moving_boundary_nodes = cp.array([],dtype=np.int32)
            host_moving_boundary_nodes = np.array([],dtype=np.int64)
        elif boundary_conditions[2] == 0:
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
        host_fixed_nodes = cp.asnumpy(fixed_nodes)
        stress_direction = 0
        stress_node_force = 0
    elif boundary_conditions[0] == 'hysteresis':
        fixed_nodes = cp.asarray(boundaries['bot'],dtype=cp.int32,order='C')
        host_fixed_nodes = cp.asnumpy(fixed_nodes)
        stressed_nodes = cp.array([],dtype=np.int32)
        host_stressed_nodes = np.array([],dtype=np.int64)
        moving_boundary_nodes = cp.array([],dtype=np.int32)
        host_moving_boundary_nodes = np.array([],dtype=np.int64)
        stress_direction = 0
        stress_node_force = 0
    num_particles = host_particles.shape[0]
    if num_particles == 0:
        nodes_per_particle = 0
    else:
        nodes_per_particle = cp.int32(particles.shape[0]/num_particles)
    particle_posns = get_particle_posns_gpu(posns,particles,num_particles,nodes_per_particle)
    # particle_posns = get_particle_posns(host_particles,last_posns)
    # particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
    a_var, normalized_magnetization = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary_nodes,starting_magnetization=normalized_magnetization)

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
    # 2024-07-09 making a mask array to avoid including particle nodes when getting convergence criteria
    my_mask = np.ones((int(N_nodes),),dtype=np.bool8)
    my_mask[np.ravel(host_particles)] = False
    my_mask[host_fixed_nodes] = False

    i = 0
    previously_converged = False
    while i < max_integrations:
        print(f'starting integration run {i+1}')
        for j in range(max_integration_steps):
            leapfrog_update(posns,velocities,step_size,size_entries)
            particle_posns = get_particle_posns_gpu(posns,particles,num_particles,nodes_per_particle)
            # host_posns = cp.asnumpy(posns)
            # host_posns = np.reshape(host_posns,(N_nodes,3))
            # particle_posns = get_particle_posns(host_particles,host_posns)
            # particle_posns = cp.asarray(particle_posns.reshape((num_particles*3,1)),dtype=cp.float32,order='C')
            if np.mod(j+i*max_integration_steps,snapshot_stepsize) == 0:
                host_posns = cp.asnumpy(posns)
                host_posns = np.reshape(host_posns,(N_nodes,3))
                host_accel = cp.asnumpy(a_var)
                host_accel = np.reshape(host_accel,(N_nodes,3))
                
                if host_stressed_nodes.shape[0] !=0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[host_stressed_nodes],axis=0)
                elif host_moving_boundary_nodes.shape[0] != 0:
                    snapshot_boundary_posn[snapshot_count] = np.mean(host_posns[host_moving_boundary_nodes],axis=0)

                host_accel_norms = np.linalg.norm(host_accel[my_mask],axis=1)
                snapshot_accel_norm[snapshot_count] = np.mean(host_accel_norms)
                snapshot_accel_norm_std[snapshot_count] = np.std(host_accel_norms)
                snapshot_accel_components_avg[snapshot_count] = np.mean(host_accel,axis=0)

                host_velocities = cp.asnumpy(velocities)
                host_velocities = np.reshape(host_velocities,(N_nodes,3))
                host_vel_norms = np.linalg.norm(host_velocities[my_mask],axis=1)
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
            a_var, normalized_magnetization = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e,moving_boundary_nodes,starting_magnetization=normalized_magnetization)
            leapfrog_update(velocities,a_var,step_size,size_entries)
        host_posns = cp.asnumpy(posns)
        host_velocities = cp.asnumpy(velocities)
        sol = np.concatenate((host_posns,host_velocities))
        host_accel = cp.asnumpy(a_var)
        host_accel = np.reshape(host_accel,(N_nodes,3))
        host_velocities = np.reshape(host_velocities,(N_nodes,3))
        # #2024-04-04 trying out a thing where, despite not doing the vector summation and distribution of the forces acting on the particles for the node updates, I do it for determining the accelerations for convergence criteria, thereby ignoring internal forces for the particles
        # for particle in host_particles:
        #     host_accel[particle] = np.sum(host_accel[particle],axis=0)/particle.shape[0]
        #     host_velocities[particle] = np.sum(host_velocities[particle],axis=0)/particle.shape[0]
        #2024-07-09 trying to separate out the contribution of the particle nodes to the convergence criteria entirely. may use the particle net acceleration and velocity component magnitudes as a separate check, but i'll just print it out for now
        
        # prior approach to the convergence criteria
        # accel_comp_magnitude = np.abs(host_accel)
        # accel_comp_magnitude_avg = np.mean(accel_comp_magnitude)
        # vel_comp_magnitude = np.abs(host_velocities)
        # vel_comp_magnitude_avg = np.mean(vel_comp_magnitude)
        # a_norms = np.linalg.norm(host_accel,axis=1)
        # a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        accel_comp_magnitude = np.abs(host_accel)
        accel_comp_magnitude_avg = np.mean(accel_comp_magnitude[my_mask])
        vel_comp_magnitude = np.abs(host_velocities)
        vel_comp_magnitude_avg = np.mean(vel_comp_magnitude[my_mask])
        if num_particles != 0:
            # particle_net_accel = np.sum(host_accel[np.ravel(host_particles),:],axis=0)/host_particles[0].shape
            particle_accel_comp_magnitude_avg = np.mean(accel_comp_magnitude[np.ravel(host_particles)])
            # particle_net_vel = np.sum(host_velocities[np.ravel(host_particles),:],axis=0)/host_particles[0].shape
            particle_vel_comp_magnitude_avg = np.mean(vel_comp_magnitude[np.ravel(host_particles)])
        else:
            particle_accel_comp_magnitude_avg = np.nan
            particle_vel_comp_magnitude_avg = np.nan

        a_norms = np.linalg.norm(host_accel[my_mask],axis=1)
        a_norm_avg = np.sum(a_norms)/np.shape(a_norms)[0]
        host_posns = np.reshape(host_posns,(N_nodes,3))
        final_posns = host_posns
        
        final_v = host_velocities
        # v_norms = np.linalg.norm(final_v,axis=1)
        # v_norm_avg = np.sum(v_norms)/N_nodes
        v_norms = np.linalg.norm(final_v[my_mask],axis=1)
        v_norm_avg = np.sum(v_norms)/np.shape(v_norms)[0]
        if False and host_stressed_nodes.shape[0] != 0:
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
            print(f'Particle acceleration component magnitude average: {np.round(particle_accel_comp_magnitude_avg,decimals=6)}\n particle velocity component magnitude average: {np.round(particle_vel_comp_magnitude_avg,decimals=6)}\n')
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
            print(f'Particle acceleration component magnitude average: {np.round(particle_accel_comp_magnitude_avg,decimals=6)}\n particle velocity component magnitude average: {np.round(particle_vel_comp_magnitude_avg,decimals=6)}\n')
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
            print(f'particle velocity = {np.round(particle_velocity,decimals=6)}\n')
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
    host_normalized_magnetization = cp.asnumpy(normalized_magnetization)
    return sol, host_normalized_magnetization, return_status#returning a solution object, that can then have it's attributes inspected

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
    a_var, _ = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
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
            a_var, _ = composite_gpu_force_calc_v3b(posns,velocities,N_nodes,elements,kappa,springs,stressed_nodes,stress_direction,stress_node_force,beta_i,drag,fixed_nodes,particles,particle_posns,num_particles,Hext,Ms,chi,particle_radius,particle_volume,beta,particle_mass,l_e)
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