#2023-09-13
#David Marchfield
#Visualize the end configuration of a two particle simulation with clustering, where the solution did not meet convergence criteria
#determine if any nodes are interpenetrating volumes they should not

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
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
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

def main():
    drag = 10
    discretization_order = 1

    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-09-13_results_order_{discretization_order}_drag_{drag}/'

    initial_node_posns, _, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(output_dir+'init.h5')
    final_posns, boundary_conditions, _, sim_time = mre.initialize.read_output_file(output_dir+'output_0.h5')

    plot_regular_spacing_volumetric(initial_node_posns,final_posns,springs_var,particles,output_dir)
    plot_regular_cuts_volumetric(initial_node_posns,final_posns,springs_var,particles,output_dir)
    #find the interior region including the particles, for plotting
    my_max = np.zeros((3,))
    my_min = np.zeros((3,))
    tmp_max = np.zeros((3,))
    tmp_min = np.zeros((3,))
    #iterate through the particles, getting the positions of the nodes making up the particles, and then find the maximum and minimum extent in each direction. Will expand a bit past those for visualization
    for particle in particles:
        tmp_node_posns = initial_node_posns[particle,:]
        for i in range(3):
            tmp_max[i] = np.max(tmp_node_posns[:,i])
            tmp_min[i] = np.min(tmp_node_posns[:,i])
            if tmp_max[i] > my_max[i]:
                my_max[i] = tmp_max[i]
            if tmp_min[i] < my_min[i]:
                my_min[i] = tmp_min[i]
    #do "cuts" of sorts on the final_posns varibale to get only the nodes that are within the region of interest
    bool_array = np.zeros((final_posns.shape[0],3),dtype=np.bool)
    for i in range(3):
        bool_array[:,i] = np.logical_and(initial_node_posns[:,i] >= my_min[i] - 2,initial_node_posns[:,i] <= my_max[i] + 2)
        bool_array[:,i] = np.logical_and(final_posns[:,i] >= my_min[i] - 2,final_posns[:,i] <= my_max[i] + 2)#This was attempting to grab the volume immediately around and between the clustered particles. the visualizationdidn't work very well for me
    final_bool_array = np.logical_and(np.logical_and(bool_array[:,0],bool_array[:,1]),bool_array[:,2])
    chosen_nodes = np.nonzero(final_bool_array)[0]
    tmp_subset_nodes = final_posns[final_bool_array,:]
    # plot_clustering_visualization(initial_node_posns,final_posns,chosen_nodes,springs_var,particles,output_dir)

def plot_clustering_visualization(eq_node_posns,final_node_posns,chosen_nodes,springs,particles,output_dir):
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(final_node_posns[chosen_nodes,0],final_node_posns[chosen_nodes,1],final_node_posns[chosen_nodes,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,chosen_nodes,springs,spring_color='b')
    cut_nodes_set = set(chosen_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r')
    ax.set_xlim((-0.3,1.2*Lx))
    ax.set_ylim((0,1.2*Ly))
    ax.set_zlim((0,1.2*Lz))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.axis('equal')
    plt.show()
    savename = output_dir + f'cluster_volumetric_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_regular_cuts_volumetric(eq_node_posns,final_node_posns,springs,particles,output_dir):
    # cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    # cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    #number of cuts to plot (trying to get some sense of the volume without plotting everything)
    N_cuts = 10
    step_size = int(Lx)//int(N_cuts)
    gets_both_ends = np.mod(Lx,N_cuts)
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot(projection= '3d')
    cut_type_index = 2
    for i in range(N_cuts):
        cut_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i,eq_node_posns[:,cut_type_index]).nonzero()[0]
        ax.scatter(final_node_posns[cut_nodes,0],final_node_posns[cut_nodes,1],final_node_posns[cut_nodes,2],color ='b',marker='o')
        mre.analyze.plot_subset_springs(ax,final_node_posns,cut_nodes,springs,spring_color='b')
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r')
    ax.set_xlim((-0.3,1.2*Lx))
    ax.set_ylim((0,1.2*Ly))
    ax.set_zlim((0,1.2*Lz))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.axis('equal')
    plt.show()
    savename = output_dir + f'series_cuts_cluster_volumetric_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_regular_spacing_volumetric(eq_node_posns,final_node_posns,springs,particles,output_dir):
    # cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    # cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    #number of cuts to plot (trying to get some sense of the volume without plotting everything)
    N_cuts = 5
    step_size = int(Lx)//int(N_cuts)
    gets_both_ends = np.mod(Lx,N_cuts)
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot(projection= '3d')
    cut_type_index = 2
    cut_nodes_bool = np.zeros((eq_node_posns.shape[0],),dtype=np.bool)
    for i in range(N_cuts):
        x_cut = np.logical_or(np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i,eq_node_posns[:,0]),np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i+1,eq_node_posns[:,0]))
        for j in range(N_cuts):
            y_cut = np.logical_or(np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*j,eq_node_posns[:,1]),np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*j+1,eq_node_posns[:,1]))
            for k in range(N_cuts):
                z_cut = np.logical_or(np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*k,eq_node_posns[:,2]),np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*k+1,eq_node_posns[:,2]))
                cut_nodes_bool = np.logical_or(np.logical_and(x_cut,np.logical_and(y_cut,z_cut)),cut_nodes_bool)
            # cut_nodes = np.isclose(np.ones((eq_node_posns[y_cut,:].shape[0],))*step_size*i,eq_node_posns[y_cut,1]).nonzero()[0]
            # cut_nodes = np.logical_and(np.logical_and(np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i,eq_node_posns[:,0]),np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i,eq_node_posns[:,1])),np.isclose(np.ones((eq_node_posns.shape[0],))*step_size*i,eq_node_posns[:,2])).nonzero()[0]
    cut_nodes = cut_nodes_bool.nonzero()[0]
    ax.scatter(final_node_posns[cut_nodes,0],final_node_posns[cut_nodes,1],final_node_posns[cut_nodes,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,cut_nodes,springs,spring_color='b')
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r')
    ax.set_xlim((-0.3,1.2*Lx))
    ax.set_ylim((0,1.2*Ly))
    ax.set_zlim((0,1.2*Lz))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.axis('equal')
    plt.show()
    savename = output_dir + f'regular_spacing_cluster_volumetric_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_cut_normalized(cut_type,eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = (np.round(np.array([Lx,Ly,Lz]))/2)
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot(projection= '3d')
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        # cut_nodes1 = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        # cut_nodes2 =np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]-1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        # cut_nodes = np.concatenate((cut_nodes1,cut_nodes2))
    ax.scatter(node_posns[cut_nodes,0],node_posns[cut_nodes,1],node_posns[cut_nodes,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,node_posns,cut_nodes,springs,spring_color='b')
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(node_posns[particle_cut_nodes,0],node_posns[particle_cut_nodes,1],node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,node_posns,particle_cut_nodes_set,springs,spring_color='r')
    ax.set_xlim((-0.3,1.2*Lx))
    ax.set_ylim((0,1.2*Ly))
    ax.set_zlim((0,1.2*Lz))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    if cut_type_index == 0:
        ax.view_init(elev=0,azim=0,roll=0)
    elif cut_type_index == 1:
        ax.view_init(elev=0,azim=-90,roll=0)
    else:
        ax.view_init(elev=90,azim=-90,roll=0)
    ax.axis('equal')
    # ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'post_plot_cut_{cut_type}_{center[cut_type_index]}' + str(np.round(boundary_conditions[2],decimals=2)) + tag +'.png'
    plt.savefig(savename)
    plt.close()

if __name__ == "__main__":
    main()