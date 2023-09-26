#2023-09-13
#David Marchfield
#Visualize the end configuration of a two particle simulation with clustering, where the solution did not meet convergence criteria
#determine if any nodes are interpenetrating volumes they should not

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import get_volume_correction_force_cy_nogil
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

def plot_element(node_posns,element,springs):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(node_posns[element,0],node_posns[element,1],node_posns[element,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,node_posns,element,springs,spring_color='b')
    plt.show()

def main():
    drag = 10
    discretization_order = 1

    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-09-13_results_order_{discretization_order}_drag_{drag}/'

    initial_node_posns, _, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(output_dir+'init.h5')
    final_posns, boundary_conditions, _, sim_time = mre.initialize.read_output_file(output_dir+'output_0.h5')

    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    avg_edge_vectors = get_volume_correction_force_cy_nogil.get_avg_edge_vectors_normalized(final_posns,elements,vectors,avg_vectors)
    approx_element_volumes = get_volume_correction_force_cy_nogil.get_element_volume_normalized(avg_edge_vectors)
    avg_edge_vector_norms = np.linalg.norm(avg_edge_vectors,axis=2)
    # fig, ax = plt.subplots()
    # counts, bins = np.histogram(approx_element_volumes,100)
    # plt.hist(bins[:-1],bins,weights=counts)
    # ax.set_title('Distribution of approximate element volumes')
    # plt.show()
    # fig, ax = plt.subplots()
    # counts, bins = np.histogram(np.ravel(avg_edge_vector_norms),100)
    # plt.hist(bins[:-1],bins,weights=counts)
    # ax.set_title('Distribution of average element edge norms')
    # plt.show()
    strongly_deformed_count = 0
    for i in range(elements.shape[0]):
        if np.abs(avg_edge_vector_norms[i,0]-1) > 0.2 or np.abs(avg_edge_vector_norms[i,1]-1) > 0.2 or np.abs(avg_edge_vector_norms[i,2]-1) > 0.2:
            strongly_deformed_count += 1
            # plot_element(final_posns,elements[i],springs_var)
    stiffness = params[0][-2]
    cut_type = 'yz'
    index = 14
    # plot_scatter_color_depth_visualization(initial_node_posns,final_posns,cut_type,index,springs_var,particles,output_dir,spring_type=stiffness[0],tag="")
    fig, ax = plt.subplots()
    scatter_interactive = ScatterIndexTracker(fig,ax,cut_type,initial_node_posns,final_posns,springs_var,particles,spring_type=stiffness[0])
    fig.canvas.mpl_connect('scroll_event', scatter_interactive.on_scroll)
    plt.show()

    # An attempt to pull out the parameter entries i want without counting which element it is. there are better ways to handle this, like an additional function that takes the params variable and returns the variables in params with appropriate names
    # print(params.dtype)
    # params_entries = params.dtype
    # for i in range(len(params_entries)):
    #     if params_entries.descr[i][0] == 'num_elements':
    #         N_elements = params[0][i]
    fig, ax = plt.subplots()
    cut_type = 'xz'
    #create an IndexTracker and make sure it lives during the whole lifetime of the figure by assignign it to a variable
    tracker = IndexTracker(fig,ax,cut_type,initial_node_posns,final_posns)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

    plot_outer_surfaces_contours(initial_node_posns,final_posns,boundaries,output_dir)
    plot_outer_surfaces(initial_node_posns,final_posns,springs_var,boundaries,output_dir,spring_type=stiffness[0])

    plot_regular_spacing_volumetric(initial_node_posns,final_posns,springs_var,particles,output_dir)
    plot_regular_cuts_volumetric(initial_node_posns,final_posns,springs_var,particles,output_dir,spring_type=stiffness[0])
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
    """Plot a set of chosen nodes, passed as a list of integers representing the row index of the node in final_node_posns"""
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

def plot_scatter_color_depth_visualization(eq_node_posns,final_node_posns,cut_type,index,springs,particles,output_dir,spring_type=None,tag=""):
    """Plot a set of chosen nodes, passed as a list of integers representing the row index of the node in final_node_posns"""
    #TODO add spring type to plot_subset_springs, and to function arguments. add argument or somehow get the type of cut, to properly label the axes on the image. use the "depth" (odd node position out) to give a list of numbers to be mapped to colors using cmap and norm (which means selecting or else using the default cmap (and most likly letting norm do it's default), and setting the colorbar)
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    chosen_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot()
    depth_color = final_node_posns[chosen_nodes,cut_type_index]
    if cut_type_index == 0:
        xvar = final_node_posns[chosen_nodes,1]
        yvar = final_node_posns[chosen_nodes,2]
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
    elif cut_type_index == 1:
        xvar = final_node_posns[chosen_nodes,0]
        yvar = final_node_posns[chosen_nodes,2]
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
    else:
        xvar = final_node_posns[chosen_nodes,0]
        yvar = final_node_posns[chosen_nodes,1]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
    sc = ax.scatter(xvar,yvar,c=depth_color,marker='o')
    plt.colorbar(sc)
    # ax.scatter(final_node_posns[chosen_nodes,0],final_node_posns[chosen_nodes,1],final_node_posns[chosen_nodes,2],color ='b',marker='o')
    #TODO add a colorscheme for springs that reflects the displacement from equilibrium length
    # mre.analyze.plot_subset_springs(ax,final_node_posns,chosen_nodes,springs,spring_color='b',spring_type=spring_type)
    cut_nodes_set = set(chosen_nodes)
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    if cut_type_index == 0:
        xvar = final_node_posns[particle_cut_nodes,1]
        yvar = final_node_posns[particle_cut_nodes,2]
    elif cut_type_index == 1:
        xvar = final_node_posns[particle_cut_nodes,0]
        yvar = final_node_posns[particle_cut_nodes,2]
    else:
        xvar = final_node_posns[particle_cut_nodes,0]
        yvar = final_node_posns[particle_cut_nodes,1]
    ax.scatter(xvar,yvar,color='k',marker='o')
    # ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
    ax.set_xlim((-0.3,1.2*Lx))
    ax.set_ylim((0,1.2*Ly))
    # ax.set_zlim((0,1.2*Lz))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_zlabel('Z (m)')
    ax.axis('equal')
    plt.show()
    savename = output_dir + f'scatter_color_depth_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_regular_cuts_volumetric(eq_node_posns,final_node_posns,springs,particles,output_dir,spring_type=None):
    """On a single figure, plot multiple regularly spaced cuts through the volume (currently cuts of constant z value)"""
    # cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    # cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    #number of cuts to plot (trying to get some sense of the volume without plotting everything)
    N_cuts = 10
    step_size = int(Lx)//int(N_cuts)
    #TODO use gets_both_ends to determine if both boundary surfaces are plotted, and plot the missing surface if it would be skipped by regular spacing
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
        mre.analyze.plot_subset_springs(ax,final_node_posns,cut_nodes,springs,spring_color='b',spring_type=spring_type)
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
    mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
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
    """Plot regularly spaced volume elements in 3D, not as cuts"""
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

#plot all 3D view planes to show each outer surface of the simulated volume
def plot_outer_surfaces(eq_node_posns,node_posns,springs,boundaries,output_dir,spring_type=None):
    """Plot the outer surfaces of the simulated volume"""
    # (plane, (elevation, azimuthal, roll))
    views = [('top', ( 90,-90, 0)),
            ('front', (  0,-90, 0)),
            ('left', (  0,  0, 0)),
            ('bot',(-90, 90, 0)),
            ('back',(  0, 90, 0)),
            ('right',(  0,180, 0))]
    # I like this approach, but i can't see the plots well enough with so many of them. the sizing just doesn't match what i need, unsurprisingly. maybe the bigger issue is that scatter is just not the rightway to handle this. i am going to try doing some surface plots.
    # fig, axs = plt.subplots(2,3,subplot_kw=dict(projection='3d'))
    # default_width,default_height = fig.get_size_inches()
    # fig.set_size_inches(3*default_width,3*default_height)
    # fig.set_dpi(200)
    # for index in range(len(views)):
    #     view, angles = views[index]
    #     row = int(np.mod(index,2))
    #     col = int(np.mod(index,3))
    #     ax = axs[row,col]
    #     ax.view_init(elev=angles[0],azim=angles[1],roll=angles[2])
    #     posns = node_posns[boundaries[view]]
    #     ax.scatter(posns[:,0],posns[:,1],posns[:,2],color='b',marker='o')
    #     mre.analyze.plot_subset_springs(ax,node_posns,boundaries[view],springs,spring_color='b',spring_type=spring_type)
    #     # ax.axis('equal')
    #     ax.set_title(view)

    # Attempt 2, using surface plots. (attempt 3 might use scatter or surface, but more importantly, i should try doing side by sides of just two of the boundaries. or maybe go back to one)
    fig, axs = plt.subplots(2,3)
    # fig, axs = plt.subplots(2,3,subplot_kw=dict(projection='3d'))
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    for index in range(len(views)):
        view, angles = views[index]
        row = int(np.mod(index,2))
        col = int(np.mod(index,3))
        ax = axs[row,col]
        # ax.view_init(elev=angles[0],azim=angles[1],roll=angles[2])
        posns = node_posns[boundaries[view]]
        eq_posns = eq_node_posns[boundaries[view]]
        if view == 'top' or view == 'bot':
            z = posns[:,2]
            x = posns[:,0]
            y = posns[:,1]
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,1]) + 1)
        elif view == 'front' or view == 'back':
            z = posns[:,1]
            x = posns[:,0]
            y = posns[:,2]
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
        else:
            z = posns[:,0]
            x = posns[:,1]
            y = posns[:,2]
            Nx = int(np.max(eq_posns[:,1]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
        #need to convert from 1D vectors to 2D matrices
        #coordinates of the corners of quadrilaterlas of a pcolormesh:
        #(X[i+1,j],Y[i+1,j])       (X[i+1,j+1],Y[i+1,j+1])
        #              *----------*
        #              |          |
        #              *----------*
        #(X[i,j],Y[i,j])           (X[i,j+1],Y[i,j+1])
        sort_indices = np.argsort(x)
        sorted_x = x[sort_indices]
        sorted_y = y[sort_indices]
        sorted_z = z[sort_indices]
        X = np.zeros((Ny,Nx))
        Y = np.zeros((Ny,Nx))
        Z = np.zeros((Ny,Nx))
        start = 0
        end = Ny
        for i in range(Nx):
            X[:,i] = sorted_x[start:(i+1)*end]
            Y[:,i] = sorted_y[start:(i+1)*end]
            Z[:,i] = sorted_z[start:(i+1)*end]
            start = (i+1)*end
        #first sorting and then distributing values has the X array behaving as necessary, with the incrememnt in the column index resulting in an increase in the x position, but I also want the Y array to have an increment in the row index result in an increase in the y position. I need to do more sorting, but without destroying the sorted nature of the X array, and I need to use argsort to sort the X and Z arrays to match the sorted Y array. Within each column of Y, I need to sort from lowest to highest. but can i also just use the sorting of the first column for all the row rearrangement?
        # sorted_indices = np.argsort(Y[:,0])
        for i in range(Nx):
            sorted_indices = np.argsort(Y[:,i])
            X[:,i] = X[sorted_indices,i]
            Y[:,i] = Y[sorted_indices,i]
            Z[:,i] = Z[sorted_indices,i]
        #SANITY CHECK: because i know how the corners need to be arranged, I can check to make sure they are arranged correctly
        for i in range(Ny-1):
            for j in range(Nx-1):
                assert X[i,j] < X[i,j+1] and X[i+1,j] < X[i+1,j+1] and Y[i,j] < Y[i+1,j] and Y[i,j+1] < Y[i+1,j+1], 'Quadrilateral corners are out of order from pmeshcolor expectations, image will draw incorrectly'
        img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        # ax.scatter(posns[:,0],posns[:,1],posns[:,2],color='b',marker='o')
        # mre.analyze.plot_subset_springs(ax,node_posns,boundaries[view],springs,spring_color='b',spring_type=spring_type)
        ax.axis('equal')
        ax.set_title(view)
    # plt.show()
    savename = output_dir + f'surface_pcolormesh_visualization.png'
    plt.savefig(savename)
    # plt.close()

def plot_outer_surfaces_contours(eq_node_posns,node_posns,boundaries,output_dir):
    """plot the outer surfaces of the simulated volume, using contours (tricontourf())"""
    # (plane, (elevation, azimuthal, roll))
    views = [('top', ( 90,-90, 0)),
            ('front', (  0,-90, 0)),
            ('left', (  0,  0, 0)),
            ('bot',(-90, 90, 0)),
            ('back',(  0, 90, 0)),
            ('right',(  0,180, 0))]
    fig, axs = plt.subplots(2,3)
    # fig, axs = plt.subplots(2,3,subplot_kw=dict(projection='3d'))
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    for index in range(len(views)):
        view, angles = views[index]
        row = int(np.mod(index,2))
        col = int(np.mod(index,3))
        ax = axs[row,col]
        # ax.view_init(elev=angles[0],azim=angles[1],roll=angles[2])
        posns = node_posns[boundaries[view]]
        eq_posns = eq_node_posns[boundaries[view]]
        if view == 'top' or view == 'bot':
            z = posns[:,2]
            x = posns[:,0]
            y = posns[:,1]
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,1]) + 1)
            xmax = np.max(eq_posns[:,0])
            xmin = -0.1
            ymax = np.max(eq_posns[:,1])
            ymin = -0.1
        elif view == 'front' or view == 'back':
            z = posns[:,1]
            x = posns[:,0]
            y = posns[:,2]
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
            xmax = np.max(eq_posns[:,0])
            xmin = -0.1
            ymax = np.max(eq_posns[:,2])
            ymin = -0.1
        else:
            z = posns[:,0]
            x = posns[:,1]
            y = posns[:,2]
            Nx = int(np.max(eq_posns[:,1]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
            xmax = np.max(eq_posns[:,1])
            xmin = -0.1
            ymax = np.max(eq_posns[:,2])
            ymin = -0.1
        img = ax.tricontourf(x,y,z,levels=10)
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))
        ax.set_title(view)
    fig.tight_layout()
    savename = output_dir + f'surface_tricontourf_visualization.png'
    plt.savefig(savename)
    return 0

#copied and modified from matplotlib gallery on event handling, image slices viewer
# class IndexTracker:
#     def __init__(self,ax, X):
#         self.index = 0
#         self.X = X
#         self.ax = ax
#         self.im = ax.imshow(self.X[:,:, self.index])
#         self.update()
    
#     def on_scroll(self, event):
#         print(event.button, event.step)
#         increment = 1 if event.button == 'up' else -1
#         max_index = self.X.shape[-1] -1
#         self.index = np.clip(self.index + increment, 0, max_index)
#         self.update()

#     def update(self):
#         self.im.set_data(self.X[:,:,self.index])
#         self.ax.set_title(f'Use scroll whell to navigate\nindex {self.index}')
#         self.im.axes.figure.canvas.draw()

# x, y, z = np.ogrid[-1:10:100j, -10:10:100j, 1:10:20j]
# X = np.sin( x * y * z) / (x*y*z)
# fig, ax = plt.subplots()
# #create an IndexTracker and make sure it lives during the whole lifetime of the figure by assignign it to a variable
# tracker = IndexTracker(ax,X)

# fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
# plt.show()
class ScatterIndexTracker:
    def __init__(self,fig,ax,cut_type,eq_node_posns,node_posns,springs,particles,spring_type=None):
        self.index = 0
        cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
        cut_type_index = cut_type_dict[cut_type]
        self.max_index = int(np.max(eq_node_posns[:,cut_type_index]))
        self.fig = fig
        self.ax = ax
        self.cut_type = cut_type
        self.eq_node_posns = eq_node_posns
        self.node_posns = node_posns
        self.springs = springs
        self.particles = particles
        self.spring_type = spring_type
        self.im = self.plot_scatter_color_depth_visualization(self.fig,self.ax,self.cut_type,self.eq_node_posns,self.node_posns,self.index,self.springs,self.particles,self.spring_type)
        self.cbar = self.fig.colorbar(self.im,ax=self.ax)
        self.update()
    
    def on_scroll(self, event):
        print(event.button, event.step)
        increment = 1 if event.button == 'up' else -1
        max_index = self.max_index
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        plt.cla()
        self.im = self.plot_scatter_color_depth_visualization(self.fig,self.ax,self.cut_type,self.eq_node_posns,self.node_posns,self.index,self.springs,self.particles,self.spring_type)
        try:
            self.cbar.update_normal(self.im)
        except:
            print('No existing colorbar to update')
        self.ax.set_title(f'Use scroll whell to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()
    
    def plot_scatter_color_depth_visualization(self,fig,ax,cut_type,eq_node_posns,final_node_posns,index,springs,particles,spring_type=None,tag=""):
        """Plot a set of chosen nodes, passed as a list of integers representing the row index of the node in final_node_posns"""
        #TODO add spring type to plot_subset_springs, and to function arguments. add argument or somehow get the type of cut, to properly label the axes on the image. use the "depth" (odd node position out) to give a list of numbers to be mapped to colors using cmap and norm (which means selecting or else using the default cmap (and most likly letting norm do it's default), and setting the colorbar)
        cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
        cut_type_index = cut_type_dict[cut_type]
        chosen_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
        Lx = eq_node_posns[:,0].max()
        Ly = eq_node_posns[:,1].max()
        Lz = eq_node_posns[:,2].max()
        # default_width,default_height = fig.get_size_inches()
        # fig.set_size_inches(3*default_width,3*default_height)
        # fig.set_dpi(200)
        depth_color = final_node_posns[chosen_nodes,cut_type_index]
        if cut_type_index == 0:
            xvar = final_node_posns[chosen_nodes,1]
            yvar = final_node_posns[chosen_nodes,2]
            xlabel = 'Y (l_e)'
            ylabel = 'Z (l_e)'
            xlim = (-0.1,Ly*1.1)
            ylim = (-0.1,Lz*1.1)
        elif cut_type_index == 1:
            xvar = final_node_posns[chosen_nodes,0]
            yvar = final_node_posns[chosen_nodes,2]
            xlabel = 'X (l_e)'
            ylabel = 'Z (l_e)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Lz*1.1)
        else:
            xvar = final_node_posns[chosen_nodes,0]
            yvar = final_node_posns[chosen_nodes,1]
            xlabel = 'X (l_e)'
            ylabel = 'Y (l_e)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Ly*1.1)
        sc = ax.scatter(xvar,yvar,c=depth_color,marker='o')
        # plt.colorbar(sc)
        # ax.scatter(final_node_posns[chosen_nodes,0],final_node_posns[chosen_nodes,1],final_node_posns[chosen_nodes,2],color ='b',marker='o')
        #TODO add a colorscheme for springs that reflects the displacement from equilibrium length
        # mre.analyze.plot_subset_springs(ax,final_node_posns,chosen_nodes,springs,spring_color='b',spring_type=spring_type)
        cut_nodes_set = set(chosen_nodes)
        particle_nodes_set = set(particles.ravel())
        particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
        #alternative to below, use set unpacking
        #*particle_cut_nodes, = particle_cut_nodes_set
        particle_cut_nodes = [x for x in particle_cut_nodes_set]
        if cut_type_index == 0:
            xvar = final_node_posns[particle_cut_nodes,1]
            yvar = final_node_posns[particle_cut_nodes,2]
        elif cut_type_index == 1:
            xvar = final_node_posns[particle_cut_nodes,0]
            yvar = final_node_posns[particle_cut_nodes,2]
        else:
            xvar = final_node_posns[particle_cut_nodes,0]
            yvar = final_node_posns[particle_cut_nodes,1]
        ax.scatter(xvar,yvar,color='k',marker='o')
        # ax.scatter(final_node_posns[particle_cut_nodes,0],final_node_posns[particle_cut_nodes,1],final_node_posns[particle_cut_nodes,2],color='k',marker='o')
        mre.analyze.plot_subset_springs(ax,final_node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_zlim((0,1.2*Lz))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_zlabel('Z (m)')
        ax.axis('equal')
        return sc

class IndexTracker:
    def __init__(self,fig,ax,cut_type,eq_node_posns,node_posns):
        self.index = 0
        cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
        cut_type_index = cut_type_dict[cut_type]
        self.max_index = int(np.max(eq_node_posns[:,cut_type_index]))
        self.fig = fig
        self.ax = ax
        self.cut_type = cut_type
        self.eq_node_posns = eq_node_posns
        self.node_posns = node_posns
        self.im = self.plot_cut_pcolormesh(self.fig,self.ax,self.cut_type,self.eq_node_posns,self.node_posns,self.index)
        self.cbar = self.fig.colorbar(self.im,ax=self.ax)
        self.update()
    
    def on_scroll(self, event):
        print(event.button, event.step)
        increment = 1 if event.button == 'up' else -1
        max_index = self.max_index
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        plt.cla()
        self.im = self.plot_cut_pcolormesh(self.fig,self.ax,self.cut_type,self.eq_node_posns,self.node_posns,self.index)
        try:
            self.cbar.update_normal(self.im)
        except:
            print('No existing colorbar to update')
        self.ax.set_title(f'Use scroll whell to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()

    def plot_cut_pcolormesh(self,fig,ax,cut_type,eq_node_posns,node_posns,index):
        """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
        
        cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
        
        tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
        cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
        cut_type_index = cut_type_dict[cut_type]
        Lx = eq_node_posns[:,0].max()
        Ly = eq_node_posns[:,1].max()
        Lz = eq_node_posns[:,2].max()
        # fig, ax = plt.subplots()
        # default_width,default_height = fig.get_size_inches()
        # fig.set_size_inches(3*default_width,3*default_height)
        # fig.set_dpi(200)
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
        posns = node_posns[cut_nodes]
        eq_posns = eq_node_posns[cut_nodes]
        # i need to sort things properly, so that the nodes defining the quadrilateral faces are in an appropriate order for pcolormesh, BUT I also need to ensure that those quadrilateral faces still reflect the initial node configurations (the faces are the faces of the cubic volume elements). if some sort of inversion or flipping of the node positions has happened in the simulation, i want to know about it. I know something about the way the nodes are initialized in position (first incrementing in z position until the maximum z value, then x, finally y. think about how an odometer works with rolling over). in that way, the xz plane is a good place to start. 
        nodes_per_row = int(Lx + 1)
        nodes_per_column = int(Lz + 1)
        nodes_per_plane = nodes_per_row*nodes_per_column
        #the first xz plane is the nodes 0 to nodes_per_plane -1, then nodes_per_plane to 2*nodes_per_plane -1, and so on
        # for reshaping, the "X" array should increment in "X" value as the column index increases, "Y" array as the row index increases. for xz planes, x -> "X", z -> "Y" is the mapping. the first column of the "X" and "Y" arrays should have a constant X value, so they belong to the nodes 0 to nodes_per_col-1, and so on for the columns until the thing is filled...  
        if cut_type == 'xy':
            z = posns[:,2]
            x = posns[:,0]
            y = posns[:,1]
            init_x = eq_posns[:,0]
            init_y = eq_posns[:,1]
            comparison_sort_indices = np.argsort(init_x)
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,1]) + 1)
        elif cut_type == 'xz':
            Nx = int(np.max(eq_posns[:,0]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
            X = np.zeros((Ny,Nx))
            Y = np.zeros((Ny,Nx))
            Z = np.zeros((Ny,Nx))
            start_index = index*nodes_per_plane
            for i in range(Nx):
                #for the moment, doing the below but with the initialized node positions to see if the layers that won't plot with the final node positions will plot
                X[:,i] = eq_node_posns[start_index+i*Ny:start_index+(i+1)*Ny,0]
                Y[:,i] = eq_node_posns[start_index+i*Ny:start_index+(i+1)*Ny,2]
                Z[:,i] = eq_node_posns[start_index+i*Ny:start_index+(i+1)*Ny,1]
                # X[:,i] = node_posns[start_index+i*Ny:start_index+(i+1)*Ny,0]
                # Y[:,i] = node_posns[start_index+i*Ny:start_index+(i+1)*Ny,2]
                # Z[:,i] = node_posns[start_index+i*Ny:start_index+(i+1)*Ny,1]
            # z = posns[:,1]
            # x = posns[:,0]
            # y = posns[:,2]
            # init_x = eq_posns[:,0]
            # init_y = eq_posns[:,2]
            # comparison_sort_indices = np.argsort(init_x)
            # Nx = int(np.max(eq_posns[:,0]) + 1)
            # Ny = int(np.max(eq_posns[:,2]) + 1)
        else:
            z = posns[:,0]
            x = posns[:,1]
            y = posns[:,2]
            init_x = eq_posns[:,1]
            init_y = eq_posns[:,2]
            comparison_sort_indices = np.argsort(init_x)
            Nx = int(np.max(eq_posns[:,1]) + 1)
            Ny = int(np.max(eq_posns[:,2]) + 1)
        #need to convert from 1D vectors to 2D matrices
        #coordinates of the corners of quadrilaterlas of a pcolormesh:
        #(X[i+1,j],Y[i+1,j])       (X[i+1,j+1],Y[i+1,j+1])
        #              *----------*
        #              |          |
        #              *----------*
        #(X[i,j],Y[i,j])           (X[i,j+1],Y[i,j+1])

        # sort_indices = np.argsort(x)
        # print(f'First sort of final and initial node positions the same: {np.allclose(sort_indices,comparison_sort_indices)}')
        # sorted_x = x[sort_indices]
        # sorted_y = y[sort_indices]
        # sorted_z = z[sort_indices]
        # X = np.zeros((Ny,Nx))
        # Y = np.zeros((Ny,Nx))
        # Z = np.zeros((Ny,Nx))
        # init_X = np.zeros((Ny,Nx))
        # init_Y = np.zeros((Ny,Nx))
        # sorted_init_x = init_x[comparison_sort_indices]
        # sorted_init_y = init_y[comparison_sort_indices]
        # start = 0
        # end = Ny
        # for i in range(Nx):
        #     X[:,i] = sorted_x[start:(i+1)*end]
        #     Y[:,i] = sorted_y[start:(i+1)*end]
        #     Z[:,i] = sorted_z[start:(i+1)*end]
        #     init_X[:,i] = sorted_init_x[start:(i+1)*end]
        #     init_Y[:,i] = sorted_init_y[start:(i+1)*end]
        #     start = (i+1)*end
        # #first sorting and then distributing values has the X array behaving as necessary, with the incrememnt in the column index resulting in an increase in the x position, but I also want the Y array to have an increment in the row index result in an increase in the y position. I need to do more sorting, but without destroying the sorted nature of the X array, and I need to use argsort to sort the X and Z arrays to match the sorted Y array. Within each column of Y, I need to sort from lowest to highest.
        # for i in range(Nx):
        #     sorted_indices = np.argsort(Y[:,i])
        #     comparison_sorted_indices = np.argsort(init_Y[:,i])
        #     print(f'Second sort, {i}th row of positions the same:{np.allclose(sorted_indices,comparison_sorted_indices)}')
        #     X[:,i] = X[sorted_indices,i]
        #     Y[:,i] = Y[sorted_indices,i]
        #     Z[:,i] = Z[sorted_indices,i]
        #     init_X[:,i] = init_X[comparison_sorted_indices,i]
        #     init_Y[:,i] = init_Y[comparison_sorted_indices,i]
        #SANITY CHECK: because i know how the corners need to be arranged, I can check to make sure they are arranged correctly
        for i in range(Ny-1):
            for j in range(Nx-1):
                assert X[i,j] < X[i,j+1] and X[i+1,j] < X[i+1,j+1] and Y[i,j] < Y[i+1,j] and Y[i,j+1] < Y[i+1,j+1], 'Quadrilateral corners are out of order from pmeshcolor expectations, image will draw incorrectly'
        # for i in range(Ny-1):
        #     for j in range(Nx-1):
        #         assert init_X[i,j] < init_X[i,j+1] and init_X[i+1,j] < init_X[i+1,j+1] and init_Y[i,j] < init_Y[i+1,j] and init_Y[i,j+1] < init_Y[i+1,j+1], 'Quadrilateral corners for initialized positions are out of order from pmeshcolor expectations, image will draw incorrectly'
        img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        # fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(f'{cut_type}' + f'layer {index}')
        # plt.show()
        return img

# fig, ax = plt.subplots()
# #create an IndexTracker and make sure it lives during the whole lifetime of the figure by assignign it to a variable
# tracker = IndexTracker(ax,X)

# fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
# plt.show()

def plot_cut_pcolormesh(cut_type,eq_node_posns,node_posns,index,particles,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig, ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
    posns = node_posns[cut_nodes]
    eq_posns = eq_node_posns[cut_nodes]
    if cut_type == 'xy':
        z = posns[:,2]
        x = posns[:,0]
        y = posns[:,1]
        Nx = int(np.max(eq_posns[:,0]) + 1)
        Ny = int(np.max(eq_posns[:,1]) + 1)
    elif 'xz':
        z = posns[:,1]
        x = posns[:,0]
        y = posns[:,2]
        Nx = int(np.max(eq_posns[:,0]) + 1)
        Ny = int(np.max(eq_posns[:,2]) + 1)
    else:
        z = posns[:,0]
        x = posns[:,1]
        y = posns[:,2]
        Nx = int(np.max(eq_posns[:,1]) + 1)
        Ny = int(np.max(eq_posns[:,2]) + 1)
    #need to convert from 1D vectors to 2D matrices
    #coordinates of the corners of quadrilaterlas of a pcolormesh:
    #(X[i+1,j],Y[i+1,j])       (X[i+1,j+1],Y[i+1,j+1])
    #              *----------*
    #              |          |
    #              *----------*
    #(X[i,j],Y[i,j])           (X[i,j+1],Y[i,j+1])
    sort_indices = np.argsort(x)
    sorted_x = x[sort_indices]
    sorted_y = y[sort_indices]
    sorted_z = z[sort_indices]
    X = np.zeros((Ny,Nx))
    Y = np.zeros((Ny,Nx))
    Z = np.zeros((Ny,Nx))
    start = 0
    end = Ny
    for i in range(Nx):
        X[:,i] = sorted_x[start:(i+1)*end]
        Y[:,i] = sorted_y[start:(i+1)*end]
        Z[:,i] = sorted_z[start:(i+1)*end]
        start = (i+1)*end
    #first sorting and then distributing values has the X array behaving as necessary, with the incrememnt in the column index resulting in an increase in the x position, but I also want the Y array to have an increment in the row index result in an increase in the y position. I need to do more sorting, but without destroying the sorted nature of the X array, and I need to use argsort to sort the X and Z arrays to match the sorted Y array. Within each column of Y, I need to sort from lowest to highest. but can i also just use the sorting of the first column for all the row rearrangement?
    # sorted_indices = np.argsort(Y[:,0])
    for i in range(Nx):
        sorted_indices = np.argsort(Y[:,i])
        X[:,i] = X[sorted_indices,i]
        Y[:,i] = Y[sorted_indices,i]
        Z[:,i] = Z[sorted_indices,i]
    #SANITY CHECK: because i know how the corners need to be arranged, I can check to make sure they are arranged correctly
    for i in range(Ny-1):
        for j in range(Nx-1):
            assert X[i,j] < X[i,j+1] and X[i+1,j] < X[i+1,j+1] and Y[i,j] < Y[i+1,j] and Y[i,j+1] < Y[i+1,j+1], 'Quadrilateral corners are out of order from pmeshcolor expectations, image will draw incorrectly'
    img = ax.pcolormesh(X,Y,Z,shading='gouraud')
    #don't forget to add a colorbar and limits
    fig.colorbar(img,ax=ax)
    # ax.scatter(posns[:,0],posns[:,1],posns[:,2],color='b',marker='o')
    # mre.analyze.plot_subset_springs(ax,node_posns,boundaries[view],springs,spring_color='b',spring_type=spring_type)
    ax.axis('equal')
    ax.set_title(f'{cut_type}' + f'layer {index}')
    # plt.show()
    savename = output_dir + f'cut_pcolormesh_visualization.png'
    plt.savefig(savename)

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