import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import get_spring_force_cy
import get_volume_correction_force_cy_nogil

def post_plot_cut_normalized(eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag=""):
    plot_cut_normalized('xy',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)
    plot_cut_normalized('xz',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)
    plot_cut_normalized('yz',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)

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
    plot_subset_springs(ax,node_posns,cut_nodes,springs,spring_color='b')
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(node_posns[particle_cut_nodes,0],node_posns[particle_cut_nodes,1],node_posns[particle_cut_nodes,2],color='k',marker='o')
    plot_subset_springs(ax,node_posns,particle_cut_nodes_set,springs,spring_color='r')
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

def post_plot_cut_normalized_hyst(eq_node_posns,node_posns,springs,particles,Hext,output_dir):
    plot_cut_normalized_hyst('xy',eq_node_posns,node_posns,springs,particles,Hext,output_dir)
    plot_cut_normalized_hyst('xz',eq_node_posns,node_posns,springs,particles,Hext,output_dir)
    plot_cut_normalized_hyst('yz',eq_node_posns,node_posns,springs,particles,Hext,output_dir)

def plot_cut_normalized_hyst(cut_type,eq_node_posns,node_posns,springs,particles,Hext,output_dir):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut."""
    cut_type_dict = {'xy':0, 'xz':1, 'yz':2}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = (np.round(np.array([Lx,Ly,Lz]))/2)
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes1 = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        cut_nodes2 =np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]-1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        cut_nodes = np.concatenate((cut_nodes1,cut_nodes2))
    ax.scatter(node_posns[cut_nodes,0],node_posns[cut_nodes,1],node_posns[cut_nodes,2],color ='b',marker='o')
    plot_subset_springs(ax,node_posns,cut_nodes,springs,spring_color='b')
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    ax.scatter(node_posns[particle_cut_nodes,0],node_posns[particle_cut_nodes,1],node_posns[particle_cut_nodes,2],color='k',marker='o')
    plot_subset_springs(ax,node_posns,particle_cut_nodes_set,springs,spring_color='r')
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
    # ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    savename = output_dir + f'post_plot_cut_{cut_type}_{center[cut_type_index]}' + 'Hext_' + str(np.round(Hext[2],decimals=2)) +'.png'
    plt.savefig(savename)
    plt.close()

def plot_subset_springs(ax,node_posns,nodes,springs,spring_color,spring_type=None):
    """Plot a subset of the springs which are connected to the nodes passed to the function"""
    if isinstance(nodes,set):
        nodes_set = nodes
    else:
        nodes_set = set(np.unique(nodes))
    for spring in springs:
        if spring_type is None:
            subset = set(spring[:2])
            if subset < nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
                x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                                np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                                np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
                ax.plot(x,y,z,color=spring_color)
        elif np.isclose(spring_type,spring[2]) or np.isclose(spring_type/2,spring[2]) or np.isclose(spring_type/4,spring[2]):
            subset = set(spring[:2])
            if subset < nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
                x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                                np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                                np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
                ax.plot(x,y,z,color=spring_color)

def plot_subset_springs_2D(ax,cut_type,node_posns,nodes,springs,spring_color,spring_type=None):
    """Plot a subset of the springs which are connected to the nodes passed to the function"""
    if isinstance(nodes,set):
        nodes_set = nodes
    else:
        nodes_set = set(np.unique(nodes))
    for spring in springs:
        if spring_type is None:
            subset = set(spring[:2])
            if subset < nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
                x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                                np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                                np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
                if cut_type == 'xy':
                    ax.plot(x,y,color=spring_color)
                elif cut_type == 'xz':
                    ax.plot(x,z,color=spring_color)
                elif cut_type == 'yz':
                    ax.plot(y,z,color=spring_color)
        elif np.isclose(spring_type,spring[2]) or np.isclose(spring_type/2,spring[2]) or np.isclose(spring_type/4,spring[2]):
            subset = set(spring[:2])
            if subset < nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
                x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                                np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                                np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
                if cut_type == 'xy':
                    ax.plot(x,y,color=spring_color)
                elif cut_type == 'xz':
                    ax.plot(x,z,color=spring_color)
                elif cut_type == 'yz':
                    ax.plot(y,z,color=spring_color)

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

def plot_scatter_color_depth_visualization(eq_node_posns,final_node_posns,cut_type,index,springs,particles,output_dir,spring_type=None,tag=""):
    """Plot the final positions of a set of nodes based on the type of cut (plane of nodes to plot) and the index (layer) they belong to in the initial configuration"""
    #TODO add spring type to plot_subset_springs, and to function arguments. add argument or somehow get the type of cut, to properly label the axes on the image. use the "depth" (odd node position out) to give a list of numbers to be mapped to colors using cmap and norm (which means selecting or else using the default cmap (and most likly letting norm do it's default), and setting the colorbar)
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    chosen_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
    if chosen_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        chosen_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*(index+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
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
    #TODO add a colorscheme for springs that reflects the displacement from equilibrium length
    marker_size = 250.0
    plot_subset_springs_2D(ax,cut_type,final_node_posns,chosen_nodes,springs,spring_color='b',spring_type=spring_type)
    sc = ax.scatter(xvar,yvar,s=marker_size,c=depth_color,marker='o',zorder=2.5)
    plt.colorbar(sc)
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
    # ax.scatter(xvar,yvar,color='k',marker='o')
    plot_subset_springs_2D(ax,cut_type,final_node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    # plt.show()
    savename = output_dir + f'scatter_color_depth_{cut_type}_layer_{index}_'+ tag + '.png'
    plt.savefig(savename)
    plt.close()

def center_cut_visualization(eq_node_posns,final_node_posns,springs,particles,output_dir,tag=""):
    """Plot node positions of central cuts of the simulated volume using a 2D scatter plot with color representing the depth of node positions."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = (np.round(np.array([Lx,Ly,Lz]))/2)
    cut_types = ['xy','xz','yz']
    spring_type = np.max(springs[:,2])
    for i in range(3):
        plot_scatter_color_depth_visualization(eq_node_posns,final_node_posns,cut_types[i],center[i],springs,particles,output_dir,spring_type=spring_type,tag=tag)

def plot_center_cuts(eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut('xy',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)
    plot_center_cut('xz',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)
    plot_center_cut('yz',eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag)

def plot_center_cut(cut_type,eq_node_posns,node_posns,springs,particles,boundary_conditions,output_dir,tag=""):
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
    ax = fig.add_subplot()
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_type_index == 0:
        xvar = node_posns[cut_nodes,1]
        yvar = node_posns[cut_nodes,2]
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Ly*1.1)
        ylim = (-0.1,Lz*1.1)
    elif cut_type_index == 1:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,2]
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Lz*1.1)
    else:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,1]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
    ax.scatter(xvar,yvar,color ='b',marker='o',zorder=2.5)
    spring_type = np.max(springs[:,2])
    plot_subset_springs_2D(ax,cut_type,node_posns,cut_nodes,springs,spring_color='b',spring_type=spring_type)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    if cut_type_index == 0:
        xvar = node_posns[particle_cut_nodes,1]
        yvar = node_posns[particle_cut_nodes,2]
    elif cut_type_index == 1:
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,2]
    else:
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,1]
    ax.scatter(xvar,yvar,color='k',marker='o',zorder=2.5)
    plot_subset_springs_2D(ax,cut_type,node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'post_plot_cut_{cut_type}_{center[cut_type_index]}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_center_cuts_contour(eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut_contour('xy',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_contour('xz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_contour('yz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)

def plot_center_cut_contour(cut_type,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = (np.round(np.array([Lx,Ly,Lz]))/2)
    fig, ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_type_index == 0:
        xvar = node_posns[cut_nodes,1]
        yvar = node_posns[cut_nodes,2]
        zvar = node_posns[cut_nodes,0]
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Ly*1.1)
        ylim = (-0.1,Lz*1.1)
    elif cut_type_index == 1:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,2]
        zvar = node_posns[cut_nodes,1]
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Lz*1.1)
    else:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,1]
        zvar = node_posns[cut_nodes,2]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
    # ax.scatter(xvar,yvar,color ='b',marker='o',zorder=2.5)
    levels = np.linspace(zvar.min(),zvar.max(),10)
    ax.plot(xvar,yvar,'o',color='b')
    sc = ax.tricontourf(xvar,yvar,zvar,levels=levels)
    # spring_type = np.max(springs[:,2])
    # plot_subset_springs_2D(ax,cut_type,node_posns,cut_nodes,springs,spring_color='b',spring_type=spring_type)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    if cut_type_index == 0:
        xvar = node_posns[particle_cut_nodes,1]
        yvar = node_posns[particle_cut_nodes,2]
    elif cut_type_index == 1:
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,2]
    else:
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,1]
    # ax.scatter(xvar,yvar,color='k',marker='o',zorder=2.5)
    ax.plot(xvar,yvar,'o',color='r')
    # plot_subset_springs_2D(ax,cut_type,node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(sc)
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'tricontourf_cut_{cut_type}_{center[cut_type_index]}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_center_cuts_wireframe(eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut_wireframe('xy',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_wireframe('xz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_wireframe('yz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    
def plot_center_cut_wireframe(cut_type,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = np.round(np.array([Lx,Ly,Lz]))/2
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(center[cut_type_index]),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(int(center[cut_type_index]+1/2)),eq_node_posns[:,cut_type_index]).nonzero()[0]
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    if cut_type_index == 0:
        idx = int(center[0])
        xvar = xposn_3D[idx,:,:]
        yvar = yposn_3D[idx,:,:]
        zvar = zposn_3D[idx,:,:]
    elif cut_type_index == 1:
        idx = int(center[1])
        xvar = xposn_3D[:,idx,:]
        yvar = yposn_3D[:,idx,:]
        zvar = zposn_3D[:,idx,:]
    else:
        idx = int(center[2])
        xvar = xposn_3D[:,:,idx]
        yvar = yposn_3D[:,:,idx]
        zvar = zposn_3D[:,:,idx]
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    # the below sets things up for everything laying in the same plane visually, for the different plot types. why not let it be the same for the different cuts, and just see how it is?
    # if cut_type_index == 0:
    #     idx = int(center[0])
    #     xvar = yposn_3D[idx,:,:]
    #     yvar = zposn_3D[idx,:,:]
    #     zvar = xposn_3D[idx,:,:]
    #     xlabel = 'Y (l_e)'
    #     ylabel = 'Z (l_e)'
    #     zlabel = 'X (l_e)'
    #     xlim = (-0.1,Ly*1.1)
    #     ylim = (-0.1,Lz*1.1)
    #     zlim = (-0.1,Lx*1.1)
    # elif cut_type_index == 1:
    #     idx = int(center[1])
    #     xvar = xposn_3D[:,idx,:]
    #     yvar = zposn_3D[:,idx,:]
    #     zvar = yposn_3D[:,idx,:]
    #     xlabel = 'X (l_e)'
    #     ylabel = 'Z (l_e)'
    #     zlabel = 'Y (l_e)'
    #     xlim = (-0.1,Lx*1.1)
    #     ylim = (-0.1,Lz*1.1)
    #     zlim = (-0.1,Ly*1.1)
    # else:
    #     idx = int(center[2])
    #     xvar = xposn_3D[:,:,idx]
    #     yvar = yposn_3D[:,:,idx]
    #     zvar = zposn_3D[:,:,idx]
    #     xlabel = 'X (l_e)'
    #     ylabel = 'Y (l_e)'
    #     zlabel = 'Z (l_e)'
    #     xlim = (-0.1,Lx*1.1)
    #     ylim = (-0.1,Ly*1.1)
    #     zlim = (-0.1,Lz*1.1)
    ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    #similar to the above for the wiremesh... letting the positions be the positions variables(x,y,z), instead of trying to have everything lie in the same plane visually when plotting (which is appropriate when i am 2D plotting, but removes a great deal of the benefit of trying the 3D plotting)
    # if cut_type_index == 0:
    #     xvar = node_posns[particle_cut_nodes,1]
    #     yvar = node_posns[particle_cut_nodes,2]
    #     zvar = node_posns[particle_cut_nodes,0]
    # elif cut_type_index == 1:
    #     xvar = node_posns[particle_cut_nodes,0]
    #     yvar = node_posns[particle_cut_nodes,2]
    #     zvar = node_posns[particle_cut_nodes,1]
    # else:
    #     xvar = node_posns[particle_cut_nodes,0]
    #     yvar = node_posns[particle_cut_nodes,1]
    #     zvar = node_posns[particle_cut_nodes,2]
    xvar = node_posns[particle_cut_nodes,0]
    yvar = node_posns[particle_cut_nodes,1]
    zvar = node_posns[particle_cut_nodes,2]
    ax.scatter(xvar,yvar,zvar,'o',color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'wireframe_cut_{cut_type}_{center[cut_type_index]}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def transform_to_3D_array(array,dimensions):
    """Given a 1D vector of node positions, or similarly structured per node values, and convert to a 3D array mapped to the grid of initial node positions for plotting and analysis. Dimensions is tuple or array of number of nodes along each direction (x,y,z)."""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    array_3D = np.zeros((xshape,yshape,zshape))
    nodes_per_column = zshape
    nodes_per_plane = zshape*xshape
    for j in range(yshape):
        for i in range(xshape):
            array_3D[i,j,:] = array[(j*nodes_per_plane)+(i*nodes_per_column):(j*nodes_per_plane)+(i+1)*nodes_per_column]
    return array_3D

def get_component_3D_arrays(array,dimensions):
    xarray_3D = transform_to_3D_array(array[:,0],dimensions)
    yarray_3D = transform_to_3D_array(array[:,1],dimensions)
    zarray_3D = transform_to_3D_array(array[:,2],dimensions)
    return xarray_3D, yarray_3D, zarray_3D

def plot_overlayed_center_cuts_wireframe(eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume on a single figure/axis, showing the configuration of the nodes that sat at the center of the initialized system.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = np.round(np.array([Lx,Ly,Lz]))/2
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    for cut_type_index in range(1,3):
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(center[cut_type_index]),eq_node_posns[:,cut_type_index]).nonzero()[0]
        # if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        #     cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
        if cut_type_index == 0:
            idx = int(center[0])
            xvar = xposn_3D[idx,:,:]
            yvar = yposn_3D[idx,:,:]
            zvar = zposn_3D[idx,:,:]
        elif cut_type_index == 1:
            idx = int(center[1])
            xvar = xposn_3D[:,idx,:]
            yvar = yposn_3D[:,idx,:]
            zvar = zposn_3D[:,idx,:]
        else:
            idx = int(center[2])
            xvar = xposn_3D[:,:,idx]
            yvar = yposn_3D[:,:,idx]
            zvar = zposn_3D[:,:,idx]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        zlabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
        zlim = (-0.1,Lz*1.1)
        ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
        #now identify which of those nodes belong to the particle and the cut. set intersection?
        cut_nodes_set = set(cut_nodes)
        #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
        particle_nodes_set = set(particles.ravel())
        particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
        #alternative to below, use set unpacking
        #*particle_cut_nodes, = particle_cut_nodes_set
        particle_cut_nodes = [x for x in particle_cut_nodes_set]
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,1]
        zvar = node_posns[particle_cut_nodes,2]
        ax.scatter(xvar,yvar,zvar,'o',color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'overlayed_wireframe_cuts_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_center_cuts_surf(eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut_surf('xy',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_surf('xz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_surf('yz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    
def plot_center_cut_surf(cut_type,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = np.round(np.array([Lx,Ly,Lz]))/2
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(center[cut_type_index]),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(int(center[cut_type_index]+1/2)),eq_node_posns[:,cut_type_index]).nonzero()[0]
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    if cut_type_index == 0:
        idx = int(center[0])
        xvar = xposn_3D[idx,:,:]
        yvar = yposn_3D[idx,:,:]
        zvar = zposn_3D[idx,:,:]
        color_dimension = xvar
    elif cut_type_index == 1:
        idx = int(center[1])
        xvar = xposn_3D[:,idx,:]
        yvar = yposn_3D[:,idx,:]
        zvar = zposn_3D[:,idx,:]
        color_dimension = yvar
    else:
        idx = int(center[2])
        xvar = xposn_3D[:,:,idx]
        yvar = yposn_3D[:,:,idx]
        zvar = zposn_3D[:,:,idx]
        color_dimension = zvar
    color_min, color_max = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(color_min,color_max)
    my_cmap = cm.ScalarMappable(norm=norm,cmap='jet')
    my_cmap.set_array([])
    fcolors = my_cmap.to_rgba(color_dimension)
    surf = ax.plot_surface(xvar,yvar,zvar,rstride=1,cstride=1,facecolors=fcolors,vmin=color_min,vmax=color_max,shade=False)
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    # the below sets things up for everything laying in the same plane visually, for the different plot types. why not let it be the same for the different cuts, and just see how it is?
    # if cut_type_index == 0:
    #     idx = int(center[0])
    #     xvar = yposn_3D[idx,:,:]
    #     yvar = zposn_3D[idx,:,:]
    #     zvar = xposn_3D[idx,:,:]
    #     xlabel = 'Y (l_e)'
    #     ylabel = 'Z (l_e)'
    #     zlabel = 'X (l_e)'
    #     xlim = (-0.1,Ly*1.1)
    #     ylim = (-0.1,Lz*1.1)
    #     zlim = (-0.1,Lx*1.1)
    # elif cut_type_index == 1:
    #     idx = int(center[1])
    #     xvar = xposn_3D[:,idx,:]
    #     yvar = zposn_3D[:,idx,:]
    #     zvar = yposn_3D[:,idx,:]
    #     xlabel = 'X (l_e)'
    #     ylabel = 'Z (l_e)'
    #     zlabel = 'Y (l_e)'
    #     xlim = (-0.1,Lx*1.1)
    #     ylim = (-0.1,Lz*1.1)
    #     zlim = (-0.1,Ly*1.1)
    # else:
    #     idx = int(center[2])
    #     xvar = xposn_3D[:,:,idx]
    #     yvar = yposn_3D[:,:,idx]
    #     zvar = zposn_3D[:,:,idx]
    #     xlabel = 'X (l_e)'
    #     ylabel = 'Y (l_e)'
    #     zlabel = 'Z (l_e)'
    #     xlim = (-0.1,Lx*1.1)
    #     ylim = (-0.1,Ly*1.1)
    #     zlim = (-0.1,Lz*1.1)
    fig.colorbar(my_cmap)
    # fig.colorbar(surf,aspect=5)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    #TODO unravel the particles variable since there might be more than one, need a onedimensional object (i think) to pass to the set() constructor
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    #similar to the above for the wiremesh... letting the positions be the positions variables(x,y,z), instead of trying to have everything lie in the same plane visually when plotting (which is appropriate when i am 2D plotting, but removes a great deal of the benefit of trying the 3D plotting)
    # if cut_type_index == 0:
    #     xvar = node_posns[particle_cut_nodes,1]
    #     yvar = node_posns[particle_cut_nodes,2]
    #     zvar = node_posns[particle_cut_nodes,0]
    # elif cut_type_index == 1:
    #     xvar = node_posns[particle_cut_nodes,0]
    #     yvar = node_posns[particle_cut_nodes,2]
    #     zvar = node_posns[particle_cut_nodes,1]
    # else:
    #     xvar = node_posns[particle_cut_nodes,0]
    #     yvar = node_posns[particle_cut_nodes,1]
    #     zvar = node_posns[particle_cut_nodes,2]
    xvar = node_posns[particle_cut_nodes,0]
    yvar = node_posns[particle_cut_nodes,1]
    zvar = node_posns[particle_cut_nodes,2]
    ax.scatter(xvar,yvar,zvar,'o',color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'surf_cut_{cut_type}_{center[cut_type_index]}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def setup_figure_and_variables_2D(cut_type,eq_node_posns,node_posns):
    """Perform setup and return figure handle, axis handle, and variables, and set axis labels and limits for 2D plots"""
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
    ax = fig.add_subplot()
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_type_index == 0:
        xvar = node_posns[cut_nodes,1]
        yvar = node_posns[cut_nodes,2]
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Ly*1.1)
        ylim = (-0.1,Lz*1.1)
    elif cut_type_index == 1:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,2]
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Lz*1.1)
    else:
        xvar = node_posns[cut_nodes,0]
        yvar = node_posns[cut_nodes,1]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax,xvar,yvar

def main():
    pass

if __name__=="__main__":
    main()