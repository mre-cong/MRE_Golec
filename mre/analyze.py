import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import simulate

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 14
BIG_FONT_SIZE = 20

plt.rc('font',size=BIG_FONT_SIZE)
plt.rc('axes',titlesize=BIG_FONT_SIZE)
plt.rc('axes',labelsize=BIG_FONT_SIZE)
plt.rc('xtick',labelsize=BIG_FONT_SIZE)
plt.rc('ytick',labelsize=BIG_FONT_SIZE)
# plt.rc('ztick',labelsize=MEDIUM_FONT_SIZE)
plt.rc('legend',fontsize=BIG_FONT_SIZE)
plt.rc('figure',titlesize=BIG_FONT_SIZE)

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

def plot_surf_cut(cut_type,layer,eq_node_posns,node_posns,boundary_conditions,output_dir,tag="",ax=None):
    """Plot a cut through the simulated volume, showing the configuration of the nodes that sat at the specified layer of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    if cut_type_index == 0:
        assert layer <= Lx and layer >= 0, f'Layer choice not within bounds of available layers [0,{Lx}]'
    elif cut_type_index == 1:
        assert layer <= Ly and layer >= 0, f'Layer choice not within bounds of available layers [0,{Ly}]'
    elif cut_type_index == 2:
        assert layer <= Lz and layer >= 0, f'Layer choice not within bounds of available layers [0,{Lz}]'
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    if ax == None:
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(3*default_width,3*default_height)
        fig.set_dpi(200)
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    idx = int(layer)
    if cut_type_index == 0:
        xvar = xposn_3D[idx,:,:]
        yvar = yposn_3D[idx,:,:]
        zvar = zposn_3D[idx,:,:]
        xlim = (xvar.min(),xvar.max())
        color_dimension = xvar
    elif cut_type_index == 1:
        xvar = xposn_3D[:,idx,:]
        yvar = yposn_3D[:,idx,:]
        zvar = zposn_3D[:,idx,:]
        ylim = (yvar.min(),yvar.max())
        color_dimension = yvar
    else:
        xvar = xposn_3D[:,:,idx]
        yvar = yposn_3D[:,:,idx]
        zvar = zposn_3D[:,:,idx]
        zlim = (zvar.min(),zvar.max())
        color_dimension = zvar
    # the below sets things up for everything laying in the same plane visually, for the different plot types. why not let it be the same for the different cuts, and just see how it is?
    # if cut_type_index == 0:
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
    #     xvar = xposn_3D[:,:,idx]
    #     yvar = yposn_3D[:,:,idx]
    #     zvar = zposn_3D[:,:,idx]
    #     xlabel = 'X (l_e)'
    #     ylabel = 'Y (l_e)'
    #     zlabel = 'Z (l_e)'
    #     xlim = (-0.1,Lx*1.1)
    #     ylim = (-0.1,Ly*1.1)
    #     zlim = (-0.1,Lz*1.1)
    color_min, color_max = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(color_min,color_max)
    my_cmap = cm.ScalarMappable(norm=norm,cmap='jet')
    my_cmap.set_array([])
    fcolors = my_cmap.to_rgba(color_dimension)
    surf = ax.plot_surface(xvar,yvar,zvar,rstride=1,cstride=1,facecolors=fcolors,vmin=color_min,vmax=color_max,shade=False)
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
    savename = output_dir + f'surf_cut_{cut_type}_{layer}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_wireframe_cut(cut_type,layer,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag="",ax=None):
    """Plot a cut through the simulated volume, showing the configuration of the nodes that sat at the specified layer of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    if cut_type_index == 0:
        assert layer <= Lx and layer >= 0, f'Layer choice not within bounds of available layers [0,{Lx}]'
    elif cut_type_index == 1:
        assert layer <= Ly and layer >= 0, f'Layer choice not within bounds of available layers [0,{Ly}]'
    elif cut_type_index == 2:
        assert layer <= Lz and layer >= 0, f'Layer choice not within bounds of available layers [0,{Lz}]'
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    if ax == None:
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        default_width,default_height = fig.get_size_inches()
        fig.set_size_inches(3*default_width,3*default_height)
        fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(layer),eq_node_posns[:,cut_type_index]).nonzero()[0]
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    idx = int(layer)
    if cut_type_index == 0:
        xvar = xposn_3D[idx,:,:]
        yvar = yposn_3D[idx,:,:]
        zvar = zposn_3D[idx,:,:]
        xlim = (xvar.min(),xvar.max())
    elif cut_type_index == 1:
        xvar = xposn_3D[:,idx,:]
        yvar = yposn_3D[:,idx,:]
        zvar = zposn_3D[:,idx,:]
        ylim = (yvar.min(),yvar.max())
    else:
        xvar = xposn_3D[:,:,idx]
        yvar = yposn_3D[:,:,idx]
        zvar = zposn_3D[:,:,idx]
        zlim = (zvar.min(),zvar.max())
    # the below sets things up for everything laying in the same plane visually, for the different plot types. why not let it be the same for the different cuts, and just see how it is?
    # if cut_type_index == 0:
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
    savename = output_dir + f'wireframe_cut_{cut_type}_{layer}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

#Center cuts
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

def plot_contour_cut(cut_type,layer,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the user passed layer of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    idx = layer
    if cut_type_index == 0:
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Ly*1.1)
        ylim = (-0.1,Lz*1.1)
    elif cut_type_index == 1:
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Lz*1.1)
    else:
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
    if cut_type_index == 0:
        xvar2D = xposn_3D[idx,:,:]
        yvar2D = yposn_3D[idx,:,:]
        zvar2D = zposn_3D[idx,:,:]
        xvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
        yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
        zvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
    elif cut_type_index == 1:
        xvar2D = xposn_3D[:,idx,:]
        yvar2D = yposn_3D[:,idx,:]
        zvar2D = zposn_3D[:,idx,:]
        xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
        yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
        zvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
    else:
        xvar2D = xposn_3D[:,:,idx]
        yvar2D = yposn_3D[:,:,idx]
        zvar2D = zposn_3D[:,:,idx]
        xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
        yvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
        zvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
    fig, ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*idx,eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(idx+1/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
    levels = np.linspace(zvar.min(),zvar.max(),10)
    ax.plot(xvar,yvar,'o',color='b')
    sc = ax.tricontourf(xvar,yvar,zvar,levels=levels)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
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
    ax.plot(xvar,yvar,'o',color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(sc)
    if tag != "":
        ax.set_title(tag)
        tag = "_" + tag
    savename = output_dir + f'tricontourf_cut_{cut_type}_{idx}_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
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
    layer = int(center[cut_type_index])
    plot_contour_cut(cut_type,layer,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag="")

def plot_center_cuts_wireframe(eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut_wireframe('xy',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_wireframe('xz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    plot_center_cut_wireframe('yz',eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag)
    
def plot_center_cut_wireframe(cut_type,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = np.round(np.array([Lx,Ly,Lz]))/2
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    layer = int(center[cut_type_index])
    plot_wireframe_cut(cut_type,layer,eq_node_posns,node_posns,particles,boundary_conditions,output_dir,tag="",ax=None)

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

def plot_center_cuts_surf_si(eq_node_posns,node_posns,l_e,particles,output_dir,plot_3D_flag=True,tag=""):
    """Plot a series of 3 cuts through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system."""
    plot_center_cut_surf_si('xy',eq_node_posns,node_posns,l_e,particles,output_dir,plot_3D_flag,tag)
    plot_center_cut_surf_si('xz',eq_node_posns,node_posns,l_e,particles,output_dir,plot_3D_flag,tag)
    plot_center_cut_surf_si('yz',eq_node_posns,node_posns,l_e,particles,output_dir,plot_3D_flag,tag)

def plot_center_cut_surf_si(cut_type,eq_node_posns,node_posns,l_e,particles,output_dir,plot_3D_flag=True,tag=""):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    center = np.round(np.array([Lx,Ly,Lz]))/2
    Lx *= l_e*1e6
    Ly *= l_e*1e6
    Lz *= l_e*1e6
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(center[cut_type_index]),eq_node_posns[:,cut_type_index]).nonzero()[0]
    if cut_nodes.shape[0] == 0:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*(int(center[cut_type_index]+1/2)),eq_node_posns[:,cut_type_index]).nonzero()[0]
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    xposn_3D *= l_e*1e6
    yposn_3D *= l_e*1e6
    zposn_3D *= l_e*1e6
    if plot_3D_flag:
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
        ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
        xlabel = 'X (um)'
        ylabel = 'Y (um)'
        zlabel = 'Z (um)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
        zlim = (-0.1,Lz*1.1)
    # the below sets things up for everything laying in the same plane visually, for the different plot types. why not let it be the same for the different cuts, and just see how it is?
    else:
        if cut_type_index == 0:
            idx = int(center[0])
            xvar = yposn_3D[idx,:,:]
            yvar = zposn_3D[idx,:,:]
            zvar = xposn_3D[idx,:,:]
            color_dimension = xvar
            xlabel = 'Y (um)'
            ylabel = 'Z (um)'
            zlabel = 'X (um)'
            xlim = (-0.1,Ly*1.1)
            ylim = (-0.1,Lz*1.1)
            zlim = (-0.1,Lx*1.1)
        elif cut_type_index == 1:
            idx = int(center[1])
            xvar = xposn_3D[:,idx,:]
            yvar = zposn_3D[:,idx,:]
            zvar = yposn_3D[:,idx,:]
            color_dimension = yvar
            xlabel = 'X (um)'
            ylabel = 'Z (um)'
            zlabel = 'Y (um)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Lz*1.1)
            zlim = (-0.1,Ly*1.1)
        else:
            idx = int(center[2])
            xvar = xposn_3D[:,:,idx]
            yvar = yposn_3D[:,:,idx]
            zvar = zposn_3D[:,:,idx]
            color_dimension = zvar
            xlabel = 'X (um)'
            ylabel = 'Y (um)'
            zlabel = 'Z (um)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Ly*1.1)
            zlim = (-0.1,Lz*1.1)
        color_min, color_max = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(color_min,color_max)
        my_cmap = cm.ScalarMappable(norm=norm,cmap='jet')
        my_cmap.set_array([])
        fcolors = my_cmap.to_rgba(color_dimension)
        surf = ax.plot_surface(xvar,yvar,zvar,rstride=1,cstride=1,facecolors=fcolors,vmin=color_min,vmax=color_max,shade=False)
        ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
    fig.colorbar(my_cmap)
    #now identify which of those nodes belong to the particle and the cut. set intersection?
    cut_nodes_set = set(cut_nodes)
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    #alternative to below, use set unpacking
    #*particle_cut_nodes, = particle_cut_nodes_set
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    #similar to the above for the wiremesh... letting the positions be the positions variables(x,y,z), instead of trying to have everything lie in the same plane visually when plotting (which is appropriate when i am 2D plotting, but removes a great deal of the benefit of trying the 3D plotting)
    if plot_3D_flag:
        xvar = node_posns[particle_cut_nodes,0]
        yvar = node_posns[particle_cut_nodes,1]
        zvar = node_posns[particle_cut_nodes,2]
    else:
        if cut_type_index == 0:
            xvar = node_posns[particle_cut_nodes,1]
            yvar = node_posns[particle_cut_nodes,2]
            zvar = node_posns[particle_cut_nodes,0]
        elif cut_type_index == 1:
            xvar = node_posns[particle_cut_nodes,0]
            yvar = node_posns[particle_cut_nodes,2]
            zvar = node_posns[particle_cut_nodes,1]
        else:
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
    # if tag != "":
    #     ax.set_title(tag)
    #     tag = "_" + tag
    savename = output_dir + f'surf_cut_center_{cut_type}_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

#Particle focused cut visualizations

def plot_particle_centric_cuts_wireframe(initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Given a simulation containing particles, plot wireframe cuts through the system based on the initial positions of the particles. Intended for single and two particle simulations"""
    centers = np.zeros((particles.shape[0],3))
    for i, particle in enumerate(particles):
        centers[i,:] = simulate.get_particle_center(particle,initialized_node_posns)
        particle_node_posns = initialized_node_posns[particle,:]
        x_max = np.max(particle_node_posns[:,0])
        y_max = np.max(particle_node_posns[:,1])
        z_max = np.max(particle_node_posns[:,2])
        x_min = np.min(particle_node_posns[:,0])
        y_min = np.min(particle_node_posns[:,1])
        z_min = np.min(particle_node_posns[:,2])
        plot_wireframe_cut('xy',int(z_min),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_wireframe_cut('xy',int(z_max),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
        plot_wireframe_cut('xz',int(y_min),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_wireframe_cut('xz',int(y_max),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
        plot_wireframe_cut('yz',int(x_min),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_wireframe_cut('yz',int(x_max),initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
    layer = int((centers[0,0]+centers[1,0])/2)
    plot_wireframe_cut('yz',layer,initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle_center")
    layer = int((centers[0,1]+centers[1,1])/2)
    plot_wireframe_cut('xz',layer,initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle_center")
    layer = int((centers[0,2]+centers[1,2])/2)
    plot_wireframe_cut('xy',layer,initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=f"particle_center")

def plot_particle_centric_cuts_surf(initialized_node_posns,current_node_posns,particles,boundary_conditions,output_dir,tag=""):
    """Given a simulation containing particles, plot surf cuts through the system based on the initial positions of the particles. Intended for single and two particle simulations"""
    centers = np.zeros((particles.shape[0],3))
    for i, particle in enumerate(particles):
        centers[i,:] = simulate.get_particle_center(particle,initialized_node_posns)
        particle_node_posns = initialized_node_posns[particle,:]
        x_max = np.max(particle_node_posns[:,0])
        y_max = np.max(particle_node_posns[:,1])
        z_max = np.max(particle_node_posns[:,2])
        x_min = np.min(particle_node_posns[:,0])
        y_min = np.min(particle_node_posns[:,1])
        z_min = np.min(particle_node_posns[:,2])
        plot_surf_cut('xy',int(z_min),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_surf_cut('xy',int(z_max),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
        plot_surf_cut('xz',int(y_min),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_surf_cut('xz',int(y_max),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
        plot_surf_cut('yz',int(x_min),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge")
        plot_surf_cut('yz',int(x_max),initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle{i+1}_edge2")
    layer = int((centers[0,0]+centers[1,0])/2)
    plot_surf_cut('yz',layer,initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle_center")
    layer = int((centers[0,1]+centers[1,1])/2)
    plot_surf_cut('xz',layer,initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle_center")
    layer = int((centers[0,2]+centers[1,2])/2)
    plot_surf_cut('xy',layer,initialized_node_posns,current_node_posns,boundary_conditions,output_dir,tag=f"particle_center")

#Outer surface plots

def plot_tiled_outer_surfaces_contours(eq_node_posns,node_posns,output_dir,tag=""):
    """plot the outer surfaces of the simulated volume in a tile plot, using contours (tricontourf())"""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    layers = (0,Lx,0,Ly,0,Lz)
    for cut_type_index in range(3):
        if cut_type_index == 0:
            xlabel = 'Y (l_e)'
            ylabel = 'Z (l_e)'
            xlim = (-0.1,Ly*1.1)
            ylim = (-0.1,Lz*1.1)
        elif cut_type_index == 1:
            xlabel = 'X (l_e)'
            ylabel = 'Z (l_e)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Lz*1.1)
        else:
            xlabel = 'X (l_e)'
            ylabel = 'Y (l_e)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Ly*1.1)
        for i in range(2):
            idx = int(layers[2*cut_type_index+i])
            if cut_type_index == 0:
                xvar2D = xposn_3D[idx,:,:]
                yvar2D = yposn_3D[idx,:,:]
                zvar2D = zposn_3D[idx,:,:]
                # xlim = (xvar.min(),xvar.max())
                xvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
                yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
                zvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
            elif cut_type_index == 1:
                xvar2D = xposn_3D[:,idx,:]
                yvar2D = yposn_3D[:,idx,:]
                zvar2D = zposn_3D[:,idx,:]
                # ylim = (yvar.min(),yvar.max())
                xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
                yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
                zvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
            else:
                xvar2D = xposn_3D[:,:,idx]
                yvar2D = yposn_3D[:,:,idx]
                zvar2D = zposn_3D[:,:,idx]
                # zlim = (zvar.min(),zvar.max())
                xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
                yvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
                zvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
            ax = axs[i,cut_type_index]
            # ax.scatter(xvar,yvar,color ='b',marker='o',zorder=2.5)
            levels = np.linspace(zvar.min(),zvar.max(),20)
            ax.scatter(xvar,yvar,marker='o',color='b',s=2.0,zorder=2.5)
            sc = ax.tricontourf(xvar,yvar,zvar,levels=levels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.colorbar(sc,ax=ax)
            #don't forget to add a colorbar and limits
    fig.tight_layout()
    savename = output_dir + f'outersurfaces_tricontourf_tile_visualization' + tag + '.png'
    plt.savefig(savename)
    return 0

def plot_tiled_outer_surfaces_contours_si(eq_node_posns,node_posns,l_e,output_dir,tag=""):
    """Plot the outer surfaces of the simulated volume in a tile plot, using contours (tricontourf()), with SI units."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    xposn_3D *= l_e*1e6
    yposn_3D *= l_e*1e6
    zposn_3D *= l_e*1e6
    layers = (0,Lx,0,Ly,0,Lz)
    Lx *= l_e*1e6
    Ly *= l_e*1e6
    Lz *= l_e*1e6
    for cut_type_index in range(3):
        if cut_type_index == 0:
            xlabel = 'Y (um)'
            ylabel = 'Z (um)'
            xlim = (-0.1,Ly*1.1)
            ylim = (-0.1,Lz*1.1)
        elif cut_type_index == 1:
            xlabel = 'X (um)'
            ylabel = 'Z (um)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Lz*1.1)
        else:
            xlabel = 'X (um)'
            ylabel = 'Y (um)'
            xlim = (-0.1,Lx*1.1)
            ylim = (-0.1,Ly*1.1)
        for i in range(2):
            idx = int(layers[2*cut_type_index+i])
            if cut_type_index == 0:
                xvar2D = xposn_3D[idx,:,:]
                yvar2D = yposn_3D[idx,:,:]
                zvar2D = zposn_3D[idx,:,:]
                # xlim = (xvar.min(),xvar.max())
                xvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
                yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
                zvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
            elif cut_type_index == 1:
                xvar2D = xposn_3D[:,idx,:]
                yvar2D = yposn_3D[:,idx,:]
                zvar2D = zposn_3D[:,idx,:]
                # ylim = (yvar.min(),yvar.max())
                xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
                yvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
                zvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
            else:
                xvar2D = xposn_3D[:,:,idx]
                yvar2D = yposn_3D[:,:,idx]
                zvar2D = zposn_3D[:,:,idx]
                # zlim = (zvar.min(),zvar.max())
                xvar = np.reshape(xvar2D,(xvar2D.shape[0]*xvar2D.shape[1],))
                yvar = np.reshape(yvar2D,(yvar2D.shape[0]*yvar2D.shape[1],))
                zvar = np.reshape(zvar2D,(zvar2D.shape[0]*zvar2D.shape[1],))
            ax = axs[i,cut_type_index]
            # ax.scatter(xvar,yvar,color ='b',marker='o',zorder=2.5)
            levels = np.linspace(zvar.min(),zvar.max(),20)
            ax.scatter(xvar,yvar,marker='o',color='b',s=2.0,zorder=2.5)
            sc = ax.tricontourf(xvar,yvar,zvar,levels=levels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.colorbar(sc,ax=ax)
            #don't forget to add a colorbar and limits
    fig.tight_layout()
    savename = output_dir + f'outersurfaces_contours_tiled_' + tag + '.png'
    plt.savefig(savename)
    return 0

def plot_outer_surfaces_wireframe(eq_node_posns,node_posns,boundary_conditions,output_dir,tag=""):
    """Plot the outer surfaces of the simulated volume as a wireframe plot.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    # cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*int(layer),eq_node_posns[:,cut_type_index]).nonzero()[0]
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    layers = (0,Lx,0,Ly,0,Lz)
    for cut_type_index in range(3):
        for i in range(2):
            idx = int(layers[2*cut_type_index+i])
            if cut_type_index == 0:
                xvar = xposn_3D[idx,:,:]
                yvar = yposn_3D[idx,:,:]
                zvar = zposn_3D[idx,:,:]
                # xlim = (xvar.min(),xvar.max())
            elif cut_type_index == 1:
                xvar = xposn_3D[:,idx,:]
                yvar = yposn_3D[:,idx,:]
                zvar = zposn_3D[:,idx,:]
                # ylim = (yvar.min(),yvar.max())
            else:
                xvar = xposn_3D[:,:,idx]
                yvar = yposn_3D[:,:,idx]
                zvar = zposn_3D[:,:,idx]
                # zlim = (zvar.min(),zvar.max())
            ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
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
    savename = output_dir + f'wireframe_outer_surfaces_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_outer_surfaces(eq_node_posns,node_posns,boundary_conditions,output_dir,tag=""):
    """Plot the outer surfaces of the simulated volume as a surface plot.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    layers = (0,Lx,0,Ly,0,Lz)
    for cut_type_index in range(3):
        for i in range(2):
            idx = int(layers[2*cut_type_index+i])
            if cut_type_index == 0:
                xvar = xposn_3D[idx,:,:]
                yvar = yposn_3D[idx,:,:]
                zvar = zposn_3D[idx,:,:]
            elif cut_type_index == 1:
                xvar = xposn_3D[:,idx,:]
                yvar = yposn_3D[:,idx,:]
                zvar = zposn_3D[:,idx,:]
            else:
                xvar = xposn_3D[:,:,idx]
                yvar = yposn_3D[:,:,idx]
                zvar = zposn_3D[:,:,idx]
            surf = ax.plot_surface(xvar,yvar,zvar,rstride=1,cstride=1)
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
    savename = output_dir + f'outer_surfaces_' + str(np.round(boundary_conditions[2],decimals=2)) + '_' + tag +'.png'
    plt.savefig(savename)
    plt.close()

def plot_outer_surfaces_si(eq_node_posns,node_posns,l_e,output_dir,tag=""):
    """Plot the outer surfaces of the simulated volume as a surface plot.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    xlabel = 'X (um)'
    ylabel = 'Y (um)'
    zlabel = 'Z (um)'
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposn_3D, yposn_3D, zposn_3D = get_component_3D_arrays(node_posns,(int(Lx+1),int(Ly+1),int(Lz+1)))
    xposn_3D *= l_e*1e6
    yposn_3D *= l_e*1e6
    zposn_3D *= l_e*1e6
    layers = (0,Lx,0,Ly,0,Lz)
    Lx *= l_e*1e6
    Ly *= l_e*1e6
    Lz *= l_e*1e6
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
    for cut_type_index in range(3):
        for i in range(2):
            idx = int(layers[2*cut_type_index+i])
            if cut_type_index == 0:
                xvar = xposn_3D[idx,:,:]
                yvar = yposn_3D[idx,:,:]
                zvar = zposn_3D[idx,:,:]
            elif cut_type_index == 1:
                xvar = xposn_3D[:,idx,:]
                yvar = yposn_3D[:,idx,:]
                zvar = zposn_3D[:,idx,:]
            else:
                xvar = xposn_3D[:,:,idx]
                yvar = yposn_3D[:,:,idx]
                zvar = zposn_3D[:,:,idx]
            surf = ax.plot_surface(xvar,yvar,zvar,rstride=1,cstride=1)
            ax.plot_wireframe(xvar,yvar,zvar,rstride=1,cstride=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    # if tag != "":
    #     ax.set_title(tag)
    savename = output_dir + f'outer_surfaces_3D_' + tag +'.png'
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

def setup_figure_and_variables_3D(cut_type,eq_node_posns,node_posns):
    """Perform setup and return figure handle, axis handle, and variables, and set axis labels and limits for 3D plots"""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    xlabel = 'X (l_e)'
    ylabel = 'Y (l_e)'
    zlabel = 'Z (l_e)'
    xlim = (-0.1,Lx*1.1)
    ylim = (-0.1,Ly*1.1)
    zlim = (-0.1,Lz*1.1)
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
        xlim = (xvar.min(),xvar.max())
    elif cut_type_index == 1:
        idx = int(center[1])
        xvar = xposn_3D[:,idx,:]
        yvar = yposn_3D[:,idx,:]
        zvar = zposn_3D[:,idx,:]
        ylim = (yvar.min(),yvar.max())
    else:
        idx = int(center[2])
        xvar = xposn_3D[:,:,idx]
        yvar = yposn_3D[:,:,idx]
        zvar = zposn_3D[:,:,idx]
        zlim = (zvar.min(),zvar.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    return fig, ax, xvar, yvar, zvar

def format_figure(ax,title_size=30,label_size=30,tick_size=30):
    """Given the axis handle, adjust the font sizes of the title, axis labels, and tick labels."""
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(ax.get_xlabel(),fontsize=label_size)
    ax.set_ylabel(ax.get_ylabel(),fontsize=label_size)
    ax.set_title(ax.get_title(),fontsize=title_size)

def main():
    pass

if __name__=="__main__":
    main()