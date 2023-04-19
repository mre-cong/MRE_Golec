import matplotlib.pyplot as plt
import numpy as np
import get_spring_force_cy
import get_volume_correction_force_cy_nogil

def post_plot(node_posns,connectivity,stiffness_constants):
    x0 = node_posns
    epsilon = np.spacing(1)
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(x0[:,0],x0[:,1],x0[:,2],'o')
    ax.set_xlim((-0.3,1.2*node_posns[:,0].max()))
    ax.set_ylim((0,1.2*node_posns[:,1].max()))
    ax.set_zlim((0,1.2*node_posns[:,2].max()))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for i in range(len(x0)-1):
        for j in range(i+1,len(x0)):
            if np.abs(connectivity[i,j] - stiffness_constants[0]) <= epsilon or np.abs(connectivity[i,j] - stiffness_constants[0]/2) <= epsilon or np.abs(connectivity[i,j] - stiffness_constants[0]/4) <= epsilon:#connectivity[i,j] != 0:
                x,y,z = (np.array((x0[i,0],x0[j,0])),
                          np.array((x0[i,1],x0[j,1])),
                          np.array((x0[i,2],x0[j,2])))
                ax.plot(x,y,z)

def post_plot_v2(node_posns,springs,boundary_conditions,output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(node_posns[:,0],node_posns[:,1],node_posns[:,2],'o')
    ax.set_xlim((-0.3,1.2*node_posns[:,0].max()))
    ax.set_ylim((0,1.2*node_posns[:,1].max()))
    ax.set_zlim((0,1.2*node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    for spring in springs:
        x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                          np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                          np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
        ax.plot(x,y,z)
    savename = output_dir + 'post_plotv2' + str(np.round(boundary_conditions[2],decimals=2)) +'.png'
    plt.savefig(savename)
    plt.close()

def post_plot_v3(eq_node_posns,node_posns,springs,boundary_conditions,boundaries,output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    boundary_nodes = np.zeros((1,),dtype=int)
    for key, val in boundaries.items():
        boundary_nodes = np.concatenate((boundary_nodes,val))
    ax.scatter(node_posns[boundary_nodes,0],node_posns[boundary_nodes,1],node_posns[boundary_nodes,2],'o')    
    ax.set_xlim((-0.3,1.2*eq_node_posns[:,0].max()))
    ax.set_ylim((0,1.2*eq_node_posns[:,1].max()))
    ax.set_zlim((0,1.2*eq_node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    boundary_nodes = set(np.unique(boundary_nodes))
    for spring in springs:
        subset = set(spring[:2])
        if subset < boundary_nodes:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
            x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                            np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                            np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
            ax.plot(x,y,z)
    savename = output_dir + 'post_plotv3' + str(np.round(boundary_conditions[2],decimals=2)) +'.png'
    plt.savefig(savename)
    plt.close()

#!!! TODO clean up this function, D.R.Y. seriously. also read up on using variable position based arguments, default arguments, and keyword arguments and use that for allowing defauly and more custom plotting behavior involving particles, etc.
def post_plot_cut(eq_node_posns,node_posns,springs,particles,dimensions,l_e,boundary_conditions,output_dir):
    plot_cut('xy',eq_node_posns,node_posns,springs,particles,dimensions,l_e,boundary_conditions,output_dir)
    plot_cut('xz',eq_node_posns,node_posns,springs,particles,dimensions,l_e,boundary_conditions,output_dir)
    plot_cut('yz',eq_node_posns,node_posns,springs,particles,dimensions,l_e,boundary_conditions,output_dir)

def plot_cut(cut_type,eq_node_posns,node_posns,springs,particles,dimensions,l_e,boundary_conditions,output_dir):
    """Plot a cut through the center of the simulated volume, showing the configuration of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut."""
    cut_type_dict = {'xy':0, 'xz':1, 'yz':2}
    cut_type_index = cut_type_dict[cut_type]
    Lx,Ly,Lz = dimensions
    center = (np.round(np.array([Lx/l_e,Ly/l_e,Lz/l_e]))/2) * l_e
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    cut_nodes = np.isclose(np.ones((node_posns.shape[0],))*center[cut_type_index],eq_node_posns[:,cut_type_index]).nonzero()[0]
    if not cut_nodes:#list is empty, central point is not aligned with nodes, try a shift
        cut_nodes1 = np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]+l_e/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
        cut_nodes2 =np.isclose(np.ones((node_posns.shape[0],))*(center[cut_type_index]-l_e/2),eq_node_posns[:,cut_type_index]).nonzero()[0]
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
    ax.set_xlim((-0.3,1.2*eq_node_posns[:,0].max()))
    ax.set_ylim((0,1.2*eq_node_posns[:,1].max()))
    ax.set_zlim((0,1.2*eq_node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    savename = output_dir + f'post_plot_cut_{cut_type}_{center[cut_type_index]}' + str(np.round(boundary_conditions[2],decimals=2)) +'.png'
    plt.savefig(savename)
    plt.close()

def plot_subset_springs(ax,node_posns,nodes,springs,spring_color):
    """Plot a subset of the springs which are connected to the nodes passed to the function"""
    if isinstance(nodes,set):
        nodes_set = nodes
    else:
        nodes_set = set(np.unique(nodes))
    for spring in springs:
        subset = set(spring[:2])
        if subset < nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
            x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                            np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                            np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
            ax.plot(x,y,z,color=spring_color)

#TODO !!! properly incorporate the fact that there might be more than one particle
def post_plot_particle(eq_node_posns,node_posns,particles,springs,boundary_conditions,output_dir):
    particle_node_posns = node_posns[particles]
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    particle_node_posns = node_posns[particles]
    ax.scatter(particle_node_posns[:,0],particle_node_posns[:,1],particle_node_posns[:,2],'o')
    ax.set_xlim((-0.3,1.2*eq_node_posns[:,0].max()))
    ax.set_ylim((0,1.2*eq_node_posns[:,1].max()))
    ax.set_zlim((0,1.2*eq_node_posns[:,2].max()))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(boundary_conditions[0] + ' ' +  boundary_conditions[1][0] + boundary_conditions[1][1] + ' ' + str(boundary_conditions[2]))
    particle_nodes_set = set(particles)
    for spring in springs:
        subset = set(spring[:2])
        if subset < particle_nodes_set:#if the two node indices for the spring are a subset of the node indices for nodes on the boundaries...
            x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                            np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                            np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
            ax.plot(x,y,z)
    savename = output_dir + 'post_plot_particle' + str(np.round(boundary_conditions[2],decimals=2)) +'.png'
    plt.savefig(savename)
    plt.close()

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
#TODO !!! optimize this function. np.any is most likely not necessary. will want to include functionality to calculate accelerations at the end of a simulation, and run a second round (or however many, doing this iteratively) if the accelerations of the nonboundary nodes are high enough to suggest that the system is far from equilibrium
def get_accelerations_post_simulation_v2(x0,boundaries,springs,elements,kappa,l_e,bc):
    N = len(x0)
    m = np.ones(x0.shape[0])*1e-2
    a = np.empty(x0.shape,dtype=float)
    # avg_vectors = get_average_edge_vectors(x0,elements)
    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    correction_force_cy_nogil = np.zeros((N,3),dtype=np.float64)
    get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, correction_force_cy_nogil)
    spring_force_cy = np.empty(x0.shape,dtype=np.float64)
    get_spring_force_cy.get_spring_forces(x0, springs, spring_force_cy)
    for i, posn in enumerate(x0):
        if (np.any(i==boundaries[bc[1][0]]) or np.any(i==boundaries[bc[1][1]])):
            a[i] = (spring_force_cy[i] + correction_force_cy_nogil[i])/m[i]
        else:
            a[i] = 0
    return a

def main():
    pass

if __name__=="__main__":
    main()