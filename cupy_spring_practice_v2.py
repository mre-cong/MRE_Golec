import cupy as cp
import numpy as np
import scipy
from cupyx.profiler import benchmark
import cupyx
import time
import get_spring_force_cy
import springs
#Given the dimensions of a rectilinear space describing the system of interest, and the side length of the unit cell that will be used to discretize the space, return list of vectors that point to the nodal positions at stress free equilibrium
def discretize_space(Lx,Ly,Lz,cube_side_length):
    """Given the side lengths of a rectilinear space and the side length of the cubic unit cell to discretize the space, return arrays of (respectively) the node positions as an N_vertices x 3 array, N_cells x 8 array, and N_vertices x 8 array"""#??? should it be N_vertices by 8? i can't mix types in the python numpy arrays. if it is N by 8 I can store index values for the unit cells that each node belongs to, and maybe negative values or NaN for the extra entries if the vertex/node doesn't belong to 8 unit cells
    #check the side length compared to the dimensions of the space of interest to determine if the side length is appropriate for the space?
    [x,y,z] = np.meshgrid(np.r_[0:Lx+cube_side_length*0.1:cube_side_length],
                          np.r_[0:Ly+cube_side_length*0.1:cube_side_length],
                          np.r_[0:Lz+cube_side_length*0.1:cube_side_length])
    #one of my ideas for implementing this was to create a single unit cell and tile it to fill the space, which could allow me to create the unit_cell_def array and maybe the node_sharing array more easily
    node_posns = np.concatenate((np.reshape(x,np.size(x))[:,np.newaxis],
                                np.reshape(y,np.size(y))[:,np.newaxis],
                                np.reshape(z,np.size(z))[:,np.newaxis]),1)
    #need to keep track of which nodes belong to a unit cell (at some point)
    # N_el_x = np.int32(round(Lx/cube_side_length))
    # N_el_y = np.int32(round(Ly/cube_side_length))
    # N_el_z = np.int32(round(Lz/cube_side_length))
    # N_el = N_el_x * N_el_y * N_el_z
    # #finding the indices for the nodes/vertices belonging to each element
    # #!!! need to check if there is any ordering to the vertices right now that I can use. I need to have each vertex for each element assigned an identity relative to the element for calculating average edge vectors to estimate the volume after deformation
    # elements = np.empty((N_el,8))
    # counter = 0
    # for i in range(N_el_z):
    #     for j in range(N_el_y):
    #         for k in range(N_el_x):
    #             elements[counter,:] = np.nonzero((node_posns[:,0] <= cube_side_length*(k+1)) & (node_posns[:,0] >= cube_side_length*k) & (node_posns[:,1] >= cube_side_length*j) & (node_posns[:,1] <= cube_side_length*(j+1)) & (node_posns[:,2] >= cube_side_length*i) & (node_posns[:,2] <= cube_side_length*(i+1)))[0]
    #             counter += 1
    # top_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].max())[0]
    # bot_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].min())[0]
    # left_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].min())[0]
    # right_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].max())[0]
    # front_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].min())[0]
    # back_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].max())[0]
    # boundaries = {'top': top_bdry, 'bot': bot_bdry, 'left': left_bdry, 'right': right_bdry, 'front': front_bdry, 'back': back_bdry}
    return np.float32(node_posns)#, np.int32(elements), boundaries

def get_boundaries(node_posns):
    top_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].max())[0]
    bot_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].min())[0]
    left_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].min())[0]
    right_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].max())[0]
    front_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].min())[0]
    back_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].max())[0]
    boundaries = {'top': top_bdry, 'bot': bot_bdry, 'left': left_bdry, 'right': right_bdry, 'front': front_bdry, 'back': back_bdry}  
    return boundaries

#given the material properties (Young's modulus, shear modulus, and poisson's ratio) of an isotropic material, calculate the spring stiffness constants for edge springs, center diagonal springs, and face diagonal springs for a cubic unit cell
def get_spring_constants(E,nu,l_e):
    """given the Young's modulus, poisson's ratio, and the length of the edge springs, calculate the edge, central diagonal, and face diagonal stiffness constants of the system"""
    A = 1 #ratio of the stiffness constants of the center diagonal to face diagonal springs
    k_e = 0.4 * (E * l_e) * (8 + 3 * A) / (4 + 3 * A)
    k_c = 1.2 * (E * l_e) / (4 + 3 * A)
    k_f = A * k_c
    k = [k_e, k_f, k_c]
    return k

def get_kappa(E,nu):
    """Given the Young's modulus and Poissons's ratio, return the value of the additional bulk modulus, kappa, for the volume correction forces"""
    kappa = E * (4 * nu - 1) / (2 * (1 + nu) * (1 - 2 * nu))
    return kappa

    #given the node positions and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
def create_springs(node_posns,stiffness_constants,cube_side_length,dimensions):
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #if unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    springs = np.zeros((1,4),dtype=np.float64)#creating an array that will hold the springs, will have to concatenate as new springs are added, and delete the first row before passing back
    for i, posn in enumerate(node_posns):
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        edge_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        face_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        # now i get to figure out how to do diagonal springs, and also how to combine all these freaking things properly
        diagonal_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[2],center_diagonal_length,max_shared_elements=1)
        springs = np.concatenate((springs,edge_springs,face_springs,diagonal_springs),dtype=np.float64)
    return np.ascontiguousarray(springs[1:],dtype=np.float32)#want a C-contiguous memory representation for using cythonized compiled functionality, where information on memory structure can provide performance speedups

#functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
def get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constant,comparison_length,max_shared_elements):
    """setting the stiffness of a particular element based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    connected_vertices = np.asarray(np.abs(rij_mag - comparison_length) < epsilon).nonzero()[0]#per numpy documentation, this method is preferred over np.where if np.where is only passed a condition, instead of a condition and two arrays to select from
    valid_connections = connected_vertices[i < connected_vertices]
    springs = np.empty((valid_connections.shape[0],4),dtype=np.float64)
    #trying to preallocate space for springs array based on the number of connected vertices, but if i am trying to not double count springs i will sometimes need less space. how do i know how many are actually going to be used? i guess another condition check?
    if max_shared_elements == 1:#if we are dealing with the center diagonal springs we don't need to count shared elements
        for row, v in enumerate(valid_connections):
            springs[row] = [i,v,stiffness_constant,comparison_length]
    else:
        node_type_i = identify_node_type(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
        for row, v in enumerate(valid_connections):
            node_type_v = identify_node_type(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
            if node_type_i == 'interior' and node_type_v == 'interior':
                if max_shared_elements == 4:
                    springs[row] = [i,v,stiffness_constant*4,comparison_length]
                else:
                    springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif (node_type_i == 'interior' and node_type_v == 'surface') or (node_type_i == 'surface' and node_type_v == 'interior'):
                if max_shared_elements == 4:
                    springs[row] = [i,v,stiffness_constant*4,comparison_length]
                else:
                    springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif (node_type_i == 'interior' and node_type_v == 'edge') or (node_type_i == 'edge' and node_type_v == 'interior'):
                springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif node_type_i == 'surface' and node_type_v == 'surface':
                if max_shared_elements == 4:#two shared elements for a cube edge spring in this case if they are both on the same surface, so check for shared surfaces. otherwise the answer is 4.
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row] = [i,v,stiffness_constant*2,comparison_length]
                    else:
                        springs[row] = [i,v,stiffness_constant*4,comparison_length]
                else:#face spring, if the two nodes are on the same surface theres only one element, if they are on two different surfaces theyre are two shared elements
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row] = [i,v,stiffness_constant,comparison_length]
                    else:#on different surfaces, two shared elements
                        springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif (node_type_i == 'surface' and node_type_v == 'edge') or (node_type_i == 'edge' and node_type_v == 'surface'):
                if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
                    springs[row] = [i,v,stiffness_constant*2,comparison_length]
                else:#this is a face spring with only a single element if the edge and surface node have a shared surface
                    node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row] = [i,v,stiffness_constant,comparison_length]
                    else:#they don't share a surface, and so they share two elements
                        springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif node_type_i == 'edge' and node_type_v == 'edge':
                #both nodes belong to two surfaces (if they are edge nodes). if the surfaces are the same, then it is a shared edge, if they are not, they are separate edges of the simulated volume. there aer 6 surfaces
                node_i_surf = get_node_surf(node_posns[i,:],dimensions[0],dimensions[1],dimensions[2])
                node_v_surf = get_node_surf(node_posns[v,:],dimensions[0],dimensions[1],dimensions[2])
                if ((node_i_surf[0] == node_v_surf[0] and node_i_surf[1] == node_v_surf[1] and (node_i_surf[0] != 0 and node_i_surf[1] != 0)) or (node_i_surf[0] == node_v_surf[0] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[0] != 0 and node_i_surf[2] != 0)) or(node_i_surf[1] == node_v_surf[1] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[1] != 0 and node_i_surf[2] != 0))):#if both nodes belong to the same two surfaces, they are on the same edge
                    springs[row] = [i,v,stiffness_constant,comparison_length]
                elif max_shared_elements == 4:#if they don't share two surfaces and it's a cube edge spring, they share two elements
                    springs[row] = [i,v,stiffness_constant*2,comparison_length]
                else:#if it's a face spring
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):#if they do share a surface, then the face spring has as single element
                        springs[row] = [i,v,stiffness_constant,comparison_length]
                    else:#they don't share a single surface, then they diagonally across one another and have two shared elements
                        springs[row] = [i,v,stiffness_constant*2,comparison_length]
            elif node_type_i == 'corner' or node_type_v == 'corner':#any spring involving a corner node covered
                springs[row] = [i,v,stiffness_constant,comparison_length]
    return springs

def identify_node_type(node_posn,Lx,Ly,Lz):
    """based on the node position and the dimensions of the simulation, identify if the node is a corner, edge, surface, or interior point
    """
    eps = np.spacing(1)
    if ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        #if at extremes in 3 of 3 position components
        return 'corner'
    elif (((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps)) or ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)) or ((node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps))):
        #if at an edge (at extremes in two of the 3 position components)
        return 'edge'
    elif ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) or (node_posn[1] == 0 or node_posn[1] == Ly) or (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        return 'surface'
    else:
        return 'interior'

def get_node_surf(node_posn,Lx,Ly,Lz):
    eps = np.spacing(1)
    surfaces = [0, 0, 0]
    if np.abs(node_posn[0]- Lx) < eps:
        surfaces[0] = 1
    elif node_posn[0] == 0:
        surfaces[0] = -1
    if np.abs(node_posn[1] -Ly) < eps:
        surfaces[1] = 1
    elif node_posn[1] == 0:
        surfaces[1] = -1
    if np.abs(node_posn[2] -Lz) < eps:
        surfaces[2] = 1
    elif node_posn[2] == 0:
        surfaces[2] = -1
    return surfaces     

def main():
    E = 1
    nu = 0.49
    l_e = 0.1
    Lx = 25.6
    Ly = 12.8
    Lz = 12.8
    node_posns = discretize_space(Lx,Ly,Lz,l_e)
    boundaries = get_boundaries(node_posns)
    k = get_spring_constants(E,nu,l_e)
    k = np.array(k,dtype=np.float64)
    dimensions = np.array([Lx,Ly,Lz])
    # edges = create_springs(node_posns,k,l_e,dimensions)
    node_types = springs.get_node_type(node_posns.shape[0],boundaries,dimensions,l_e)
    max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
    edges = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, edges, max_springs, k, dimensions, l_e)
    edges = edges[:num_springs,:]
    
    #raw kernels are important for dealing with more complicated user defined kernels.
    #proper calculation of the index each thread should use requires some thought based on the grid size(number of blocks) and block size (number of threads per block)
    #atomic operations prevent threads from accessing for writing to the same memory address, but slow things down by forcing those threads to operate sequentially instead of concurrently
    #atomci operations can force single access across a system (multipleGPUs) a single device, or across a thread block/thread group depending on what is invoked
    #global memory access is slow compared to shared memory, but shared memory (shared across a thread block or a group of thread blocks with certain semantics)
    #still requires careful thought due to size limits and the need to update a larger data structure later anyway
    #conditional checking needs to be used to prevent accessing data outside of bounds when the number of threads is larger than the number of operations/elements to operate on
    #conditional checks are slow in general, and across a warp (32 threads which can operate together quickly on read and execution) each thread should traverse the same conditional ath or issues arse
    #it would be great to coalesce access to global memory if possible to reduce the overhead of global memory access
    #cuda kernels really don't like arguments that are pointers to pointers (float**, etc), which is a bit of an issue, since it means i need to reshape every 2d or 3d thing to a vector. but at least it is manageable... then i need to take into account "striding", but just in terms of indices, not memory size/bytes per element
    spring_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void spring_force(const float* edges, const float* node_posns, float* forces, const int size_edges) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size_edges)
        {
            int iid = edges[4*tid];
            int jid = edges[4*tid+1];
            //printf("iid = %i, jid = %i\n",iid,jid);
            float rij[3];
            rij[0] = node_posns[3*iid]-node_posns[3*jid];
            rij[1] = node_posns[3*iid+1]-node_posns[3*jid+1];
            rij[2] = node_posns[3*iid+2]-node_posns[3*jid+2];
            //printf("tid = %i, node_posns[3*%i] = %f, node_posns[3*%i+1] = %f, node_posns[3*%i+2] = %f\n",tid,iid,node_posns[3*iid],iid,node_posns[3*iid+1],iid,node_posns[3*iid+2]);
            //printf("tid = %i, node_posns[3*%i] = %f, node_posns[3*%i+1] = %f, node_posns[3*%i+2] = %f\n",tid,jid,node_posns[3*jid],jid,node_posns[3*jid+1],jid,node_posns[3*jid+2]);
            //printf("tid = %i, rij[0] = %f, rij[1] = %f, rij[2] = %f\n",tid,rij[0],rij[1],rij[2]);
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

    node_posns[:,0] *= 1.05
    # cupy_node_posns = cp.array(node_posns).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
    cupy_edges = cp.array(edges.astype(np.float32)).reshape((edges.shape[0]*edges.shape[1],1),order='C')
    N_nodes = node_posns.shape[0]
    # cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)#TODO implement the function to include the instantiation of the GPU memory for the forces, edges, and positions (since that overhead will exist every time prior to executing the kernel)
    size_edges = edges.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_edges/block_size)))/14)*14))
    N_iterations = 100
    def spring_func_w_less_transfers(cupy_edges,node_posns,N_nodes,size_edges):
        cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
        cupy_node_posns = cp.array(node_posns).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
        spring_kernel((grid_size,),(block_size,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
        host_cupy_forces = cp.asnumpy(cupy_forces)
    def spring_func_w_transfers(edges,node_posns,N_nodes,size_edges):
        cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
        cupy_node_posns = cp.array(node_posns).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
        cupy_edges = cp.array(edges.astype(np.float32)).reshape((edges.shape[0]*edges.shape[1],1),order='C')
        spring_kernel((grid_size,),(block_size,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
        host_cupy_forces = cp.asnumpy(cupy_forces)
    def spring_func(cupy_edges,cupy_node_posns,cupy_forces,size_edges):
        spring_kernel((grid_size,),(block_size,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
    execution_gpu = cupyx.profiler.benchmark(spring_func_w_less_transfers,(cupy_edges,node_posns,N_nodes,size_edges),n_repeat=N_iterations)
    # execution_gpu = cupyx.profiler.benchmark(spring_func_w_transfers,(edges,node_posns,N_nodes,size_edges),n_repeat=N_iterations)
    # execution_gpu = cupyx.profiler.benchmark(spring_func_w_transfers,(cupy_edges,cupy_node_posns,cupy_forces,size_edges),n_repeat=N_iterations)
    delta_gpu_naive = np.sum(execution_gpu.gpu_times)
    # start = time.perf_counter()
    # for i in range(N_iterations):
    #     spring_kernel((grid_size,),(block_size,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
    # end = time.perf_counter()
    # grid_size = (int (np.ceil(size_edges/1024)))
    # start = time.perf_counter()
    # spring_kernel((grid_size,),(1024,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
    # cp.cuda.runtime.deviceSynchronize()
    # print(cp.ndarray.get(cupy_forces).reshape((N_nodes,3)))
    # delta_gpu_naive = end-start
    # below this point i am using the code snippet from timestep() to calculate elastic forces using pure python/numpy expressions
    # this code is ripe for improvement even just using pure python and numpy. for example, the enforcement of constant strain
    # could be handled by setting the accelerations of those nodes to zero, which would require doing the normal calculations but which would
    # avoid the costly conditional checks (mostly costly because of the np.any() function calls, which are slow, probably slower as the number of nodes increases
    # and which wouldn't be an issue )
    x0 = np.array(node_posns,dtype=np.float64)
    spring_force = np.empty(x0.shape,dtype=np.float64)
    edges = np.array(edges,dtype=np.float64)
    start = time.perf_counter()
    for i in range(N_iterations):
        get_spring_force_cy.get_spring_forces(x0, edges, spring_force)
    end = time.perf_counter()
    delta_np = end-start
    spring_force = spring_force.astype(np.float32)
    cupy_cython_spring_force = cp.asarray(spring_force).reshape((spring_force.shape[0]*spring_force.shape[1],1),order='C')
    # host_cupy_forces = cp.asnumpy(cupy_forces)
    # try:
    #     correctness = np.allclose(host_cupy_forces,cupy_cython_spring_force)
    # except Exception as inst:
    #         print('Exception raised during calculation')
    #         print(type(inst))
    #         print(inst)
    # print("GPU and CPU based calculations of forces agree?: " + str(correctness))
    print("CPU time is {} seconds".format(delta_np))
    print("GPU time is {} seconds".format(delta_gpu_naive))
    print("GPU is {}x faster than CPU".format(delta_np/delta_gpu_naive))
    return

if __name__ == "__main__":
    main()