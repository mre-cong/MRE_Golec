import cupy as cp
import numpy as np
import scipy
from cupyx.profiler import benchmark
import cupyx
import time

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
    N_el_x = np.int32(round(Lx/cube_side_length))
    N_el_y = np.int32(round(Ly/cube_side_length))
    N_el_z = np.int32(round(Lz/cube_side_length))
    N_el = N_el_x * N_el_y * N_el_z
    #finding the indices for the nodes/vertices belonging to each element
    #!!! need to check if there is any ordering to the vertices right now that I can use. I need to have each vertex for each element assigned an identity relative to the element for calculating average edge vectors to estimate the volume after deformation
    elements = np.empty((N_el,8))
    counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                elements[counter,:] = np.nonzero((node_posns[:,0] <= cube_side_length*(k+1)) & (node_posns[:,0] >= cube_side_length*k) & (node_posns[:,1] >= cube_side_length*j) & (node_posns[:,1] <= cube_side_length*(j+1)) & (node_posns[:,2] >= cube_side_length*i) & (node_posns[:,2] <= cube_side_length*(i+1)))[0]
                counter += 1
    top_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].max())[0]
    bot_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].min())[0]
    left_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].min())[0]
    right_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].max())[0]
    front_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].min())[0]
    back_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].max())[0]
    boundaries = {'top': top_bdry, 'bot': bot_bdry, 'left': left_bdry, 'right': right_bdry, 'front': front_bdry, 'back': back_bdry}
    return np.float32(node_posns), np.int32(elements), boundaries


def create_connectivity_sparse(node_posns,elements,stiffness_constants,cube_side_length):
    """Returns a scipy.sparse CSR array of the connectivity and separation matrices."""
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #besides the node positions and stiffness constants (which is more complicated than just 3 numbers for the MRE case where there are two isotropic phases present with different material properties), we need to determine the connectivity matrix by summing up the stiffness contribution from each cell that share the vertices whose element is being calculated. A vertex in the inner part of the body will be shared by 8 cells, while one on a surface boundary may be shared by 4, or at a corner, only belonging to one unit cell. Similarly, if those unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    #since i have a unit cell, and since the connectivity matrix looks the same for each unit cell of a material type, can I construct the connectivity matrix for a single unit cell of each type, and combine them into the overall connectivity matrix? that way I can avoid calculating separations for all nodes when I know that there's only so many connections per node possible (26 total for edge, face diagonal, and center diagonal springs in a cubic cell)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    data = []
    separation_data = []
    row_ind = []
    col_ind = []
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    i = 0
    for posn in node_posns:
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        result = set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(cube_side_length)
        result = set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(face_diagonal_length)
        center_diagonal_indices = np.where(np.abs(rij_mag - center_diagonal_length) < epsilon)[0]
        result = (center_diagonal_indices, np.ones(center_diagonal_indices.size,)*stiffness_constants[2])
        data.extend(result[1])
        col_ind.extend(result[0])
        for index in result[0]:
            row_ind.append(i)
            separation_data.append(center_diagonal_length)
        i += 1
    sparse_connectivity = scipy.sparse.csr_array((data, (row_ind, col_ind)), shape=(N, N),dtype=np.float32)
    sparse_separations = scipy.sparse.csr_array((separation_data, (row_ind, col_ind)), shape=(N, N),dtype=np.float32)
    return (sparse_connectivity,sparse_separations)
            
def set_stiffness_shared_elements_sparse(i,rij_mag,elements,stiffness_constant,comparison_length,max_shared_elements):
    """Return list of tuples of index of connected node and stiffness constant, setting the stiffness of a particular connection based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    shared_elements = 0
    result = []
    indices = []
    effective_stiffness =[]
    connected_vertices = np.nonzero(np.abs(rij_mag - comparison_length) < epsilon)[0]
    for v in connected_vertices:
        for el in elements:
            if np.any(el == i) & np.any(el == v):
                shared_elements += 1
                if shared_elements == max_shared_elements:
                    break
        tmp_stiffness_constant = stiffness_constant * shared_elements
        indices.append(v)
        effective_stiffness.append(tmp_stiffness_constant)
        shared_elements = 0
    result = (indices,effective_stiffness)
    return result
            
#first goal is to calculate the distance between one node and the other nodes in the cell. then you can use a for loop to iterate over each node. when you know the distances for a single node, can do a conditional check on the magnitude of separation to determine which stiffness constant belongs to which entry in the 8x8 matrix representing connectivity within that unit cell. then can think about how to do that process simultaneously using numpy and scipy functionality
#after that process works, the next step is figuring out how to combine connectivity matrix entries. the more i think about it the less convinced i am that i can combine unit cell connectivity matrices together in a reasonable matter, as the order of the nodes in the connectivity matrix for one unit cell doesn't have particular meaning related to their spacing, and the order of nodes in the overall connectivity matrix involving more than one unit cell will also not be related to the relative positions in space
#i don't need to be clever for this bit, at least not right now. first write something that works, even if it is brute force. try to be clever when it works.


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

#given the node positions and stiffness constants for the different types of springs, calculate and return the connectivity matrix, and equilibrium separation matrix
def create_connectivity(node_posns,elements,stiffness_constants,cube_side_length):
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #besides the node positions and stiffness constants (which is more complicated than just 3 numbers for the MRE case where there are two isotropic phases present with different material properties), we need to determine the connectivity matrix by summing up the stiffness contribution from each cell that share the vertices whose element is being calculated. A vertex in the inner part of the body will be shared by 8 cells, while one on a surface boundary may be shared by 4, or at a corner, only belonging to one unit cell. Similarly, if those unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    #since i have a unit cell, and since the connectivity matrix looks the same for each unit cell of a material type, can I construct the connectivity matrix for a single unit cell of each type, and combine them into the overall connectivity matrix? that way I can avoid calculating separations for all nodes when I know that there's only so many connections per node possible (26 total for edge, face diagonal, and center diagonal springs in a cubic cell)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    connectivity = np.zeros((N,N),dtype=np.float32)
    separations = np.empty((N,N),dtype=np.float32)
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    i = 0
    for posn in node_posns:
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        set_stiffness_shared_elements(i,rij_mag,elements,connectivity,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        set_stiffness_shared_elements(i,rij_mag,elements,connectivity,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        connectivity[np.abs(rij_mag - center_diagonal_length) < epsilon,i] = stiffness_constants[2]
        separations[:,i] = rij_mag
        i += 1
    return (connectivity,separations)

#functionalizing the setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
def set_stiffness_shared_elements(i,rij_mag,elements,connectivity,stiffness_constant,comparison_length,max_shared_elements):
    """setting the stiffness of a particular element based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    shared_elements = 0
    connected_vertices = np.where(np.abs(rij_mag - comparison_length) < epsilon)[0]
    for v in connected_vertices:
        if connectivity[i,v] == 0:
            for el in elements:
                if np.any(el == i) & np.any(el == v):
                    shared_elements += 1
                    if shared_elements == max_shared_elements:
                        break
            connectivity[i,v] = stiffness_constant * shared_elements
            connectivity[v,i] = connectivity[i,v]
            shared_elements = 0

def main():
    E = 1
    nu = 0.49
    l_e = 0.1
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    node_posns, elements, boudnaries = discretize_space(Lx,Ly,Lz,l_e)
    k = get_spring_constants(E,nu,l_e)
    sparse_connectivity, sparse_separation = create_connectivity_sparse(node_posns,elements,k,l_e)
    connectivity, eq_separations = create_connectivity(node_posns,elements,k,l_e)
    # x_gpu = cp.array([1,2,3])
    # l2_gpu = cp.linalg.norm(x_gpu)
    # x_cpu = np.array([1,2,3])
    # newx_gpu = cp.asarray(x_cpu)

    # squared_diff = cp.ElementwiseKernel(
    #     'float32 x, float32 y',
    #     'float32 z',
    #     'z = (x - y) * (x - y)',
    #     'squared_diff')

    # x0 = cp.array([[0.,0.,0.],[1.,0,0]],dtype=cp.float32)
    # x1 = cp.array([[1.,0.,0.],[0.,0,0]],dtype=cp.float32)
    # z = cp.empty((2,3),dtype = cp.float32)
    # z = squared_diff(x0,x1)
    # print(z)

    # l2norm_kernel = cp.ReductionKernel(
    #     'T x',
    #     'T y',
    #     'x * x',
    #     'a + b',
    #     'y = sqrt(a)',
    #     '0',
    #     'l2norm'
    # )
    # x = cp.arange(10,dtype = np.float32).reshape(2,5)
    # y = l2norm_kernel(x,axis=1)
    # print(x)
    # print(y)

    # rij_kernel = cp.ElementwiseKernel(
    #     'float32 x, float32 y',
    #     'float32 z',
    #     'z = x - y',
    #     'rij_kernel'
    # )
    # x1 = cp.array([3.,0.,1.],dtype=np.float32)
    # x2 = cp.array([0.,1.,0.],dtype=np.float32)
    # x_12 = rij_kernel(x1,x2)
    # print(x_12)

    # invl2norm_kernel = cp.ReductionKernel(
    #     'T x',
    #     'T y',
    #     'x * x',
    #     'a + b',
    #     'y = 1/sqrt(a)',
    #     '0',
    #     'invl2norm'
    # )
    # x_12_mag = l2norm_kernel(x_12)
    # print(x_12_mag)
    # x_12_mag_inv = invl2norm_kernel(x_12)
    # print(x_12_mag_inv)

    # rij_hat_kernel = cp.ElementwiseKernel(
    #     'float32 x, float32 y',
    #     'float32 z',
    #     'z = x * y',
    #     'rij_hat_kernel' 
    # )
    # x_12_hat = rij_hat_kernel(x_12,x_12_mag_inv)
    # print(x_12_hat)
    # print(l2norm_kernel(x_12_hat))

    # combined_kernel = cp.ReductionKernel(
    #     'T x, T y',
    #     'T z',
    #     '(x - y) * (x - y)',
    #     'a + b',
    #     'y = 1/sqrt(a)',
    #     '0',
    #     'invl2norm'
    # )

    # add_kernel = cp.RawKernel(r'''
    # extern "C" __global__
    # void my_add(const float* x1, const float* x2, float* y) {
    #     int tid = blockDim.x * blockIdx.x + threadIdx.x;
    #     y[tid] = x1[tid] + x2[tid];
    #     }
    #     ''', 'my_add')
    # x1 = cp.arange(25,dtype = cp.float32).reshape(5,5)
    # x2 = cp.arange(25,dtype = cp.float32).reshape(5,5)
    # y = cp.zeros((5,5),dtype = cp.float32)
    # add_kernel((5,),(5,),(x1,x2,y)) #grid, block and arguments
    # print(y)

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

    print(sparse_connectivity.nnz)
    #adjust the sparse matrix entries into an appropriate data structue called edges, which contains rows describing individual springs
    #the first entry is index i, the second is index j (the row indices in node_posns of the corresponding nodes at the ends of the spring)
    #entry three is the stiffness constant, entry four is the equilibrium length
    edges = cp.zeros((int(sparse_connectivity.nnz/2),4),dtype = cp.float32)
    cupy_sparse_connectivity = cupyx.scipy.sparse.csr_matrix(sparse_connectivity)
    cupy_sparse_separation = cupyx.scipy.sparse.csr_matrix(sparse_separation)
    #returns a tuple of arrays, the first array has the row indices of the nonzero elements, and the second array has the column indices of the nonzero elements
    (row_ids,col_ids) = sparse_connectivity.nonzero()
    #now i need to create the entries for the edges data structure, but i want to avoid doing an edge/spring twice, so i need to keep track of the i,j combinations somehow
    count = 0
    for i, row_id in enumerate(row_ids):
        #since the matrix is symmetric, let's only take the entries in the upper triangle
        if col_ids[i] > row_id:
            edges[count,0] = row_id
            edges[count,1] = col_ids[i]
            edges[count,2] = sparse_connectivity.data[i]
            edges[count,3] = sparse_separation.data[i]
            count += 1
    # print(edges)

    node_posns[:,0] *= 1.05
    cupy_node_posns = cp.array(node_posns).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
    cupy_edges = cp.array(edges).reshape((edges.shape[0]*edges.shape[1],1),order='C')
    N_nodes = node_posns.shape[0]
    cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)
    size_edges = edges.shape[0]
    grid_size = (int (np.ceil(size_edges/1024)))
    start = time.perf_counter()
    spring_kernel((grid_size,),(1024,),(cupy_edges,cupy_node_posns,cupy_forces,size_edges))
    # cp.cuda.runtime.deviceSynchronize()
    print(cp.ndarray.get(cupy_forces).reshape((N_nodes,3)))
    end = time.perf_counter()
    delta_gpu_naive = end-start
    # below this point i am using the code snippet from timestep() to calculate elastic forces using pure python/numpy expressions
    # this code is ripe for improvement even just using pure python and numpy. for example, the enforcement of constant strain
    # could be handled by setting the accelerations of those nodes to zero, which would require doing the normal calculations but which would
    # avoid the costly conditional checks (mostly costly because of the np.any() function calls, which are slow, probably slower as the number of nodes increases
    # and which wouldn't be an issue )
    x0 = node_posns
    separations = np.empty((N_nodes,N_nodes))
    np_forces = np.empty(x0.shape,dtype=np.float32)
    start = time.perf_counter()
    i = 0
    for posn in x0:
        rij = posn - x0
        rij_mag = np.sqrt(np.sum(rij**2,1))
        separations[:,i] = rij_mag
        i += 1
    displacement = separations - eq_separations
    force = -1*connectivity * displacement
    # correction_force = get_volume_correction_force(elements, avg_vectors, kappa, l_e,N)
    i = 0
    for posn in x0:
        rij = posn - x0
        rij_mag = np.sqrt(np.sum(rij**2,1))
        rij_mag[rij_mag == 0] = 1#this shouldn't cause issues, it is here to prevent a load of divide by zero errors occuring. if rij is zero length, it is the vector pointing from the vertex to itself, and so rij/rij_mag will cause a divide by zero warning. by setting the magnitude to 1 in that case we avoid that error, and that value should only occur for the vector pointing to itself, which shouldn't contirbute to the force
        # while i understand at the moment why i calculated the elastic forces that way i did, it is unintuitive. I am trying to use numpy's broadcasting and matrix manipulation to improve speeds, but the transformations aren't obviously useful. maybe there is a clearer way to do this that is still fast enough. or maybe this is the best i can do (though i will need to use cython to create compiled code literally everywhere i have for loops over anything, which means getting more comfortable with cython and cythonic code)
        force_vector = np.transpose(np.tile(force[i,:],(3,1)))*(rij/np.tile(rij_mag[:,np.newaxis],(1,3)))
        force_vector = np.nan_to_num(force_vector,posinf=0)
        np_forces[i] = np.sum(force_vector,0)
        i +=1
    end = time.perf_counter()
    delta_np = end-start
    correctness = cp.allclose(cupy_forces,cp.array(np_forces).reshape((np_forces.shape[0]*np_forces.shape[1],1),order='C'))
    print("GPU and CPU based calculations of forces agree?: " + str(correctness))
    print("CPU time is {} seconds".format(delta_np))
    print("GPU time is {} seconds".format(delta_gpu_naive))
    print("GPU is {}x faster than CPU".format(delta_np/delta_gpu_naive))
    return

if __name__ == "__main__":
    main()