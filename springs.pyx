cimport cython
from libc.math cimport sqrt
from libc.math cimport fabs
cimport libc.math
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulate_spring_forces(float[:,::1] gpu_spring_force, double[:,::1] springs, float[:,::1] spring_force):
    cdef int i
    cdef int node_idx_i
    cdef int node_idx_j
    for i in range(gpu_spring_force.shape[0]):
        node_idx_i = int(springs[i,0])
        node_idx_j = int(springs[i,1])
        spring_force[node_idx_i,0] += gpu_spring_force[i,0]
        spring_force[node_idx_i,1] += gpu_spring_force[i,1]
        spring_force[node_idx_i,2] += gpu_spring_force[i,2]
        spring_force[node_idx_j,0] -= gpu_spring_force[i,0]
        spring_force[node_idx_j,1] -= gpu_spring_force[i,1]
        spring_force[node_idx_j,2] -= gpu_spring_force[i,2]

#alternative implementation. start with node 0, counting up elements until you hit N_el_x. with node 0 get the other node indices by a similar mechanism used for the spring variable setting, how many nodes per line, or per plane, to get the other 7 entries. keeping track of the counter index compared to the number of elements in each direction to avoid continuing when you've just finished a boundary element. should be significantly faster than generating the positions and then going backwards to the index, and i need the additional speed up. that being said, the current implementation is, for 100x100x10 elements, ~26 times faster than the pure python. but i can do better than that.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] get_elements_v2(double[:] dimensions, double cube_side_length):
    cdef int N_el_x = np.int32(round(dimensions[0]/cube_side_length))
    cdef int N_el_y = np.int32(round(dimensions[1]/cube_side_length))
    cdef int N_el_z = np.int32(round(dimensions[2]/cube_side_length))
    cdef int N_el = N_el_x * N_el_y * N_el_z
    cdef np.ndarray[np.int32_t, ndim=2] elements = np.empty((N_el,8),dtype=np.int32)
    cdef int i
    cdef int j
    cdef int k
    cdef int nodes_per_line = N_el_z + 1
    cdef int nodes_per_plane = (N_el_x + 1)*(N_el_z + 1)
    cdef int counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                elements[counter,0] = k*nodes_per_line + j*nodes_per_plane + i 
                elements[counter,1] = k*nodes_per_line + j*nodes_per_plane + i + 1
                elements[counter,2] = (k+1)*nodes_per_line + j*nodes_per_plane + i 
                elements[counter,3] = (k+1)*nodes_per_line + j*nodes_per_plane + i + 1
                elements[counter,4] = k*nodes_per_line + (j+1)*nodes_per_plane + i 
                elements[counter,5] = k*nodes_per_line + (j+1)*nodes_per_plane + i + 1 
                elements[counter,6] = (k+1)*nodes_per_line + (j+1)*nodes_per_plane + i 
                elements[counter,7] = (k+1)*nodes_per_line + (j+1)*nodes_per_plane + i + 1
                counter += 1
    return elements

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] get_elements_v2_normalized(double N_nodes_x, double N_nodes_y, double N_nodes_z):
    """given the number of nodes in each direction, return an N_elements by 8 array where each row represents a single volume element and each column is the associated row index in node_posns of a vertex of the volume element"""
    #need to keep track of which nodes belong to a unit cell (at some point)""""
    #TODO move this part outside of the cdef function... and call a cdef function that returns void but sets the elements variable
    cdef int N_el_x = np.int32(N_nodes_x-1)
    cdef int N_el_y = np.int32(N_nodes_y-1)
    cdef int N_el_z = np.int32(N_nodes_z-1)
    cdef int N_el = N_el_x * N_el_y * N_el_z
    cdef np.ndarray[np.int32_t, ndim=2] elements = np.empty((N_el,8),dtype=np.int32)
    cdef int i
    cdef int j
    cdef int k
    cdef int nodes_per_line = N_el_z + 1
    cdef int nodes_per_plane = (N_el_x + 1)*(N_el_z + 1)
    cdef int counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                elements[counter,0] = k*nodes_per_line + j*nodes_per_plane + i 
                elements[counter,1] = k*nodes_per_line + j*nodes_per_plane + i + 1
                elements[counter,2] = (k+1)*nodes_per_line + j*nodes_per_plane + i 
                elements[counter,3] = (k+1)*nodes_per_line + j*nodes_per_plane + i + 1
                elements[counter,4] = k*nodes_per_line + (j+1)*nodes_per_plane + i 
                elements[counter,5] = k*nodes_per_line + (j+1)*nodes_per_plane + i + 1 
                elements[counter,6] = (k+1)*nodes_per_line + (j+1)*nodes_per_plane + i 
                elements[counter,7] = (k+1)*nodes_per_line + (j+1)*nodes_per_plane + i + 1
                counter += 1
    return elements

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] get_elements(double[:,::1] node_posns, double[:] dimensions, double cube_side_length):
    """given the node/vertex positions, dimensions of the simulated volume, and volume element edge length, return an N_elements by 8 array where each row represents a single volume element and each column is the associated row index in node_posns of a vertex of the volume element"""
    #need to keep track of which nodes belong to a unit cell (at some point)
    #TODO move this part outside of the cdef function... and call a cdef function that returns void but sets the elements variable
    cdef int N_el_x = np.int32(round(dimensions[0]/cube_side_length))
    cdef int N_el_y = np.int32(round(dimensions[1]/cube_side_length))
    cdef int N_el_z = np.int32(round(dimensions[2]/cube_side_length))
    cdef int N_el = N_el_x * N_el_y * N_el_z
    #finding the indices for the nodes/vertices belonging to each element
    #!!! need to check if there is any ordering to the vertices right now that I can use. I need to have each vertex for each element assigned an identity relative to the element for calculating average edge vectors to estimate the volume after deformation
    cdef np.ndarray[np.int32_t, ndim=2] elements = np.empty((N_el,8),dtype=np.int32)
    cdef int i
    cdef int j
    cdef int k
    cdef double[8][3] posns = np.empty((8,3),dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] element = np.empty((8,),dtype=np.int32)
    cdef int counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                posns[0][0] = k*cube_side_length
                posns[0][1] = j*cube_side_length
                posns[0][2] = i*cube_side_length
                posns[1][0] = k*cube_side_length
                posns[1][1] = j*cube_side_length
                posns[1][2] = (i+1)*cube_side_length
                posns[2][0] = (k+1)*cube_side_length
                posns[2][1] = j*cube_side_length
                posns[2][2] = i*cube_side_length
                posns[3][0] = (k+1)*cube_side_length
                posns[3][1] = j*cube_side_length
                posns[3][2] = (i+1)*cube_side_length
                posns[4][0] = k*cube_side_length
                posns[4][1] = (j+1)*cube_side_length
                posns[4][2] = i*cube_side_length
                posns[5][0] = k*cube_side_length
                posns[5][1] = (j+1)*cube_side_length
                posns[5][2] = (i+1)*cube_side_length
                posns[6][0] = (k+1)*cube_side_length
                posns[6][1] = (j+1)*cube_side_length
                posns[6][2] = i*cube_side_length
                posns[7][0] = (k+1)*cube_side_length
                posns[7][1] = (j+1)*cube_side_length
                posns[7][2] = (i+1)*cube_side_length
                element = get_row_indices(posns,cube_side_length,dimensions)
                elements[counter,:] = element
                counter +=1
    return elements
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] get_elements_normalized(double N_nodes_x, double N_nodes_y, double N_nodes_z):
    """given the number of nodes in each direction, return an N_elements by 8 array where each row represents a single volume element and each column is the associated row index in node_posns of a vertex of the volume element"""
    #need to keep track of which nodes belong to a unit cell (at some point)""""
    #TODO move this part outside of the cdef function... and call a cdef function that returns void but sets the elements variable
    cdef int N_el_x = np.int32(N_nodes_x-1)
    cdef int N_el_y = np.int32(N_nodes_y-1)
    cdef int N_el_z = np.int32(N_nodes_z-1)
    cdef int N_el = N_el_x * N_el_y * N_el_z
    cdef double cube_side_length = 1.0
    #finding the indices for the nodes/vertices belonging to each element
    #!!! need to check if there is any ordering to the vertices right now that I can use. I need to have each vertex for each element assigned an identity relative to the element for calculating average edge vectors to estimate the volume after deformation
    cdef np.ndarray[np.int32_t, ndim=2] elements = np.empty((N_el,8),dtype=np.int32)
    cdef int i
    cdef int j
    cdef int k
    cdef double[8][3] posns = np.empty((8,3),dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] element = np.empty((8,),dtype=np.int32)
    cdef int counter = 0
    for i in range(N_el_z):
        for j in range(N_el_y):
            for k in range(N_el_x):
                posns[0][0] = k*cube_side_length
                posns[0][1] = j*cube_side_length
                posns[0][2] = i*cube_side_length
                posns[1][0] = k*cube_side_length
                posns[1][1] = j*cube_side_length
                posns[1][2] = (i+1)*cube_side_length
                posns[2][0] = (k+1)*cube_side_length
                posns[2][1] = j*cube_side_length
                posns[2][2] = i*cube_side_length
                posns[3][0] = (k+1)*cube_side_length
                posns[3][1] = j*cube_side_length
                posns[3][2] = (i+1)*cube_side_length
                posns[4][0] = k*cube_side_length
                posns[4][1] = (j+1)*cube_side_length
                posns[4][2] = i*cube_side_length
                posns[5][0] = k*cube_side_length
                posns[5][1] = (j+1)*cube_side_length
                posns[5][2] = (i+1)*cube_side_length
                posns[6][0] = (k+1)*cube_side_length
                posns[6][1] = (j+1)*cube_side_length
                posns[6][2] = i*cube_side_length
                posns[7][0] = (k+1)*cube_side_length
                posns[7][1] = (j+1)*cube_side_length
                posns[7][2] = (i+1)*cube_side_length
                element = get_row_indices_normalized(posns,N_nodes_x,N_nodes_z)
                elements[counter,:] = element
                counter +=1
    return elements

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=1] get_row_indices_normalized(double[:,:] node_posns, double N_nodes_x, double N_nodes_z):
    """Return the row indices corresponding to the normalized node positions of interest given the simulation dimension parameters"""
    cdef np.int32_t nodes_per_col = np.round(N_nodes_z).astype(np.int32)
    cdef np.int32_t nodes_per_row = np.round(N_nodes_x).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] row_index = np.empty((node_posns.shape[0],),dtype=np.int32)
    cdef int i
    for i in range(node_posns.shape[0]):
        row_index[i] =  int(((nodes_per_col * node_posns[i,0]) + (nodes_per_col * nodes_per_row *node_posns[i,1]) + node_posns[i,2]))
    return row_index

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=1] get_row_indices(double[:,:] node_posns, double l_e, double[:] dim):
    """Return the row indices corresponding to the node positions of interest given the simulation dimension parameters"""
    cdef double Lx = dim[0]
    cdef double Lz = dim[2]
    cdef double inv_l_e = 1/l_e
    cdef np.int32_t nodes_per_col = np.round(Lz/l_e + 1).astype(np.int32)
    cdef np.int32_t nodes_per_row = np.round(Lx/l_e + 1).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] row_index = np.empty((node_posns.shape[0],),dtype=np.int32)
    cdef int i
    for i in range(node_posns.shape[0]):
        row_index[i] =  int(((nodes_per_col * inv_l_e * node_posns[i,0]) + (nodes_per_col * nodes_per_row * inv_l_e *node_posns[i,1]) + inv_l_e *node_posns[i,2]))
    return row_index

@cython.boundscheck(False)
@cython.wraparound(False)
# cpdef np.ndarray[np.float64_t,ndim=2] get_springs(np.ndarray[np.int8_t,ndim=1] node_type,  int max_springs, double[:] k, double[:] dim, double l_e):
cpdef int get_springs(np.ndarray[np.int8_t,ndim=1] node_type, double[:,::1] springs, int max_springs, double[:] k, double[:] dim, double l_e):
#used in get springs to define which adjacent nodes would be above/below/etc the current node of interest, for deciding which connections are possible based on the node type (a node on the top surface can't be connected to a node above it, there are no nodes above it)
    cdef int i
    cdef np.ndarray[np.npy_bool, ndim=1] ABOVE = np.array([True,False,False,False,True,False,False,True,False,False,True,False,True],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] BELOW = np.array([False,False,False,True,False,False,True,False,False,True,False,True,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LEFT = np.array([False,False,False,False,False,True,False,False,False,True,True,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RIGHT = np.array([False,True,False,False,False,False,True,True,True,False,False,True,True],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] FRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] BACK = np.array([False,False,True,True,True,True,False,False,True,True,True,True,True],dtype=np.bool_)
    #combinations of the arrays above to define above and to the left nodes, etc. for deciding which connections are possible based on node type
    cdef np.ndarray[np.npy_bool, ndim=1] LTOP = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LBOT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RTOP = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RBOT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] TOPFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] TOPBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] BOTFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] BOTBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    #combinations of the first set of arrays to define nodes which are placed above to the let and behind, etc.
    cdef np.ndarray[np.npy_bool, ndim=1] LTOPFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LTOPBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LBOTFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] LBOTBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RTOPFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RTOPBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RBOTFRONT = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=1] RBOTBACK = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False],dtype=np.bool_)

    for i in range(13):
        LTOP[i] = LEFT[i] or ABOVE[i]
        LBOT[i] = LEFT[i] or BELOW[i]
        LFRONT[i] = LEFT[i] or FRONT[i]
        LBACK[i] = LEFT[i] or BACK[i]
        RTOP[i] = RIGHT[i] or ABOVE[i]
        RBOT[i] = RIGHT[i] or BELOW[i]
        RFRONT[i] = RIGHT[i] or FRONT[i]
        RBACK[i] = RIGHT[i] or BACK[i]
        TOPFRONT[i] = ABOVE[i] or FRONT[i]
        TOPBACK[i] = ABOVE[i] or BACK[i]
        BOTFRONT[i] = BELOW[i] or FRONT[i]
        BOTBACK[i] = BELOW[i] or BACK[i]
        #combinations of the first set of arrays to define nodes which are placed above to the let and behind, etc.
        LTOPFRONT[i] = LEFT[i] or ABOVE[i] or FRONT[i]
        LTOPBACK[i] = LEFT[i] or ABOVE[i] or BACK[i]
        LBOTFRONT[i] = LEFT[i] or BELOW[i] or FRONT[i]
        LBOTBACK[i] = LEFT[i] or BELOW[i] or BACK[i]
        RTOPFRONT[i] = RIGHT[i] or ABOVE[i] or FRONT[i]
        RTOPBACK[i] = RIGHT[i] or ABOVE[i] or BACK[i]
        RBOTFRONT[i] = RIGHT[i] or BELOW[i] or FRONT[i]
        RBOTBACK[i] = RIGHT[i] or BELOW[i] or BACK[i]

    cdef int j
    cdef int Nz = np.round(dim[2]/l_e + 1).astype(np.int64)
    cdef int Nx = np.round(dim[0]/l_e + 1).astype(np.int64)
    cdef int nodes_per_line = Nz
    cdef int nodes_per_plane = Nx*Nz
    cdef double face_spring_length = sqrt(2)*l_e
    cdef double center_diagonal_length = sqrt(3)*l_e
    # cdef np.ndarray[np.float64_t,ndim=2] springs = np.empty((max_springs,4),dtype=np.float64)
    cdef int spring_counter = 0
    cdef int[13] adjacent_node_indices = np.empty((13,),dtype=np.int32)
    for i in range(node_type.shape[0]):#there are 26 adjacent nodes that could be connected, but because i only want unique connections (i < other_node_index) and i'm iterating over i, i can ignore the 13 that would have lower node indices
        #edge type connections
        adjacent_node_indices[0] = i + 1
        adjacent_node_indices[1] = i + nodes_per_line
        adjacent_node_indices[2] = i + nodes_per_plane
        #face type connections
        adjacent_node_indices[3] = i + nodes_per_plane - 1
        adjacent_node_indices[4] = i + nodes_per_plane + 1
        adjacent_node_indices[5] = i - nodes_per_line + nodes_per_plane
        adjacent_node_indices[6] = i + nodes_per_line - 1
        adjacent_node_indices[7] = i + nodes_per_line + 1
        adjacent_node_indices[8] = i + nodes_per_line + nodes_per_plane
        #center diagonal type connections
        adjacent_node_indices[9] = i - nodes_per_line + nodes_per_plane - 1 
        adjacent_node_indices[10] = i - nodes_per_line + nodes_per_plane + 1 
        adjacent_node_indices[11] = i + nodes_per_line + nodes_per_plane - 1
        adjacent_node_indices[12] = i + nodes_per_line + nodes_per_plane + 1
        #now i need to include the logic for checking based on the node type whether or not there would be nodes to the left, right, above, below or behind (any node in front (lower y value) will always be a lower node index and so won't need to be considered)
        #i should group the adjacent nodes so that they are edge springs, then face springs, then center diagonal springs
#defining which of the potential node connections involve translations above/below/etc
#just define the connections (first two elements of the spring variable. maybe the length? and get the stiffness set later with it's own functionality?)
        if node_type[i] == 0:
            for j in range(3):
                springs[spring_counter,0] = i
                springs[spring_counter,1] = adjacent_node_indices[j]
                springs[spring_counter,2] = k[0]
                springs[spring_counter,3] = l_e
                spring_counter += 1
            for j in range(3,9):
                springs[spring_counter,0] = i
                springs[spring_counter,1] = adjacent_node_indices[j]
                springs[spring_counter,2] = k[1]
                springs[spring_counter,3] = face_spring_length
                spring_counter += 1
            for j in range(9,13):
                springs[spring_counter,0] = i
                springs[spring_counter,1] = adjacent_node_indices[j]
                springs[spring_counter,2] = k[2]
                springs[spring_counter,3] = center_diagonal_length
                spring_counter += 1
            #the logic for springs tiffness setting should be established in a separate function(s)
        elif node_type[i] == 1:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LEFT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 2:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RIGHT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 3:
            spring_counter = set_connection_type_conditional(i, node_type, springs, ABOVE, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 4:
            spring_counter = set_connection_type_conditional(i, node_type, springs, BELOW, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 5:
            spring_counter = set_connection_type_conditional(i, node_type, springs, FRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 6:
            spring_counter = set_connection_type_conditional(i, node_type, springs, BACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 7:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LTOP, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 8:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LBOT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 9:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 10:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 11:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RTOP, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 12:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RBOT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 13:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 14:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 15:
            spring_counter = set_connection_type_conditional(i, node_type, springs, TOPFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 16:
            spring_counter = set_connection_type_conditional(i, node_type, springs, TOPBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 17:
            spring_counter = set_connection_type_conditional(i, node_type, springs, BOTFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 18:
            spring_counter = set_connection_type_conditional(i, node_type, springs, BOTBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 19:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LBOTFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 20:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RBOTFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 21:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LBOTBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 22:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LTOPFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 23:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RBOTBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 24:
           spring_counter = set_connection_type_conditional(i, node_type, springs, RTOPFRONT, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 25:
            spring_counter = set_connection_type_conditional(i, node_type, springs, LTOPBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
        elif node_type[i] == 26:
            spring_counter = set_connection_type_conditional(i, node_type, springs, RTOPBACK, adjacent_node_indices, spring_counter, l_e, face_spring_length, center_diagonal_length, k)
    return spring_counter#springs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int8_t, ndim=1] get_node_type_normalized(int n_nodes, dict boundaries, double[:] dim):
    cdef np.ndarray[np.int8_t, ndim=1] node_type = np.zeros((n_nodes,),dtype=np.int8)
    #zero represents interior nodes, first identify surface nodes, then edges, then corners (edge nodes are surface nodes, corner nodes are edge nodes)
    #using sets for intersection method to determine edges
    cdef set left = set(boundaries['left'])
    cdef set right = set(boundaries['right'])
    cdef set top = set(boundaries['top'])
    cdef set bottom = set(boundaries['bot'])
    cdef set front = set(boundaries['front'])
    cdef set back = set(boundaries['back'])
    node_type[boundaries['left']] = 1
    node_type[boundaries['right']] = 2
    node_type[boundaries['top']] = 3
    node_type[boundaries['bot']] = 4
    node_type[boundaries['front']] = 5
    node_type[boundaries['back']] = 6
    #now edges
    #unpacking the set members, as sets can't be used for magic indexing
    *tmp_var, = left.intersection(top)
    node_type[tmp_var] = 7
    *tmp_var, = left.intersection(bottom)
    node_type[tmp_var] = 8
    *tmp_var, = left.intersection(front)
    node_type[tmp_var] = 9
    *tmp_var, = left.intersection(back)
    node_type[tmp_var] = 10
    *tmp_var, = right.intersection(top)
    node_type[tmp_var] = 11
    *tmp_var, = right.intersection(bottom)
    node_type[tmp_var] = 12
    *tmp_var, = right.intersection(front)
    node_type[tmp_var] = 13
    *tmp_var, = right.intersection(back)
    node_type[tmp_var] = 14
    *tmp_var, = top.intersection(front)
    node_type[tmp_var] = 15
    *tmp_var, = top.intersection(back)
    node_type[tmp_var] = 16
    *tmp_var, = bottom.intersection(front)
    node_type[tmp_var] = 17
    *tmp_var, = bottom.intersection(back)
    node_type[tmp_var] = 18
    cdef np.ndarray[np.float64_t, ndim=2] corners = np.zeros((8,3),dtype=np.float64)
    corners[1][0] = dim[0]
    corners[2][1] = dim[1]
    corners[3][2] = dim[2]
    corners[4][0] = dim[0]
    corners[4][1] = dim[1]
    corners[5][0] = dim[0]
    corners[5][2] = dim[2]
    corners[6][1] = dim[1]
    corners[6][2] = dim[2]
    corners[7][0] = dim[0]
    corners[7][1] = dim[1]
    corners[7][2] = dim[2]
    cdef np.ndarray[np.int32_t, ndim=1] corner_indices = np.empty((8,),dtype=np.int32)
    cdef double N_nodes_x = dim[0] + 1.0
    cdef double N_nodes_z = dim[2] + 1.0
    corner_indices = get_row_indices_normalized(corners,N_nodes_x,N_nodes_z)
    node_type[corner_indices[0]] = 19#leftbotfront
    node_type[corner_indices[1]] = 20#rightbotfront
    node_type[corner_indices[2]] = 21#leftbotback
    node_type[corner_indices[3]] = 22#lefttopfront
    node_type[corner_indices[4]] = 23#rightbotback
    node_type[corner_indices[5]] = 24#righttopfront
    node_type[corner_indices[6]] = 25#lefttopback
    node_type[corner_indices[7]] = 26#righttopback
    #i can adjust this function to return values from 0 to 26, reflecting interior, the 6 different surfaces, 12 edges, and 8 corner types, to get more detailed information about the node type for assigning spring stiffness and determining adjacent node connectivity
    return node_type

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int8_t, ndim=1] get_node_type(int n_nodes, dict boundaries, double[:] dim, double l_e):
    cdef np.ndarray[np.int8_t, ndim=1] node_type = np.zeros((n_nodes,),dtype=np.int8)
    #zero represents interior nodes, first identify surface nodes, then edges, then corners (edge nodes are surface nodes, corner nodes are edge nodes)
    #using sets for intersection method to determine edges
    cdef set left = set(boundaries['left'])
    cdef set right = set(boundaries['right'])
    cdef set top = set(boundaries['top'])
    cdef set bottom = set(boundaries['bot'])
    cdef set front = set(boundaries['front'])
    cdef set back = set(boundaries['back'])
    node_type[boundaries['left']] = 1
    node_type[boundaries['right']] = 2
    node_type[boundaries['top']] = 3
    node_type[boundaries['bot']] = 4
    node_type[boundaries['front']] = 5
    node_type[boundaries['back']] = 6
    # node_type[boundaries['left']] = 1
    # node_type[boundaries['right']] = 1
    # node_type[boundaries['top']] = 1
    # node_type[boundaries['bot']] = 1
    # node_type[boundaries['front']] = 1
    # node_type[boundaries['back']] = 1
    #now edges
    #unpacking the set members, as sets can't be used for magic indexing
    *tmp_var, = left.intersection(top)
    node_type[tmp_var] = 7
    *tmp_var, = left.intersection(bottom)
    node_type[tmp_var] = 8
    *tmp_var, = left.intersection(front)
    node_type[tmp_var] = 9
    *tmp_var, = left.intersection(back)
    node_type[tmp_var] = 10
    *tmp_var, = right.intersection(top)
    node_type[tmp_var] = 11
    *tmp_var, = right.intersection(bottom)
    node_type[tmp_var] = 12
    *tmp_var, = right.intersection(front)
    node_type[tmp_var] = 13
    *tmp_var, = right.intersection(back)
    node_type[tmp_var] = 14
    *tmp_var, = top.intersection(front)
    node_type[tmp_var] = 15
    *tmp_var, = top.intersection(back)
    node_type[tmp_var] = 16
    *tmp_var, = bottom.intersection(front)
    node_type[tmp_var] = 17
    *tmp_var, = bottom.intersection(back)
    node_type[tmp_var] = 18
    # *tmp_var, = left.intersection(top)
    # node_type[tmp_var] = 2
    # *tmp_var, = left.intersection(bottom)
    # node_type[tmp_var] = 2
    # *tmp_var, = left.intersection(front)
    # node_type[tmp_var] = 2
    # *tmp_var, = left.intersection(back)
    # node_type[tmp_var] = 2
    # *tmp_var, = right.intersection(top)
    # node_type[tmp_var] = 2
    # *tmp_var, = right.intersection(bottom)
    # node_type[tmp_var] = 2
    # *tmp_var, = right.intersection(front)
    # node_type[tmp_var] = 2
    # *tmp_var, = right.intersection(back)
    # node_type[tmp_var] = 2
    # *tmp_var, = top.intersection(front)
    # node_type[tmp_var] = 2
    # *tmp_var, = top.intersection(back)
    # node_type[tmp_var] = 2
    # *tmp_var, = bottom.intersection(front)
    # node_type[tmp_var] = 2
    # *tmp_var, = bottom.intersection(back)
    # node_type[tmp_var] = 2
    #
    cdef np.ndarray[np.float64_t, ndim=2] corners = np.zeros((8,3),dtype=np.float64)
    corners[1][0] = dim[0]
    corners[2][1] = dim[1]
    corners[3][2] = dim[2]
    corners[4][0] = dim[0]
    corners[4][1] = dim[1]
    corners[5][0] = dim[0]
    corners[5][2] = dim[2]
    corners[6][1] = dim[1]
    corners[6][2] = dim[2]
    corners[7][0] = dim[0]
    corners[7][1] = dim[1]
    corners[7][2] = dim[2]
    cdef np.ndarray[np.int32_t, ndim=1] corner_indices = np.empty((8,),dtype=np.int32)
    corner_indices = get_row_indices(corners,l_e,dim)
    # node_type[corner_indices] = 3
    node_type[corner_indices[0]] = 19#leftbotfront
    node_type[corner_indices[1]] = 20#rightbotfront
    node_type[corner_indices[2]] = 21#leftbotback
    node_type[corner_indices[3]] = 22#lefttopfront
    node_type[corner_indices[4]] = 23#rightbotback
    node_type[corner_indices[5]] = 24#righttopfront
    node_type[corner_indices[6]] = 25#lefttopback
    node_type[corner_indices[7]] = 26#righttopback
    #i can adjust this function to return values from 0 to 26, reflecting interior, the 6 different surfaces, 12 edges, and 8 corner types, to get more detailed information about the node type for assigning spring stiffness and determining adjacent node connectivity
    return node_type



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int set_connection_type_conditional(int i, np.ndarray[np.int8_t,ndim=1] node_type, double[:,::1] springs, np.ndarray[np.npy_bool, ndim=1] disallowed_connections, int[13] adjacent_node_indices, int spring_counter, double l_e, double face_spring_length, double center_diagonal_length, double[:] k):
    cdef int j
    for j in range(3):
        if not disallowed_connections[j]:
            springs[spring_counter,0] = i
            springs[spring_counter,1] = adjacent_node_indices[j]
            springs[spring_counter,3] = l_e
            #set spring value
            springs[spring_counter,2] = get_edge_stiffness(node_type[i], node_type[adjacent_node_indices[j]], k[0])
            spring_counter += 1
    for j in range(3,9):
        if not disallowed_connections[j]:
            springs[spring_counter,0] = i
            springs[spring_counter,1] = adjacent_node_indices[j]
            springs[spring_counter,3] = face_spring_length
            #set spring value
            springs[spring_counter,2] = get_face_stiffness(node_type[i], node_type[adjacent_node_indices[j]], k[1])
            spring_counter += 1
    for j in range(9,13):
        if not disallowed_connections[j]:
            springs[spring_counter,0] = i
            springs[spring_counter,1] = adjacent_node_indices[j]
            springs[spring_counter,3] = center_diagonal_length
            #set spring value
            springs[spring_counter,2] = k[2]
            spring_counter += 1
    return spring_counter

cdef double get_edge_stiffness(int node_type_i,int node_type_j, double k):
    if node_type_i > 0 and node_type_i < 7:#surface
        if node_type_j == 0:
            return k
        if node_type_j >0 and node_type_j < 7:
            if node_type_i == node_type_j:
                return k/2
            else:
                return k
        if node_type_j > 6 and node_type_j < 19: #edge
            return k/2
        else:
            return k/4
    if node_type_i > 6 and node_type_i < 19: #edge
        if node_type_j > 0 and node_type_j < 7:
            return k/2
        if node_type_j > 6 and node_type_j < 19:
            if node_type_i == node_type_j:
                return k/4
            else:
                return k/2
        else:
            return k/4
    else:
        return k/4

cdef double get_face_stiffness(int node_type_i,int node_type_j, double k):
    if node_type_i > 0 and node_type_i < 7:#surface
        if node_type_j == 0:
            return k
        if node_type_j >0 and node_type_j < 7:
            if node_type_i == node_type_j:#same surface
                return k/2
            else:
                return k
        if node_type_j > 6 and node_type_j < 19: #edge
            if node_type_i == 1 and (node_type_j == 7 or node_type_j == 8 or node_type_j == 9 or node_type_j == 10):
                return k/2
            elif node_type_i == 2 and (node_type_j == 11 or node_type_j == 12 or node_type_j == 13 or node_type_j == 14):
                return k/2
            elif node_type_i == 3 and (node_type_j == 7 or node_type_j == 11 or node_type_j == 15 or node_type_j == 16):
                return k/2
            elif node_type_i == 4 and (node_type_j == 8 or node_type_j == 12 or node_type_j == 17 or node_type_j == 18):
                return k/2
            elif node_type_i == 5 and (node_type_j == 9 or node_type_j == 13 or node_type_j == 15 or node_type_j == 17):
                return k/2
            elif node_type_i == 6 and (node_type_j == 10 or node_type_j == 14 or node_type_j == 16 or node_type_j == 18):
                return k/2
            else:
                return k
        else:
            return k/2
    if node_type_i > 6 and node_type_i < 19: #edge
        if node_type_j == 0:
            return k
        if node_type_j > 0 and node_type_j < 7:
            if node_type_j == 1 and (node_type_i == 7 or node_type_i == 8 or node_type_i == 9 or node_type_i == 10):
                return k/2
            elif node_type_j == 2 and (node_type_i == 11 or node_type_i == 12 or node_type_i == 13 or node_type_i == 14):
                return k/2
            elif node_type_j == 3 and (node_type_i == 7 or node_type_i == 11 or node_type_i == 15 or node_type_i == 16):
                return k/2
            elif node_type_j == 4 and (node_type_i == 8 or node_type_i == 12 or node_type_i == 17 or node_type_i == 18):
                return k/2
            elif node_type_j == 5 and (node_type_i == 9 or node_type_i == 13 or node_type_i == 15 or node_type_i == 17):
                return k/2
            elif node_type_j == 6 and (node_type_i == 10 or node_type_i == 14 or node_type_i == 16 or node_type_i == 18):
                return k/2
            else:
                return k
        if node_type_j > 6 and node_type_j < 19:
            #if they don't share a single surface, then they diagonally across one another and have two shared elements, so we return k. if they do share a surface, they have on shared element, and we return k/2
            if node_type_i == 7 and (node_type_j == 8 or node_type_j == 9 or node_type_j == 10 or node_type_j == 11 or node_type_j == 15 or node_type_j == 16):
                return k/2
            elif node_type_i == 8 and (node_type_j == 7 or node_type_j == 9 or node_type_j == 10 or node_type_j == 12 or node_type_j == 17 or node_type_j == 18):
                return k/2
            elif node_type_i == 9 and (node_type_j == 7 or node_type_j == 8 or node_type_j == 10 or node_type_j == 13 or node_type_j == 15 or node_type_j == 17):
                return k/2
            elif node_type_i == 10 and (node_type_j == 7 or node_type_j == 8 or node_type_j == 9 or node_type_j == 14 or node_type_j == 16 or node_type_j == 18):
                return k/2
            elif node_type_i == 11 and (node_type_j == 12 or node_type_j == 13 or node_type_j == 14 or node_type_j == 7 or node_type_j == 15 or node_type_j == 16):
                return k/2
            elif node_type_i == 12 and (node_type_j == 11 or node_type_j == 13 or node_type_j == 14 or node_type_j == 8 or node_type_j == 17 or node_type_j == 18):
                return k/2
            elif node_type_i == 13 and (node_type_j == 11 or node_type_j == 12 or node_type_j == 14 or node_type_j == 9 or node_type_j == 15 or node_type_j == 17):
                return k/2
            elif node_type_i == 14 and (node_type_j == 11 or node_type_j == 12 or node_type_j == 13 or node_type_j == 10 or node_type_j == 16 or node_type_j == 18):
                return k/2
            elif node_type_i == 15 and (node_type_j == 16 or node_type_j == 7 or node_type_j == 11 or node_type_j == 9 or node_type_j == 13 or node_type_j == 17):
                return k/2
            elif node_type_i == 16 and (node_type_j == 15 or node_type_j == 7 or node_type_j == 11 or node_type_j == 10 or node_type_j == 14 or node_type_j == 18):
                return k/2
            elif node_type_i == 17 and (node_type_j == 18 or node_type_j == 8 or node_type_j == 12 or node_type_j == 9 or node_type_j == 13 or node_type_j == 15):
                return k/2
            elif node_type_i == 18 and (node_type_j == 17 or node_type_j == 8 or node_type_j == 12 or node_type_j == 10 or node_type_j == 14 or node_type_j == 16):
                return k/2
            else:
                return k
        else:
            return k/2
    else:
        return k/2
    #surface type that shares a surface with edge types
    # 1,, 7,8,9,10
    # 2,, 11,12,13,14
    # 3,, 7,11,15,16
    # 4,, 8,12,17,18
    # 5,, 9,13,15,17
    # 6,, 10,14,16,18
    #edge types that share a surface with edge types
    # 7,, 8,9,10,11,15,16
    # 8,, 7,9,10,12,17,18
    # 9,, 7,8,10,13,15,17
    # 10,, 7,8,9,14,16,18
    # 11,, 12,13,14,7,15,16
    # 12,, 11,13,14,8,17,18
    # 13,, 11,12,14,9,15,17
    # 14,, 11,12,13,10,16,18
    # 15,, 16,7,11,9,13,17
    # 16,, 15,7,11,10,14,18
    # 17,, 18,8,12,9,13,15
    # 18,, 17,8,12,10,14,16