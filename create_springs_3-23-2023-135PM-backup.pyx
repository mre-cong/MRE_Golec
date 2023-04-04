cimport cython
from libc.math cimport sqrt
from libc.math cimport fabs
cimport libc.math
cimport numpy as np
import numpy as np

cdef eps = np.spacing(1)
#given the node positiondds and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int create_springs(double[:,::1] node_posns, double[::1] stiffness_constants, double cube_side_length, double Lx, double Ly, double Lz, double[:,::1] springs, double[::1] rij_mag):
    cdef int i
    cdef int j
    cdef int spring_idx = 0
    cdef int current_spring_idx
    cdef int current_sub_idx = 0
    cdef int my_counter
    cdef double[3] posn
    cdef double face_diagonal_length = sqrt(2)*cube_side_length
    cdef double center_diagonal_length = sqrt(3)*cube_side_length
    cdef double[3] mydim
    mydim[0] = Lx
    mydim[1] = Ly
    mydim[2] = Lz
    cdef double[6][4] edge_springs = np.empty((6,4),np.dtype(np.float64))
    cdef double[12][4] face_springs = np.empty((12,4),np.dtype(np.float64))
    cdef double[8][4] diagonal_springs = np.empty((8,4),np.dtype(np.float64))
    cdef double[3] rij
    cdef double rijsquared
    cdef int num_springs = 0
    cdef int num_edge_springs = 0
    cdef int num_face_springs = 0
    cdef int num_center_springs = 0
    # # springs = np.zeros((1,4),dtype=np.float64)#creating an array that will hold the springs, will have to concatenate as new springs are added, and delete the first row before passing back
    # rij = np.empty(node_posns.shape,dtype=np.float64)
    # rij_mag = np.empty((node_posns.shape[0],),dtype=np.float64)
    for i in range(node_posns.shape[0]):
        # posn[0] = node_posns[i,0]
        # posn[1] = node_posns[i,1]
        # posn[2] = node_posns[i,2]
        for j in range(node_posns.shape[0]):
            # rij[j,0] = posn[0] - node_posns[j,0]
            # rij[j,1] = posn[1] - node_posns[j,1]
            # rij[j,2] = posn[2] - node_posns[j,2]
            rij[0] = node_posns[i,0] - node_posns[j,0]
            rij[1] = node_posns[i,1] - node_posns[j,1]
            rij[2] = node_posns[i,2] - node_posns[j,2]
            rijsquared = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2]
            rij_mag[j] = libc.math.sqrt(rijsquared)
    #     # rij_mag = np.sqrt(np.sum(rij**2,1))
        #get the row indices of the connected nodes that are connected by edge springs
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*cube_side_length).nonzero()[0]).astype(np.int32)
        num_edge_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[0]/4,cube_side_length,4,edge_springs)
        num_springs += num_edge_springs
        #use the count of number of edge springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = edge_springs[current_sub_idx][my_counter]    
            current_sub_idx += 1
        spring_idx += num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*face_diagonal_length).nonzero()[0]).astype(np.int32)
        num_face_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[1]/2,face_diagonal_length,2,face_springs)
        num_springs += num_face_springs
        #use the count of number of face springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = face_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx += num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*center_diagonal_length).nonzero()[0]).astype(np.int32)
        num_center_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[2],center_diagonal_length,1,diagonal_springs)
        num_springs += num_center_springs
        #use the count of number of diagonal springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = diagonal_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx += num_springs
        current_sub_idx = 0
        #functionalize the above, D.R.Y.
        # get_node_springs(i,node_posns,rij_mag,mydim,stiffness_constants[0]/4,cube_side_length,4,edge_springs)
        # get_node_springs(i,node_posns,rij_mag,mydim,stiffness_constants[1]/2,face_diagonal_length,2,face_springs)
        # get_node_springs(i,node_posns,rij_mag,mydim,stiffness_constants[2],center_diagonal_length,1,diagonal_springs)
        # springs = np.concatenate((springs,edge_springs,face_springs,diagonal_springs),dtype=np.float64)
    # return np.ascontiguousarray(springs[1:],dtype=np.float64)#want a C-contiguous memory representation for using cythonized compiled functionality, where information on memory structure can provide performance speedups
    return num_springs
# def create_springs(node_posns,stiffness_constants,cube_side_length,dimensions):
#     face_diagonal_length = np.sqrt(2)*cube_side_length
#     center_diagonal_length = np.sqrt(3)*cube_side_length
#     springs = np.zeros((1,4),dtype=np.float64)#creating an array that will hold the springs, will have to concatenate as new springs are added, and delete the first row before passing back
#     for i, posn in enumerate(node_posns):
#         rij = posn - node_posns
#         rij_mag = np.sqrt(np.sum(rij**2,1))
#         edge_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
#         face_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
#         diagonal_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[2],center_diagonal_length,max_shared_elements=1)
#         springs = np.concatenate((springs,edge_springs,face_springs,diagonal_springs),dtype=np.float64)
#     return np.ascontiguousarray(springs[1:],dtype=np.float64)#want a C-contiguous memory representation for using cythonized compiled functionality, where information on memory structure can provide performance speedups

#functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
@cython.boundscheck(False)
@cython.wraparound(False)
# cdef double[:,::1] get_node_springs(int i):
cdef int get_node_springs(int idx, double[:,::1] node_posns, int[::1] connected_vertices, double[3] dimensions, double stiffness_constant, double comparison_length, int max_shared_elements, double[:,::1] springs):#, dimensions, double stiffness_constant, double comparison_length, int max_shared_elements, double[:,::1] springs):
# cdef void get_node_springs(int idx, double[:,::1] node_posns, double[::1] rij_mag, dimensions, double stiffness_constant, double comparison_length, int max_shared_elements, double[:,::1] springs):
    """Set the stiffness of a particular spring based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    cdef double[3] node_posn_v
    cdef double[3] node_posn_i
    node_posn_i[0] = node_posns[idx,0]
    node_posn_i[1] = node_posns[idx,1]
    node_posn_i[2] = node_posns[idx,2]
    cdef int i
    cdef int v
    cdef int count = 0
    cdef int row
    cdef int[12] valid_connections#since i am doing each spring type separately, the max size is really 12 (12 face springs max per node)
    # moved to parent function # connected_vertices = np.isclose(rij_mag,comparison_length).nonzero()[0]
    # # TODO !!! remove the below line that has been commented out, if the results are accurate between the method above and the method below (which they should be and look to be based on tests so far)
    # # connected_vertices = np.asarray(np.abs(rij_mag - comparison_length)/comparison_length < epsilon).nonzero()[0]#per numpy documentation, this method is preferred over np.where if np.where is only passed a condition, instead of a condition and two arrays to select from
    # valid_connections = connected_vertices[idx < connected_vertices]
    #TODO get valid_connections for a cythonized version. I'm not sure if the pythonic masking will work as well here compared to looping with conditional checking
    for i in range(connected_vertices.shape[0]):
        if idx < connected_vertices[i]:
            valid_connections[count] = connected_vertices[i]
            count += 1
    # springs = np.empty((valid_connections.shape[0],4),dtype=np.float64)
    # #trying to preallocate space for springs array based on the number of connected vertices, but if i am trying to not double count springs i will sometimes need less space. how do i know how many are actually going to be used? i guess another condition check?
    if max_shared_elements == 1:#if we are dealing with the center diagonal springs we don't need to count shared elements
        # for row, v in enumerate(valid_connections):#switch to range. for row in range(count): springs[row] = [idx,v,stiffness_constant,comparison_length]#if i have to I can assign each element individually. I can assign 3 of the 4 at the top and the stiffness constant based on the decision tree
        #     springs[row] = [idx,v,stiffness_constant,comparison_length]
        for row in range(count):
            v = valid_connections[row]
            springs[row,0] = idx
            springs[row,1] = v
            springs[row,2] = stiffness_constant
            springs[row,3] = comparison_length
    else:
        node_type_i = identify_node_type(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
        for row in range(count):#for row, v in enumerate(valid_connections):
            v = valid_connections[row]
            springs[row,0] = idx
            springs[row,1] = v
            springs[row,3] = comparison_length
            node_posn_v[0] = node_posns[v,0]
            node_posn_v[1] = node_posns[v,1]
            node_posn_v[2] = node_posns[v,2]
            node_type_v = identify_node_type(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
            if node_type_i == 'i'.encode('ascii') and node_type_v == 'i'.encode('ascii'):
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == 'i'.encode('ascii') and node_type_v == 's'.encode('ascii')) or (node_type_i == 's'.encode('ascii') and node_type_v == 'i'.encode('ascii')):
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == 'i'.encode('ascii') and node_type_v == 'e'.encode('ascii')) or (node_type_i == 'e'.encode('ascii') and node_type_v == 'i'.encode('ascii')):
                springs[row,2] = stiffness_constant*2
            elif node_type_i == 's'.encode('ascii') and node_type_v == 's'.encode('ascii'):
                if max_shared_elements == 4:#two shared elements for a cube edge spring in this case if they are both on the same surface, so check for shared surfaces. otherwise the answer is 4.
                    node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row,2] = stiffness_constant*2
                    else:
                        springs[row,2] = stiffness_constant*4
                else:#face spring, if the two nodes are on the same surface theres only one element, if they are on two different surfaces theyre are two shared elements
                    node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row,2] = stiffness_constant
                    else:#on different surfaces, two shared elements
                        springs[row,2] = stiffness_constant*2
            elif (node_type_i == 's'.encode('ascii') and node_type_v == 'e'.encode('ascii')) or (node_type_i == 'e'.encode('ascii') and node_type_v == 's'.encode('ascii')):
                if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
                    springs[row,2] = stiffness_constant*2
                else:#this is a face spring with only a single element if the edge and surface node have a shared surface
                    node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row,2] = stiffness_constant
                    else:#they don't share a surface, and so they share two elements
                        springs[row,2] = stiffness_constant*2
            elif node_type_i == 'e'.encode('ascii') and node_type_v == 'e'.encode('ascii'):
                #both nodes belong to two surfaces (if they are edge nodes). if the surfaces are the same, then it is a shared edge, if they are not, they are separate edges of the simulated volume. there aer 6 surfaces
                node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                if ((node_i_surf[0] == node_v_surf[0] and node_i_surf[1] == node_v_surf[1] and (node_i_surf[0] != 0 and node_i_surf[1] != 0)) or (node_i_surf[0] == node_v_surf[0] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[0] != 0 and node_i_surf[2] != 0)) or(node_i_surf[1] == node_v_surf[1] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[1] != 0 and node_i_surf[2] != 0))):#if both nodes belong to the same two surfaces, they are on the same edge
                    springs[row,2] = stiffness_constant
                elif max_shared_elements == 4:#if they don't share two surfaces and it's a cube edge spring, they share two elements
                    springs[row,2] = stiffness_constant*2
                else:#if it's a face spring
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):#if they do share a surface, then the face spring has as single element
                        springs[row,2] = stiffness_constant
                    else:#they don't share a single surface, then they diagonally across one another and have two shared elements
                        springs[row,2] = stiffness_constant*2
            elif node_type_i == 'c'.encode('ascii') or node_type_v == 'c'.encode('ascii'):#any spring involving a corner node covered
                springs[row,2] = stiffness_constant
    # return springs
    return count

@cython.boundscheck(False)
@cython.wraparound(False)
cdef char identify_node_type(const double[3] node_posn,const double Lx,const double Ly,const double Lz):
    """Identify the node type (corner, edge, surface, or interior point) based on the node position and the dimensions of the simulation. 
    """
    if ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        #if at extremes in 3 of 3 position components
        return b'c'
    elif (((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps)) or ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)) or ((node_posn[1] == 0 or np.abs(node_posn[1] -Ly) < eps) and (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps))):
        #if at an edge (at extremes in two of the 3 position components)
        return b'e'
    elif ((node_posn[0] == 0 or np.abs(node_posn[0]- Lx) < eps) or (node_posn[1] == 0 or node_posn[1] == Ly) or (node_posn[2] == 0 or np.abs(node_posn[2] -Lz) < eps)):
        return b's'
    else:
        return b'i'

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (int,int,int) get_node_surf(const double[3] node_posn,const double Lx,const double Ly,const double Lz):
    """Return a triplet that provides information on what surface or surfaces, if any, a node is part of."""
    cdef (int,int,int) surfaces = (0, 0, 0)
    if fabs(node_posn[0] - Lx) < eps:
        surfaces[0] = 1
    elif node_posn[0] == 0:
        surfaces[0] = -1
    if fabs(node_posn[1] - Ly) < eps:
        surfaces[1] = 1
    elif node_posn[1] == 0:
        surfaces[1] = -1
    if fabs(node_posn[2] - Lz) < eps:
        surfaces[2] = 1
    elif node_posn[2] == 0:
        surfaces[2] = -1
    return surfaces     