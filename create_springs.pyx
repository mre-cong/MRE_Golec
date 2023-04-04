cimport cython
from libc.math cimport sqrt
from libc.math cimport fabs
cimport libc.math
cimport numpy as np
import numpy as np

def get_row_indices(node_posns,l_e,dim):
    """Return the row indices corresponding to the node positions of interest given the simulation dimension parameters"""
    Lx,Ly,Lz = dim
    inv_l_e = 1/l_e
    nodes_per_col = np.round(Lz/l_e + 1).astype(np.int32)
    nodes_per_row = np.round(Lx/l_e + 1).astype(np.int32)
    row_index = ((nodes_per_col * inv_l_e * node_posns[:,0]) + (nodes_per_col * nodes_per_row * inv_l_e *node_posns[:,1]) + inv_l_e *node_posns[:,2]).astype(np.int32)
    return row_index

def create_springs_v3(node_posns,stiffness_constants,cube_side_length,Lx,Ly,Lz,springs):
    cdef double[26][3] relative_positions = np.empty((26,3),np.dtype(np.float64))
    cdef double[26][3] tmp_positions = np.empty((26,3),np.dtype(np.float64))
    cdef double[3] mydim
    mydim[0] = Lx
    mydim[1] = Ly
    mydim[2] = Lz
    cdef int i
    cdef int j
    cdef int[26] row_indices = np.empty((26,),np.dtype(np.int32))
    cdef int spring_idx = 0
    cdef double face_spring_length = sqrt(2)*cube_side_length
    cdef double center_spring_length = sqrt(3)*cube_side_length
    relative_positions[0,:] = cube_side_length*np.array([1,0,0],np.dtype(np.float64))
    relative_positions[1,:] = cube_side_length*np.array([0,1,0],np.dtype(np.float64))
    relative_positions[2,:] = cube_side_length*np.array([0,0,1],np.dtype(np.float64))
    relative_positions[3,:] = cube_side_length*np.array([-1,0,0],np.dtype(np.float64))
    relative_positions[4,:] = cube_side_length*np.array([0,-1,0],np.dtype(np.float64))
    relative_positions[5,:] = cube_side_length*np.array([0,0,-1],np.dtype(np.float64))
    #edge springs
    relative_positions[6,:] = cube_side_length*np.array([1,1,0],np.dtype(np.float64))
    relative_positions[7,:] = cube_side_length*np.array([1,0,1],np.dtype(np.float64))
    relative_positions[8,:] = cube_side_length*np.array([0,1,1],np.dtype(np.float64))
    relative_positions[9,:] = cube_side_length*np.array([1,-1,0],np.dtype(np.float64))
    relative_positions[10,:] = cube_side_length*np.array([1,0,-1],np.dtype(np.float64))
    relative_positions[11,:] = cube_side_length*np.array([0,-1,1],np.dtype(np.float64))
    relative_positions[12,:] = cube_side_length*np.array([0,1,-1],np.dtype(np.float64))
    relative_positions[13,:] = cube_side_length*np.array([-1,1,0],np.dtype(np.float64))
    relative_positions[14,:] = cube_side_length*np.array([-1,0,1],np.dtype(np.float64))
    relative_positions[15,:] = cube_side_length*np.array([0,-1,-1],np.dtype(np.float64))
    relative_positions[16,:] = cube_side_length*np.array([-1,0,-1],np.dtype(np.float64))
    relative_positions[17,:] = cube_side_length*np.array([-1,-1,0],np.dtype(np.float64))
    #face springs
    relative_positions[18,:] = cube_side_length*np.array([1,1,1],np.dtype(np.float64))
    relative_positions[19,:] = cube_side_length*np.array([1,1,-1],np.dtype(np.float64))
    relative_positions[20,:] = cube_side_length*np.array([1,-1,1],np.dtype(np.float64))
    relative_positions[21,:] = cube_side_length*np.array([-1,1,1],np.dtype(np.float64))
    relative_positions[22,:] = cube_side_length*np.array([-1,-1,1],np.dtype(np.float64))
    relative_positions[23,:] = cube_side_length*np.array([-1,1,-1],np.dtype(np.float64))
    relative_positions[24,:] = cube_side_length*np.array([1,-1,-1],np.dtype(np.float64))
    relative_positions[25,:] = cube_side_length*np.array([-1,-1,-1],np.dtype(np.float64))
    #center diagonal springs
    for i in range(node_posns.shape[0]):
        for j in range(26):
            tmp_positions[j,:] = relative_positions[j,:] + node_posns[i,:]
            row_indices = get_row_indices(tmp_positions,cube_side_length,[Lx,Ly,Lz])
            if not ((tmp_positions[j][0] < 0 or tmp_positions[j][0] <=Lx) or (tmp_positions[j][1] < 0 or tmp_positions[j][1] <=Ly) or (tmp_positions[j][2] < 0 or tmp_positions[j][2] <=Lz)):
                # instead of saving a new variable, i can just say, alright, this is in the right spot. let's use the j index to decide what the separation and spring type are
                if i < row_indices[j]:
                    springs[spring_idx,0] = i
                    springs[spring_idx,1] = row_indices[j]
                    if j < 6:
                        springs[spring_idx,3] = cube_side_length
                        springs[spring_idx,2] = get_spring_stiffness(i,row_indices[j],node_posns,mydim,stiffness_constants[0]/4,4)
                    elif j < 18:
                        springs[spring_idx,3] = face_spring_length
                        springs[spring_idx,2] = get_spring_stiffness(i,row_indices[j],node_posns,mydim,stiffness_constants[1]/2,2)
                    else:
                        springs[spring_idx,3] = center_spring_length
                        springs[spring_idx,2] = stiffness_constants[2]
                    spring_idx +=1
        
#get the spring stiffness based on the node types of the connected nodes
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_spring_stiffness(int i, int v, double[:,::1] node_posns, double[3] dimensions, double stiffness_constant, int max_shared_elements):
    """Set the stiffness of a particular spring based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    cdef double[3] node_posn_v
    cdef double[3] node_posn_i
    node_posn_i[0] = node_posns[i,0]
    node_posn_i[1] = node_posns[i,1]
    node_posn_i[2] = node_posns[i,2]
    node_type_i = identify_node_type(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
    node_posn_v[0] = node_posns[v,0]
    node_posn_v[1] = node_posns[v,1]
    node_posn_v[2] = node_posns[v,2]
    node_type_v = identify_node_type(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
    if node_type_i == b'i' and node_type_v == b'i':
        if max_shared_elements == 4:
            return stiffness_constant*4
        else:
            return stiffness_constant*2
    elif (node_type_i == b'i' and node_type_v == b's') or (node_type_i == b's' and node_type_v == b'i'):
        if max_shared_elements == 4:
            return stiffness_constant*4
        else:
            return stiffness_constant*2
    elif (node_type_i == b'i' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b'i'):
        return stiffness_constant*2
    elif node_type_i == b's' and node_type_v == b's':
        if max_shared_elements == 4:#two shared elements for a cube edge spring in this case if they are both on the same surface, so check for shared surfaces. otherwise the answer is 4.
            node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
            node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
            if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                return stiffness_constant*2
            else:
                return stiffness_constant*4
        else:#face spring, if the two nodes are on the same surface theres only one element, if they are on two different surfaces theyre are two shared elements
            node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
            node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
            if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                return stiffness_constant
            else:#on different surfaces, two shared elements
                return stiffness_constant*2
    elif (node_type_i == b's' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b's'):
        if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
            return stiffness_constant*2
        else:#this is a face spring with only a single element if the edge and surface node have a shared surface
            node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
            node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
            if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                return stiffness_constant
            else:#they don't share a surface, and so they share two elements
                return stiffness_constant*2
    elif node_type_i == b'e' and node_type_v == b'e':
        #both nodes belong to two surfaces (if they are edge nodes). if the surfaces are the same, then it is a shared edge, if they are not, they are separate edges of the simulated volume. there aer 6 surfaces
        node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
        node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
        if ((node_i_surf[0] == node_v_surf[0] and node_i_surf[1] == node_v_surf[1] and (node_i_surf[0] != 0 and node_i_surf[1] != 0)) or (node_i_surf[0] == node_v_surf[0] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[0] != 0 and node_i_surf[2] != 0)) or(node_i_surf[1] == node_v_surf[1] and node_i_surf[2] == node_v_surf[2] and (node_i_surf[1] != 0 and node_i_surf[2] != 0))):#if both nodes belong to the same two surfaces, they are on the same edge
            return stiffness_constant
        elif max_shared_elements == 4:#if they don't share two surfaces and it's a cube edge spring, they share two elements
            return stiffness_constant*2
        else:#if it's a face spring
            if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):#if they do share a surface, then the face spring has as single element
                return stiffness_constant
            else:#they don't share a single surface, then they diagonally across one another and have two shared elements
                return stiffness_constant*2
    elif node_type_i == b'c' or node_type_v == b'c':#any spring involving a corner node covered
        return stiffness_constant
    return -1

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
    for i in range(node_posns.shape[0]):
        for j in range(node_posns.shape[0]):
            rij[0] = node_posns[i,0] - node_posns[j,0]
            rij[1] = node_posns[i,1] - node_posns[j,1]
            rij[2] = node_posns[i,2] - node_posns[j,2]
            rijsquared = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2]
            rij_mag[j] = libc.math.sqrt(rijsquared)
        #get the row indices of the connected nodes that are connected by edge springs
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*cube_side_length).nonzero()[0]).astype(np.int32)
        num_edge_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[0]/4,cube_side_length,4,edge_springs)
        num_springs += num_edge_springs
        #use the count of number of edge springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = edge_springs[current_sub_idx][my_counter]    
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*face_diagonal_length).nonzero()[0]).astype(np.int32)
        num_face_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[1]/2,face_diagonal_length,2,face_springs)
        num_springs += num_face_springs
        #use the count of number of face springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = face_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*center_diagonal_length).nonzero()[0]).astype(np.int32)
        num_center_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[2],center_diagonal_length,1,diagonal_springs)
        num_springs += num_center_springs
        #use the count of number of diagonal springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = diagonal_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        #functionalize the above, D.R.Y.
    return num_springs

@cython.boundscheck(True)
@cython.wraparound(False)
cpdef int create_springs_v2(double[:,::1] node_posns, int[:,::1] elements, double[::1] stiffness_constants, double cube_side_length, double Lx, double Ly, double Lz, double[:,::1] springs):
    cdef int i
    cdef int j
    cdef int i_idx
    cdef int j_idx
    cdef int element_idx
    cdef int spring_idx = 0
    cdef int current_spring_idx
    cdef int current_sub_idx = 0
    cdef int my_counter
    cdef double face_diagonal_length = sqrt(2)*cube_side_length
    cdef double center_diagonal_length = sqrt(3)*cube_side_length
    cdef double[3] mydim
    mydim[0] = Lx
    mydim[1] = Ly
    mydim[2] = Lz
    cdef double[3][4] edge_springs = np.empty((3,4),np.dtype(np.float64))
    cdef double[3][4] face_springs = np.empty((3,4),np.dtype(np.float64))
    cdef double[1][4] diagonal_springs = np.empty((1,4),np.dtype(np.float64))
    cdef double[3] rij
    cdef double rijsquared
    cdef double[8] rij_mag
    cdef int nodes_per_element = 8
    cdef int num_springs = 0
    cdef int num_edge_springs = 0
    cdef int num_face_springs = 0
    cdef int num_center_springs = 0
    cdef int[8] element = np.empty((8,),np.dtype(np.int32))
    for element_idx in range(elements.shape[0]):
        for i in range(8):
            element[i] = elements[element_idx,i]
        for i in range(8):
            i_idx = element[i]
            for j in range(8):
                j_idx = element[j]
                rij[0] = node_posns[i_idx,0] - node_posns[j_idx,0]
                rij[1] = node_posns[i_idx,1] - node_posns[j_idx,1]
                rij[2] = node_posns[i_idx,2] - node_posns[j_idx,2]
                rijsquared = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2]
                rij_mag[j] = libc.math.sqrt(rijsquared)
            #get the row indices of the connected nodes that are connected by edge springs
            connected_vertices = (np.isclose(rij_mag,np.ones((nodes_per_element,),dtype=np.float64)*cube_side_length).nonzero()[0]).astype(np.int32)
            num_edge_springs = get_node_springs_v2(i_idx,node_posns,element,connected_vertices,mydim,stiffness_constants[0]/4,cube_side_length,4,edge_springs)
            num_springs += num_edge_springs
            print(f'spring_idx = {spring_idx}, num_springs = {num_springs}')
            #use the count of number of edge springs that were filled to then assign to the springs variable
            for current_spring_idx in range(spring_idx,num_springs):
                for my_counter in range(4):
                    springs[current_spring_idx,my_counter] = edge_springs[current_sub_idx][my_counter]    
                current_sub_idx += 1
            spring_idx = num_springs
            current_sub_idx = 0
            connected_vertices = (np.isclose(rij_mag,np.ones((nodes_per_element,),dtype=np.float64)*face_diagonal_length).nonzero()[0]).astype(np.int32)
            num_face_springs = get_node_springs_v2(i_idx,node_posns,element,connected_vertices,mydim,stiffness_constants[1]/2,face_diagonal_length,2,face_springs)
            num_springs += num_face_springs
            print(f'spring_idx = {spring_idx}, num_springs = {num_springs}')
            #use the count of number of face springs that were filled to then assign to the springs variable
            for current_spring_idx in range(spring_idx,num_springs):
                for my_counter in range(4):
                    springs[current_spring_idx,my_counter] = face_springs[current_sub_idx][my_counter] 
                current_sub_idx += 1
            spring_idx = num_springs
            current_sub_idx = 0
            connected_vertices = (np.isclose(rij_mag,np.ones((nodes_per_element,),dtype=np.float64)*center_diagonal_length).nonzero()[0]).astype(np.int32)
            num_center_springs = get_node_springs_v2(i_idx,node_posns,element,connected_vertices,mydim,stiffness_constants[2],center_diagonal_length,1,diagonal_springs)
            num_springs += num_center_springs
            print(f'spring_idx = {spring_idx}, num_springs = {num_springs}')
            #use the count of number of diagonal springs that were filled to then assign to the springs variable
            for current_spring_idx in range(spring_idx,num_springs):
                for my_counter in range(4):
                    springs[current_spring_idx,my_counter] = diagonal_springs[current_sub_idx][my_counter] 
                current_sub_idx += 1
            spring_idx = num_springs
            current_sub_idx = 0
            #functionalize the above, D.R.Y.
    return num_springs

#given the node positiondds and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int create_springs_wrapper(double[:,::1] node_posns, double[::1] stiffness_constants, double cube_side_length, double Lx, double Ly, double Lz, double[:,::1] springs, double[::1] rij_mag):
    cdef int num_springs
    num_springs = create_springs_c(node_posns, stiffness_constants, cube_side_length, Lx, Ly, Lz, springs, rij_mag)
    return num_springs

#given the node positiondds and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int create_springs_c(double[:,::1] node_posns, double[::1] stiffness_constants, double cube_side_length, double Lx, double Ly, double Lz, double[:,::1] springs, double[::1] rij_mag):
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
    for i in range(node_posns.shape[0]):
        for j in range(node_posns.shape[0]):
            rij[0] = node_posns[i,0] - node_posns[j,0]
            rij[1] = node_posns[i,1] - node_posns[j,1]
            rij[2] = node_posns[i,2] - node_posns[j,2]
            rijsquared = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2]
            rij_mag[j] = libc.math.sqrt(rijsquared)
        #get the row indices of the connected nodes that are connected by edge springs
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*cube_side_length).nonzero()[0]).astype(np.int32)
        num_edge_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[0]/4,cube_side_length,4,edge_springs)
        num_springs += num_edge_springs
        #use the count of number of edge springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = edge_springs[current_sub_idx][my_counter]    
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*face_diagonal_length).nonzero()[0]).astype(np.int32)
        num_face_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[1]/2,face_diagonal_length,2,face_springs)
        num_springs += num_face_springs
        #use the count of number of face springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = face_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        connected_vertices = (np.isclose(rij_mag,np.ones((rij_mag.shape[0],),dtype=np.float64)*center_diagonal_length).nonzero()[0]).astype(np.int32)
        num_center_springs = get_node_springs(i,node_posns,connected_vertices,mydim,stiffness_constants[2],center_diagonal_length,1,diagonal_springs)
        num_springs += num_center_springs
        #use the count of number of diagonal springs that were filled to then assign to the springs variable
        for current_spring_idx in range(spring_idx,num_springs):
            for my_counter in range(4):
                springs[current_spring_idx,my_counter] = diagonal_springs[current_sub_idx][my_counter] 
            current_sub_idx += 1
        spring_idx = num_springs
        current_sub_idx = 0
        #functionalize the above, D.R.Y.
    return num_springs

#functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_node_springs(int idx, double[:,::1] node_posns, int[::1] connected_vertices, double[3] dimensions, double stiffness_constant, double comparison_length, int max_shared_elements, double[:,::1] springs):
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
    for i in range(connected_vertices.shape[0]):
        if idx < connected_vertices[i]:
            valid_connections[count] = connected_vertices[i]
            count += 1
    if max_shared_elements == 1:#if we are dealing with the center diagonal springs we don't need to count shared elements
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
            if node_type_i == b'i' and node_type_v == b'i':
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == b'i' and node_type_v == b's') or (node_type_i == b's' and node_type_v == b'i'):
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == b'i' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b'i'):
                springs[row,2] = stiffness_constant*2
            elif node_type_i == b's' and node_type_v == b's':
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
            elif (node_type_i == b's' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b's'):
                if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
                    springs[row,2] = stiffness_constant*2
                else:#this is a face spring with only a single element if the edge and surface node have a shared surface
                    node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row,2] = stiffness_constant
                    else:#they don't share a surface, and so they share two elements
                        springs[row,2] = stiffness_constant*2
            elif node_type_i == b'e' and node_type_v == b'e':
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
            elif node_type_i == b'c' or node_type_v == b'c':#any spring involving a corner node covered
                springs[row,2] = stiffness_constant
    # return springs
    return count

#functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_node_springs_v2(int idx, double[:,::1] node_posns, int[8] element, int[::1] connected_vertices, double[3] dimensions, double stiffness_constant, double comparison_length, int max_shared_elements, double[:,::1] springs):
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
    cdef int[3] valid_connections#since i am doing each spring type separately, the max size is really 12 (12 face springs max per node)
    for i in range(connected_vertices.shape[0]):
        if idx < element[connected_vertices[i]]:
            valid_connections[count] = element[connected_vertices[i]]
            count += 1
    if max_shared_elements == 1:#if we are dealing with the center diagonal springs we don't need to count shared elements
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
            if node_type_i == b'i' and node_type_v == b'i':
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == b'i' and node_type_v == b's') or (node_type_i == b's' and node_type_v == b'i'):
                if max_shared_elements == 4:
                    springs[row,2] = stiffness_constant*4
                else:
                    springs[row,2] = stiffness_constant*2
            elif (node_type_i == b'i' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b'i'):
                springs[row,2] = stiffness_constant*2
            elif node_type_i == b's' and node_type_v == b's':
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
            elif (node_type_i == b's' and node_type_v == b'e') or (node_type_i == b'e' and node_type_v == b's'):
                if max_shared_elements == 4:#if the max_shared_elements is 4, this is a cube edge spring, and an edge-surface connection has two shared elements
                    springs[row,2] = stiffness_constant*2
                else:#this is a face spring with only a single element if the edge and surface node have a shared surface
                    node_i_surf = get_node_surf(node_posn_i,dimensions[0],dimensions[1],dimensions[2])
                    node_v_surf = get_node_surf(node_posn_v,dimensions[0],dimensions[1],dimensions[2])
                    if ((node_i_surf[0] == node_v_surf[0]and node_i_surf[0] != 0 ) or (node_i_surf[1] == node_v_surf[1] and node_i_surf[1] != 0) or (node_i_surf[2] == node_v_surf[2] and node_i_surf[2] != 0)):
                        springs[row,2] = stiffness_constant
                    else:#they don't share a surface, and so they share two elements
                        springs[row,2] = stiffness_constant*2
            elif node_type_i == b'e' and node_type_v == b'e':
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
            elif node_type_i == b'c' or node_type_v == b'c':#any spring involving a corner node covered
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