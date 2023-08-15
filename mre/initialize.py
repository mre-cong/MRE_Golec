# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 02:36:19 2023

@author: David Marchfield
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import lib_programname
import tables as tb#pytables, for HDF5 interface
# import create_springs
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
    return node_posns

def discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z):
    """Given the number of nodes in each direction, set up the normalized grid of points"""
    [x,y,z] = np.meshgrid(np.r_[0:N_nodes_x:1],
                          np.r_[0:N_nodes_y:1],
                          np.r_[0:N_nodes_z:1])
    node_posns = np.concatenate((np.reshape(x,np.size(x))[:,np.newaxis],
                                np.reshape(y,np.size(y))[:,np.newaxis],
                                np.reshape(z,np.size(z))[:,np.newaxis]),1)
    return node_posns

def get_boundaries(node_posns):
    top_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].max())[0]
    bot_bdry = np.nonzero(node_posns[:,2] == node_posns[:,2].min())[0]
    left_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].min())[0]
    right_bdry = np.nonzero(node_posns[:,0] == node_posns[:,0].max())[0]
    front_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].min())[0]
    back_bdry = np.nonzero(node_posns[:,1] == node_posns[:,1].max())[0]
    boundaries = {'top': top_bdry, 'bot': bot_bdry, 'left': left_bdry, 'right': right_bdry, 'front': front_bdry, 'back': back_bdry}  
    return boundaries
 
 #TODO check performance of get_elements method for scaled up systems to see if performance improvements are necessary/if bottlenecking is occurring
def get_elements(node_posns,Lx,Ly,Lz,cube_side_length):
    """given the node/vertex positions, dimensions of the simulated volume, and volume element edge length, return an N_elements by 8 array where each row represents a single volume element and each column is the associated row index in node_posns of a vertex of the volume element"""
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
    return elements.astype(np.int32)

    #given the node positions and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
    #TODO improve create_springs function performance by switching to a divide and conquer approach. see notes from March 15th 2023
def create_springs_old(node_posns,stiffness_constants,cube_side_length,dimensions):
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    springs = np.zeros((1,4),dtype=np.float64)#creating an array that will hold the springs, will have to concatenate as new springs are added, and delete the first row before passing back
    for i, posn in enumerate(node_posns):
        rij = posn - node_posns
        rij_mag = np.sqrt(np.sum(rij**2,1))
        edge_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
        face_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
        diagonal_springs = get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constants[2],center_diagonal_length,max_shared_elements=1)
        springs = np.concatenate((springs,edge_springs,face_springs,diagonal_springs),dtype=np.float64)
    return np.ascontiguousarray(springs[1:],dtype=np.float64)#want a C-contiguous memory representation for using cythonized compiled functionality, where information on memory structure can provide performance speedups

def create_springs_v2(node_posns,elements,stiffness_constants,cube_side_length,dimensions):
    face_diagonal_length = np.sqrt(2)*cube_side_length
    center_diagonal_length = np.sqrt(3)*cube_side_length
    springs_tmp = np.zeros((1,4),dtype=np.float64)#creating an array that will hold the springs, will have to concatenate as new springs are added, and delete the first row before passing back
    for element in elements:
        posns_tmp = node_posns[element]
        for i in element:
            posn = node_posns[i,:]
            rij = posn - posns_tmp
            rij_mag = np.sqrt(np.sum(rij**2,1))
            edge_springs = get_node_springs_v2(i,node_posns,element,rij_mag,dimensions,stiffness_constants[0]/4,cube_side_length,max_shared_elements=4)
            face_springs = get_node_springs_v2(i,node_posns,element,rij_mag,dimensions,stiffness_constants[1]/2,face_diagonal_length,max_shared_elements=2)
            diagonal_springs = get_node_springs_v2(i,node_posns,element,rij_mag,dimensions,stiffness_constants[2],center_diagonal_length,max_shared_elements=1)
            springs_tmp = np.concatenate((springs_tmp,edge_springs,face_springs,diagonal_springs),dtype=np.float64)
    springs = np.unique(springs_tmp,axis=0)
    return np.ascontiguousarray(springs[1:],dtype=np.float64)#want a C-contiguous memory representation for using cythonized compiled functionality, where information on memory structure can provide performance speedups    

# functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
def get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constant,comparison_length,max_shared_elements):
    """Set the stiffness of a particular spring based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    connected_vertices = np.isclose(rij_mag,comparison_length).nonzero()[0]
    # TODO !!! remove the below line that has been commented out, if the results are accurate between the method above and the method below (which they should be and look to be based on tests so far)
    # connected_vertices = np.asarray(np.abs(rij_mag - comparison_length)/comparison_length < epsilon).nonzero()[0]#per numpy documentation, this method is preferred over np.where if np.where is only passed a condition, instead of a condition and two arrays to select from
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
def get_node_springs_v2(i,node_posns,element,rij_mag,dimensions,stiffness_constant,comparison_length,max_shared_elements):
    """Set the stiffness of a particular spring based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
    epsilon = np.spacing(1)
    connected_vertices = np.asarray(np.abs(rij_mag - comparison_length) < epsilon).nonzero()[0]#per numpy documentation, this method is preferred over np.where if np.where is only passed a condition, instead of a condition and two arrays to select from
    valid_connections = np.asarray([val for val in element[connected_vertices] if i < val])
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
    """Identify the node type (corner, edge, surface, or interior point) based on the node position and the dimensions of the simulation. 
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
    """Return a triplet that provides information on what surface or surfaces, if any, a node is part of."""
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

def get_row_indices(node_posns,l_e,dim):
    """Return the row indices corresponding to the node positions of interest given the simulation dimension parameters"""
    Lx,Ly,Lz = dim
    inv_l_e = 1/l_e
    nodes_per_col = np.round(Lz/l_e + 1).astype(np.int32)
    nodes_per_row = np.round(Lx/l_e + 1).astype(np.int32)
    row_index = ((nodes_per_col * inv_l_e * node_posns[:,0]) + (nodes_per_col * nodes_per_row * inv_l_e *node_posns[:,1]) + inv_l_e *node_posns[:,2]).astype(np.int32)
    return row_index

#given the material properties (Young's modulus, shear modulus, and poisson's ratio) of an isotropic material, calculate the spring stiffness constants for edge springs, center diagonal springs, and face diagonal springs for a cubic unit cell
def get_spring_constants(E,l_e):
    """Return the edge, central diagonal, and face diagonal stiffness constants of the system from the Young's modulus, poisson's ratio, and the length of the edge springs."""
    A = 1 #ratio of the stiffness constants of the center diagonal to face diagonal springs
    k_e = 0.4 * (E * l_e) * (8 + 3 * A) / (4 + 3 * A)
    k_c = 1.2 * (E * l_e) / (4 + 3 * A)
    k_f = A * k_c
    k = [k_e, k_f, k_c]
    return k

def get_kappa(E,nu):
    """Return the value of the additional bulk modulus, kappa, for the volume correction force given the Young's modulus and Poissons's ratio."""
    kappa = E * (4 * nu - 1) / (2 * (1 + nu) * (1 - 2 * nu))
    return kappa

def get_node_mass(N_nodes,dimensions,particles,particle_size):
    """Return the mass values of the nodes based on the system size, matrix mass density, particle size and particle mass density."""
    system_volume = dimensions[0]*dimensions[1]*dimensions[2]
    matrix_density = 0.965 #kg/m^3
    matrix_node_mass = matrix_density*system_volume/N_nodes
    m = np.ones((N_nodes,))*matrix_node_mass
    particle_mass_density = 7.86 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    if particles.size != 0:
        particle_node_mass = particle_mass_density*((4/3)*np.pi*(particle_size**3))/particles[0,:].shape[0]
        for particle in particles:
            m[particle] = particle_node_mass
    return m

def get_node_mass_v2(N_nodes,node_types,l_e,particles,particle_size):
    """Return the mass values of the nodes, the volume element mass, and the particle mass based on the size of the cubic volume elements, matrix mass density, node type (corner, edge, surface, interior), particle size and particle mass density."""
    matrix_density = 0.965 #kg/m^3
    volume_element_mass = matrix_density*(l_e**3)
    m = np.ones((N_nodes,))*volume_element_mass
    m[node_types!=0] = volume_element_mass/2#all non-interior nodes set to half the interior node values
    m[node_types>=7] = volume_element_mass/4#setting the edges and corners to half the surface node values
    m[node_types>=19] = volume_element_mass/8#setting the corner nodes to 1/8 the interior node mass
    particle_mass_density = 7.86 #kg/m^3, americanelements.com/carbonyl-iron-powder-7439-89-6, young's modulus 211 GPa
    if particles.size != 0:
        particle_mass = particle_mass_density*((4/3)*np.pi*(particle_size**3))
        particle_node_mass = particle_mass/particles[0,:].shape[0]
        for particle in particles:
            m[particle] = particle_node_mass
    return m, volume_element_mass, particle_mass

#function which plots with a 3D scatter and lines, the connectivity of the unit cell
def plot_unit_cell(node_posns,connectivity):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(node_posns[:,0],node_posns[:,1],node_posns[:,2],'o')
    for i in range(len(node_posns)-1):
        for j in range(i+1,len(node_posns)):
            if connectivity[i,j] != 0:#add conditional to check equilibrium separation and only show edge springs
                x,y,z = (np.array((node_posns[i,0],node_posns[j,0])),
                         np.array((node_posns[i,1],node_posns[j,1])),
                         np.array((node_posns[i,2],node_posns[j,2])))
                ax.plot(x,y,z)
        #others = remove_i(node_posns,i)
        #feels like this should be recursive. I have a listof points, I want to draw lines from each pair of points but without redrawing lines. I have one point, I remove it from the list, if there's nothing left in the list I have nothing to draw, if there's one thing left in the list, I draw the line connecting this point to that point, if I have more than one point left in the list, I 

#!!! construct the boundary conditions data structure
#TODO
def get_boundary_conditions(boundary_condition_type,):
    #given a few experimental setups (plus fixed displacement type boundary conditions...)
    #experimental setups: shear, compression, tension, torsion, bending
    if boundary_condition_type == 'shear':
        return 0
    elif boundary_condition_type == 'compression':
        return 0
    elif boundary_condition_type == 'tension':
        return 0
    elif boundary_condition_type == 'torsion':
        return 0
    elif boundary_condition_type == 'bending':
        return 0
    elif boundary_condition_type == 'displacement':
        return 0
    elif boundary_condition_type == 'mixed':
        return 0

class Simulation(object):
    """A simulation has properties which define the simulation. These include the Modulus, Poisson's ratio, cubic element side length, simulation dimensions.
    
    Attributes
    ----------
    E : Young's modulus [Pa]
    nu : Poisson's ratio []
    l_e : side length of an element [m]
    Lx : length in x direction of the object [m]
    Ly : length in y direction of the object [m]
    Lz : length in z direction of the object [m]
    """
    #TODO flesh out this class based approach to the simulation interface
    def __init__(self,E=1,nu=0.49,l_e=0.1,Lx=0.4,Ly=0.4,Lz=0.4):
        """Initializes simulation with default values if they are not passed"""
        self.E = E
        self.nu = nu
        self.l_e = l_e
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.t_f = 0
        self.N_iter = 0
        N_nodes_x = np.round(Lx/l_e + 1)
        N_nodes_y = np.round(Ly/l_e + 1)
        N_nodes_z = np.round(Lz/l_e + 1)
        N_el_x = N_nodes_x - 1
        N_el_y = N_nodes_y - 1
        N_el_z = N_nodes_z - 1
        
    def set_time(self,time):
        self.t_f = time
        
    def set_iterations(self,N_iter):
        self.N_iter = N_iter
        
    def report(self):
        """Using using hand written string of the instance variables of the object to create a descriptuion of the simulation parameters (useful for writing a log file)"""
        report_string = 'E = ' + str(self.E) + ' m\n'+ 'nu = ' + str(self.nu) + '\n'+'l_e = ' + str(self.l_e) + ' m\n'+'Lx = ' + str(self.Lx) + ' m\n'+'Ly = ' + str(self.Ly) + ' m\n'+'Lz = ' + str(self.Lz) + ' m\n'+ 'total_time = ' + str(self.t) + ' s\n'+ 'N_iterations = ' + str(self.N_iter) + '  iterations\n'
        return report_string
    
    def report2(self):
        """Using built-in python features to iterate over the instance variables of the object to create a set of strings describing the simulation parameters (useful for writing a log file)"""
        my_keys = list(vars(self).keys())
        # my_vals = list(vars(self).values())
        report_string = ''
        for key in my_keys:
            report_string += key + ' = ' + str(vars(self).get(key)) + ' \n'
        return report_string
                
    def write_log(self,output_dir):
        timestamp = time.ctime()
        script_name = lib_programname.get_path_executed_script()
        explanation = "default explanation empty"#input("Add a sentence or two describing this simulation and/or explaining it's purpose:\n")
        with open(output_dir+'logfile.txt','a') as f_obj:
            f_obj.writelines([explanation+'\n',self.report2(),str(script_name)+'\n',timestamp+'\n'])

    #TODO make functionality that converts the boundaries variable data into a format that can be stored in hdf5 format and functionality that reads in from the hdf5 format to the typical boundaries variable format
def write_init_file(posns,springs,elements,particles,boundaries,output_dir):
    """Write out the node positions, springs are N_springs by 4, first two columns are row indices in posns for nodes connected by springs, 3rd column is stiffness, 4th is equilibrium separation, and the nodes that make up each cubic element as .csv files (or HDF5 files). To be modified in the future, to handle large systems (which will require sparse matrix representations due to memory limits)"""
    f = tb.open_file(output_dir+'init.h5','w')
    f.create_array('/','node_posns',posns)
    f.create_array('/','springs',springs)
    f.create_array('/','elements',elements)
    f.create_group('/','boundaries',"Boundary Nodes")
    for key in boundaries.keys():
        f.create_array('/boundaries',key,boundaries[key])
    f.create_array('/','particles',particles)
    f.close()

def read_init_file(fn):
    f = tb.open_file(fn,'r')
    node_object = f.get_node('/','node_posns')
    node_posns = node_object.read()
    spring_object = f.get_node('/','springs')
    springs = spring_object.read()
    element_object = f.get_node('/','elements')
    elements = element_object.read()
    # particle_object = f.get_node('/','particles')
    # particles = particle_object.read()
    boundaries = {}
    for leaf in f.root.boundaries._f_walknodes('Leaf'):
        boundaries[leaf.name] = leaf.read()
    f.close()
    return node_posns, springs, elements, boundaries

#TODO make functionality that converts boundary_conditions variable data into a format that can be stored in hdf5 format, and a function that reverses this process (reading from hdf5 format to a variable in the format of boundary_conditions)
def write_output_file(count,posns,boundary_conditions,output_dir):
    """Write out the vertex positions, connectivity matrix defined by equilibrium separation, connectivity matrix defined by stiffness constant, and the nodes that make up each cubic element as .csv files (or HDF5 files). To be modified in the future, to handle large systems (which will require sparse matrix representations due to memory limits)"""
    f = tb.open_file(f'{output_dir}output_{count}.h5','w')
    f.create_array('/','node_posns',posns)
    # f.create_group('/','node_posns','Final Configurations')
    # f.create_array('/node_posns',str(count),posns)
    # f.create_group('/','boundary_conditions','Applied Boundary Conditions')
    # f.create_group('/','applied_field')
    dt = np.dtype([('bc_type','S6'),('surf1','S6'),('surf2','S6'),('value',np.float64)])
    f.create_table('/','boundary_conditions',dt)
    bc = np.array([(boundary_conditions[0],boundary_conditions[1][0],boundary_conditions[1][1],boundary_conditions[2])],dtype=dt)
    f.root.boundary_conditions.append(bc)
    f.close()

def read_output_file(fn):
    f = tb.open_file(fn,'r')
    node_object = f.get_node('/','node_posns')
    node_posns = node_object.read()
    bc_object = f.get_node('/','boundary_conditions')
    bc = bc_object.read()
    boundary_condition = bc[0]
    f.close()
    return node_posns, boundary_condition

def test_element_setting():
    import time
    E = 1
    nu = 0.4999
    l_e = .1
    lx = range(100,101)
    ly = range(100,101)
    lz = range(10,11)
    for countlx in range(len(lx)):
        for countly in range(len(ly)):
            for countlz in range(len(lz)):
                Lx = lx[countlx]*l_e
                Ly = ly[countly]*l_e
                Lz = lz[countlz]*l_e
                node_posns = discretize_space(Lx,Ly,Lz,l_e)
                start = time.perf_counter()
                elements = get_elements(node_posns,Lx,Ly,Lz,l_e)
                end = time.perf_counter()
                py_time = end-start
                # boundaries = get_boundaries(node_posns)
                dimensions = np.array([Lx,Ly,Lz])
                start = time.perf_counter()
                new_elements = springs.get_elements(node_posns, dimensions, l_e)
                end = time.perf_counter()
                cy_time = end-start
                start = time.perf_counter()
                newer_elements = springs.get_elements_v2(dimensions, l_e)
                end = time.perf_counter()
                cy_time2 = end-start
                correctness = np.allclose(elements,new_elements)
                if not correctness:
                    print(f'consider sorting mechanism for both axes to perform meaningul correctness comparison (use argsort)')
                else:
                    print("New Cython and Old implementations agree?: " + str(correctness))
                    print(f"Original implementation took {py_time}s")
                    print(f"Cython implementation took {cy_time}s")
                    print(f"Cython implementation {py_time/cy_time}x times faster")
                correctness = np.allclose(elements,newer_elements)
                if not correctness:
                    print(f'consider sorting mechanism for both axes to perform meaningul correctness comparison (use argsort)')
                else:
                    print("Newer Cython and Old implementations agree?: " + str(correctness))
                    print(f"Original implementation took {py_time}s")
                    print(f"Newer Cython implementation took {cy_time2}s")
                    print(f"Newer Cython implementation {py_time/cy_time2}x times faster")

def test_spring_connection_setting():
    import time
    E = 1
    nu = 0.4999
    l_e = .1
    lx = range(1,11)
    ly = range(1,11)
    lz = range(1,6)
    for countlx in range(len(lx)):
        for countly in range(len(ly)):
            for countlz in range(len(lz)):
                Lx = lx[countlx]*l_e
                Ly = ly[countly]*l_e
                Lz = lz[countlz]*l_e
                node_posns = discretize_space(Lx,Ly,Lz,l_e)
                elements = get_elements(node_posns,Lx,Ly,Lz,l_e)
                boundaries = get_boundaries(node_posns)
                dimensions = np.array([Lx,Ly,Lz])
                node_types = springs.get_node_type(node_posns.shape[0],boundaries,dimensions,l_e)
                k = get_spring_constants(E,nu,l_e)
                k = np.array(k,dtype=np.float64)
                # start = time.perf_counter()
                # #estimate upper bound on number of springs
                # max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
                # # lesser_upper_bound = max_springs - 2*((np.round(Lx/l_e - 1).astype(np.int32)*np.round(Ly/l_e - 1)) + (np.round(Lx/l_e - 1).astype(np.int32)*np.round(Lz/l_e - 1)) + (np.round(Ly/l_e - 1).astype(np.int32)*np.round(Lz/l_e - 1)))*13/2 - (np.round(Lx/l_e - 1).astype(np.int32) + np.round(Ly/l_e - 1).astype(np.int32) +  np.round(Lz/l_e - 1).astype(np.int32))*26*3/2 - 7*26/2
                # edges_cy = np.empty((max_springs,4),dtype=np.float64)
                # rij_mag = np.empty((node_posns.shape[0],),dtype=np.float64)
                # num_springs = create_springs.create_springs(node_posns,k,l_e,Lx,Ly,Lz,edges_cy,rij_mag)
                # edges_cy = edges_cy[:num_springs,:]
                # end = time.perf_counter()
                # cy_time = end-start
                start = time.perf_counter()
                edges_original = create_springs_old(node_posns,k,l_e,dimensions)
                end = time.perf_counter()
                original_time = end-start
                # correctness_cy = np.allclose(np.sort(edges_original,axis=0),np.sort(edges_cy,axis=0))
                # print("New Cython and Old implementations agree?: " + str(correctness_cy))
                # print("Original implementation took {}s".format(original_time))
                # print("Cython implementation took {}s".format(cy_time))
                # print("Cython implementation {}x times faster".format(original_time/cy_time))
                start = time.perf_counter()
                #estimate upper bound on number of springs
                max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
                other_springs = np.empty((max_springs,4),dtype=np.float64)
                num_springs = springs.get_springs(node_types, other_springs, max_springs, k, dimensions, l_e)
                other_springs = other_springs[:num_springs,:]
                end = time.perf_counter()
                newest_time = end-start
                argsort_indices = np.argsort(edges_original[:,0],axis=0)
                sorted_original = edges_original[argsort_indices,:]
                j = 0
                for i in range(sorted_original.shape[0]):
                    if i == sorted_original.shape[0] - 1:
                        break
                    if sorted_original[i,0] == sorted_original[i+1,0]:
                        pass
                    else:
                        subarray = sorted_original[j:i+1,:]
                        argsort_indices = np.argsort(subarray[:,1])
                        sorted_original[j:i+1,:] = subarray[argsort_indices,:]
                        j = i+1
                argsort_indices = np.argsort(other_springs[:,0],axis=0)
                sorted_other = other_springs[argsort_indices,:]
                j = 0
                for i in range(sorted_original.shape[0]):
                    if i == sorted_other.shape[0] - 1:
                        break
                    if sorted_other[i,0] == sorted_other[i+1,0]:
                        pass
                    else:
                        subarray = sorted_other[j:i+1,:]
                        argsort_indices = np.argsort(subarray[:,1])
                        sorted_other[j:i+1,:] = subarray[argsort_indices,:]
                        j = i+1
                correctness = np.allclose(sorted_original,sorted_other)
                if not correctness:
                    print(f'{np.isclose(sorted_original[:,0],sorted_other[:,0])}')
                    print(f'{np.isclose(sorted_original[:,1],sorted_other[:,1])}')
                    print(f'{np.isclose(sorted_original[:,2],sorted_other[:,2])}')
                    print(f'{np.logical_not(np.isclose(sorted_original[:,2],sorted_other[:,2])).nonzero()[0]}')
                    print(f'{np.isclose(sorted_original[:,3],sorted_other[:,3])}')
                    print(f'{sorted_original[np.logical_not(np.isclose(sorted_original[:,2],sorted_other[:,2])).nonzero()[0],2]}')
                    print(f'{sorted_other[np.logical_not(np.isclose(sorted_original[:,2],sorted_other[:,2])).nonzero()[0],2]}')
                    print(f'original\n{sorted_original[np.logical_not(np.isclose(sorted_original[:,2],sorted_other[:,2])).nonzero()[0],:]}')
                    print(f'new\n{sorted_other[np.logical_not(np.isclose(sorted_original[:,2],sorted_other[:,2])).nonzero()[0],:]}')
                print("New Cython and Old implementations agree?: " + str(correctness))
                print("Original implementation took {}s".format(original_time))
                print("Cython implementation took {}s".format(newest_time))
                print("Cython implementation {}x times faster".format(original_time/newest_time))

def main():
    try:
        test_element_setting()
    except Exception as inst:
        print('Exception raised during testing of element setting')
        print(type(inst))
        print(inst)
    pass
    # try:
    #     test_spring_connection_setting()
    # except Exception as inst:
    #     print('Exception raised during testing of spring connection setting')
    #     print(type(inst))
    #     print(inst)
    # pass

if __name__ == "__main__":
    main()