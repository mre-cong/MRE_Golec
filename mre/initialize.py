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
    elements = get_elements(node_posns,Lx,Ly,Lz,cube_side_length)
    boundaries = get_boundaries(node_posns)
    return node_posns, np.int32(elements), boundaries

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
    return elements

    #given the node positions and stiffness constants for the different types of springs, calculate and return a list of springs, which is  N_springs x 4, where each row represents a spring, and the columns are [node_i_rowidx, node_j_rowidx, stiffness, equilibrium_length]
    #TODO improve create_springs function performance by switching to a divide and conquer approach. see notes from March 15th 2023
def create_springs(node_posns,stiffness_constants,cube_side_length,dimensions):
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

#functionalizing the construction of springs including setting of stiffness constants based on number of shared elements for different spring types (edge and face diagonal). will need to be extended for the case of materials with two phases
def get_node_springs(i,node_posns,rij_mag,dimensions,stiffness_constant,comparison_length,max_shared_elements):
    """Set the stiffness of a particular spring based on the number of shared elements (and spring type: edge or face diagaonal). assumes a single material phase"""
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

#given the material properties (Young's modulus, shear modulus, and poisson's ratio) of an isotropic material, calculate the spring stiffness constants for edge springs, center diagonal springs, and face diagonal springs for a cubic unit cell
def get_spring_constants(E,nu,l_e):
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
                
def write_log(simulation,output_dir):
    timestamp = time.ctime()
    script_name = lib_programname.get_path_executed_script()
    with open(output_dir+'logfile.txt','a') as f_obj:
        f_obj.writelines([simulation.report2(),str(script_name)+'\n',timestamp+'\n'])

    #TODO make functionality that converts the boundaries variable data into a format that can be stored in hdf5 format and functionality that reads in from the hdf5 format to the typical boundaries variable format
def write_init_file(posns,springs,boundaries,output_dir):
    """Write out the vertex positions, springs are N_springs by 4, first two columns are row indices in posns for nodes connected by springs, 3rd column is stiffness, 4th is equilibrium separation, and the nodes that make up each cubic element as .csv files (or HDF5 files). To be modified in the future, to handle large systems (which will require sparse matrix representations due to memory limits)"""
    f = tb.open_file(output_dir+'init.h5','w')
    f.create_array('/','vertex_posns',posns)
    f.create_array('/','springs',springs)
    # f.create_array('/','boundary_nodes',boundaries)
    # posn_dt = np.dtype([('x',np.float64),('y',np.float64),('z',np.float64)])
    # f.create_table('/','vertex_posns',posn_dt)
    # f.root.vertex_posns.append(posns)
    f.close()
#TODO make functionality that converts boundary_conditions variable data into a format that can be stored in hdf5 format, and a function that reverses this process (reading from hdf5 format to a variable in the format of boundary_conditions)
def write_output_file(count,posns,boundary_conditions,output_dir):
    """Write out the vertex positions, connectivity matrix defined by equilibrium separation, connectivity matrix defined by stiffness constant, and the nodes that make up each cubic element as .csv files (or HDF5 files). To be modified in the future, to handle large systems (which will require sparse matrix representations due to memory limits)"""
    f = tb.open_file(output_dir+'final_posns.h5','w')
    f.create_array('/','node_posns'+str(count),posns)
    # datatype = np.dtype((('bc_type',str),(('boundary',str),('boundary',str)),('value',float)))
    # f.create_table('/','boundary_conditions',datatype)
    # f.root.boundary_conditions.append(boundary_conditions)
    # posn_dt = np.dtype([('x',np.float64),('y',np.float64),('z',np.float64)])
    # f.create_table('/','vertex_posns',posn_dt)
    # f.root.vertex_posns.append(posns)
    f.close()
    
def read_init_file(fn):
    f = tb.open_file(fn)
    return f

def main():
    pass

if __name__ == "__main__":
    main()