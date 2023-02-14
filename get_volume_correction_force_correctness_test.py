# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:01:24 2023

@author: bagaw
"""

# compare the numpy and cythonized volume correction force expressions for correctness
import numpy as np
import timeit
import get_volume_correction_force_cy
import get_volume_correction_force_cy_nogil
import time
from multiprocessing import Pool
import numba 
import scipy

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
    return node_posns, np.int32(elements), boundaries

#given the node positions and stiffness constants for the different types of springs, calculate and return the connectivity matrix, and equilibrium separation matrix
def create_connectivity(node_posns,elements,stiffness_constants,cube_side_length):
    #!!! need to include the elements array, and take into account the number of elements an edge or face diagonal spring is shared with (due to kirchoff's law)
    #besides the node positions and stiffness constants (which is more complicated than just 3 numbers for the MRE case where there are two isotropic phases present with different material properties), we need to determine the connectivity matrix by summing up the stiffness contribution from each cell that share the vertices whose element is being calculated. A vertex in the inner part of the body will be shared by 8 cells, while one on a surface boundary may be shared by 4, or at a corner, only belonging to one unit cell. Similarly, if those unit cells represent different materials the stiffness for an edge spring made of one phase will be different than the second. While I can ignore this for the time being (as i am only going to consider a single unit cell to begin with), I will need some way to keep track of individual unit cells and their properties (keeping track of the individual unit cells will be necessary for iterating over unit cells when calculating volume preserving energy/force)
    #since i have a unit cell, and since the connectivity matrix looks the same for each unit cell of a material type, can I construct the connectivity matrix for a single unit cell of each type, and combine them into the overall connectivity matrix? that way I can avoid calculating separations for all nodes when I know that there's only so many connections per node possible (26 total for edge, face diagonal, and center diagonal springs in a cubic cell)
    N = np.shape(node_posns)[0]
    epsilon = np.spacing(1)
    connectivity = np.zeros((N,N))
    separations = np.empty((N,N))
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

#volume correction force calculation. I need to calculate all the forces acting on each vertex (or gradient of the energy wrt the vertex position, which is the negative of the force vector acting on the vertex). I can write this to work for a single unit cell at first, but I need to think about how I will store the forces before I add them up, and assign them to the correct vertices.
def get_volume_correction_force(elements, avg_vectors, kappa, l_e,N_vertices):
    """calculate the volume correction force on each of the vertices of the unit cell"""
    #maybe this should return the force on each vertex of the unit cell
    correction_force = np.zeros((N_vertices,3))
    counter = 0
    for el in elements:
        avg_vec = avg_vectors[:,:,counter]
        acrossb = np.cross(avg_vec[0,:],avg_vec[1,:])
        bcrossc = np.cross(avg_vec[1,:],avg_vec[2,:])
        ccrossa = np.cross(avg_vec[2,:],avg_vec[0,:])
        adotbcrossc = np.dot(avg_vec[0,:],bcrossc)
        gradV1 = -bcrossc - ccrossa - acrossb
        gradV8 = -gradV1
        gradV3 = bcrossc - ccrossa - acrossb
        gradV6 = -gradV3
        gradV7 = bcrossc + ccrossa - acrossb
        gradV2 = -gradV7
        gradV5 = -bcrossc + ccrossa - acrossb
        gradV4 = -gradV5
        prefactor = -kappa * ((1/(l_e**3) * adotbcrossc - 1))
        correction_force[el] += prefactor * np.array([gradV1,gradV2,gradV3,gradV4,gradV5,gradV6,gradV7,gradV8])
        counter += 1
    return correction_force

# need to create a multiprocessing version of the pure python version of the volume_correction_force function first, to use as a comparison point. if I can do this succesfully I should then try to do the same with the cython version with GIL. cython nogil is going to be complicated, and would use actual threading instead of multiprocessing. i need to make a list of tasks, run the tasks, and then rearrange the results since they can come in any order. actual multiprocessing would be used on the volume_correction_force and/or the spring stiffness and/or the whole timestep function.
# to start with i'm going to copy the code from the book "effective computation in physics" for the n-body problem, at least in part, to get a feel for the pattern(these first two , simulate and timestep, are for the threaded version which is not effective with the GIL working)
# def simulate(P,N,D,S,G,dt):
    # x0,v0,m=initial_cond(N,D)
    # pool=Pool(P)
    # for s in range(S):
        # x1,v1 = timestep(x0,v0,G,m,dt,pool)
        # x0,v0=x1,v1

# def timestep(x0,v0,G,m,dt,pool):
#     """computes the enxt position and velocity for all masses given initial conditions and a time step size.
#     """
#     N = len(x0)
#     # generating a task for each body
#     tasks = [(i,x0,v0,G,m,dt) for i in range(N)]
#     results = pool.do(tasks)
#     x1 = np.empty(x0.shape,dtype=float)
#     v1 = np.empty(v0.shape,dtype=float)
#     for i, x_i1, v_i1 in results:
#         x1[i] = x_i1
#         v1[i] = v_i1
#     return x1,v1

# Pool.map() has a similar interface to the built-in map() function. it tkaes two arguments, a function and an iterable of arguments to pass intio that function and returns a list of values in the same order that was given in the original iterable. the major limitation is that Pool.map() takes a function of only one argument, overcome this by storing the arguments you neeed in a typle or dictionary before calling it

# def timestep_i(args):
#     """computes the next position and velocity for the ith mass"""
#     i,x0,v0,G,m,dt = args
#     # unpacking the arguments
#     # a_i0 = a(i,x0,G,m)
#     # v_i1 = a_i0*dt + v0[i]
#     # x_i1 = a_i0 * dt**2 + v0[i] * dt + x0[i]
#     # return i,x_i1, v_i1

# actual timestep function needs to be modified as well. need to swap out the old do() call for the new Pool.map() and pass it timestep_i as well as the tasks

# def timestep(x0,v0,G,m,dt,pool):
#     """computes next position and velcoity for all masses given initial conditions and a time step size"""
#     N = len(x0)
#     # generating a task for each body
#     tasks = [(i,x0,v0,G,m,dt) for i in range(N)]
#     results = pool.map(timestep_i,tasks)
#     x1 = np.empty(x0.shape,dtype=float)
#     v1 = np.empty(v0.shape,dtype=float)
#     for i, x_i1, v_i1 in results:
#         x1[i] = x_i1
#         v1[i] = v_i1
#     return x1,v1
    
def get_volume_correction_force_mp(elements, avg_vectors, kappa, l_e,N_vertices,pool):
    """calculate the volume correction force on each of the vertices of the unit cell using multiprocessing pool"""
    # list of tasks is just the list of elements plus avg_vectors. additional information is the row of the element whose force is being calculated. return is the 8x3 matrix for forces (and the row of the element?). recombining results involves summations. i'm replacing the loop below, as that loop will be the task sent to the pool of workers. after getting the results (appending each result to a list of results?) I will then sort the results and perform the necessary summations. should i be instantiating a pool each time this function is called? maybe to start with. in an implementation for the whole simulation method I would instantiate the pool of workers and dish tasks out to them as needed for normal spring force calculations and the volume correction force, but that would be an instantiation at the start of the simulation, not when the spring forces need to be evaluated, or the volume correction force.
    correction_force = np.zeros((N_vertices,3))
    tasks = [(i,elements[i,:],avg_vectors[:,:,i],kappa,l_e) for i in range(elements.shape[0])]
    results = pool.map(get_volume_correction_force_el_mp,tasks,chunksize=None)
    for i, correction_force_el in results:
        correction_force[elements[i]] += correction_force_el
    return correction_force

def get_volume_correction_force_el_mp(args):
    """computes the volume correction force for a single cubic element"""
    i, el, avg_vec, kappa, l_e = args
    # print('got here somehow')
    correction_force_el = np.zeros((8,3),dtype=np.float64)
    # unpacking arguments
    acrossb = np.cross(avg_vec[0,:],avg_vec[1,:])
    bcrossc = np.cross(avg_vec[1,:],avg_vec[2,:])
    ccrossa = np.cross(avg_vec[2,:],avg_vec[0,:])
    adotbcrossc = np.dot(avg_vec[0,:],bcrossc)
    gradV1 = -bcrossc - ccrossa - acrossb
    gradV8 = -gradV1
    gradV3 = bcrossc - ccrossa - acrossb
    gradV6 = -gradV3
    gradV7 = bcrossc + ccrossa - acrossb
    gradV2 = -gradV7
    gradV5 = -bcrossc + ccrossa - acrossb
    gradV4 = -gradV5
    prefactor = -kappa * ((1/(l_e**3) * adotbcrossc - 1))
    correction_force_el += prefactor * np.array([gradV1,gradV2,gradV3,gradV4,gradV5,gradV6,gradV7,gradV8])
    return (i, correction_force_el)

def main():
    E = 1
    nu = 0.49
    l_e = .1
    Lx = 1.
    Ly = 1.
    Lz = 1.

    k = get_spring_constants(E, nu, l_e)
    kappa = get_kappa(E, nu)


    node_posns,elements,boundaries = discretize_space(Lx,Ly,Lz,l_e)
    (c,s) = create_connectivity(node_posns,elements,k,l_e)

    x0 = node_posns
    x0[boundaries['right']] *= 1.05
    N = len(x0)
    avg_vectors = get_average_edge_vectors(x0,elements)

    correction_force_cy = np.zeros((N,3))
    correction_force_cy_mp = np.zeros((N,3))

    N_runs_perf = 1000
    start = time.perf_counter()
    for i in range(N_runs_perf):
        correction_force_npy = get_volume_correction_force(elements, avg_vectors, kappa, l_e,N)

    end = time.perf_counter()
    delta_npy = end - start

    start = time.perf_counter()
    for i in range(N_runs_perf):
        correction_force_cy = np.zeros((N,3))
        get_volume_correction_force_cy.get_volume_correction_force(elements, avg_vectors, kappa, l_e, correction_force_cy)
        
        
    end = time.perf_counter()
    delta_cy = end - start

    # testvec1 = np.array([1.,0.,0.])
    # testvec2 = np.array([0.,1.,0.])
    # testvec3 = np.array([0.,0.,1.])
    # testvec4 = np.array([1.,1.,1.])
    # testvec5 = np.array([2.1,-.8,1.1])
    # nogil_crossresultvec = np.array([0.,0.,0.])
    # npdotrseult = np.dot(testvec4,testvec5)
    # npcrossresult = np.cross(testvec4,testvec5)
    # cydotresult = get_volume_correction_force_cy.dot_prod(testvec4,testvec5)
    # cycrossresult = get_volume_correction_force_cy.cross_prod(testvec4,testvec5)
    # nogil_dotresult = get_volume_correction_force_cy_nogil.dot_prod(testvec1,testvec2)
    # get_volume_correction_force_cy_nogil.cross_prod(testvec1,testvec2,nogil_crossresultvec)

    print(delta_cy,delta_npy)
    print('Cython is {}x faster'.format(delta_npy/delta_cy))
    
    # multiprocess version of pure python using different numbers of cores. i will scale up the problem as well to see how the performance difference occurs
    P = 8
    start = time.perf_counter()
    with Pool(P) as pool:
        for i in range(N_runs_perf):
            correction_force_npy_mp = get_volume_correction_force_mp(elements, avg_vectors, kappa, l_e,N,pool)
    end = time.perf_counter()
    delta_npy_mp = end-start


    print(delta_cy,delta_npy,delta_npy_mp)
    print('Multiprocessing is {}x faster'.format(delta_npy/delta_npy_mp))

    correction_force_el = np.empty((8,3),dtype=np.float64)
    vectors = np.empty((8,3),dtype=np.float64)
    avg_vectors = np.empty((3,3),dtype=np.float64)
    correction_force_cy_nogil = np.zeros((N,3),dtype=np.float64)
    start = time.perf_counter()
    for i in range(N_runs_perf):
    #    correction_force_cy_nogil = np.zeros((N,3),dtype=np.float64) 
       get_volume_correction_force_cy_nogil.get_volume_correction_force(x0,elements,kappa,l_e,correction_force_el,vectors,avg_vectors, correction_force_cy_nogil)
       
        
    end = time.perf_counter()
    delta_cy_nogil = end-start
    print(delta_cy,delta_npy,delta_npy_mp,delta_cy_nogil)
    print('nogil Cython is {}x faster'.format(delta_npy/delta_cy_nogil))

    # assert ((correction_force_npy == correction_force_cy).all())
    if (correction_force_npy == correction_force_cy).all():
        print('same correction force calculated between python and cython methods')
    else:
        print('difference in calculated correction force between python and cython methods')
        print('maximum difference is {}x'.format(np.max(correction_force_npy-correction_force_cy)))
        print('mean diffrence is {}x'.format(np.mean(correction_force_npy-correction_force_cy)))
    if (correction_force_npy == correction_force_npy_mp).all():
        print('same correction force calculated between python and python_mp methods')
    if (correction_force_npy == correction_force_cy_nogil).all():
        print('same correction force calculated between python and cython_nogil methods')
    else:
        print('difference in calculated correction force between python and cython_nogil methods')
        print('maximum difference is {}x'.format(np.max(correction_force_npy-correction_force_cy_nogil)))
        print('mean diffrence is {}x'.format(np.mean(correction_force_npy-correction_force_cy_nogil)))
    if (correction_force_cy == correction_force_cy_nogil).all():
        print('same correction force calculated between cython and cython_nogil methods')
    else:
        print('difference in calculated correction force between cython and cython_nogil methods')
        print('maximum difference is {}x'.format(np.max(correction_force_cy-correction_force_cy_nogil)))
        print('mean diffrence is {}x'.format(np.mean(correction_force_cy-correction_force_cy_nogil)))


if __name__ == "__main__":
    main()