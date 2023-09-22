# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:46:56 2023

@author: bagaw
"""

cimport cython
# from cython.parallel import prange
# import numpy as np
cimport numpy as np
import numpy as np
#cstruct for a 3D vector
cdef struct Vec3:
    double x, y, z

#volume correction force calculation. I need to calculate all the forces acting on each vertex (or gradient of the energy wrt the vertex position, which is the negative of the force vector acting on the vertex). I can write this to work for a single unit cell at first, but I need to think about how I will store the forces before I add them up, and assign them to the correct vertices.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_volume_correction_force(double[:,::1] node_posns,int[:,::1] elements, double kappa, double l_e,double[:,::1] correction_force_el,double[:,::1] vectors, double[:,::1] avg_vectors, double[:,::1] correction_force):
    """calculate the volume correction force on each of the vertices of the unit cell of each element. must pass an array for correction_force that will be modified"""
    cdef int i
    cdef int j
    # cdef double[8][3] correction_force_el
    # because of the way that the nogil statements require things to be handled without interfacing with python, and because the elements of the volume correction force are accumulated, it is necessary to instantiate(or reset) the values of the elements to zero whenever the function is invoked so that garbage data is not used and forces across timesteps are not accumulated
    for i in range(correction_force.shape[0]):
        for j in range(3):
            correction_force[i][j] = 0.0
    # iterating over elements, get an 8x3 array, then add to each entry in the correction_force array by iterating over j, getting the index from elements[i,j] (the jth node of the ith element), where correction_force_el[j,0] is the x component of the jth node due to the volume correction force 
    # cdef double vectors[8][3]
    # cdef double avg_vectors[3][3]
    with nogil:
        for i in range(elements.shape[0]):
            get_average_edge_vectors(node_posns,elements,i,vectors,avg_vectors)
            get_volume_correction_force_el(avg_vectors, kappa, l_e,correction_force_el)
            for j in range(8):
                correction_force[elements[i,j]][0] += correction_force_el[j][0]
                correction_force[elements[i,j]][1] += correction_force_el[j][1]
                correction_force[elements[i,j]][2] += correction_force_el[j][2]

#let's create a cdef function that returns the volume correction forces on the nodes of a single element, and call that from the cpdef function that we will call in the python script to get the volume correction force on each object. I should also write a cdef cross function, since that takes up a significant portion of the volume correction force calculation according to the profiler results. I should try writing some example cython functions first before trying to convert these things. I also need to make sure i set up a git repository, or ensure i am using the one i had already set up rather than continuing to do things without a record. try writing the cross product function and compare the results to the numpy function to ensure the result is correct.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_volume_correction_force_el(double[:,::1] avg_vec, double kappa, double l_e, double[:,::1] correction_force) nogil:
    cdef double[3] acrossb
    cdef double[3] bcrossc
    cdef double[3] ccrossa
    cdef int i
    cdef double[3] avg_a
    cdef double[3] avg_b
    cdef double[3] avg_c
    for i in range(3):
        avg_a[i] = 0.
        avg_b[i] = 0.
        avg_c[i] = 0.
    for i in range(3):
        avg_a[i] += avg_vec[0,i]
        avg_b[i] += avg_vec[1,i]
        avg_c[i] += avg_vec[2,i]
    cross_prod(avg_a,avg_b,acrossb)
    cross_prod(avg_b,avg_c,bcrossc)
    cross_prod(avg_c,avg_a,ccrossa)
    cdef double adotbcrossc = dot_prod(avg_a,bcrossc)
    cdef double[3] gradV1
    cdef double[3] gradV8
    cdef double[3] gradV3
    cdef double[3] gradV6
    cdef double[3] gradV7
    cdef double[3] gradV2
    cdef double[3] gradV5
    cdef double[3] gradV4
    for i in range(3):
        gradV1[i] = -1*bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV8[i] = -1*gradV1[i]
        gradV3[i] = bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV6[i] = -1*gradV3[i]
        gradV7[i] = bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV2[i] = -1*gradV7[i]
        gradV5[i] = -1*bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV4[i] = -1*gradV5[i]
    cdef double prefactor = -kappa * ((1/(l_e*l_e*l_e) * adotbcrossc - 1))
    for i in range(3):
        correction_force[0][i] = prefactor*gradV1[i]
        correction_force[1][i] = prefactor*gradV2[i]
        correction_force[2][i] = prefactor*gradV3[i]
        correction_force[3][i] = prefactor*gradV4[i]
        correction_force[4][i] = prefactor*gradV5[i]
        correction_force[5][i] = prefactor*gradV6[i]
        correction_force[6][i] = prefactor*gradV7[i]
        correction_force[7][i] = prefactor*gradV8[i]

#volume correction force calculation taking into account scaling of the positions/lengths in the simulation. I need to calculate all the forces acting on each vertex (or gradient of the energy wrt the vertex position, which is the negative of the force vector acting on the vertex). I can write this to work for a single unit cell at first, but I need to think about how I will store the forces before I add them up, and assign them to the correct vertices.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_volume_correction_force_normalized(double[:,::1] node_posns,int[:,::1] elements, double kappa, double[:,::1] correction_force_el,double[:,::1] vectors, double[:,::1] avg_vectors, double[:,::1] correction_force):
    """calculate the volume correction force on each of the vertices of the unit cell of each element. must pass an array for correction_force that will be modified"""
    cdef int i
    cdef int j
    # cdef double[8][3] correction_force_el
    # because of the way that the nogil statements require things to be handled without interfacing with python, and because the elements of the volume correction force are accumulated, it is necessary to instantiate(or reset) the values of the elements to zero whenever the function is invoked so that garbage data is not used and forces across timesteps are not accumulated
    for i in range(correction_force.shape[0]):
        for j in range(3):
            correction_force[i][j] = 0.0
    # iterating over elements, get an 8x3 array, then add to each entry in the correction_force array by iterating over j, getting the index from elements[i,j] (the jth node of the ith element), where correction_force_el[j,0] is the x component of the jth node due to the volume correction force 
    # cdef double vectors[8][3]
    # cdef double avg_vectors[3][3]
    with nogil:
        for i in range(elements.shape[0]):
            get_average_edge_vectors(node_posns,elements,i,vectors,avg_vectors)
            get_volume_correction_force_el_normalized(avg_vectors, kappa,correction_force_el)
            for j in range(8):
                correction_force[elements[i,j]][0] += correction_force_el[j][0]
                correction_force[elements[i,j]][1] += correction_force_el[j][1]
                correction_force[elements[i,j]][2] += correction_force_el[j][2]

#let's create a cdef function that returns the volume correction forces on the nodes of a single element, and call that from the cpdef function that we will call in the python script to get the volume correction force on each object. I should also write a cdef cross function, since that takes up a significant portion of the volume correction force calculation according to the profiler results. I should try writing some example cython functions first before trying to convert these things. I also need to make sure i set up a git repository, or ensure i am using the one i had already set up rather than continuing to do things without a record. try writing the cross product function and compare the results to the numpy function to ensure the result is correct.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_volume_correction_force_el_normalized(double[:,::1] avg_vec, double kappa, double[:,::1] correction_force) nogil:
    cdef double[3] acrossb
    cdef double[3] bcrossc
    cdef double[3] ccrossa
    cdef int i
    cdef double[3] avg_a
    cdef double[3] avg_b
    cdef double[3] avg_c
    for i in range(3):
        avg_a[i] = 0.
        avg_b[i] = 0.
        avg_c[i] = 0.
    for i in range(3):
        avg_a[i] += avg_vec[0,i]
        avg_b[i] += avg_vec[1,i]
        avg_c[i] += avg_vec[2,i]
    cross_prod(avg_a,avg_b,acrossb)
    cross_prod(avg_b,avg_c,bcrossc)
    cross_prod(avg_c,avg_a,ccrossa)
    cdef double adotbcrossc = dot_prod(avg_a,bcrossc)
    cdef double[3] gradV1
    cdef double[3] gradV8
    cdef double[3] gradV3
    cdef double[3] gradV6
    cdef double[3] gradV7
    cdef double[3] gradV2
    cdef double[3] gradV5
    cdef double[3] gradV4
    for i in range(3):
        gradV1[i] = -1*bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV8[i] = -1*gradV1[i]
        gradV3[i] = bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV6[i] = -1*gradV3[i]
        gradV7[i] = bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV2[i] = -1*gradV7[i]
        gradV5[i] = -1*bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV4[i] = -1*gradV5[i]
    cdef double prefactor = -kappa * ((adotbcrossc - 1))
    for i in range(3):
        correction_force[0][i] = prefactor*gradV1[i]
        correction_force[1][i] = prefactor*gradV2[i]
        correction_force[2][i] = prefactor*gradV3[i]
        correction_force[3][i] = prefactor*gradV4[i]
        correction_force[4][i] = prefactor*gradV5[i]
        correction_force[5][i] = prefactor*gradV6[i]
        correction_force[6][i] = prefactor*gradV7[i]
        correction_force[7][i] = prefactor*gradV8[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cross_prod(double[3] vec1, double[3] vec2, double[3] result) nogil:
    result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot_prod(double[3] vec1, double[3] vec2) nogil:
    cdef double result = 0 
    cdef int i
    for i in range(3):
        result += vec1[i]*vec2[i]
    return result

#helper function for getting the unit cell volume. I need the averaged edge vectors used in the volume calculation for other calculations later (the derivative of the deformed volume with respect to the position of each vertex is used to calculate the volume correction force). However, the deformed volume is also used in that expression. Really these are two helper functions for the volume correction force
#need to cythonize this. what should the return datatype look like, what shape should it have? last index will be contiguous if i make this a numpy array, which sure, whynot
#but i need to look at instantiating numpy arrays in cython code. include a datatype for the numpy array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_average_edge_vectors(double[:,::1] node_posns,int[:,::1] elements, int index, double[:,::1] vectors, double[:,::1]avg_vectors) nogil:
    """passing in the node positions for vertices, the row indices of the element of interest, a numpy array (8,3) for storing the positions of the nodes of the element, and a (3,3) numpy array that will be manipulated to get the resulting average edge vectors of the element"""
    cdef int[8] element
    cdef int i
    for i in range(8):
        element[i] = elements[index,i]
    for i in range(8):
        vectors[i,:] = node_posns[element[i],:]
    for i in range(3):
        avg_vectors[0,i] = vectors[2,i] - vectors[0,i] + vectors[3,i] - vectors[1,i] + vectors[6,i] - vectors[4,i] + vectors[7,i] - vectors[5,i]
        avg_vectors[1,i] = vectors[4,i] - vectors[0,i] + vectors[6,i] - vectors[2,i] + vectors[5,i] - vectors[1,i] + vectors[7,i] - vectors[3,i]
        avg_vectors[2,i] = vectors[1,i] - vectors[0,i] + vectors[3,i] - vectors[2,i] + vectors[5,i] - vectors[4,i] + vectors[7,i] - vectors[6,i]
    cdef int j
    for i in range(3):
        for j in range(3):
            avg_vectors[i,j] *= 0.25


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] get_avg_edge_vectors_normalized(double[:,::1] node_posns,int[:,::1] elements, double[:,::1] vectors, double[:,::1] avg_vectors):
    """calculate the average edge vectors of each element. must pass an (3,3) shape array for avg_vectors that will be modified"""
    cdef int i
    cdef int j
    cdef np.ndarray[np.float64_t, ndim=3] element_avg_vectors = np.zeros((elements.shape[0],3,3),dtype=np.float64)
    for i in range(elements.shape[0]):
        get_average_edge_vectors(node_posns,elements,i,vectors,avg_vectors)
        for j in range(3):
            element_avg_vectors[i,0,j] = avg_vectors[0,j]
            element_avg_vectors[i,1,j] = avg_vectors[1,j]
            element_avg_vectors[i,2,j] = avg_vectors[2,j]
    return element_avg_vectors

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] get_element_volume_normalized(double[:,:,::1] avg_vectors):
    """calculate the volume of each element."""
    cdef np.ndarray[np.float64_t, ndim=1] element_volumes = np.zeros((avg_vectors.shape[0],),dtype=np.float64)
    cdef double[3] bcrossc
    cdef double adotbcrossc
    cdef int i
    cdef int j
    cdef double[3] avg_a
    cdef double[3] avg_b
    cdef double[3] avg_c
    for j in range(avg_vectors.shape[0]):
        for i in range(3):
            avg_a[i] = 0.
            avg_b[i] = 0.
            avg_c[i] = 0.
        for i in range(3):
            avg_a[i] += avg_vectors[j,0,i]
            avg_b[i] += avg_vectors[j,1,i]
            avg_c[i] += avg_vectors[j,2,i]
        cross_prod(avg_b,avg_c,bcrossc)
        adotbcrossc = dot_prod(avg_a,bcrossc)
        element_volumes[j] = adotbcrossc
    return element_volumes

# cdef void get_average_edge_vectors(node_posns,elements,avg_vectors):
#     counter = 0
#     for el in elements:
#         vectors = node_posns[el]
#         avg_vectors[0,:,counter] = vectors[2] - vectors[0] + vectors[3] - vectors[1] + vectors[6] - vectors[4] + vectors[7] - vectors[5]
#         avg_vectors[1,:,counter] = vectors[4] - vectors[0] + vectors[6] - vectors[2] + vectors[5] - vectors[1] + vectors[7] - vectors[3]
#         avg_vectors[2,:,counter] = vectors[1] - vectors[0] + vectors[3] - vectors[2] + vectors[5] - vectors[4] + vectors[7] - vectors[6]
#         counter += 1
#     avg_vectors *= 0.25
#     return avg_vectors

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef Vec3 cross_prod(Vec3 vec1, Vec3 vec2):
#    cdef Vec3 result
#    result.x = vec1.y*vec2.z - vec1.z*vec2.y
#    result.y = vec1.z*vec2.x - vec1.x*vec2.z
#    result.z = vec1.x*vec2.y - vec1.y*vec2.x
#    return result

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef double dot_prod(Vec3 vec1, Vec3 vec2):
    #cdef double result = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z
    #return result