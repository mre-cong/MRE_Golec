# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:46:56 2023

@author: bagaw
"""

cimport cython
#volume correction force calculation. I need to calculate all the forces acting on each vertex (or gradient of the energy wrt the vertex position, which is the negative of the force vector acting on the vertex). I can write this to work for a single unit cell at first, but I need to think about how I will store the forces before I add them up, and assign them to the correct vertices.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_volume_correction_force(int[:,:] elements, double[:,:,:] avg_vectors, double kappa, double l_e, double[:,:] correction_force):
    """calculate the volume correction force on each of the vertices of the unit cell of each element. must pass an array for correction_force that will be modified"""
    cdef int i
    cdef int j
    cdef double correction_force_el[8][3]
    for i in range(8):
        for j in range(3):
            correction_force_el[i][j] = 0.0
    # iterating over elements, get an 8x3 array, then add to each entry in the correction_force array by iterating over j, getting the index from elements[i,j] (the jth node of the ith element), where correction_force_el[j,0] is the x component of the jth node due to the volume correction force 
    for i in range(elements.shape[0]):
        get_volume_correction_force_el(elements[i,:], avg_vectors[:,:,i], kappa, l_e,correction_force_el)
        for j in range(8):
            correction_force[elements[i,j]][0] += correction_force_el[j][0]
            correction_force[elements[i,j]][1] += correction_force_el[j][1]
            correction_force[elements[i,j]][2] += correction_force_el[j][2]


#let's create a cdef function that returns the volume correction forces on the nodes of a single element, and call that from the cpdef function that we will call in the python script to get the volume correction force on each object. I should also write a cdef cross function, since that takes up a significant portion of the volume correction force calculation according to the profiler results. I should try writing some example cython functions first before trying to convert these things. I also need to make sure i set up a git repository, or ensure i am using the one i had already set up rather than continuing to do things without a record. try writing the cross product function and compare the results to the numpy function to ensure the result is correct.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_volume_correction_force_el(int[:] element, double[:,:] avg_vec, double kappa, double l_e, double[:,:] correction_force):
    cdef double[3] acrossb = cross_prod(avg_vec[0,:],avg_vec[1,:])
    cdef double[3] bcrossc = cross_prod(avg_vec[1,:],avg_vec[2,:])
    cdef double[3] ccrossa = cross_prod(avg_vec[2,:],avg_vec[0,:])
    cdef double adotbcrossc = dot_prod(avg_vec[0,:],bcrossc)
    cdef double[3] gradV1
    cdef double[3] gradV8
    cdef double[3] gradV3
    cdef double[3] gradV6
    cdef double[3] gradV7
    cdef double[3] gradV2
    cdef double[3] gradV5
    cdef double[3] gradV4
    cdef int i
    for i in range(3):
        gradV1[i] = -1*bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV8[i] = -1*gradV1[i]
        gradV3[i] = bcrossc[i] -1*ccrossa[i] -1*acrossb[i]
        gradV6[i] = -1*gradV3[i]
        gradV7[i] = bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV2[i] = -1*gradV7[i]
        gradV5[i] = -1*bcrossc[i] + ccrossa[i] -1*acrossb[i]
        gradV4[i] = -1*gradV5[i]
    cdef double prefactor = -kappa * ((1/pow(l_e,3) * adotbcrossc - 1))
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
cpdef cross_prod(double[:] vec1, double[:] vec2):
    cdef double result[3]
    result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dot_prod(double[:] vec1, double[:] vec2):
    cdef double result = 0 
    cdef int i
    for i in range(vec1.shape[0]):
        result += vec1[i]*vec2[i]
    return result