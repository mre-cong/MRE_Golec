import numpy as np
import matplotlib.pyplot as plt
import time
import os
import lib_programname
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization

#the purpose of this script: To compare teh final node positions given the same simulation parameters when using a method involving mks units versus a scaled approach (the purpose of the scaling is to allow micron sized systems to be simulated). This approach is not sufficient for verification. The numerical integration will proceed differently for the scaled versus unscaled systems, as such the final node configurations given the same number of iterations for integration will not be the same. If a set of convergence criteria were chosen, there would still be issues regarding equivalence of the convergence criteria between unscaled (mks) and scaled systems. As an alternative approach, plotting the RMS acceleration versus integration iteration to see if both systems converge, and to observe the convergence behavior, will be done. Because of the nature of the numerical integration, the pltos will have to be produced during the simulation, and saved out. When a magnetic field is applied, certain aspects of the particle behavior can be compared, like the particle positions/separations and magnetizations, as a verification of the results between the scaled and unscaled systems.
scaled_system_dir = '/mnt/c/Users/bagaw/Desktop/scaled_system_verification/'
unscaled_system_dir = '/mnt/c/Users/bagaw/Desktop/unscaled_system_results/'
#should be reading in the simulation log file, or the init file, and using the stored values to determine the scaling factor for the positions
scaling_factor = 0.1
unmatched_nodes = []
for i in range(0,3):
    unscaled_posns, bc1 = mre.initialize.read_output_file(unscaled_system_dir+f'output_{i}.h5')
    scaled_posns, bc2 = mre.initialize.read_output_file(scaled_system_dir+f'output_{i}.h5')
    resized_posns = scaled_posns*scaling_factor
    for j in range(unscaled_posns.shape[0]):
        correctness = np.allclose(unscaled_posns[j,:],resized_posns[j,:])
        if not correctness:
            unmatched_nodes.append(j)
    print(f'total number of unmatched nodes:{len(unmatched_nodes)}')
    print(f'unmatched node IDs are:{unmatched_nodes}')
    unmatched_nodes = []
    # correctness = np.allclose(unscaled_posns,resized_posns)
    # print(f'Final node configurations are close?:{correctness}')