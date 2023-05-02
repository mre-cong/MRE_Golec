import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import springs


def place_two_particles(radius,l_e,dimensions,separation):
    """Return a 2D array where each row lists the indices making up a rigid particle. radius is the size in meters, l_e is the cubic element edge length in meters, dimensions are the simulated volume size in meters, separation is the center to center particle separation in cubic elements."""
    Lx, Ly, Lz = dimensions
    # radius = 0.5*l_e# radius = l_e*(4.5)
    assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
    radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
    Nel_x = np.round(Lx/l_e).astype(np.int32)
    Nel_y = np.round(Ly/l_e).astype(np.int32)
    Nel_z = np.round(Lz/l_e).astype(np.int32)
    #find the center of the simulated system
    center = (np.array([Nel_x,Nel_y,Nel_z])/2) * l_e
    #if there are an even number of elements in a direction, need to increment the central position by half an edge length so the particle centers match up with the centers of cubic elements
    if np.mod(Nel_x,2) == 0:
        center[0] += l_e/2
    if np.mod(Nel_y,2) == 0:
        center[1] += l_e/2
    if np.mod(Nel_z,2) == 0:
        center[2] += l_e/2
    #check particle separation to see if it is acceptable or not for the shift in particle placement from the simulation "center" to align with the cubic element centers
    if np.mod(separation,2) == 1:
        shift_l = (separation-1)*l_e/2
        shift_r = (separation+1)*l_e/2
    else:
        shift_l = separation*l_e/2
        shift_r = shift_l
    particle_nodes = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center-np.array([shift_l,0,0]),dimensions)
    particle_nodes2 = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center+np.array([shift_r,0,0]),dimensions)
    particles = np.vstack((particle_nodes,particle_nodes2))
    return particles

def main():
    E = 1
    nu = 0.499
    l_e = .1#cubic element side length
    Lx = 1.0
    Ly = 1.1
    Lz = 1.1
    t_f = 30
    dimensions = np.array([Lx,Ly,Lz])
    #TODO
    #need functionality to check some central directory containing initialization files
    system_string = f'E_{E}_le_{l_e}_Lx_{Lx}_Ly_{Ly}_Lz_{Lz}'
    current_dir = os.path.abspath('.')
    input_dir = current_dir + f'/init_files/{system_string}/'
    if not (os.path.isdir(input_dir)):#TODO add and statement that checks if the init file also exists?
        os.mkdir(input_dir)
        node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
        elements = springs.get_elements(node_posns, dimensions, l_e)
        boundaries = mre.initialize.get_boundaries(node_posns)
        k = mre.initialize.get_spring_constants(E, nu, l_e)
        node_types = springs.get_node_type(node_posns.shape[0],boundaries,dimensions,l_e)
        k = np.array(k,dtype=np.float64)
        # springs = mre.initialize.create_springs(node_posns,k,l_e,dimensions)
        max_springs = np.round(Lx/l_e + 1).astype(np.int32)*np.round(Ly/l_e + 1).astype(np.int32)*np.round(Lz/l_e + 1).astype(np.int32)*13
        springs_var = np.empty((max_springs,4),dtype=np.float64)
        num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions, l_e)
        springs_var = springs_var[:num_springs,:]
        separation = 5
        radius = 0.5*l_e# radius = l_e*(4.5)
        # assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
        # radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
        # center = (np.round(np.array([Lx/l_e,Ly/l_e,Lz/l_e]))/2) * l_e
        # particle_nodes = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center-np.array([separation*l_e/2,0,0]),dimensions)
        # particle_nodes2 = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center+np.array([separation*l_e/2,0,0]),dimensions)
        # particles = np.vstack((particle_nodes,particle_nodes2))
        particles = place_two_particles(radius,l_e,dimensions,separation)
        mre.initialize.write_init_file(node_posns,springs_var,elements,particles,boundaries,input_dir)
    else:
        node_posns, springs_var, elements, boundaries = mre.initialize.read_init_file(input_dir+'init.h5')
        #TODO implement support functions for particle placement to ensure matching to existing grid of points and avoid unnecessary repetition
        #radius = l_e*0.5
        separation = 5
        radius = l_e*0.5# radius = l_e*(4.5)
        particles = place_two_particles(radius,l_e,dimensions,separation)
        # assert(radius < np.min(dimensions)/2), f"Particle size greater than the smallest dimension of the simulation"
        # radius_voxels = np.round(radius/l_e,decimals=1).astype(np.float32)
        # center = (np.round(np.array([Lx/l_e,Ly/l_e,Lz/l_e]))/2) * l_e
        # particle_nodes = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center-np.array([separation*l_e/2,0,0]),dimensions)
        # print(f'particle center position {center[0]-separation*l_e/2}, {center[1]}, {center[2]}')
        print(f'particle node positions based on the returned nodes')
        for particle in particles:
            for node in particle:
               print(f'{node_posns[node]}')
        # particle_nodes2 = mre.sphere_rasterization.place_sphere(radius_voxels,l_e,center+np.array([separation*l_e/2,0,0]),dimensions)
        #TODO do better at placing multiple particles, make the helper functionality to ensure placement makes sense
        # particles = np.vstack((particle_nodes,particle_nodes2)) 

if __name__ == "__main__":
    main()