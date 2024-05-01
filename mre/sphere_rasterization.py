
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba.experimental import jitclass
from numba import types, njit, prange
import mre.initialize
#this code was taken from https://stackoverflow.com/questions/41656006/how-to-rasterize-a-sphere
#originally written by Mitchell Walls on july 16 2021, adapted from a method written by Matt Timmerman in the same thread
#i will/have made changes to comment out how the code works and changes to adapt for my purposes in simulating MREs, since i need to rasterize spherical particles that will be treated as rigid objects
#this code apparently only works for spheres instantiated in the positive octant (I)


#below is text from Matt Timmerman, explaining the method
# I think the easiest way to do this is something like the Midpoint Circle Algorithm, extended to 3D.

# First, lets figure out which blocks we want to fill. Assuming an origin in the middle of block (0,0,0) and radius R:

#     We only want to fill boxes inside the sphere. Those are exactly the boxes (x,y,z) such that x²+y²+z² <= R²; and
#     We only want to fill boxes with a face showing. If a box has a face showing, then at least one of its neighbors is not in the sphere, so: (|x|+1)²+y²+z² > R² OR x²+(|y|+1)²+z² > R² OR x²+y²+(|z|+1)² > R²

# It's the 2nd part that makes it tricky, but remember that (|a|+1)² = |a|² + 2|a| + 1. If, say, z is the largest coordinate of a box that is inside the sphere, and if that box has a face showing, then the z face in particular will be showing, because x²+y²+(|z|+1)² = x²+y²+z²+2|z|+1, and that will be at least as big as the analogous values for x and y.

# So, it's pretty easy to calculate the boxes that are 1) inside the sphere, 2) have z as their largest coordinate, and 3) have the largest possible z value, i.e., adding 1 to z results in a box outside the sphere. Additionally, 4) have positive values for all x,y,z.

# The coordinates of these boxes can then be reflected 24 different ways to generate all the boxes on the surface of the sphere. Those are all 8 combinations of signs of the coordinates times all 3 choices for which axis has the largest coordinate.

#this is a type definition for the just-in-time compiler of numba. specifying types is important (maybe even necessary) for the compiler to function
spec = [
    ('radius', types.float64),
    ('svoxel', types.int64[:]),
    ('grid', types.b1[:,:,:])#,
    #('inner_grid', types.b1[:,:,:])
]
#why this is implemented as a class is unclear to me, but it provides an interface for the functionality, and i suppose that is a good enough reason
@jitclass(spec)
class RasterizeSphere(object):
    def __init__(self, svoxel, radius, grid):
        #grid is a 3rd rank tensor/3D array of boolean values, where true describes a voxel that belongs to the rasterized sphere on a grid of voxels
        self.grid = grid
        #same as grid, but will only refer to the inner voxels
        # self.inner_grid = grid
        #svoxel appears to be a 1D array of values describing the central voxel of the sphere
        self.svoxel = svoxel
        #radius of the sphere in voxels. since the diameter should be an integer value, the radius must be a half integer value
        self.radius = radius
        R2 = np.floor(self.radius**2)
        #z value such that z is the maximum value for a voxel in the sphere for the given x (or x and y) value(s). as x changes, the maximum z value will change
        zmax_given_x = np.int64(np.floor(self.radius))
        x = 0
        #assuming grid points at the center of voxels and starting at (0,0,0), we only want to fill voxels such that x**2 + y **2 + z**2 <= R**2
        #and, we only want to fill voxels who have a face showing (since we are only concerned with the surface of the sphere) so these are voxels such that at least one neighbor is NOT in the sphere (|x| + 1)**2 + y**2 + z**2 > R**2 or y, or z
        while True:
            while x**2 + zmax_given_x**2 > R2 and zmax_given_x >= x:
                zmax_given_x -= 1
            if zmax_given_x < x:
                break
            z = zmax_given_x
            y = 0
            #now repeat the same modification of the max z value taking into account the current y coordinate
            while True:
                while x**2 + y**2 + z**2 > R2 and z >= x and z >= y:
                    z -= 1
                if z < x or z < y:
                    break
                self.fill_all(x, y, z)
                ###### Optional fills the inside of sphere as well. #######
                #useful for getting the internal voxels on the grid, so i can remove them from my larger MRE system later
                for nz in range(z):
                    self.fill_all(x, y, nz)
                y += 1
            x += 1

    #fill_signs assigns true values to each voxel on the grid belonging to the sphere based on the x,y,z values given by performing reflections
    # z,y,x 
    # + + +
    # - + +
    # + - +
    # - - +
    # + + -
    # - + -
    # + - -
    # - - -
    #order of the coordinate signs as the reflections are performed and the grid value assigned to True
    def fill_signs(self, x, y, z):
        self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = True
        while True:
            z = -z
            if z >= 0:
                y = -y
                if y >= 0:
                    x = -x
                    if x >= 0:
                        break
            self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = True

    # for fill_all, fill_signs is called. i need to work through an example, but it almost seems like there would be cases where the if statements result in voxels that are already assigned True being assigned True again, but the calculation of zmax_given_x seems to preclue z every being greater than y or x (only ever equal or less than), in which case the conditional checks are unnecessary.
    def fill_all(self, x, y, z):
        self.fill_signs(x, y, z)
        if z > y:
            self.fill_signs(x, z, y)
        if z > x and z > y:
            self.fill_signs(z, y, x)

#receiving errors using njit for parallel computation, seemingly due to type mismatch between expected and provided return types (is it allowed to provide return types with a parallel function? CUDA would require a kernel to be of type void, and return no values, instead mutating an argument in place)
#@njit(parallel=True, cache=True)
def parallel_spheres(grid):
    # prange just to show the insane speedup for large number of spheres disable visualization below if using large amount of prange.
    for i in prange(2):
        radius = 1.5
        svoxel = np.array([5, 5, 5])
        max = np.int64(np.ceil(radius**2))
        rs = RasterizeSphere(svoxel, radius, grid)
        points = np.where(rs.grid)
        return np.array([*points])
#______________________________
#below this line is all my code, above it is mostly code from the authors credited above
def get_sphere_on_grid(radius):
    """given a sphere radius in voxels (half integer), return a 2D array whose row entries define which grid points are voxels belonging to the sphere"""
    #TODO better use of assert statement to ensure appropriate sphere size
    assert radius > 0,f'particle radius in voxels must be a positive half integer value, you used {radius}'
    max = np.int64(np.ceil(radius**2))
    center = np.int64(np.ceil(radius)) - 1 #zero based indexing... need the offset
    svoxel = np.array([center,center,center])
    grid = np.zeros((max,max,max),dtype=bool)
    rs = RasterizeSphere(svoxel, radius, grid)
    grid_points = np.transpose(np.where(rs.grid))
    # print(rs.grid)
    # print(grid_points)
    return np.array([*grid_points])

def get_sphere_on_grid_for_voxel_plotting(radius):
    """given a sphere radius in voxels (half integer), return a 3D boolean array where True values define which grid points are voxels belonging to the sphere"""
    #TODO better use of assert statement to ensure appropriate sphere size
    assert radius > 0,f'particle radius in voxels must be a positive half integer value, you used {radius}'
    max = np.int64(np.ceil(radius**2))
    center = np.int64(np.ceil(radius)) - 1 #zero based indexing... need the offset
    svoxel = np.array([center,center,center])
    grid = np.zeros((max,max,max),dtype=bool)
    rs = RasterizeSphere(svoxel, radius, grid)
    return rs.grid

def get_nodes_from_grid_voxels(grid_points,l_e,translation):
    """given the grid coordinates of the voxels and the voxel size, get the nodes/vertices of the voxels"""
    basis_vec_len = l_e/2
    v1 = np.array([1,0,0])*basis_vec_len
    v2 = np.array([0,1,0])*basis_vec_len
    v3 = np.array([0,0,1])*basis_vec_len
    voxel_diameter = np.max(grid_points) + 1
    # print(f'maximum and minimum x grid point:{np.max(grid_points[:,0])}, {np.min(grid_points[:,0])}')
    # print(f'maximum and minimum y grid point:{np.max(grid_points[:,1])}, {np.min(grid_points[:,1])}')
    # print(f'maximum and minimum z grid point:{np.max(grid_points[:,2])}, {np.min(grid_points[:,2])}')
    adjusted_translation = translation - (voxel_diameter-1)/2
    nodes = np.zeros((1,3))
    for point in grid_points:
        center = (np.array([point[0], point[1], point[2]])*l_e) + adjusted_translation #+ basis_vec_len
        # center = np.array([point[0]*l_e, point[1]*l_e, point[2]*l_e])
        node0 = center + v1 + v2 + v3
        node1 = center - v1 + v2 + v3
        node2 = center + v1 - v2 + v3
        node3 = center + v1 + v2 - v3
        node4 = center - v1 - v2 + v3
        node5 = center - v1 + v2 - v3
        node6 = center + v1 - v2 - v3
        node7 = center - v1 - v2 - v3
        points = np.vstack((node0[np.newaxis,:],node1[np.newaxis,:],node2[np.newaxis,:],node3[np.newaxis,:],node4[np.newaxis,:],node5[np.newaxis,:],node6[np.newaxis,:],node7[np.newaxis,:]))
        nodes = np.concatenate((nodes,points))
    unique_nodes = np.unique(nodes[1:,:],axis=0)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(*np.transpose(unique_nodes))
    # plt.savefig('./sphere_voxel_to_nodes.png')
    return unique_nodes

def get_row_indices(node_posns,l_e,dim):
    Lx,Ly,Lz = dim
    inv_l_e = 1/l_e
    nodes_per_col = np.round(Lz/l_e + 1).astype(np.int32)
    nodes_per_row = np.round(Lx/l_e + 1).astype(np.int32)
    row_index = (np.round(nodes_per_col * inv_l_e * node_posns[:,0]) + np.round(nodes_per_col * nodes_per_row * inv_l_e *node_posns[:,1]) + np.round(inv_l_e *node_posns[:,2])).astype(np.int32)
    row_index_noround = ((nodes_per_col * inv_l_e * node_posns[:,0]) + (nodes_per_col * nodes_per_row * inv_l_e *node_posns[:,1]) + inv_l_e *node_posns[:,2]).astype(np.int32)
    assert np.all(row_index == row_index_noround), f'difference with and without rounding before conversion to integer, with rounding {row_index}, without{row_index_noround}'
    return np.unique(row_index)

def get_row_indices_normalized(node_posns, dim):
    """Return the row indices corresponding to the normalized node positions of interest given the simulation dimension parameters (number of cubic volume ellements along each direction)"""
    N_nodes_x = dim[0] + 1
    N_nodes_z = dim[2] + 1
    nodes_per_col = np.round(N_nodes_z).astype(np.int32)
    nodes_per_row = np.round(N_nodes_x).astype(np.int32)
    row_index = np.empty((node_posns.shape[0],),dtype=np.int32)
    row_index = ((nodes_per_col * node_posns[:,0]) + (nodes_per_col * nodes_per_row *node_posns[:,1]) + node_posns[:,2]).astype(np.int32)
    # for i in range(node_posns.shape[0]):
        # row_index[i] =  int(((nodes_per_col * node_posns[i,0]) + (nodes_per_col * nodes_per_row *node_posns[i,1]) + node_posns[i,2]))
    return row_index

def get_row_indices_old(node_posns,l_e,dim):
    row_index = np.empty((node_posns.shape[0],),dtype=np.int32)
    for i in range(node_posns.shape[0]):
        row_index[i] = get_row_index_node(i,node_posns[i,:],l_e,dim)
    return row_index

def get_row_index_node(i,node_posn,l_e,dim):
    Lx,Ly,Lz = dim
    inv_l_e = 1/l_e
    nodes_per_col = np.round(Lz/l_e + 1).astype(np.int32)
    nodes_per_row = np.round(Lx/l_e + 1).astype(np.int32)
    # row_index = (inv_l_e * ((nodes_per_col * node_posn[0]) + (nodes_per_col * nodes_per_row * node_posn[1]) + node_posn[2])).astype(np.int32)
    row_index = (np.round(nodes_per_col * inv_l_e * node_posn[0]) + np.round(nodes_per_col * nodes_per_row * inv_l_e *node_posn[1]) + np.round(inv_l_e *node_posn[2])).astype(np.int32)
    row_index_noround = ((nodes_per_col * inv_l_e * node_posn[0]) + (nodes_per_col * nodes_per_row * inv_l_e *node_posn[1]) + (inv_l_e *node_posn[2])).astype(np.int32)
    row_index_2 = ((1/l_e * (Lz/l_e + 1)) * node_posn[0] + (1/l_e * (Lz/l_e + 1) * (Lx/l_e + 1)) * node_posn[1] + (1/l_e) * node_posn[2]).astype(np.int32)
    # try:
    #     assert(row_index == row_index_2)
    # except:
    #     print('methods to calculate row index {} do not agree'.format(i))
    #     print('method one {}'.format(row_index))
    #     print('method two {}'.format(row_index_2))
    #     print('node position {} x {} y {} z'.format(node_posn[0],node_posn[1],node_posn[2]))
    return row_index

def test_get_row_index_node():
    l_e = 1
    dim = [3,3,3]
    Lx,Ly,Lz = dim
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    for i in range(node_posns.shape[0]):
        node_posn = node_posns[i,:]
        row_index = get_row_index_node(-1,node_posn,l_e,dim)
        assert np.allclose(node_posn,node_posns[row_index]), f"calculated row index does not point back to proper node position for {i}"
    l_e = 0.1
    dim = [0.3,0.3,0.3]
    Lx,Ly,Lz = dim
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    for i in range(node_posns.shape[0]):
        node_posn = node_posns[i,:]
        row_index = get_row_index_node(-1,node_posn,l_e,dim)
        assert np.allclose(node_posn,node_posns[row_index]), f"calculated row index does not point back to proper node position for {i}"
    l_e = 0.1
    dim = [20.,10.,2.]
    Lx,Ly,Lz = dim
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    for i in range(node_posns.shape[0]):
        node_posn = node_posns[i,:]
        row_index = get_row_index_node(-1,node_posn,l_e,dim)
        assert np.allclose(node_posn,node_posns[row_index]), f"calculated row index does not point back to proper node position for {i}"

def test_get_row_indices():
    l_e = 0.1
    dim = [10.1,5.3,2.]
    Lx,Ly,Lz = dim
    node_posns = mre.initialize.discretize_space(Lx,Ly,Lz,l_e)
    row_index = get_row_indices(node_posns,l_e,dim)
    assert np.allclose(node_posns,node_posns[row_index]), f"calculated row indices do not map back to proper node positions"

def old_main():
    # Make max large enough to hold the spheres.
    max = 100
    points = parallel_spheres(np.zeros((max, max, max), dtype=bool))
    print(points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(*points)
    # plt.show()
    plt.savefig('./spheres.png')

def plotter(node_posns,springs,radius):
    fig = plt.figure
    ax = plt.axes(projection='3d')
    ax.scatter3D(node_posns[:,0],node_posns[:,1],node_posns[:,2])
    for spring in springs:
        if not np.allclose(spring[2],0):
            x,y,z = (np.array((node_posns[int(spring[0]),0],node_posns[int(spring[1]),0])),
                            np.array((node_posns[int(spring[0]),1],node_posns[int(spring[1]),1])),
                            np.array((node_posns[int(spring[0]),2],node_posns[int(spring[1]),2])))
            ax.plot(x,y,z)
    # for i in range(node_posns.shape[0] - 1):
    #     x,y,z = (np.array((node_posns[i,0],node_posns[i+1,0])),
    #                       np.array((node_posns[i,1],node_posns[i+1,1])),
    #                       np.array((node_posns[i,2],node_posns[i+1,2])))
    #     ax.plot(x,y,z)
    save_string = f'./rasterized_sphere_R_{radius}.png'
    plt.savefig(save_string)

def plot_sphere_voxels(grid_points,radius,output_dir):
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.voxels(grid_points,edgecolor='k')
    ax.set_xlim((0,2*radius+1))
    ax.set_ylim((0,2*radius+1))
    ax.set_zlim((0,2*radius+1))
    save_name = f'voxel_sphere_R_{radius}.png'
    plt.savefig(output_dir+save_name)

def place_sphere(radius,l_e,center,dim):
    """Given the sphere radius in voxels, the position of the center of the spherical particle, and the dimensions of the simulation (number of elements in each direction), return the indices of the nodes that make up the spherical particle."""
    grid_points = get_sphere_on_grid(radius)
    node_posns = get_nodes_from_grid_voxels(grid_points,l_e,center)
    row_indices = get_row_indices(node_posns,l_e,dim)
    return row_indices

def place_sphere_normalized(radius,center,dim):
    """Given the sphere radius in voxels, the position of the center of the spherical particle in scaled units, and the normalized dimensions of the simulation (number of elements in each direction), return the indices of the nodes that make up the spherical particle."""
    grid_points = get_sphere_on_grid(radius)
    node_posns = get_nodes_from_grid_voxels(grid_points,1,center)
    row_indices = get_row_indices_normalized(node_posns,dim)
    return row_indices

def place_spheres_normalized(radius,centers,dim):
    """Given the sphere radius in voxels, the positions of the centers of the spherical particles in scaled units (should be at half integer values), and the normalized dimensions of the simulation (number of elements in each direction), return the indices of the nodes that make up the spherical particles."""
    grid_points = get_sphere_on_grid(radius)
    node_posns = get_nodes_from_grid_voxels(grid_points,1,centers[0])
    row_indices = get_row_indices_normalized(node_posns,dim)
    particles = np.zeros((centers.shape[0],row_indices.shape[0]),dtype=np.int64)
    particles[0] = row_indices
    for i in range(1,centers.shape[0]):
        rij = centers[i] - centers[0]
        next_particle_node_posns = node_posns + rij
        next_particle_indices = get_row_indices_normalized(next_particle_node_posns,dim)
        particles[i] = next_particle_indices
    return particles

def place_spheres(radius,l_e,centers,dim):
    """returns a 2D array of row indices defining the vertices making up a rasterized spherical particle. Assumes all particles have the same size"""
    grid_points = get_sphere_on_grid(radius)
    row_indices = np.array([],dtype=np.int32)
    for center in centers:
        node_posns = get_nodes_from_grid_voxels(grid_points,l_e,center)
        current_row_indices = get_row_indices(node_posns,l_e,dim)
        if not row_indices:
            row_indices = current_row_indices
        else:
            row_indices = np.concatenate((row_indices,current_row_indices))
    return row_indices

def main():
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/'
    radius = 3.5
    # l_e = 0.1
    # grid_points = get_sphere_on_grid(radius)
    for i in range(11):
        radius = i + 0.5
        grid_points_3D = get_sphere_on_grid_for_voxel_plotting(radius)
        plot_sphere_voxels(grid_points_3D,radius,output_dir)
    # particle_center = np.array([3 + l_e/2,3 + l_e/2,3 + l_e/2],dtype=np.float64)
    # node_posns = get_nodes_from_grid_voxels(grid_points,l_e,particle_center)
    # dim = [32,32,32]
    # Lx,Ly,Lz = dim
    # row_indices = get_row_indices(node_posns,l_e,dim)
    # k = [1,0,0]
    # springs = mre.initialize.create_springs(node_posns,k,l_e,dim)
    # plotter(node_posns,springs,radius)
    # fig = plt.figure
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(node_posns[:,0],node_posns[:,1],node_posns[:,2])
    # save_string = f'./rasterized_sphere_R_{radius}.png'
    # plt.savefig(save_string)

def thesis_plot_voxel_spheres():
    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/'
    for i in range(11):
        radius = i + 0.5
        grid_points_3D = get_sphere_on_grid_for_voxel_plotting(radius)
        plot_sphere_voxels(grid_points_3D,radius,output_dir)

if __name__ == '__main__':
    # test_get_row_indices_new()
    # test_get_row_index_node()
    main()