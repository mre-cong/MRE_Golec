#2023-09-29
#David Marchfield
#Calculate displacement vector field and from that, calculate deformation/displacement gradient, strain tensor (linear), Green strain tensor (nonlinear), and left/right deformation tensor (FF^T, (F^T)F)
#from scaled accelerations, calculate forces and then stress tensor field

#TODO because i am dealing with a scaled system, i need to do conversions, either before calculating or after calculating relevant quantities (displacement, displacement gradient, etc.)

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import time
import os
import tables as tb#pytables, for HDF5 interface
import mre.initialize
import mre.analyze
import mre.sphere_rasterization
import get_volume_correction_force_cy_nogil
import simulate
#magnetic permeability of free space
mu0 = 4*np.pi*1e-7

def plot_element(node_posns,element,springs):
    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(node_posns[element,0],node_posns[element,1],node_posns[element,2],color ='b',marker='o')
    mre.analyze.plot_subset_springs(ax,node_posns,element,springs,spring_color='b')
    plt.show()

def main():
    drag = 10
    discretization_order = 1

    output_dir = f'/mnt/c/Users/bagaw/Desktop/MRE/two_particle/2023-10-06_results_order_{discretization_order}_drag_{drag}/'

    initial_node_posns, node_mass, springs_var, elements, boundaries, particles, params, series, series_descriptor = mre.initialize.read_init_file(output_dir+'init.h5')
    final_posns, applied_field, boundary_conditions, sim_time = mre.initialize.read_output_file(output_dir+'output_0.h5')
    criteria = mre.initialize.read_criteria_file(output_dir+'field_0_Bext_1.0/criteria.h5')
    checkpoint_solution, _, _, integration_number = mre.initialize.read_checkpoint_file(output_dir+'field_0_Bext_1.0/checkpoint.h5')

    # print(f'{params.dtype}')

    for i in range(len(params[0])):
        if params.dtype.descr[i][0] == 'num_elements':
            num_elements = params[0][i]
            num_nodes = num_elements + 1
        if params.dtype.descr[i][0] == 'poisson_ratio':
            nu = params[0][i]
        if params.dtype.descr[i][0] == 'young_modulus':
            E = params[0][i]
        if params.dtype.descr[i][0] == 'kappa':
            kappa = params[0][i]
        if params.dtype.descr[i][0] == 'scaling_factor':
            beta = params[0][i]
        if params.dtype.descr[i][0] == 'element_length':
            l_e = params[0][i]
        if params.dtype.descr[i][0] == 'particle_mass':
            particle_mass = params[0][i]
        if params.dtype.descr[i][0] == 'particle_radius':
            particle_radius = params[0][i]
        if params.dtype.descr[i][0] == 'particle_Ms':
            Ms = params[0][i]
        if params.dtype.descr[i][0] == 'particle_chi':
            chi = params[0][i]
        if params.dtype.descr[i][0] == 'young_modulus':
            E = params[0][i]

    # sim_criteria_time = criteria['time']
    # timestep = sim_criteria_time[1:] - sim_criteria_time[:-1]
    # iter_number = criteria['iter_number']
    # fig, axs = plt.subplots(1,3)
    # default_width, default_height = fig.get_size_inches()
    # fig.set_size_inches(2*default_width,2*default_height)
    # fig.set_dpi(100)
    # axs[0].plot(sim_criteria_time[:timestep.shape[0]],timestep,'.')
    # axs[0].set_title('Time Step Taken')
    # axs[0].set_xlabel('scaled time')
    # axs[0].set_ylabel('time step')
    # axs[1].plot(iter_number[:timestep.shape[0]],timestep,'.')
    # axs[1].set_title('Time Step Taken')
    # axs[1].set_xlabel('integration number')
    # axs[1].set_ylabel('time step')
    # axs[2].plot(iter_number,sim_criteria_time,'.')
    # axs[2].set_title('Total Time')
    # axs[2].set_xlabel('integration number')
    # axs[2].set_ylabel('total scaled time')
    # # plt.show()
    # savename = output_dir + 'timestep_per_iteration_and_time.png'
    # plt.savefig(savename)
    # plt.close()

    dimensions = (np.max(initial_node_posns[:,0]),np.max(initial_node_posns[:,1]),np.max(initial_node_posns[:,2]))
    beta_i = beta/node_mass
    Hext = applied_field
    #TODO adjust the read output file function to return a proper boundary conditions variable
    boundary_conditions = (str(boundary_conditions[0][0]),(str(boundary_conditions[0][1]),str(boundary_conditions[0][2])),boundary_conditions[0][3])
    a_var = simulate.get_accel_scaled(checkpoint_solution,elements,springs_var,particles,kappa,l_e,beta,beta_i,boundary_conditions,boundaries,dimensions,Hext,particle_radius,particle_mass,chi,Ms,drag)
    a_norms = np.linalg.norm(a_var,axis=1)

    # plot_residual_acceleration_hist(a_norms,output_dir)

    # visualize residual acceleration with pcolormesh or i guess the scatter plot with color for depth could be modified
    # use argsort to find the strongest acceleration norm nodes, and find out how many of them there are, and where they are (so that you can visualize the parts of the simulation volume that have high accelerations)

    accel_sorted_node_indices = np.argsort(a_norms)[::-1]#returns sorting indices from highest to lowest acceleration norm
    top_fifty = accel_sorted_node_indices[:50]

    strong_resid_accel_nodes = initial_node_posns[top_fifty]
    strong_resid_accel_vals = a_norms[top_fifty]

    # for particle in particles:
    #     a_var[particle,:] = 0

    # plot_residual_acceleration_hist(a_norms[top_fifty],output_dir)
    index = 13
    cut_type = 'xy'
    subplot_cut_pcolormesh_vectorfield(cut_type,initial_node_posns,a_var,index,output_dir,tag="residual_accelerations")

    displacement = get_displacement_field(initial_node_posns,final_posns)

    #getting the Lame parameters from Young's modulus and poisson ratio. see en.wikipedia.org/wiki/Lame_parameters
    lame_lambda = E*nu/((1+nu)*(1-2*nu))
    shear_modulus = E/(2*(1+nu))
    xdisplacement_3D, ydisplacement_3D, zdisplacement_3D = get_component_3D_arrays(displacement,num_nodes)
    xdisplacement_gradient = np.gradient(xdisplacement_3D)
    ydisplacement_gradient = np.gradient(ydisplacement_3D)
    zdisplacement_gradient = np.gradient(zdisplacement_3D)
    xshape = int(num_nodes[0])
    yshape = int(num_nodes[1])
    zshape = int(num_nodes[2])
    # aka deformation gradient. F = I + \frac{\partial u}{\partial X}, where u is the displacement: u = x - X, where x is the current position, and X is the reference configuration of the point of the body
    # deformation_tensor = np.zeros((3,3))
    # deformation_tensor[0,0] = xdisplacement_gradient[0][0,0,0] + 1
    # deformation_tensor[1,1] = ydisplacement_gradient[1][0,0,0] + 1
    # deformation_tensor[2,2] = zdisplacement_gradient[2][0,0,0] + 1
    # deformation_tensor[0,1] = xdisplacement_gradient[1][0,0,0]
    # deformation_tensor[0,2] = xdisplacement_gradient[2][0,0,0]
    # deformation_tensor[1,2] = ydisplacement_gradient[2][0,0,0]
    # deformation_tensor[1,0] = ydisplacement_gradient[0][0,0,0]
    # deformation_tensor[2,0] = zdisplacement_gradient[0][0,0,0]
    # deformation_tensor[2,1] = zdisplacement_gradient[1][0,0,0]
    #is this suppose to be symmetric?. i don't think so. the strain tensors will be, because they will be based on F F^T or sums and multiples of grad u and (grad u)^T
    gradu = get_gradu(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,num_nodes)
    strain_tensor = get_strain_tensor(gradu)
    green_strain_tensor = get_green_strain_tensor(gradu)

    stress_tensor = get_isotropic_medium_stress(shear_modulus,lame_lambda,strain_tensor)

    index = 13
    cut_type = 'xy'
    displacements = (xdisplacement_3D,ydisplacement_3D,zdisplacement_3D)
    # subplot_cut_pcolormesh_displacement(cut_type,initial_node_posns,displacements,index,output_dir,tag="")

    subplot_cut_pcolormesh_strain(cut_type,initial_node_posns,green_strain_tensor,index,output_dir,tag="nonlinear")

    subplot_cut_pcolormesh_strain(cut_type,initial_node_posns,strain_tensor,index,output_dir,tag="")

    subplot_cut_pcolormesh_tensorfield(cut_type,initial_node_posns,stress_tensor,index,output_dir,tag="stress")

    # plot_cut_pcolormesh_displacement(cut_type,initial_node_posns,xdisplacement_3D,index,output_dir,tag="")

    #i'm fairly confident that i have correctly calculated the gradient of the displacement field, the deformation gradient, and the linear and nonlinear strain tensors. i know that in the case of small displacements the two strain tensors should coincide. i also know that if there were rigid body rotations the strain linear tensor would no longer be useful, and I would need to use the deformation tensor to calculate F F^T (or F^T F) and get V V^T or (U^T U) and then find the square root of the matrix, or else use the iterative algorithm for polar decompositions to go from the deformation gradient to the rotation matrix, then calculate the inverse of the rotation matrix and pre or post multiply F to get V or U. V and U are good strain tensors even for large deformations and rigid body rotations, where the lienar strain tensor can have large differences for the same deformations with and without rigid body rotations. However, the Green strain tensor is supposed to not be corrupted by rigid body rotations.

    #don't forget that shear terms (off diagonal) of strain are half of the engineering shear values because they are components in a strain tensor

    #i suppose the next step would be some visualization of the displacement field as a quiver plot and as a color plot (pmeshcolor, pcolormesh?). then I think i need to try and get a stress tensor

    #I can run strain simulations to try and get effective moduli. to get moduli as a function of position I need to be more thoughtful. the stiffness is a fourth rank tensor, with symmetry arguments, and for isotropic homogeneous media, there are only 2 independent elements. I still need to better understand the symmetry arguments, and see if i can go from the strain functions to stress using the stiffness tensor. or if i can use the forces on the nodes, can i get the stress tensor directly? I need to review my notes from days in the prior few weeks. I can draw imaginary planes through the node positions and then calculate force per (undeformed) area. is that appropriate? the normal example is an infinitessimal cube volume, where the direction and magnitude of forces on opposite faces have to balance to avoid a rotation (angular momentum balance, same for linear momentum (or torque/force))
    print('hm')

def get_isotropic_medium_stress(shear_modulus,lame_lambda,strain):
    """stress for homogeneous isotropic material defined by Hooke's law in 3D"""
    stress = np.zeros((np.shape(strain)))
    print(f'The shape of the result of the trace function on the strain tensor variable is {np.shape(np.trace(strain,axis1=3,axis2=4))}')
    stress[:,:,:,0,0] = 2*shear_modulus*strain[:,:,:,0,0] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,1,1] = 2*shear_modulus*strain[:,:,:,1,1] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,2,2] = 2*shear_modulus*strain[:,:,:,2,2] + lame_lambda*np.trace(strain,axis1=3,axis2=4)
    stress[:,:,:,0,1] = 2*shear_modulus*strain[:,:,:,0,1]
    stress[:,:,:,0,2] = 2*shear_modulus*strain[:,:,:,0,2]
    stress[:,:,:,1,2] = 2*shear_modulus*strain[:,:,:,1,2]
    stress[:,:,:,1,0] = 2*shear_modulus*strain[:,:,:,1,0]
    stress[:,:,:,2,0] = 2*shear_modulus*strain[:,:,:,2,0]
    stress[:,:,:,2,1] = 2*shear_modulus*strain[:,:,:,2,1]
    return stress
    
def get_strain_tensor(gradu):
    """Calculate the symmetric, linear strain tensor at each point on the grid representing the initial node positions using the gradient of the displacement field."""
    strain_tensor = np.zeros(np.shape(gradu))
    strain_tensor[:,:,:,0,0] = gradu[:,:,:,0,0]
    strain_tensor[:,:,:,1,1] = gradu[:,:,:,1,1]
    strain_tensor[:,:,:,2,2] = gradu[:,:,:,2,2]
    strain_tensor[:,:,:,0,1] = 0.5*(gradu[:,:,:,0,1] + gradu[:,:,:,1,0])
    strain_tensor[:,:,:,0,2] = 0.5*(gradu[:,:,:,0,2] + gradu[:,:,:,2,0])
    strain_tensor[:,:,:,1,2] = 0.5*(gradu[:,:,:,1,2] + gradu[:,:,:,2,1])
    strain_tensor[:,:,:,1,0] = strain_tensor[:,:,:,0,1]
    strain_tensor[:,:,:,2,0] = strain_tensor[:,:,:,0,2]
    strain_tensor[:,:,:,2,1] = strain_tensor[:,:,:,1,2]
    return strain_tensor

def get_green_strain_tensor(gradu):
    """Calculate the symmetric, nonlinear strain tensor at each point on the grid representing the initial node positions using the gradient of the displacement field."""
    strain_tensor = np.zeros(np.shape(gradu))
    strain_tensor[:,:,:,0,0] = gradu[:,:,:,0,0] + 0.5*(np.power(gradu[:,:,:,0,0],2) + np.power(gradu[:,:,:,1,0],2) + np.power(gradu[:,:,:,2,0],2))
    strain_tensor[:,:,:,1,1] = gradu[:,:,:,1,1] + 0.5*(np.power(gradu[:,:,:,0,1],2) + np.power(gradu[:,:,:,1,1],2) + np.power(gradu[:,:,:,2,1],2))
    strain_tensor[:,:,:,2,2] = gradu[:,:,:,2,2] + 0.5*(np.power(gradu[:,:,:,0,2],2) + np.power(gradu[:,:,:,1,2],2) + np.power(gradu[:,:,:,2,2],2))
    strain_tensor[:,:,:,0,1] = 0.5*(gradu[:,:,:,0,1] + gradu[:,:,:,1,0]) + 0.5*(gradu[:,:,:,0,0]*gradu[:,:,:,0,1] + gradu[:,:,:,1,0]*gradu[:,:,:,1,1] + gradu[:,:,:,2,0]*gradu[:,:,:,2,1])
    strain_tensor[:,:,:,0,2] = 0.5*(gradu[:,:,:,0,2] + gradu[:,:,:,2,0]) + 0.5*(gradu[:,:,:,0,0]*gradu[:,:,:,0,2] + gradu[:,:,:,1,0]*gradu[:,:,:,1,2] + gradu[:,:,:,2,0]*gradu[:,:,:,2,2])
    strain_tensor[:,:,:,1,2] = 0.5*(gradu[:,:,:,1,2] + gradu[:,:,:,2,1]) + 0.5*(gradu[:,:,:,0,1]*gradu[:,:,:,0,2] + gradu[:,:,:,1,1]*gradu[:,:,:,1,2] + gradu[:,:,:,2,1]*gradu[:,:,:,2,2])
    strain_tensor[:,:,:,1,0] = strain_tensor[:,:,:,0,1]
    strain_tensor[:,:,:,2,0] = strain_tensor[:,:,:,0,2]
    strain_tensor[:,:,:,2,1] = strain_tensor[:,:,:,1,2]
    return strain_tensor

def get_gradu(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,dimensions):
    """Get the tensor at each point on the grid representing the initial node positions which is the gradient of the displacement field"""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    gradu = np.zeros((xshape,yshape,zshape,3,3))
    gradu[:,:,:,0,0] = xdisplacement_gradient[0][:,:,:]
    gradu[:,:,:,1,1] = ydisplacement_gradient[1][:,:,:]
    gradu[:,:,:,2,2] = zdisplacement_gradient[2][:,:,:]
    gradu[:,:,:,0,1] = xdisplacement_gradient[1][:,:,:]
    gradu[:,:,:,0,2] = xdisplacement_gradient[2][:,:,:]
    gradu[:,:,:,1,2] = ydisplacement_gradient[2][:,:,:]
    gradu[:,:,:,1,0] = ydisplacement_gradient[0][:,:,:]
    gradu[:,:,:,2,0] = zdisplacement_gradient[0][:,:,:]
    gradu[:,:,:,2,1] = zdisplacement_gradient[1][:,:,:]
    return gradu

def get_deformation_gradient(xdisplacement_gradient,ydisplacement_gradient,zdisplacement_gradient,dimensions):
    """Calculate the deformation gradient at each point on the grid representing the initial node positions. Equal to gradu plus the identity matrix"""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    deformation_gradient = np.zeros((xshape,yshape,zshape,3,3))
    deformation_gradient[:,:,:,0,0] = xdisplacement_gradient[0][0,0,0] + 1
    deformation_gradient[:,:,:,1,1] = ydisplacement_gradient[1][0,0,0] + 1
    deformation_gradient[:,:,:,2,2] = zdisplacement_gradient[2][0,0,0] + 1
    deformation_gradient[:,:,:,0,1] = xdisplacement_gradient[1][0,0,0]
    deformation_gradient[:,:,:,0,2] = xdisplacement_gradient[2][0,0,0]
    deformation_gradient[:,:,:,1,2] = ydisplacement_gradient[2][0,0,0]
    deformation_gradient[:,:,:,1,0] = ydisplacement_gradient[0][0,0,0]
    deformation_gradient[:,:,:,2,0] = zdisplacement_gradient[0][0,0,0]
    deformation_gradient[:,:,:,2,1] = zdisplacement_gradient[1][0,0,0]
    return deformation_gradient

def get_displacement_field(initial_posns,final_posns):
    """Calculate the displacement of each node from its initial position."""
    displacement = final_posns - initial_posns
    return displacement

def transform_to_3D_array(array,dimensions):
    """Given a 1D vector of node positions, or similarly structured per node values, and convert to a 3D array mapped to the grid of initial node positions for plotting and analysis. Dimensions is tuple or array of number of nodes along each direction (x,y,z)."""
    xshape = int(dimensions[0])
    yshape = int(dimensions[1])
    zshape = int(dimensions[2])
    array_3D = np.zeros((xshape,yshape,zshape))
    nodes_per_column = zshape
    nodes_per_plane = zshape*xshape
    for j in range(yshape):
        for i in range(xshape):
            array_3D[i,j,:] = array[(j*nodes_per_plane)+(i*nodes_per_column):(j*nodes_per_plane)+(i+1)*nodes_per_column]
    return array_3D

def get_component_3D_arrays(array,dimensions):
    xarray_3D = transform_to_3D_array(array[:,0],dimensions)
    yarray_3D = transform_to_3D_array(array[:,1],dimensions)
    zarray_3D = transform_to_3D_array(array[:,2],dimensions)
    return xarray_3D, yarray_3D, zarray_3D
#i want to try to use some of the structure of this function for plotting the displacement field, but i'm not sure the best way to plot the displacement field. i can use pcolormesh. i can use quiver. i should probably try those two before using scatter and color of markers. i want to use the initial node positions, since they are on a grid, to show the deformation of the initial node positions to the final node positions. i want to plot displacement components as separate plots on a subplot figure, and the displacement norm as well... different cuts would be possible for each component/norm.
def plot_displacement_field_visualization(eq_node_posns,final_node_posns,cut_type,index,springs,particles,output_dir,spring_type=None,tag=""):
    """Plot a set of chosen nodes, passed as a list of integers representing the row index of the node in final_node_posns"""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    chosen_nodes = np.isclose(np.ones((eq_node_posns.shape[0],))*index,eq_node_posns[:,cut_type_index]).nonzero()[0]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    fig = plt.figure()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    ax = fig.add_subplot()
    depth_color = final_node_posns[chosen_nodes,cut_type_index]
    if cut_type_index == 0:
        xvar = final_node_posns[chosen_nodes,1]
        yvar = final_node_posns[chosen_nodes,2]
        xlabel = 'Y (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Ly*1.1)
        ylim = (-0.1,Lz*1.1)
    elif cut_type_index == 1:
        xvar = final_node_posns[chosen_nodes,0]
        yvar = final_node_posns[chosen_nodes,2]
        xlabel = 'X (l_e)'
        ylabel = 'Z (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Lz*1.1)
    else:
        xvar = final_node_posns[chosen_nodes,0]
        yvar = final_node_posns[chosen_nodes,1]
        xlabel = 'X (l_e)'
        ylabel = 'Y (l_e)'
        xlim = (-0.1,Lx*1.1)
        ylim = (-0.1,Ly*1.1)
    # mre.analyze.plot_subset_springs_2D(ax,cut_type,final_node_posns,chosen_nodes,springs,spring_color='b',spring_type=spring_type)
    sc = ax.scatter(xvar,yvar,c=depth_color,marker='o',zorder=2.5)
    plt.colorbar(sc)
    cut_nodes_set = set(chosen_nodes)
    particle_nodes_set = set(particles.ravel())
    particle_cut_nodes_set = cut_nodes_set.intersection(particle_nodes_set)
    particle_cut_nodes = [x for x in particle_cut_nodes_set]
    if cut_type_index == 0:
        xvar = final_node_posns[particle_cut_nodes,1]
        yvar = final_node_posns[particle_cut_nodes,2]
    elif cut_type_index == 1:
        xvar = final_node_posns[particle_cut_nodes,0]
        yvar = final_node_posns[particle_cut_nodes,2]
    else:
        xvar = final_node_posns[particle_cut_nodes,0]
        yvar = final_node_posns[particle_cut_nodes,1]
    # mre.analyze.plot_subset_springs_2D(ax,cut_type,final_node_posns,particle_cut_nodes_set,springs,spring_color='r',spring_type=spring_type)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    # plt.show()
    savename = output_dir + f'displacement_field_visualization_{cut_type}_{index}.png'
    plt.savefig(savename)
    plt.close()

def plot_cut_pcolormesh_displacement(cut_type,eq_node_posns,displacement_component,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the displacement components of the nodes that sat at the center of the initialized system.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)

    if cut_type == 'xy':
        Z = displacement_component[:,:,index]
        X = xposns[:,:,index]
        Y = yposns[:,:,index]
    elif 'xz':
        Z = displacement_component[:,index,:]
        X = xposns[:,index,:]
        Y = zposns[:,index,:]
    else:
        Z = displacement_component[index,:,:]
        X = yposns[index,:,:]
        Y = zposns[index,:,:]
    #need to convert from 1D vectors to 2D matrices
    #coordinates of the corners of quadrilaterlas of a pcolormesh:
    #(X[i+1,j],Y[i+1,j])       (X[i+1,j+1],Y[i+1,j+1])
    #              *----------*
    #              |          |
    #              *----------*
    #(X[i,j],Y[i,j])           (X[i,j+1],Y[i,j+1])

    img = ax.pcolormesh(X,Y,Z,shading='gouraud')
    #don't forget to add a colorbar and limits
    fig.colorbar(img,ax=ax)
    ax.axis('equal')
    ax.set_title('displacement ' + f'{cut_type}' + f'layer {index}')
    plt.show()
    savename = output_dir + f'cut_pcolormesh_displacement_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_cut_pcolormesh_displacement(cut_type,eq_node_posns,displacements,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the displacement components of the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)

    component_dict = {0:'x',1:'y',2:'z',3:'norm'}
    for i in range(4):
        row = np.floor_divide(i,2)
        col = i%2
        ax = axs[row,col]
        if i != 3:
            displacement_component = displacements[i]
        else:
            displacement_component = np.sqrt(np.power(displacements[0],2) +np.power(displacements[1],2) + np.power(displacements[2],2)) 
        if cut_type == 'xy':
            Z = displacement_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = displacement_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = displacement_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(f'displacement {component_dict[i]}' + f'{cut_type}' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_displacement_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_cut_pcolormesh_vectorfield(cut_type,eq_node_posns,vectorfield,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the vectorfield components of some property of the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,2)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)
    #if the shape has 2 members, reshape the vectorfield variable to a 3D grid of values
    if len(np.shape(vectorfield)) ==  2:
        xvectorfield_3D, yvectorfield_3D, zvectorfield_3D = get_component_3D_arrays(vectorfield,dimensions)
        vectorfield = (xvectorfield_3D,yvectorfield_3D,zvectorfield_3D)
    #if the shape has 3 members, assume the vectorfield variable is already in the appropriate format of a 3D grid of values for plotting with pcolormesh
    elif len(np.shape(vectorfield)) == 3:
        pass
    else:
        raise ValueError(f'shape of vector field {np.shape(vectorfield)} must be 2D or 3D')

    component_dict = {0:'x',1:'y',2:'z',3:'norm'}
    for i in range(4):
        row = np.floor_divide(i,2)
        col = i%2
        ax = axs[row,col]
        if i != 3:
            vectorfield_component = vectorfield[i]
        else:
            vectorfield_component = np.sqrt(np.power(vectorfield[0],2) +np.power(vectorfield[1],2) + np.power(vectorfield[2],2)) 
        if cut_type == 'xy':
            Z = vectorfield_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = vectorfield_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = vectorfield_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z)
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(tag+ f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_' + tag + '_vectorfield_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_cut_pcolormesh_strain(cut_type,eq_node_posns,strain,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the strain tensor components of the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)

    component_dict = {0:'xx',1:'yy',2:'zz',3:'xy',4:'xz',5:'yz'}
    for i in range(6):
        row = np.floor_divide(i,3)
        col = i%3
        ax = axs[row,col]
        if i == 0:
            strain_component = strain[:,:,:,0,0]
        elif i == 1:
            strain_component = strain[:,:,:,1,1]
        elif i == 2:
            strain_component = strain[:,:,:,2,2]
        elif i == 3:
            strain_component = strain[:,:,:,0,1]
        elif i == 4:
            strain_component = strain[:,:,:,0,2]
        elif i == 5:
            strain_component = strain[:,:,:,1,2]
        if cut_type == 'xy':
            Z = strain_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = strain_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = strain_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z)
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(tag+f'strain {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_'+tag+'strain_visualization.png'
    plt.savefig(savename)
    plt.close()

def subplot_cut_pcolormesh_tensorfield(cut_type,eq_node_posns,tensorfield,index,output_dir,tag=""):
    """Plot a cut through the simulated volume, showing the symmetric tensor components of some tensor field defined at the nodes.
    
    cut_type must be one of three: 'xy', 'xz', 'yz' describing the plane spanned by the cut.
    
    tag is an optional argument that can be used to provide additional detail in the title and save name of the figure."""
    if len(np.shape(tensorfield)) != 5:
        raise ValueError(f'shape of tensor field {np.shape(tensorfield)} must be 5D')
    cut_type_dict = {'xy':2, 'xz':1, 'yz':0}
    cut_type_index = cut_type_dict[cut_type]
    Lx = eq_node_posns[:,0].max()
    Ly = eq_node_posns[:,1].max()
    Lz = eq_node_posns[:,2].max()
    dimensions = (int(Lx+1),int(Ly+1),int(Lz+1))
    fig, axs = plt.subplots(2,3)
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(3*default_width,3*default_height)
    fig.set_dpi(200)
    xposns, yposns, zposns = get_component_3D_arrays(eq_node_posns,dimensions)

    component_dict = {0:'xx',1:'yy',2:'zz',3:'xy',4:'xz',5:'yz'}
    for i in range(6):
        row = np.floor_divide(i,3)
        col = i%3
        ax = axs[row,col]
        if i == 0:
            tensor_component = tensorfield[:,:,:,0,0]
        elif i == 1:
            tensor_component = tensorfield[:,:,:,1,1]
        elif i == 2:
            tensor_component = tensorfield[:,:,:,2,2]
        elif i == 3:
            tensor_component = tensorfield[:,:,:,0,1]
        elif i == 4:
            tensor_component = tensorfield[:,:,:,0,2]
        elif i == 5:
            tensor_component = tensorfield[:,:,:,1,2]
        if cut_type == 'xy':
            Z = tensor_component[:,:,index]
            X = xposns[:,:,index]
            Y = yposns[:,:,index]
            xlabel = 'X'
            ylabel = 'Y'
        elif 'xz':
            Z = tensor_component[:,index,:]
            X = xposns[:,index,:]
            Y = zposns[:,index,:]
            xlabel = 'X'
            ylabel = 'Z'
        else:
            Z = tensor_component[index,:,:]
            X = yposns[index,:,:]
            Y = zposns[index,:,:]
            xlabel = 'Y'
            ylabel = 'Z'
        img = ax.pcolormesh(X,Y,Z)
        # img = ax.pcolormesh(X,Y,Z,shading='gouraud')
        #don't forget to add a colorbar and limits
        fig.colorbar(img,ax=ax)
        ax.axis('equal')
        ax.set_title(tag+f' {component_dict[i]} ' + f'{cut_type} ' + f'layer {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()
    savename = output_dir + f'subplots_cut_pcolormesh_'+tag+'_tensorfield_visualization.png'
    plt.savefig(savename)
    plt.close()

def plot_residual_acceleration_hist(a_norms,output_dir):
    """Plot a histogram of the acceleration of the nodes. Intended for analyzing the behavior at the end of simulations that are ended before convergence criteria are met."""
    max_accel = np.max(a_norms)
    mean_accel = np.mean(a_norms)
    rms_accel = np.sqrt(np.sum(np.power(a_norms,2))/np.shape(a_norms)[0])
    counts, bins = np.histogram(a_norms, bins=20)
    fig,ax = plt.subplots()
    default_width,default_height = fig.get_size_inches()
    fig.set_size_inches(2*default_width,2*default_height)
    ax.hist(bins[:-1], bins, weights=counts)
    sigma = np.std(a_norms)
    mu = mean_accel
    ax.set_title(f'Residual Acceleration Histogram\nMaximum {max_accel}\nMean {mean_accel}\n$\sigma={sigma}$\nRMS {rms_accel}')
    ax.set_xlabel('acceleration norm')
    ax.set_ylabel('counts')
    savename = output_dir +'node_residual_acceleration_hist.png'
    plt.savefig(savename)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()