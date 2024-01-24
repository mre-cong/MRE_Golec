import cupy as cp
import numpy as np
from cupyx.profiler import benchmark
import cupyx
import time
import mre.initialize
import get_volume_correction_force_cy_nogil
import springs

def main():
    start = time.time()
    E = 9e3
    nu = 0.499
    # max_integrations = 20
    # max_integration_steps = 300
    # #based on the particle diameter, we want the discretization, l_e, to match with the size, such that the radius in terms of volume elements is N + 1/2 elements, where each element is l_e in side length. N is then a sort of "order of discreitzation", where larger N values result in finer discretizations. if N = 0, l_e should equal the particle diameter
    # particle_diameter = 3e-6
    # #discretization order
    # discretization_order = 2
    # l_e = (particle_diameter/2) / (discretization_order + 1/2)
    # #particle separation
    # separation_meters = 9e-6
    # separation_volume_elements = int(separation_meters / l_e)
    # separation = separation_volume_elements
    # radius = (discretization_order + 1/2)*l_e

    l_e = 1
    # Lx = 47
    # Ly = 35
    # Lz = Ly
    Lx = 256#separation_meters + particle_diameter + 1.8*separation_volume_elements*l_e
    Ly = 128#particle_diameter * 7
    Lz = Ly
    t_f = 30
    drag = 1
    N_nodes_x = np.round(Lx/l_e + 1)
    N_nodes_y = np.round(Ly/l_e + 1)
    N_nodes_z = np.round(Lz/l_e + 1)
    N_el_x = N_nodes_x - 1
    N_el_y = N_nodes_y - 1
    N_el_z = N_nodes_z - 1
    normalized_dimensions = np.array([N_el_x,N_el_y,N_el_z],dtype=np.int32)
    normalized_posns = mre.initialize.discretize_space_normalized(N_nodes_x,N_nodes_y,N_nodes_z)
    Lx = N_el_x*l_e
    Ly = N_el_y*l_e
    Lz = N_el_z*l_e
    dimensions = np.array([Lx,Ly,Lz])
    elements = springs.get_elements_v2_normalized(N_nodes_x, N_nodes_y, N_nodes_z)
    num_elements = elements.shape[0]
    boundaries = mre.initialize.get_boundaries(normalized_posns)
    k = mre.initialize.get_spring_constants(E, l_e)
    kappa = mre.initialize.get_kappa(E, nu)
    dimensions_normalized = np.array([N_nodes_x-1,N_nodes_y-1,N_nodes_z-1])
    node_types = springs.get_node_type_normalized(normalized_posns.shape[0],boundaries,dimensions_normalized)
    k = np.array(k,dtype=np.float64)
    max_springs = N_nodes_x.astype(np.int32)*N_nodes_y.astype(np.int32)*N_nodes_z.astype(np.int32)*13
    springs_var = np.empty((max_springs,4),dtype=np.float64)
    num_springs = springs.get_springs(node_types, springs_var, max_springs, k, dimensions_normalized, 1)
    springs_var = springs_var[:num_springs,:]

    print(f'Comparing run times for CPU bound versus GPU implementations of volume correction force calculation for system with {num_elements} elements')
    
    #raw kernels are important for dealing with more complicated user defined kernels.
    #proper calculation of the index each thread should use requires some thought based on the grid size(number of blocks) and block size (number of threads per block)
    #atomic operations prevent threads from accessing for writing to the same memory address, but slow things down by forcing those threads to operate sequentially instead of concurrently
    #atomci operations can force single access across a system (multipleGPUs) a single device, or across a thread block/thread group depending on what is invoked
    #global memory access is slow compared to shared memory, but shared memory (shared across a thread block or a group of thread blocks with certain semantics)
    #still requires careful thought due to size limits and the need to update a larger data structure later anyway
    #conditional checking needs to be used to prevent accessing data outside of bounds when the number of threads is larger than the number of operations/elements to operate on
    #conditional checks are slow in general, and across a warp (32 threads which can operate together quickly on read and execution) each thread should traverse the same conditional ath or issues arse
    #it would be great to coalesce access to global memory if possible to reduce the overhead of global memory access
    #cuda kernels really don't like arguments that are pointers to pointers (float**, etc), which is a bit of an issue, since it means i need to reshape every 2d or 3d thing to a vector. but at least it is manageable... then i need to take into account "striding", but just in terms of indices, not memory size/bytes per element
    
    element_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void element_force(const int* elements, const float* node_posns, const float kappa, const float l_e, float* forces, const int size_elements) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size_elements)
        {
            int index0 = elements[8*tid+0];
            int index1 = elements[8*tid+1];
            int index2 = elements[8*tid+2];
            int index3 = elements[8*tid+3];
            int index4 = elements[8*tid+4];
            int index5 = elements[8*tid+5];
            int index6 = elements[8*tid+6];
            int index7 = elements[8*tid+7];
                                  
            //printf("tid = %i, index0 = %i, index1 = %i, index2 = %i, index3 = %i, index4 = %i, index5 = %i, index6 = %i, index7 = %i\n",tid,index0,index1,index2,index3,index4,index5,index6,index7);
            //printf("tid = %i, node_posns[3*index2] = %f, node_posns[3*index0]= %f\n",tid,node_posns[3*index2],node_posns[3*index0]);                     
            //for(int i = 0; i < 24; i++){
            //    printf("node_posns[%i] = %f\n",i,node_posns[i]);
            //}
            //for the element, get the average edge vectors, then using the average edge vectors, get the volume correction force
            float avg_vector_i[3];
            float avg_vector_j[3];
            float avg_vector_k[3];
                                  
            avg_vector_i[0] = (node_posns[3*index2] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index1] + node_posns[3*index6] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index5])/4;
            avg_vector_i[1] = (node_posns[1+3*index2] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index1] + node_posns[1+3*index6] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index5])/4;
            avg_vector_i[2] = (node_posns[2+3*index2] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index1] + node_posns[2+3*index6] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index5])/4;

            //printf("tid = %i, avg_vector_i[0] = %f, avg_vector_i[1] = %f, avg_vector_i[2] = %f\n",tid,avg_vector_i[0],avg_vector_i[1],avg_vector_i[2]);
                                  
            avg_vector_j[0] = (node_posns[3*index4] - node_posns[3*index0] + node_posns[3*index6] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index1] + node_posns[3*index7] - node_posns[3*index3])/4;
            avg_vector_j[1] = (node_posns[1+3*index4] - node_posns[1+3*index0] + node_posns[1+3*index6] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index1] + node_posns[1+3*index7] - node_posns[1+3*index3])/4;
            avg_vector_j[2] = (node_posns[2+3*index4] - node_posns[2+3*index0] + node_posns[2+3*index6] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index1] + node_posns[2+3*index7] - node_posns[2+3*index3])/4;
                                  
            //printf("tid = %i, avg_vector_j[0] = %f, avg_vector_j[1] = %f, avg_vector_j[2] = %f\n",tid,avg_vector_j[0],avg_vector_j[1],avg_vector_j[2]);

            avg_vector_k[0] = (node_posns[3*index1] - node_posns[3*index0] + node_posns[3*index3] - node_posns[3*index2] + node_posns[3*index5] - node_posns[3*index4] + node_posns[3*index7] - node_posns[3*index6])/4;
            avg_vector_k[1] = (node_posns[1+3*index1] - node_posns[1+3*index0] + node_posns[1+3*index3] - node_posns[1+3*index2] + node_posns[1+3*index5] - node_posns[1+3*index4] + node_posns[1+3*index7] - node_posns[1+3*index6])/4;
            avg_vector_k[2] = (node_posns[2+3*index1] - node_posns[2+3*index0] + node_posns[2+3*index3] - node_posns[2+3*index2] + node_posns[2+3*index5] - node_posns[2+3*index4] + node_posns[2+3*index7] - node_posns[2+3*index6])/4;

            //printf("tid = %i, avg_vector_k[0] = %f, avg_vector_k[1] = %f, avg_vector_k[2] = %f\n",tid,avg_vector_k[0],avg_vector_k[1],avg_vector_k[2]);                      
            
            //need to get cross products of average vectors, stored as variables for gradient vectors, prefactor, then atomicAdd for assignment to forces
            float acrossb[3];
            float bcrossc[3];
            float ccrossa[3];
            float adotbcrossc;
                                  
            acrossb[0] = avg_vector_i[1]*avg_vector_j[2] - avg_vector_i[2]*avg_vector_j[1];
            acrossb[1] = avg_vector_i[2]*avg_vector_j[0] - avg_vector_i[0]*avg_vector_j[2];
            acrossb[2] = avg_vector_i[0]*avg_vector_j[1] - avg_vector_i[1]*avg_vector_j[0];
                                  
            bcrossc[0] = avg_vector_j[1]*avg_vector_k[2] - avg_vector_j[2]*avg_vector_k[1];
            bcrossc[1] = avg_vector_j[2]*avg_vector_k[0] - avg_vector_j[0]*avg_vector_k[2];
            bcrossc[2] = avg_vector_j[0]*avg_vector_k[1] - avg_vector_j[1]*avg_vector_k[0];
                                  
            ccrossa[0] = avg_vector_k[1]*avg_vector_i[2] - avg_vector_k[2]*avg_vector_i[1];
            ccrossa[1] = avg_vector_k[2]*avg_vector_i[0] - avg_vector_k[0]*avg_vector_i[2];
            ccrossa[2] = avg_vector_k[0]*avg_vector_i[1] - avg_vector_k[1]*avg_vector_i[0];
                                  
            adotbcrossc = avg_vector_i[0]*bcrossc[0] + avg_vector_i[1]*bcrossc[1] + avg_vector_i[2]*bcrossc[2];
                                  
            float gradV1[3];
            float gradV8[3];
            float gradV3[3];
            float gradV6[3];
            float gradV7[3];
            float gradV2[3];
            float gradV5[3];
            float gradV4[3];
                                  
            gradV1[0] = -1*bcrossc[0] -1*ccrossa[0] -1*acrossb[0];
            gradV8[0] = -1*gradV1[0];
            gradV3[0] = bcrossc[0] -1*ccrossa[0] -1*acrossb[0];
            gradV6[0] = -1*gradV3[0];
            gradV7[0] = bcrossc[0] + ccrossa[0] -1*acrossb[0];
            gradV2[0] = -1*gradV7[0];
            gradV5[0] = -1*bcrossc[0] + ccrossa[0] -1*acrossb[0];
            gradV4[0] = -1*gradV5[0];
            
            gradV1[1] = -1*bcrossc[1] -1*ccrossa[1] -1*acrossb[1];
            gradV8[1] = -1*gradV1[1];
            gradV3[1] = bcrossc[1] -1*ccrossa[1] -1*acrossb[1];
            gradV6[1] = -1*gradV3[1];
            gradV7[1] = bcrossc[1] + ccrossa[1] -1*acrossb[1];
            gradV2[1] = -1*gradV7[1];
            gradV5[1] = -1*bcrossc[1] + ccrossa[1] -1*acrossb[1];
            gradV4[1] = -1*gradV5[1];
                                  
            gradV1[2] = -1*bcrossc[2] -1*ccrossa[2] -1*acrossb[2];
            gradV8[2] = -1*gradV1[2];
            gradV3[2] = bcrossc[2] -1*ccrossa[2] -1*acrossb[2];
            gradV6[2] = -1*gradV3[2];
            gradV7[2] = bcrossc[2] + ccrossa[2] -1*acrossb[2];
            gradV2[2] = -1*gradV7[2];
            gradV5[2] = -1*bcrossc[2] + ccrossa[2] -1*acrossb[2];
            gradV4[2] = -1*gradV5[2];

            float inverse_V0 = 1/(l_e*l_e*l_e);
            float prefactor = -1*kappa * ((inverse_V0*adotbcrossc - 1));
            atomicAdd(&forces[3*index0],prefactor*gradV1[0]);
            atomicAdd(&forces[3*index0+1],prefactor*gradV1[1]);
            atomicAdd(&forces[3*index0+2],prefactor*gradV1[2]);
            atomicAdd(&forces[3*index1],prefactor*gradV2[0]);
            atomicAdd(&forces[3*index1+1],prefactor*gradV2[1]);
            atomicAdd(&forces[3*index1+2],prefactor*gradV2[2]);
            atomicAdd(&forces[3*index2],prefactor*gradV3[0]);
            atomicAdd(&forces[3*index2+1],prefactor*gradV3[1]);
            atomicAdd(&forces[3*index2+2],prefactor*gradV3[2]);
            atomicAdd(&forces[3*index3],prefactor*gradV4[0]);
            atomicAdd(&forces[3*index3+1],prefactor*gradV4[1]);
            atomicAdd(&forces[3*index3+2],prefactor*gradV4[2]);
            atomicAdd(&forces[3*index4],prefactor*gradV5[0]);
            atomicAdd(&forces[3*index4+1],prefactor*gradV5[1]);
            atomicAdd(&forces[3*index4+2],prefactor*gradV5[2]);
            atomicAdd(&forces[3*index5],prefactor*gradV6[0]);
            atomicAdd(&forces[3*index5+1],prefactor*gradV6[1]);
            atomicAdd(&forces[3*index5+2],prefactor*gradV6[2]);
            atomicAdd(&forces[3*index6],prefactor*gradV7[0]);
            atomicAdd(&forces[3*index6+1],prefactor*gradV7[1]);
            atomicAdd(&forces[3*index6+2],prefactor*gradV7[2]);
            atomicAdd(&forces[3*index7],prefactor*gradV8[0]);
            atomicAdd(&forces[3*index7+1],prefactor*gradV8[1]);
            atomicAdd(&forces[3*index7+2],prefactor*gradV8[2]);
        }
        }
        ''', 'element_force')

    normalized_posns[:,0] *= 1.01
    # cupy_node_posns = cp.array(node_posns).reshape((node_posns.shape[0]*node_posns.shape[1],1),order='C')
    cupy_elements = cp.array(elements.astype(np.int32)).reshape((elements.shape[0]*elements.shape[1],1),order='C')
    N_nodes = normalized_posns.shape[0]
    # cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32)#TODO implement the function to include the instantiation of the GPU memory for the forces, edges, and positions (since that overhead will exist every time prior to executing the kernel)
    size_elements = elements.shape[0]
    block_size = 128
    grid_size = (int (np.ceil((int (np.ceil(size_elements/block_size)))/14)*14))
    N_iterations = 10
    def element_func_w_transfers(cupy_elements,normalized_posns,kappa,l_e,N_nodes,size_elements):
        cupy_forces = cp.zeros((N_nodes*3,1),dtype=cp.float32) 
        cupy_node_posns = cp.array(np.float32(normalized_posns)).reshape((normalized_posns.shape[0]*normalized_posns.shape[1],1),order='C')
        element_kernel((grid_size,),(block_size,),(cupy_elements,cupy_node_posns,cp.float32(kappa),cp.float32(l_e),cupy_forces,size_elements))
        host_cupy_forces = cp.asnumpy(cupy_forces)
        return host_cupy_forces

    execution_gpu = cupyx.profiler.benchmark(element_func_w_transfers,(cupy_elements,normalized_posns,kappa,l_e,N_nodes,size_elements),n_repeat=N_iterations)
    delta_gpu_naive = np.sum(execution_gpu.gpu_times)

    correction_force_el = np.empty((8,3),dtype=np.float32)
    vectors = np.empty((8,3),dtype=np.float32)
    avg_vectors = np.empty((3,3),dtype=np.float32)
    volume_correction_force = np.zeros((N_nodes,3),dtype=np.float32)
    start = time.perf_counter()
    for i in range(N_iterations):
        get_volume_correction_force_cy_nogil.get_volume_correction_force_32bit(np.float32(normalized_posns),elements,np.float32(kappa),np.float32(l_e),correction_force_el,vectors,avg_vectors,volume_correction_force)
    end = time.perf_counter()
    delta_cy = end-start
    #using the 64 bit version, commented out so i can try and see how results compare if i have an implemented 32 bit version of the function
    # correction_force_el = np.empty((8,3),dtype=np.float64)
    # vectors = np.empty((8,3),dtype=np.float64)
    # avg_vectors = np.empty((3,3),dtype=np.float64)
    # volume_correction_force = np.zeros((N_nodes,3),dtype=np.float64)
    # start = time.perf_counter()
    # for i in range(N_iterations):
    #     get_volume_correction_force_cy_nogil.get_volume_correction_force(normalized_posns,elements,kappa,l_e,correction_force_el,vectors,avg_vectors,volume_correction_force)
    # end = time.perf_counter()
    # delta_cy = end-start
    host_cupy_forces = element_func_w_transfers(cupy_elements,normalized_posns,kappa,l_e,N_nodes,size_elements)
    host_cupy_forces = np.reshape(host_cupy_forces,(volume_correction_force.shape[0],volume_correction_force.shape[1]))
    try:
        correctness = np.allclose(host_cupy_forces,np.float32(volume_correction_force))
    except Exception as inst:
            print('Exception raised during calculation')
            print(type(inst))
            print(inst)
    print("GPU and CPU based calculations of forces agree?: " + str(correctness))
    if not correctness:
        difference = host_cupy_forces-volume_correction_force
        max_norm_diff = np.max(np.linalg.norm(difference,axis=1))
        print(f'maximum difference in volume correction force norm is {max_norm_diff}')
    print("CPU time is {} seconds".format(delta_cy))
    print("GPU time is {} seconds".format(delta_gpu_naive))
    print("GPU is {}x faster than CPU".format(delta_cy/delta_gpu_naive))
    return

if __name__ == "__main__":
    main()