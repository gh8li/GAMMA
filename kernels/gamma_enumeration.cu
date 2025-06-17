#include "graph/graph.h"
#include "graph/match_gpu.h"
#include "utils/config.h"
#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "utils/globals.h"
#include "utils/search.cuh"

// #define IS_DEBUGGING
namespace {
__device__ void warp_print(const uint8_t lane_id, const uint32_t warp_id, const unsigned long long global_warp_id, const char *content) {
    if(lane_id == 0)
    {
        printf("warp_id: %d, %lld, %s", warp_id, global_warp_id, content);
        // printf(content);
    }
    __syncwarp();
}
}  // namespace

template<typename T> __device__ __host__ inline T div_cell(T a, T b)
{
    return a / b + (a % b != 0);
}

// __forceinline__ __device__ bool lock(int* mutex) {
//     int count = 0;
//     while (atomicCAS((int*)mutex, 0, 1) != 0) {
//         count++;
//         if (count > 10000) {
//             printf("locking stalled!\n");
//             return false;
//         }
//     }
//     return true;
// }

// __forceinline__ __device__ bool lock(int* mutex, const uint32_t warp_id, const unsigned long long global_warp_id) {
//     int count = 0;
//     while (atomicCAS((int*)mutex, 0, 1) != 0) {
//         count++;
//         if (count > 10000) {
//             printf("warp_id: %d, global_warp_id: %llu, locking stalled!\n", warp_id, global_warp_id);
//             return false;
//         }
//     }
//     return true;
// }

__forceinline__ __device__ bool lock(int* mutex) {
    while (atomicCAS((int*)mutex, 0, 1) != 0) { }
    return true;
}

__forceinline__ __device__ void unlock(int* mutex) {
    atomicExch((int*)mutex, 0);
}

// __forceinline__ __device__ const uint8_t& get_first_backward_neighbor_idx(const uint8_t order_idx, const uint8_t cur_depth)
// {
//     return C_ORDERS[order_idx].bni_[C_ORDERS[order_idx].bni_offs_[cur_depth]];
// }

__forceinline__ __device__ const uint8_t& get_first_backward_neighbor_idx(const OrderPerEdge &order, const uint8_t cur_depth)
{
    return order.bni_[order.bni_offs_[cur_depth]];
}

__forceinline__ __device__ const uint8_t& get_query_vertex(const uint8_t order_idx, const uint8_t idx)
{
    return C_ORDERS[order_idx].vs_[idx];
}

__forceinline__ __device__ const uint8_t& get_query_vertex(const OrderPerEdge &order, const uint8_t idx)
{
    return order.vs_[idx];
}

__forceinline__ __device__ const uint8_t& get_query_edge_idx(const uint8_t src, const uint8_t dst)
{
    return C_EIDX[src * C_QV_COUNT + dst];
}

__forceinline__ __device__ const uint32_t& get_nbr_size(const RelationsGPU graph, const uint8_t query_edge_idx, const uint32_t data_vertex)
{
    return graph.sizes_[query_edge_idx][data_vertex];
}

__forceinline__ __device__ const uint32_t& get_nbr(const RelationsGPU graph, const uint8_t query_edge_idx, const uint32_t data_vertex, const uint32_t nbr_idx)
{
    return graph.nbrs_[query_edge_idx][data_vertex][nbr_idx];
}

__forceinline__ __device__ bool find_nbr(const RelationsGPU graph, const uint8_t query_edge_idx, const uint32_t data_vertex, const uint32_t nbr)
{
    const uint32_t res = lower_bound(graph.nbrs_[query_edge_idx][data_vertex], graph.sizes_[query_edge_idx][data_vertex], nbr);
    if (res == graph.sizes_[query_edge_idx][data_vertex] || graph.nbrs_[query_edge_idx][data_vertex][res] != nbr){
        return false;
    }
    else{
        return true;
    }
}

// __forceinline__ __device__ uint32_t get_previous_result(const unsigned long long previous_result_ptr, const unsigned long long result_idx, const uint8_t start_depth, const uint32_t depth)
// {
//     return C_RES_QUEUE.array_[(previous_result_ptr + result_idx * (start_depth + 1) + (depth + 1)) % C_RES_QUEUE.capacity_];
// }

__forceinline__ __device__ uint32_t gamma_get_previous_result(const OrderPerEdge &order, const unsigned long long previous_result_ptr, 
    const unsigned long long result_idx, const uint8_t num_query_vertices, const uint32_t depth)
{
    uint8_t query_vertex_id = order.vs_[depth];
    return C_RES_QUEUE.array_[(previous_result_ptr + result_idx * (num_query_vertices + 1) + (query_vertex_id + 1)) % C_RES_QUEUE.capacity_];
}

// __forceinline__ __device__ uint8_t get_matching_order_idx(const unsigned long long previous_result_ptr, const unsigned long long result_idx, const uint8_t start_depth)
// {
//     return (uint8_t)(C_RES_QUEUE.array_[(previous_result_ptr + result_idx * (start_depth + 1)) % C_RES_QUEUE.capacity_]);
// }

__forceinline__ __device__ uint8_t gamma_get_matching_order_idx(const unsigned long long previous_result_ptr, const unsigned long long result_idx, const uint8_t num_query_vertices)
{
    return (uint8_t)(C_RES_QUEUE.array_[(previous_result_ptr + result_idx * (num_query_vertices + 1)) % C_RES_QUEUE.capacity_]);
}

__forceinline__ __device__ uint8_t get_enumerate_end_depth(const uint8_t matching_order_idx)
{
    return C_NON_TAIL_LEAF_DEPTHS[matching_order_idx];
}

// __forceinline__ __device__ void write_a_result_vertex(const unsigned long long new_result_ptr, const unsigned long long result_idx, const uint8_t end_depth, const uint32_t depth, const uint32_t result)
// {
//     C_RES_QUEUE.array_[(new_result_ptr + result_idx * (end_depth + 1) + (depth + 1)) % C_RES_QUEUE.capacity_] = result;
// }

__forceinline__ __device__ void gamma_write_a_result_vertex(const OrderPerEdge &order, const unsigned long long new_result_ptr, const unsigned long long result_idx, const uint8_t num_query_vertices, const uint32_t depth, const uint32_t result)
{
    uint8_t query_vertex_id = get_query_vertex(order, depth);
    C_RES_QUEUE.array_[(new_result_ptr + result_idx * (num_query_vertices + 1) + (query_vertex_id + 1)) % C_RES_QUEUE.capacity_] = result;
}

// __forceinline__ __device__ void write_matching_order_idx(const unsigned long long new_result_ptr, const unsigned long long result_idx, const uint8_t end_depth, const uint8_t matching_order_idx)
// {
//     C_RES_QUEUE.array_[(new_result_ptr + result_idx * (end_depth + 1)) % C_RES_QUEUE.capacity_] = (uint32_t)matching_order_idx;
// }

__forceinline__ __device__ void gamma_write_matching_order_idx(const unsigned long long new_result_ptr, const unsigned long long result_idx, const uint8_t num_query_vertices, const uint8_t matching_order_idx)
{
    C_RES_QUEUE.array_[(new_result_ptr + result_idx * (num_query_vertices + 1)) % C_RES_QUEUE.capacity_] = (uint32_t)matching_order_idx;
}

__global__ void gamma_write_initial_partial_results(
    RelationsGPU local_index_gpu,  // In fact, local index are passed in.
    const uint8_t edge_idx,
    const uint8_t edge_list_idx,
    const uint8_t num_query_vertices,
    const unsigned long long int new_res_start,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    __shared__ unsigned long long int write_pos[NUM_WARP_PER_BLOCK];
    __shared__ OrderPerEdge order[NUM_WARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long global_warp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;
    
    if (lane_id == 0) {
        order[warp_id] = C_ORDERS[edge_list_idx];
    }
    __syncwarp();

#ifdef IS_DEBUGGING
if(global_warp_id == 0 and lane_id == 0)
{
    printf("warp_id: %d, %lld, result_idx: %llu, write matching_order_idx: %d \n", warp_id, global_warp_id, *new_res_size, (uint32_t)edge_list_idx);
    // printf(content);
}
__syncwarp();
#endif

    for (uint32_t v = global_warp_id; v < C_DV_COUNT; v+= num_warps)
    {
        uint32_t v_neighbor_size = 0;
        if (lane_id == 0) {
            v_neighbor_size = local_index_gpu.sizes_[edge_idx][v];
            write_pos[warp_id] = atomicAdd(new_res_size, (unsigned long long)v_neighbor_size);
        }
        // __shfl_sync(0xFFFFFFFF, v_neighbor_size, 0);
        v_neighbor_size = __shfl_sync(0xFFFFFFFF, v_neighbor_size, 0);
        
        // __syncwarp();
        //if (*new_res_size >= h_max_new_res_size) return;

        for (uint32_t j = lane_id; j < v_neighbor_size; j += WARP_SIZE)
        {
            // C_RES_QUEUE.array_[new_res_start + (write_pos[warp_id] + j) * (num_query_vertices + 1)] = (uint32_t)edge_list_idx;
            gamma_write_matching_order_idx(new_res_start, write_pos[warp_id] + j, num_query_vertices, edge_list_idx);
            gamma_write_a_result_vertex(order[warp_id], new_res_start, write_pos[warp_id] + j, num_query_vertices, 0, v);
            gamma_write_a_result_vertex(order[warp_id], new_res_start, write_pos[warp_id] + j, num_query_vertices, 1, local_index_gpu.nbrs_[edge_idx][v][j]);

            // C_RES_QUEUE.array_[new_res_start + (write_pos[warp_id] + j) * (num_query_vertices + 1) + order[warp_id].vs_[0] + 1] = v;
            // C_RES_QUEUE.array_[new_res_start + (write_pos[warp_id] + j) * (num_query_vertices + 1) + order[warp_id].vs_[1] + 1] = local_index_gpu.nbrs_[edge_idx][v][j];
        }
    }
}

__device__ bool steal_work(uint8_t warp_id, unsigned long long global_warp_id, int *mutex_this_block, uint8_t start_depth, uint8_t num_query_vertices,
uint8_t *matching_order_idx_in_block, uint8_t *end_depths_in_block, uint32_t (*finished_nbr_count)[MAX_QV_COUNT], uint32_t (*total_nbr_count)[MAX_QV_COUNT],
uint32_t (*temp_result_buffer)[MAX_QV_COUNT][WARP_SIZE], uint8_t (*temp_ptr)[MAX_QV_COUNT], uint8_t &output_target_depth)
{

    while (true) {

        // Find a warp with remaining work.
        uint8_t target_warp_id = warp_id;
        uint8_t target_depth = UINT8_MAX;


    // #ifdef IS_DEBUGGING
    //     if(global_warp_id < 2624)
    //     {
    //         printf("warp_id: %d, %lld, inside steal_work, before first for-loop.\n", warp_id, global_warp_id);
    //     }
    // #endif

        for (uint8_t j = start_depth; j < num_query_vertices; j++)
        {
            uint32_t max_remaining_work = 0;
            for (uint8_t i = 0; i < NUM_WARP_PER_BLOCK; i++)
            {
                if (i == warp_id) continue;
                // lock(&(mutex_this_block[i]));
                if (lock(&(mutex_this_block[i]))) {
                    if (j < end_depths_in_block[i]){
                        if (total_nbr_count[i][j] != UINT32_MAX)
                        {
                            uint32_t cur_remaining_work = total_nbr_count[i][j] - finished_nbr_count[i][j];
                            if (cur_remaining_work > max_remaining_work)
                            {
                                max_remaining_work = cur_remaining_work;
                                target_warp_id = i;
                                target_depth = j;
                            }
                        }
                    }
                    unlock(&(mutex_this_block[i]));
                }
            }
            if (target_warp_id != warp_id){
                break;
            }
        }

    // #ifdef IS_DEBUGGING
    //     if(global_warp_id < 2624)
    //     {
    //         printf("warp_id: %d, %lld, inside steal_work, before if (target_warp_id == warp_id) return false;\n", warp_id, global_warp_id);
    //     }
    // #endif

        if (target_warp_id == warp_id) return false;  // target_depth == UINT8_MAX

    // #ifdef IS_DEBUGGING
    //     if(global_warp_id < 2624)
    //     {
    //         printf("warp_id: %d, %lld, inside steal_work, after if (target_warp_id == warp_id) return false; before two locks.\n", warp_id, global_warp_id);
    //     }
    // #endif



                // int count = 0;
                // for (int i = 0; i < 10000000; i++) {
                //     count += 1;
                // }

            // if (warp_id < target_warp_id) {
            //     lock(&(mutex_this_block[warp_id]));
            //     lock(&(mutex_this_block[target_warp_id]));
            // }
            // else {
            //     lock(&(mutex_this_block[target_warp_id]));
            //     lock(&(mutex_this_block[warp_id]));
            // }


        // #ifdef IS_DEBUGGING
        //     if(global_warp_id < 2624)
        //     {
        //         printf("warp_id: %d, %lld, inside steal_work, after two locks.\n", warp_id, global_warp_id);
        //     }
        // #endif

        // Steal half of the remaining work from the target warp. 
        bool stolen = false;
        lock(&(mutex_this_block[target_warp_id]));
        if (total_nbr_count[target_warp_id][target_depth] != UINT32_MAX)
        {
            lock(&(mutex_this_block[warp_id]));

            uint32_t remaining_work = total_nbr_count[target_warp_id][target_depth] - finished_nbr_count[target_warp_id][target_depth];

            if (remaining_work != 0) {

                uint32_t half_remaining_work = div_cell(remaining_work, 2u);
                // (finished) TODO: This is not accurate if remaining_work is an odd number.
                total_nbr_count[warp_id][target_depth] = total_nbr_count[target_warp_id][target_depth];
                total_nbr_count[target_warp_id][target_depth] -= half_remaining_work;
                finished_nbr_count[warp_id][target_depth] = total_nbr_count[target_warp_id][target_depth];
                // finished_nbr_count[warp_id][target_depth] = finished_nbr_count[target_warp_id][target_depth] + half_remaining_work;

                matching_order_idx_in_block[warp_id] = matching_order_idx_in_block[target_warp_id];
                end_depths_in_block[warp_id] = end_depths_in_block[target_warp_id];

                for (uint8_t idx = 0; idx < target_depth; idx++)
                {
                    uint8_t temp_ptr_ij = temp_ptr[target_warp_id][idx];
                    temp_result_buffer[warp_id][idx][0] = temp_result_buffer[target_warp_id][idx][temp_ptr_ij];
                    // temp_ptr[warp_id][idx] = 0;
                    // total_nbr_count[warp_id][idx] = 0u;
                }
                // temp_ptr[warp_id][target_depth] = 0;

                output_target_depth = target_depth;

                // if (total_nbr_count[warp_id][target_depth] == UINT32_MAX) {
                //     printf("warp_id: %d, %llu, target_warp_id: %d,  target_depth: %d, inside steal_work, STRANGE VALUE!, total_nbr_count[warp_id][target_depth] == UINT32_MAX, total_nbr_count[target_warp_id][target_depth] == %d \n", (uint32_t)warp_id, global_warp_id, (uint32_t)target_warp_id, (uint32_t)target_depth, total_nbr_count[target_warp_id][target_depth]);
                // }
                // if (finished_nbr_count[warp_id][target_depth] == total_nbr_count[warp_id][target_depth]) {
                //     printf("warp_id: %d, %llu, target_warp_id: %d,  target_depth: %d, inside steal_work, STRANGE EQUALITY! finished_nbr_count[warp_id][target_depth] == total_nbr_count[warp_id][target_depth], total_nbr_count[target_warp_id][target_depth] == %d, remaining_work: %d, half_remaining_work: %d. \n", (uint32_t)warp_id, global_warp_id, (uint32_t)target_warp_id, (uint32_t)target_depth, total_nbr_count[target_warp_id][target_depth], remaining_work, half_remaining_work);
                // }
                
                // printf("warp_id: %d, %lld, target_warp_id: %d, cur_start_depth: %d, finished_nbr_count: %d, total_nbr_count: %d, steal_success! \n", (uint32_t)warp_id, global_warp_id, (uint32_t)target_warp_id, (uint32_t)target_depth, finished_nbr_count[warp_id][target_depth], total_nbr_count[warp_id][target_depth]);
                

                stolen = true;
            }

            unlock(&(mutex_this_block[warp_id]));
        }
        unlock(&(mutex_this_block[target_warp_id]));

        if (stolen) return true;

    }

// #ifdef IS_DEBUGGING
//     if(global_warp_id < 2624)
//     {
//         printf("warp_id: %d, %lld, inside steal_work, after two unlocks.\n", warp_id, global_warp_id);
//     }
// #endif
    
    // return true;
}

__global__ void gamma_enumerate(
    const unsigned long long previous_result_ptr,
    const unsigned long long previous_num_results,
    const unsigned long long new_result_ptr,
    unsigned long long *new_num_results_dptr,
    const unsigned long long _h_max_new_num_results,
    const RelationsGPU data_graph,
    const uint8_t start_depth,
    const uint8_t num_query_vertices,
    const bool enable_work_stealing,
    const bool enable_cartesian_product,
    const bool write_results,
    unsigned long long *effective_num_dptr
) {

    if (start_depth >= num_query_vertices) return;

    __shared__ uint32_t temp_result_buffer[NUM_WARP_PER_BLOCK][MAX_QV_COUNT][WARP_SIZE];
    __shared__ uint8_t temp_ptr[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];
    __shared__ uint8_t temp_size[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];

    __shared__ uint32_t finished_nbr_count[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];
    __shared__ uint32_t total_nbr_count[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];

    __shared__ OrderPerEdge matching_orders[MAX_QE_COUNT];
    __shared__ uint8_t matching_order_idx_in_block[NUM_WARP_PER_BLOCK];
    __shared__ uint8_t end_depths_in_block[NUM_WARP_PER_BLOCK];

    __shared__ int mutex_this_block[NUM_WARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long global_warp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint8_t& matching_order_idx = matching_order_idx_in_block[warp_id];

    // if (global_warp_id >= previous_num_results) {
    //     // printf("global_warp_id: %d, return.", global_warp_id);
    //     return;
    // }

// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before reading all matching orders.\n");
// #endif

    mutex_this_block[warp_id] = 0;

    for (int i = threadIdx.x; i < C_QE_COUNT; i += blockDim.x)
    {
        if (i < C_QE_COUNT){
            matching_orders[i] = C_ORDERS[i];
        }
    }

// #ifdef IS_DEBUGGING
//     if (global_warp_id < 2624){
//         warp_print(lane_id, warp_id, global_warp_id, "before get matching order idx and end depth.\n");
//     }
// #endif

    if (lane_id == 0 && global_warp_id < previous_num_results)
    {
        uint8_t temp_matching_order_idx = gamma_get_matching_order_idx(previous_result_ptr, global_warp_id, num_query_vertices);

// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == NUM_WARP_PER_BLOCK - 1)
//     {
//         // if (temp_matching_order_idx > 4) {
//         printf("warp_id: %d, %lld, temp_matching_order_idx: %d, before get_end_depth.\n", warp_id, global_warp_id, (uint32_t)temp_matching_order_idx);
//         // }
//     }
// #endif

        matching_order_idx_in_block[warp_id] = temp_matching_order_idx;
        if (enable_cartesian_product) {
            end_depths_in_block[warp_id] = get_enumerate_end_depth(temp_matching_order_idx);
        } else {
            end_depths_in_block[warp_id] = num_query_vertices;
        }

// #ifdef IS_DEBUGGING
//     // warp_print(lane_id, warp_id, global_warp_id, "after get_end_depth.\n");
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == NUM_WARP_PER_BLOCK - 1)
//     // if(warp_id == NUM_WARP_PER_BLOCK - 1)
//     if(warp_id == 0 && matching_order_idx > 3u)
//     {
//         printf("warp_id: %d, %lld, temp_matching_order_idx: %d, end_depth: %d, after get_end_depth.\n", warp_id, global_warp_id, (uint32_t)temp_matching_order_idx, (uint32_t)(end_depths_in_block[warp_id]));
//     }
// #endif

    }

    if (lane_id < C_QV_COUNT)
    {
        temp_ptr[warp_id][lane_id] = 0u;
        temp_size[warp_id][lane_id] = (lane_id < start_depth) ? 1u: 0u;
        finished_nbr_count[warp_id][lane_id] = 0u;
        total_nbr_count[warp_id][lane_id] = (lane_id < start_depth && global_warp_id < previous_num_results) ? 0u: UINT32_MAX;
    }
    __syncthreads();
    
    if (global_warp_id >= previous_num_results) {
// #ifdef IS_DEBUGGING
//         warp_print(lane_id, warp_id, global_warp_id, "return.\n");
// #endif
        return;
    }


    if (lane_id < start_depth)
    {
        temp_result_buffer[warp_id][lane_id][0] = gamma_get_previous_result(matching_orders[matching_order_idx], previous_result_ptr, global_warp_id, num_query_vertices, lane_id);
    }
    __syncwarp();


    if (start_depth >= end_depths_in_block[warp_id]) {

        unsigned long long write_pos = UINT32_MAX;
        if (lane_id == 0) 
        {
            write_pos = atomicAdd(new_num_results_dptr, 1u);
            atomicAdd(effective_num_dptr, C_EQUIV_EDGE_COUNT[matching_order_idx]);
        }

        if (write_results)
        {
            unsigned long long new_num_results;
            if (lane_id == 0)
            {
                new_num_results = *new_num_results_dptr;
            }
            new_num_results = __shfl_sync(0xffffffff, new_num_results, 0);
            if (new_num_results >= _h_max_new_num_results) return;
            
            if (lane_id == 0) {
                gamma_write_matching_order_idx(new_result_ptr, write_pos, num_query_vertices, matching_order_idx_in_block[warp_id]);
                for (uint8_t j = 0u; j < end_depths_in_block[warp_id]; j++)
                {
                    uint8_t cur_temp_ptr = temp_ptr[warp_id][j];
                    uint32_t cur_result = temp_result_buffer[warp_id][j][cur_temp_ptr];
                    gamma_write_a_result_vertex(matching_orders[matching_order_idx], new_result_ptr, write_pos, num_query_vertices, j, cur_result);
// #ifdef IS_DEBUGGING
// if (matching_order_idx == 2 && j == 1){
// printf("warp_id: %d, %lld, write_pos: %llu, rank: %d, depth-j: %d, before enumeration, gamma_write_a_result_vertex: %d.\n", warp_id, global_warp_id, write_pos, rank, (uint32_t)j, cur_result);
// }
// #endif
                }
            }
            
        }

        return;

    }


    uint8_t cur_start_depth = start_depth;

// #ifdef IS_DEBUGGING
//     // if(lane_id == 0)
//     if(lane_id == 0 && global_warp_id < 2624) {
//         printf("warp_id: %d, %lld, order_idx: %d, before while loop.\n", warp_id, global_warp_id, (uint32_t)matching_order_idx_in_block[warp_id]);
//     }
//     __syncwarp();
//     // warp_print(lane_id, warp_id, global_warp_id, "before while loop.\n");
// #endif

    while (true){

        uint8_t cur_depth = cur_start_depth;
        while (cur_depth >= cur_start_depth)
        {
            __syncwarp();


// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before get_first_backward_neighbor_idx\n");
//     __syncwarp();
// #endif
            const uint8_t& pre_qv_depth = get_first_backward_neighbor_idx(matching_orders[matching_order_idx], cur_depth);

// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before pre_qv = get_query_vertex\n");
//     __syncwarp();
// #endif
            const uint8_t& pre_qv = get_query_vertex(matching_orders[matching_order_idx], pre_qv_depth);

// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before cur_qv = get_query_vertex\n");
//     __syncwarp();
// #endif
            const uint8_t& cur_qv = get_query_vertex(matching_orders[matching_order_idx], cur_depth);

// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before get_query_edge_idx\n");
//     __syncwarp();
// #endif
            const uint8_t& pre_qe_idx = get_query_edge_idx(pre_qv, cur_qv);

// #ifdef IS_DEBUGGING
//     if (global_warp_id < 2624){
//         warp_print(lane_id, warp_id, global_warp_id, "after get_query_edge_idx\n");
//     }
// #endif

// #ifdef IS_DEBUGGING
// if(lane_id == 0)
// {
//     printf("warp_id: %d, %lld, pre_qv_depth: %u, pre_qv: %u, cur_qv: %u, pre_qe_idx: %u \n", warp_id, global_warp_id, (uint32_t)pre_qv_depth, (uint32_t)pre_qv, (uint32_t)cur_qv, (uint32_t)pre_qe_idx);
// }
// __syncwarp();
// #endif


            // if (temp_ptr[warp_id][cur_depth] < temp_size[warp_id][cur_depth] && cur_depth < end_depths_in_block[warp_id] - 1)
            if (temp_ptr[warp_id][cur_depth] < temp_size[warp_id][cur_depth])
            {
// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "inside while loop.\n");
// #endif
                cur_depth++;
                __syncwarp();
            }
            else
            {
// #ifdef IS_DEBUGGING
//     warp_print(lane_id, warp_id, global_warp_id, "before lock.\n");
// #endif
                if (lane_id == 0)
                {
                    lock(&(mutex_this_block[warp_id]));
                }
                __syncwarp();  // VERY IMPORTANT!

                if (finished_nbr_count[warp_id][cur_depth] >= total_nbr_count[warp_id][cur_depth])
                {
// #ifdef IS_DEBUGGING
//     if (global_warp_id < 2624) {
//         warp_print(lane_id, warp_id, global_warp_id, "inside branch finished_nbr_count[warp_id][cur_depth] >= total_nbr_count[warp_id][cur_depth].\n");
//     }
// #endif
                    if (lane_id == 0)
                    {
                        temp_ptr[warp_id][cur_depth] = 0u;
                        temp_size[warp_id][cur_depth] = 0u;
                        finished_nbr_count[warp_id][cur_depth] = 0u;
                        total_nbr_count[warp_id][cur_depth] = UINT32_MAX;

                        cur_depth--;

                        if (cur_depth >= cur_start_depth) {
                            temp_ptr[warp_id][cur_depth]++;
                        }

                        unlock(&(mutex_this_block[warp_id]));
                    }
                    cur_depth = __shfl_sync(0xffffffff, cur_depth, 0);
                    // __syncwarp();
                }
                else
                {
// #ifdef IS_DEBUGGING
//     if (global_warp_id < 2624) {
//         warp_print(lane_id, warp_id, global_warp_id, "inside branch !finished_nbr_count[warp_id][cur_depth] >= total_nbr_count[warp_id][cur_depth].\n");
//     }
// #endif                    
                    uint32_t pre_dv = UINT32_MAX;
                    uint8_t rem_nbr_count = UINT8_MAX;
                    uint32_t pre_finished_nbr_count = UINT32_MAX;
                    if (lane_id == 0)
                    {
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == 0)
//     {
//         printf("warp_id: %d, %lld, pre_qv_depth: %d, before pre_dv_ptr = temp_ptr[warp_id][pre_qv_depth].\n", warp_id, global_warp_id, (uint32_t)pre_qv_depth);
//     }
// #endif
                        uint8_t pre_dv_ptr = temp_ptr[warp_id][pre_qv_depth];
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == 0)
//     {
//         printf("warp_id: %d, %lld, pre_qv_depth: %d, pre_dv_ptr: %d, before pre_dv = temp_result_buffer[warp_id][pre_qv_depth][pre_dv_ptr].\n", warp_id, global_warp_id, (uint32_t)pre_qv_depth, (uint32_t)pre_dv_ptr);
//     }
// #endif
                        // Attention! uint32_t pre_dv = ... is WRONG, because this introduce a NEW local variable, masking the outside `pre_dv`.!
                        // uint32_t pre_dv = temp_result_buffer[warp_id][pre_qv_depth][pre_dv_ptr];
                        pre_dv = temp_result_buffer[warp_id][pre_qv_depth][pre_dv_ptr];
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == 0)
//     {
//         printf("warp_id: %d, %lld, pre_qv_depth: %d, pre_dv_ptr: %d, after pre_dv = temp_result_buffer[warp_id][pre_qv_depth][pre_dv_ptr].\n", warp_id, global_warp_id, (uint32_t)pre_qv_depth, (uint32_t)pre_dv_ptr);
//     }
// #endif
                        if (total_nbr_count[warp_id][cur_depth] == UINT32_MAX){
                            
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == 0)
//     // if (warp_id == NUM_WARP_PER_BLOCK -1)
//     if (matching_order_idx == 4u)
//     {
//         printf("warp_id: %d, %lld, matchint_order_idx: %d, pre_qe_idx %d, pre_dv: %d, before get_ngr_size.\n", warp_id, global_warp_id, (uint32_t)matching_order_idx, (uint32_t)pre_qe_idx, pre_dv);
//     }
// #endif
                            total_nbr_count[warp_id][cur_depth] = get_nbr_size(data_graph, pre_qe_idx, pre_dv);

#ifdef IS_DEBUGGING
    // if(blockIdx.x % 200 == 0 && warp_id == 0)
    // if(warp_id == 0)
    // if (warp_id == NUM_WARP_PER_BLOCK -1)
    // if (matching_order_idx == 2u && cur_depth == 2)
    if (matching_order_idx == 2u && cur_depth < 2)
    {
        // printf("warp_id: %d, %lld, matching_order_idx: %d, cur_depth: %d, pre_qe_idx %d, pre_dv: %d, nbr_size: %d, end_depths_in_block[warp_id]: %d, after get_nbr_size.\n", warp_id, global_warp_id, (uint32_t)matching_order_idx, (uint32_t)cur_depth, (uint32_t)pre_qe_idx, pre_dv, total_nbr_count[warp_id][cur_depth], (uint32_t)(end_depths_in_block[warp_id]));
        printf("warp_id: %d, %lld, cur_depth: %d, pre_qe_idx %d, pre_dv: %d, nbr_size: %d, end_depths_in_block[warp_id]: %d, after get_nbr_size.\n", warp_id, global_warp_id, (uint32_t)cur_depth, (uint32_t)pre_qe_idx, pre_dv, total_nbr_count[warp_id][cur_depth], (uint32_t)(end_depths_in_block[warp_id]));
    }
#endif


                        }
                        rem_nbr_count =  min(total_nbr_count[warp_id][cur_depth] - finished_nbr_count[warp_id][cur_depth], 64u);  // use 'min' to avoid uint8_t overflow
                        pre_finished_nbr_count = finished_nbr_count[warp_id][cur_depth];
                        finished_nbr_count[warp_id][cur_depth] += min(rem_nbr_count, WARP_SIZE);

                        unlock(&(mutex_this_block[warp_id]));
                    }

                    pre_dv = __shfl_sync(0xffffffff, pre_dv, 0);
                    rem_nbr_count = __shfl_sync(0xffffffff, rem_nbr_count, 0);
                    pre_finished_nbr_count = __shfl_sync(0xffffffff, pre_finished_nbr_count, 0);

                    bool found = lane_id < rem_nbr_count;
                    uint32_t temp_nbr = UINT32_MAX;
                    // if (lane_id == 0) {
                    //     temp_ptr[warp_id][cur_depth] = 0u;
                    //     temp_size[warp_id][cur_depth] = 0u;
                    // }
                    // __syncwarp();

                    if (found)
                    {
                        uint32_t nbr_idx = pre_finished_nbr_count + lane_id;
                         
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == NUM_WARP_PER_BLOCK - 1)
//     // if(warp_id == 0)
//     if (matching_order_idx == 4u)
//     {
//         printf("warp_id: %d, %lld, matching_order_idx: %d pre_qe_idx %d, pre_dv: %d, nbr_idx: %d, before get_nbr.\n", warp_id, global_warp_id, (uint32_t)matching_order_idx, (uint32_t)pre_qe_idx, pre_dv, nbr_idx);
//     }
// #endif
                        temp_nbr = get_nbr(data_graph, pre_qe_idx, pre_dv, nbr_idx);

                     
// #ifdef IS_DEBUGGING
//     // if(blockIdx.x % 200 == 0 && warp_id == 0)
//     // if(warp_id == NUM_WARP_PER_BLOCK - 1)
//     // if(warp_id == 0)
//     if(lane_id == 0 && global_warp_id < 2624) {
//         printf("warp_id: %d, %lld, pre_qe_idx %d, pre_dv: %d, nbr_idx: %d, after get_nbr.\n", warp_id, global_warp_id, (uint32_t)pre_qe_idx, pre_dv, nbr_idx);
//     }
// #endif
                    }
                    __syncwarp();

                    if (found)
                    {
                        for (uint8_t i = 0u; i < cur_depth; i++)
                        {
                            uint8_t cur_pre_dv_ptr = temp_ptr[warp_id][i];
                            uint32_t cur_pre_dv = temp_result_buffer[warp_id][i][cur_pre_dv_ptr];
                            if (temp_nbr == cur_pre_dv)
                            {
                                found = false;
                                break;
                            }
                        }
                    }
                    __syncwarp();
                    
                    if (found)
                    {
                        for (uint8_t idx = C_ORDERS[matching_order_idx].bni_offs_[cur_depth] + 1; idx < C_ORDERS[matching_order_idx].bni_offs_[cur_depth + 1]; idx++)
                        {
                            const uint8_t& backward_nbr_idx = C_ORDERS[matching_order_idx].bni_[idx];
                            const uint8_t& cur_pre_qv = get_query_vertex(matching_order_idx, backward_nbr_idx);
                            const uint8_t& cur_pre_qe_idx = get_query_edge_idx(cur_pre_qv, cur_qv);
                            
                            const uint8_t cur_pre_dv_ptr = temp_ptr[warp_id][backward_nbr_idx];
                            const uint32_t& cur_pre_dv = temp_result_buffer[warp_id][backward_nbr_idx][cur_pre_dv_ptr];

                            found = find_nbr(data_graph, cur_pre_qe_idx, cur_pre_dv, temp_nbr);

                            if(found == false) break;
                        }
                    }
                    // __syncwarp();

                    // 6. write the local candidates to result_queue and their group id to group_id
                    const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                    const uint8_t temp_found_nbr_count = __popc(found_mask);
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);

                    if (found_mask)
                    {
                        if (cur_depth < end_depths_in_block[warp_id] - 1)
                        {
                            if (found) 
                            {
                                temp_result_buffer[warp_id][cur_depth][rank] = temp_nbr;
                            }
                            if (lane_id == 0)
                            {
                                temp_ptr[warp_id][cur_depth] = 0u;
                                temp_size[warp_id][cur_depth] = temp_found_nbr_count;
                            }
                        }
                        else
                        {
                            unsigned long long write_pos;
                            if (lane_id == 0) 
                            {
                                write_pos = atomicAdd(new_num_results_dptr, temp_found_nbr_count);
                                uint8_t num_equiv = C_EQUIV_EDGE_COUNT[matching_order_idx];
                                uint8_t temp_effective_num_results = temp_found_nbr_count * num_equiv;
// #ifdef IS_DEBUGGING
//                             printf("warp_id: %d, %lld, matching_order_idx: %d, num_equiv: %d, temp_found_nbr_count: %d, temp_effective_num_results: %d, before atomicAdd(effective_num_dptr, ...) .\n", warp_id, global_warp_id, (uint32_t)matching_order_idx, (uint32_t)num_equiv, (uint32_t)temp_found_nbr_count, (uint32_t)temp_effective_num_results);
// #endif
                                atomicAdd(effective_num_dptr, temp_found_nbr_count * C_EQUIV_EDGE_COUNT[matching_order_idx]);
                            }

                            if (write_results)
                            {
                                unsigned long long new_num_results;
                                if (lane_id == 0)
                                {
                                    new_num_results = *new_num_results_dptr;
                                }
                                // new_num_results = __shfl_sync(0xffffffff, write_pos, 0);
                                new_num_results = __shfl_sync(0xffffffff, new_num_results, 0);
                                if (new_num_results >= _h_max_new_num_results) return;
                                
                                write_pos = __shfl_sync(0xffffffff, write_pos, 0);
                                if (found)
                                {
                                    write_pos += rank;
                                    gamma_write_matching_order_idx(new_result_ptr, write_pos, num_query_vertices, matching_order_idx_in_block[warp_id]);
// #ifdef IS_DEBUGGING
//     printf("warp_id: %d, %lld, rank: %d, gamma_write_matching_order_idx: %d.\n", warp_id, global_warp_id, rank, matching_order_idx_in_block[warp_id]);
// #endif
                                    for (uint8_t j = 0u; j < end_depths_in_block[warp_id] - 1; j++)
                                    {
                                        uint8_t cur_temp_ptr = temp_ptr[warp_id][j];
                                        uint32_t cur_result = temp_result_buffer[warp_id][j][cur_temp_ptr];
                                        gamma_write_a_result_vertex(matching_orders[matching_order_idx], new_result_ptr, write_pos, num_query_vertices, j, cur_result);
// #ifdef IS_DEBUGGING
//     printf("warp_id: %d, %lld, rank: %d, depth: %d, gamma_write_a_result_vertex: %d.\n", warp_id, global_warp_id, rank, (uint32_t)j, cur_result);
// #endif
#ifdef IS_DEBUGGING
    if (matching_order_idx == 2 && j == 1){
        printf("warp_id: %d, %lld, write_pos: %llu, rank: %d, depth-j: %d, gamma_write_a_result_vertex: %d.\n", warp_id, global_warp_id, write_pos, rank, (uint32_t)j, cur_result);
    }
#endif
                                    }
                                    gamma_write_a_result_vertex(matching_orders[matching_order_idx], new_result_ptr, write_pos, num_query_vertices, end_depths_in_block[warp_id] - 1, temp_nbr);
#ifdef IS_DEBUGGING
    if (matching_order_idx == 2 && end_depths_in_block[warp_id] - 1 == 1){
        printf("warp_id: %d, %lld, write_pos: %llu, rank: %d, depth-final: %d, gamma_write_a_result_vertex: %d.\n", warp_id, global_warp_id, write_pos, rank, (uint32_t)(end_depths_in_block[warp_id] - 1), temp_nbr);
    }
#endif
                                }
                            }
                            if (lane_id == 0)
                            {
                                temp_ptr[warp_id][cur_depth] = temp_found_nbr_count;
                                temp_size[warp_id][cur_depth] = temp_found_nbr_count;
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            
        }

// #ifdef IS_DEBUGGING
//     if (global_warp_id < 2624) {
//         warp_print(lane_id, warp_id, global_warp_id, "before steal work.\n");
//     }
// #endif

        bool steal_success = false;
        // if (enable_work_stealing) {
        //     if (lane_id == 0)
        //     {
        //         steal_success = steal_work(warp_id, global_warp_id, mutex_this_block, start_depth, num_query_vertices, matching_order_idx_in_block, 
        //         end_depths_in_block, finished_nbr_count, total_nbr_count, temp_result_buffer, temp_ptr, cur_start_depth);
        //         // if (steal_success) {
        //         //     // if (global_warp_id < 2624){
        //         //         printf("warp_id: %d, %lld !!! steal_success !!! after an iteration of inner while-loop. \n", warp_id, global_warp_id);
        //         //     // }
        //         // }

        //         // if (warp_id == 0 && steal_success) {
        //         if (steal_success) {
        //             printf("warp_id: %d, %lld, cur_start_depth: %d, finished_nbr_count: %d, total_nbr_count: %d, steal_success! \n", warp_id, global_warp_id, (uint32_t)cur_start_depth, finished_nbr_count[warp_id][cur_start_depth], total_nbr_count[warp_id][cur_start_depth]);
        //         }
        //     }

        //     steal_success = __shfl_sync(0xffffffff, steal_success, 0);
        // }
        if (!steal_success) break;

        cur_start_depth = __shfl_sync(0xffffffff, cur_start_depth, 0);
    }
}

__global__ void GammaGetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t num_query_vertices
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long global_warp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    for (auto i = tid; i < res_size; i += num_threads)
    {
        uint8_t order_idx = gamma_get_matching_order_idx(res, i, num_query_vertices);
        uint8_t start_depth = get_enumerate_end_depth(order_idx);
        max_num_matches[i] = 1ul;
        const OrderPerEdge cur_order = C_ORDERS[order_idx];
        for (uint8_t j = start_depth; j < num_query_vertices; j++)
        {
            const uint8_t& pre_qv_idx = get_first_backward_neighbor_idx(cur_order, j);
            const uint8_t& pre_qe_idx = get_query_edge_idx(cur_order.vs_[pre_qv_idx], cur_order.vs_[j]);
            const uint32_t v = gamma_get_previous_result(cur_order, res, i, num_query_vertices, pre_qv_idx);
            
            max_num_matches[i] *= index.sizes_[pre_qe_idx][v];

// #ifdef IS_DEBUGGING
//     // if (warp_id == 0 && lane_id == 0)
//     if (lane_id == 0 && order_idx == 2)
//     {
//         printf("warp_id: %d, %lld, thread_id: %d, order_idx: %d, res: %llu, num_query_vertices: %d, start_depth: %d, cur_depth: %d, pre_qv_idx: %d, pre_qe_idx: %d, v: %d, index.sizes_[pre_qe_idx][v]: %d, max_num_matches[%d]: %lu.\n", warp_id, global_warp_id, tid, (uint32_t)order_idx, res, (uint32_t)num_query_vertices, (uint32_t)start_depth, (uint32_t)j, (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx, v, index.sizes_[pre_qe_idx][v], i, max_num_matches[i]);
//     }
// #endif

        }
    }

}

namespace {
__device__ __forceinline__ unsigned long long calculate_result_write_start_pointer(
        const unsigned long long previous_result_ptr, 
        const unsigned long long previous_result_size, 
        const uint8_t num_query_vertices) {
    return (previous_result_ptr + previous_result_size * (num_query_vertices + 1)) % C_RES_QUEUE.capacity_ ;
}
}  // namespace


__global__ void gammaEnumerateCartesianProductTree(
        const unsigned long long res,
        const unsigned long long res_size,
        const unsigned long *max_num_matches_prefix_sum,
        const unsigned long *max_num_total_matches,
        const RelationsGPU index,
        const uint8_t num_query_vertices,
        unsigned long long *new_res_size,
        unsigned long long *effective_num_dptr,
        const bool write_results) {
    const unsigned long long new_result_ptr = calculate_result_write_start_pointer(res, res_size, num_query_vertices);
    __shared__ uint32_t visited[NUM_WARP_PER_BLOCK][WARP_SIZE][MAX_QV_COUNT - 1];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long global_warp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    for (auto i = global_warp_id; i < DIV_CEIL(*max_num_total_matches, WARP_SIZE); i += num_warps)
    {
        const unsigned long ii = i * WARP_SIZE + lane_id;

        unsigned long long res_index = -1;
        uint32_t map_v = UINT32_MAX;
        unsigned long res_index_index = -1;

        // binary search
        bool found = true;
        bool need_exploration = true;

        if (ii >= *max_num_total_matches) found = false;

        uint8_t order_idx = UINT8_MAX;
        uint8_t start_depth = UINT8_MAX;
        OrderPerEdge cur_order;

        if (found)
        {
            res_index = lower_bound(max_num_matches_prefix_sum + 1, res_size, ii + 1);
            order_idx = gamma_get_matching_order_idx(res, res_index, num_query_vertices);
            cur_order = C_ORDERS[order_idx];
            start_depth = get_enumerate_end_depth(order_idx);
            if (start_depth >= num_query_vertices) {
                need_exploration = false;                
            }
        }

        if (found && need_exploration)
        {
            for (uint8_t k = 0u; k < start_depth; k++)
            {
                visited[warp_id][lane_id][k] = gamma_get_previous_result(cur_order, res, res_index, num_query_vertices, k);
            }
            
            res_index_index = ii - max_num_matches_prefix_sum[res_index];
        }

        __syncwarp();

        for (uint8_t j = start_depth; j < num_query_vertices; j++)
        {
            if (found && need_exploration)
            {
                const uint8_t& pre_qv_idx = get_first_backward_neighbor_idx(cur_order, j);
                const uint8_t& pre_qe_idx = get_query_edge_idx(cur_order.vs_[pre_qv_idx], cur_order.vs_[j]);
                const uint32_t v = gamma_get_previous_result(cur_order, res, res_index, num_query_vertices, pre_qv_idx);

                map_v = index.nbrs_[pre_qe_idx][v][res_index_index % index.sizes_[pre_qe_idx][v]];
                res_index_index /= index.sizes_[pre_qe_idx][v];

                for (uint8_t k = 0u; k < j; k++)
                {
                    if (map_v == visited[warp_id][lane_id][k])
                    {
                        found = false;
                        break;
                    }
                }
                if (!found) break;
                if (j != num_query_vertices - 1) visited[warp_id][lane_id][j] = map_v;
            }
        }

        uint32_t effective_num_cur_thread = 0;
        if (found) {
            effective_num_cur_thread = C_EQUIV_EDGE_COUNT[order_idx];
        }
        const uint32_t effective_num_cur_warp = __reduce_add_sync(0xffffffff, effective_num_cur_thread);

        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
        
        unsigned long long write_position;
        if (lane_id == 0)
        {
            write_position = atomicAdd(new_res_size, __popc(found_mask));
            atomicAdd(effective_num_dptr, effective_num_cur_warp);
        }
        write_position = __shfl_sync(0xffffffff, write_position, 0);
        // __syncwarp();

        if (write_results && found) {
            write_position += rank;
            gamma_write_matching_order_idx(new_result_ptr, write_position, num_query_vertices, order_idx);

            if (need_exploration){
                for (uint8_t j = 0u; j < num_query_vertices - 1; j++)
                {
                    uint32_t cur_result = visited[warp_id][lane_id][j];
                    gamma_write_a_result_vertex(cur_order, new_result_ptr, write_position, num_query_vertices, j, cur_result);
if (warp_id == 0 && rank == 0) {
    printf("warp_id: %d, %lld, \"need_exploration_loop\", cur_result: %d, j: %d, res_index: %llu, res_index_index: %lu, new_result_ptr: %llu, write_position: %llu, rank: %d.\n", warp_id, global_warp_id, cur_result, (uint32_t)j, res_index, res_index_index, new_result_ptr, write_position, rank);
}
                }
if (warp_id == 0 && rank == 0) {
    printf("warp_id: %d, %lld, \"need_exploration_outside_loop\", cur_result: %d, res_index: %llu, res_index_index: %lu, new_result_ptr: %llu, write_position: %llu, rank: %d.\n", warp_id, global_warp_id, map_v, res_index, res_index_index, new_result_ptr, write_position, rank);
}
                gamma_write_a_result_vertex(cur_order, new_result_ptr, write_position, num_query_vertices, num_query_vertices - 1, map_v);
            }
            else
            {
                for (uint8_t j = 0u; j < num_query_vertices; j++)
                {
                    uint32_t cur_result = gamma_get_previous_result(cur_order, res, res_index, num_query_vertices, j);
if (warp_id == 0 && rank == 0) {
    printf("warp_id: %d, %lld, cur_result: %d, j: %d, res_index: %llu, res_index_index: %lu, new_result_ptr: %llu, write_position: %llu, rank: %d.\n", warp_id, global_warp_id, cur_result, (uint32_t)j, res_index, res_index_index, new_result_ptr, write_position, rank);
}
                    gamma_write_a_result_vertex(cur_order, new_result_ptr, write_position, num_query_vertices, j, cur_result);
                }
            }
        }
    }
}