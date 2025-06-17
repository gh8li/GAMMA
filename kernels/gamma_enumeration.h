#ifndef KERNELS_GAMMA_ENUMERATION_H
#define KERNELS_GAMMA_ENUMERATION_H

#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/match_gpu.h"

__global__ void gamma_write_initial_partial_results(
    RelationsGPU local_index_gpu,  // In fact, local index are passed in.
    const uint8_t edge_idx,
    const uint8_t edge_list_idx,
    const uint8_t num_query_vertices,
    unsigned long long int new_res_start,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
);

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
);

__global__ void GammaGetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t num_query_vertices
);

__global__ void gammaEnumerateCartesianProductTree(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t num_query_vertices,
    unsigned long long *new_res_size,
    unsigned long long *effective_num_dptr,
    const bool write_results=false
);

#endif  //KERNELS_GAMMA_ENUMERATION_H