
#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/match_gpu.h"

__global__ void generateMaxNumMatchesTree2(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi
);

__global__ void generateMaxNumMatchesTree3(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi
);

__global__ void enumerateCartesianProductTree2(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size
);

__global__ void enumerateCartesianProductTree3(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size
);

__global__ void GetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs
);

__global__ void enumerateCartesianProductTree(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size,
    const uint8_t num_vs
);
