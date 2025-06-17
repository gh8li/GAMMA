#include <cstdint>

#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/match_gpu.h"


__global__ void generateMaxNumMatchesTree2(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    const uint8_t& pre_qv_idx1 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qv_idx2 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 1]];
    const uint8_t& pre_qe_idx1 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx1] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qe_idx2 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx2] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 1]];

    for (auto i = tid; i < res_size; i += num_threads)
    {
        const uint32_t& v1 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - 2) + pre_qv_idx1) % C_RES_QUEUE.capacity_];
        const uint32_t& v2 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - 2) + pre_qv_idx2) % C_RES_QUEUE.capacity_];

        max_num_matches[i] = (long)index.sizes_[pre_qe_idx1][v1] * index.sizes_[pre_qe_idx2][v2];
    }
}

__global__ void generateMaxNumMatchesTree3(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    const uint8_t& pre_qv_idx1 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 3]];
    const uint8_t& pre_qv_idx2 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qv_idx3 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 1]];
    const uint8_t& pre_qe_idx1 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx1] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 3]];
    const uint8_t& pre_qe_idx2 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx2] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qe_idx3 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx3] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 1]];

    for (auto i = tid; i < res_size; i += num_threads)
    {
        const uint32_t& v1 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - 3) + pre_qv_idx1) % C_RES_QUEUE.capacity_];
        const uint32_t& v2 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - 3) + pre_qv_idx2) % C_RES_QUEUE.capacity_];
        const uint32_t& v3 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - 3) + pre_qv_idx3) % C_RES_QUEUE.capacity_];

        max_num_matches[i] = (long)index.sizes_[pre_qe_idx1][v1] * index.sizes_[pre_qe_idx2][v2] * index.sizes_[pre_qe_idx3][v3];
    }
}

__global__ void enumerateCartesianProductTree2(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    const uint8_t& pre_qv_idx1 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qv_idx2 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 1]];
    const uint8_t& pre_qe_idx1 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx1] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qe_idx2 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx2] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 1]];

    unsigned long long res_index;
    uint32_t map_v1, map_v2;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        // binary search
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);
            const uint32_t& v1 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 2) + pre_qv_idx1) % C_RES_QUEUE.capacity_];
            const uint32_t& v2 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 2) + pre_qv_idx2) % C_RES_QUEUE.capacity_];

            const long res_index_index = ii - max_num_matches[res_index];
            map_v1 = index.nbrs_[pre_qe_idx1][v1][res_index_index / index.sizes_[pre_qe_idx2][v2]];
            map_v2 = index.nbrs_[pre_qe_idx2][v2][res_index_index % index.sizes_[pre_qe_idx2][v2]];
            if (map_v1 == map_v2) found = false;
        }
        __syncwarp();
        if (found)
        {
            for (uint8_t j = 0u; j < C_QV_COUNT - 2; j++)
            {
                if (
                    C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 2) + j) % C_RES_QUEUE.capacity_] == map_v1 || 
                    C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 2) + j) % C_RES_QUEUE.capacity_] == map_v2)
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp();
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}

__global__ void enumerateCartesianProductTree3(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    const uint8_t& pre_qv_idx1 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 3]];
    const uint8_t& pre_qv_idx2 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qv_idx3 = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - 1]];
    const uint8_t& pre_qe_idx1 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx1] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 3]];
    const uint8_t& pre_qe_idx2 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx2] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 2]];
    const uint8_t& pre_qe_idx3 = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx3] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - 1]];

    unsigned long long res_index;
    uint32_t map_v1, map_v2, map_v3;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        // binary search
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);
            const uint32_t& v1 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + pre_qv_idx1) % C_RES_QUEUE.capacity_];
            const uint32_t& v2 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + pre_qv_idx2) % C_RES_QUEUE.capacity_];
            const uint32_t& v3 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + pre_qv_idx3) % C_RES_QUEUE.capacity_];

            const long res_index_index = ii - max_num_matches[res_index];
            map_v1 = index.nbrs_[pre_qe_idx1][v1][res_index_index / index.sizes_[pre_qe_idx3][v3] / index.sizes_[pre_qe_idx2][v2]];
            map_v2 = index.nbrs_[pre_qe_idx2][v2][(res_index_index / index.sizes_[pre_qe_idx3][v3]) % index.sizes_[pre_qe_idx2][v2]];
            map_v3 = index.nbrs_[pre_qe_idx3][v3][res_index_index % index.sizes_[pre_qe_idx3][v3]];
            if (map_v1 == map_v2 || map_v1 == map_v3 || map_v2 == map_v3) found = false;
        }
        __syncwarp();
        if (found)
        {
            for (uint8_t j = 0u; j < C_QV_COUNT - 3; j++)
            {
                if (
                    C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + j) % C_RES_QUEUE.capacity_] == map_v1 || 
                    C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + j) % C_RES_QUEUE.capacity_] == map_v2 || 
                    C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - 3) + j) % C_RES_QUEUE.capacity_] == map_v3)
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp();
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}


__global__ void GetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (auto i = tid; i < res_size; i += num_threads)
    {
        max_num_matches[i] = 1ul;
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
            const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
            const uint32_t& v = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capacity_];

            max_num_matches[i] *= index.sizes_[pre_qe_idx][v];
        }
    }
}

__global__ void enumerateCartesianProductTree(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size,
    const uint8_t num_vs
) {
    __shared__ uint32_t visited[NUM_WARP_PER_BLOCK][WARP_SIZE][MAX_QV_COUNT - 1];
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    unsigned long long res_index;
    uint32_t map_v;
    unsigned long res_index_index;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        // binary search
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);

            res_index_index = ii - max_num_matches[res_index];
            for (uint8_t k = 0u; k < C_QV_COUNT - num_vs; k++)
            {
                visited[warp_id][lane_id][k] = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + k) % C_RES_QUEUE.capacity_];
            }
        }
        __syncwarp();
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            if (found)
            {
                const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
                const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
                const uint32_t& v = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capacity_];

                map_v = index.nbrs_[pre_qe_idx][v][res_index_index % index.sizes_[pre_qe_idx][v]];
                res_index_index /= index.sizes_[pre_qe_idx][v];

                for (uint8_t k = 0u; k < C_QV_COUNT - num_vs + j; k++)
                {
                    if (map_v == visited[warp_id][lane_id][k])
                    {
                        found = false;
                        break;
                    }
                }
                if (!found) break;
                if (j != num_vs - 1) visited[warp_id][lane_id][C_QV_COUNT - num_vs + j] = map_v;
            }
        }
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}
