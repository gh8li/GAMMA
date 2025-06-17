#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/match_gpu.h"


__global__ void setCapacities(
    const CSR_GPU csr_gpu,
    uint32_t *capacity
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < csr_gpu.vs_size_; i += num_threads)
    {
        // auto off_upper = csr_gpu.offs_[i + 1];
        // auto off_lower = csr_gpu.offs_[i];
        // auto cur_vertex = csr_gpu.vs_[i];
        // if (cur_vertex >= C_DV_COUNT){
        //     printf("csr_gpu.vs_[i]: %u ", cur_vertex);
        //     // printf("csr_gpu.vs_[i]: %u", csr_gpu.vs_[i]);
        // }
        // capacity[C_DV_COUNT - 1] = 100;
        capacity[csr_gpu.vs_[i]] = csr_gpu.offs_[i + 1] - csr_gpu.offs_[i];
    }
}

__global__ void roundCapacities(
    uint32_t* array,
    const uint32_t size
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
    {
        array[i] = max(MIN_NBR_SIZE, (uint32_t)exp2f(ceilf(log2f(array[i]) + 1)));
    }
}

__global__ void roundCapacities(
    uint32_t *array,
    const uint32_t size,
    const uint32_t *label_array,
    const uint32_t q_label
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
    {
        array[i] = label_array[i] == q_label ? max(MIN_NBR_SIZE, (uint32_t)exp2f(ceilf(log2f(array[i]) + 1))) : 0u;
    }
}

__global__ void setNeighborPointers(
    uint32_t *base,
    const uint32_t *offsets,
    const uint32_t size,
    uint32_t **ptr
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
    {
        ptr[i] = base + offsets[i];
    }
}

__global__ void addTriesToGraph(
    const CSR_GPU csr_gpu,
    RelationsGPU data,
    const uint32_t idx,
    MemPool<uint32_t> nbr_mem_pool
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    const uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    __shared__ uint32_t insert_num[NUM_WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t target_pos[NUM_WARP_PER_BLOCK][WARP_SIZE];
    // Allocate WarpScan shared memory for 4 warps
    __shared__ typename cub::WarpScan<uint32_t>::TempStorage temp_storage[NUM_WARP_PER_BLOCK];


    for (uint32_t i = gwarp_id; i < csr_gpu.vs_size_; i += num_warps)
    {
        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capacity_) return;

        const uint32_t& v = csr_gpu.vs_[i];

        // merge data.nbrs_[idx][v] and csr_gpu.nbrs_[csr_gpu.offs_[i]: csr_gpu.offs_[i + 1]]
        const uint32_t* a = data.nbrs_[idx][v];
        const uint32_t* b = csr_gpu.nbrs_ + csr_gpu.offs_[i];
        // Doubt: If c_size < data.capacity_[idx][v], will writing to c[x] spoil the value in a[x]? (Seems not)
        uint32_t *c = data.nbrs_[idx][v];
        const uint32_t a_size = data.sizes_[idx][v];
        const uint32_t b_size = csr_gpu.offs_[i + 1] - csr_gpu.offs_[i];
        const uint32_t c_size = a_size + b_size;
        if (c_size >= data.capacity_[idx][v])
        {
            if (lane_id == 0)
            {
                //printf("reallocate %d %d ", idx, v);
                unsigned long long int new_capacity = max((size_t)exp2(ceilf(log2f(c_size) + 1)), 8ul);
                // (Solved) Doubt: Why the old value of nbr_mem_pool.occupy_ is new_array_start?
                // new_array_start is in fact the new_array_start_offset.
                unsigned long long int new_array_start = atomicAdd(nbr_mem_pool.occupy_, new_capacity);
                if (new_array_start + new_capacity < nbr_mem_pool.capacity_)
                {
                    c = data.nbrs_[idx][v] = nbr_mem_pool.array_ + new_array_start;
                    data.capacity_[idx][v] = new_capacity;
                }
                else
                {
                    printf("out of memory!\n");
                }
            }
            c = (uint32_t*)__shfl_sync(0xffffffff, (unsigned long)c, 0, 64);
        }
        __syncwarp();
        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capacity_) return;

        data.sizes_[idx][v] = c_size;

        uint32_t a_start = a_size - 1, a_end;
        uint32_t b_start = b_size - 1, b_end;
        uint32_t c_start = c_size - 1;

        // Doubt: Does a_start < a_size mean a_start >= 0 ? Same for b_start. (a_start has been assigned with a_size -1)
        while (a_start < a_size || b_start < b_size)
        {
            if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capacity_) return;

            if (a_start >= a_size)
            {
                // write the remaining b to c
                // Doubt: Seems that usint32_t j will overflow (smaller than 0). Does j < b_size mean j >= 0 ?
                for (uint32_t j = b_start - lane_id; j < b_size; j -= WARP_SIZE)
                {
                    c[j] = b[j];
                }
                b_start = UINT32_MAX;
                continue;
            }
            if (b_start >= b_size)
            {
                // write the remaining a to c
                for (uint32_t j = a_start - lane_id; j < a_size; j -= WARP_SIZE)
                {
                    c[j] = a[j];
                }
                a_start = UINT32_MAX;
                continue;
            }
            insert_num[warp_id][lane_id] = 0u;
            target_pos[warp_id][lane_id] = 0u;

            // include both start and end
            // at first loop, a_end == a_size - WAPR_SIZE or 0, b_end = b_size - WARP_SIZE or 0
            a_end = a_start + 1 - WARP_SIZE < a_size ? a_start + 1 - WARP_SIZE : 0;
            b_end = b_start + 1 - WARP_SIZE < b_size ? b_start + 1 - WARP_SIZE : 0;
            //if (lane_id == 0) printf("a_start=%d, a_end=%d, b_start=%d, b_end=%d, c_start=%d\n", a_start, a_end, b_start, b_end, c_start);

            // (Solved) Doubt: Why compare two vertex IDs? Answer: result array should be sorted by vertex IDs.
            if (a[a_end] < b[b_end])
            {
                // write the entire b[b_start:b_end] and part of a[a_start:a_end] to c
                //if (lane_id == 0) printf("case 3a\n");
                // at first loop, b_start - b_end == WARP_SIZE - 1 or b_start
                if (lane_id <= b_start - b_end)
                {
                    // at first loop, a_start + 1 - a_end == WARP_SIZE or a_size (smaller one)
                    // insert_pos range: [0, WARP_SIZE or a_size (smaller one)]
                    uint32_t insert_pos = lower_bound(a + a_end, a_start + 1 - a_end, b[b_start - lane_id]);
                    // insert_pos range: [0, WARP_SIZE or a_size (smaller one)], insert_pos is reversed.
                    insert_pos = a_start + 1 - a_end - insert_pos;
                    //printf("find %d at %d\n", b[b_start - lane_id], insert_pos);
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                // insert_num[warp_id] <- exclusive_sum(insert_num[warp_id])
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);
                //if (lane_id == 0) printf("insert_num ");
                //printf("%d ", insert_num[warp_id][lane_id]);
                //if (lane_id == 0) printf("target_pos ");
                //printf("%d ", target_pos[warp_id][lane_id]);

                bool write_a = true;
                if (lane_id <= a_start - a_end && a[a_start - lane_id] > b[b_end])
                {
                    //printf("write a c[%d]=%d\n", c_start - insert_num[warp_id][lane_id + 1] - lane_id, a[a_start - lane_id]);
                    write_a = false;
                    // at first loop, if lane_id + 1 == WARP_SIZE (largest possible lane_id), a[a_start - lane_id] > b[b_end] will not pass
                    // because a_start - lane_id == a_end, and a[a_end] < b[b_end] according to the condition of outer if block.
                    // Thus, `insert_num[warp_id][lane_id + 1] - lane_id]` is safe.
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                if (lane_id <= b_start - b_end)
                {
                    //printf("write b c[%d]=%d\n", c_start - target_pos[warp_id][lane_id] - lane_id, b[b_start - lane_id]);
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = b[b_start - lane_id];
                }
                c_start = c_start - b_start + b_end - 1;
                b_start = b_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                //if (lane_id == 0) printf("next_first=%d \n", next_first);
                __syncwarp();
                c_start = c_start - next_first + 1;
                a_start = a_start - next_first + 1;
            }
            else
            {
                // write the entire a[a_start:a_end] and part of b[b_start:b_end] to c
                //if (lane_id == 0) printf("case 3b\n");
                if (lane_id <= a_start - a_end)
                {
                    uint32_t insert_pos = lower_bound(b + b_end, b_start + 1 - b_end, a[a_start - lane_id]);
                    insert_pos = b_start + 1 - b_end - insert_pos;
                    //printf("find %d at %d\n", a[a_start - lane_id], insert_pos);
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);
                //if (lane_id == 0) printf("insert_num ");
                //printf("%d ", insert_num[warp_id][lane_id]);
                //if (lane_id == 0) printf("target_pos ");
                //printf("%d ", target_pos[warp_id][lane_id]);

                if (lane_id <= a_start - a_end)
                {
                    //printf("write a c[%d]=%d\n", c_start - target_pos[warp_id][lane_id] - lane_id, a[a_start - lane_id]);
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                bool write_a = true;
                if (lane_id <= b_start - b_end && b[b_start - lane_id] > a[a_end])
                {
                    //printf("write b c[%d]=%d\n", c_start - insert_num[warp_id][lane_id + 1] - lane_id, b[b_start - lane_id]);
                    write_a = false;
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = b[b_start - lane_id];
                }
                c_start = c_start - a_start + a_end - 1;
                a_start = a_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                //if (lane_id == 0) printf("nxt_first=%d \n", next_first);
                __syncwarp();
                c_start = c_start - next_first + 1;
                b_start = b_start - next_first + 1;
            }
        }
    }
}

__global__ void statisticIndex(
    const RelationsGPU global_index,
    const uint32_t idx,
    uint32_t *sum,
    uint32_t *count
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < C_DV_COUNT; i += num_threads)
    {
        if (global_index.sizes_[idx][i] != 0)
        {
            atomicAdd(sum, global_index.sizes_[idx][i]);
            atomicAdd(count, 1u);
        }
    }
    /*// Allocate WarpReduce shared memory for 4 warps
    __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage[NWARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    for (auto i = gwarp_id; i < DIV_CEIL(C_DV_COUNT, WARP_SIZE); i += num_warps)
    {
        const unsigned long ii = i * WARP_SIZE + lane_id;
        uint32_t thread_sum = ii < C_DV_COUNT ? global_index.sizes_[idx][ii] : 0u;
        bool thread_count = thread_sum != 0u ? true : false;

        uint32_t warp_sum = cub::WarpReduce<uint32_t>(temp_storage[warp_id]).Sum(thread_sum);
        if (lane_id == 0u) atomicAdd(sum, warp_sum);
        uint32_t warp_count = __popc(__ballot_sync(0xffffffff, thread_count));
        if (lane_id == 0u) atomicAdd(count, warp_count);
    }*/
}

__global__ void getGlobalCandidates(
    const RelationsGPU data,
    const uint32_t idx, // current edge_idx
    const CSR_GPU csr_gpu, // CSR representation of data graph edges that has the same (src_label, edge_label, dst_label) with the query graph edge of current edge_idx
    uint32_t *cand_bits,
    bool *cand_flag,
    const uint32_t u // source vertex of current edge_idx
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < csr_gpu.vs_size_; i += num_threads)
    {
        uint32_t& dv = csr_gpu.vs_[i];
        bool pass = true;
        // Apply NLF not only to current edge_idx, but also all edges of u
        for (uint32_t j = C_QV_OFFS[u]; j < C_QV_OFFS[u + 1]; j++)
        {
            // Doubt (solved): (1) Why + csr_gpu.offs_[i + 1] - csr_gpu.offs_[i]? Seems it is redundant. 
            // Answer: `data` contains initail edges and `csr_gpu` contains updated edges.
            // Doubt: (2) Seems all values of sizes_[x][x] are zero. (When invoked at the inital steps.)
            // When invoked at each update batch, this is OK.
            if ((j == idx ? data.sizes_[j][dv] + csr_gpu.offs_[i + 1] - csr_gpu.offs_[i] : data.sizes_[j][dv]) < C_NLF[j])
            {
                pass = false;
                break;
            }
        }
        if (pass && ((cand_bits[dv / 32] & (1u << (dv % 32))) == 0))
        {
            atomicOr(&cand_bits[dv / 32], 1u << (dv % 32));
            cand_flag[i] = true;
        }
        __syncwarp();
    }
}

__global__ void getGlobalCandidateEdgesCount(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < new_cand_size; i += num_warps)
    {
        const uint32_t v = new_cand[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        /*for (uint32_t j = lane_id; j < data.sizes_[idx][v]; j+= WARP_SIZE)
        {
            const uint32_t nbr = data.nbrs_[idx][v][j];
            if ((other_cand_bits[nbr / 32] & (1 << (nbr % 32))) > 0)
            {
                atomicAdd(&temp_num[warp_id], 1u);
            }
        }*/
        for (uint32_t j = 0; j < DIV_CEIL(data.sizes_[idx][v], WARP_SIZE); j++)
        {
            const uint32_t jj = j * WARP_SIZE + lane_id;
            const uint32_t warp_sum = __popc(__ballot_sync(
                0xffffffff, 
                jj < data.sizes_[idx][v] &&
                (other_cand_bits[data.nbrs_[idx][v][jj] / 32] & (1 << (data.nbrs_[idx][v][jj] % 32))) > 0
            ));
            if (lane_id == 0u)
            {
                atomicAdd(&temp_num[warp_id], warp_sum);
            }
            __syncwarp();
        }
        __syncwarp();
        if (lane_id == 0) cand_e_count[i] = temp_num[warp_id];
        __syncwarp();
    }
}

__global__ void getGlobalCandidateEdgesWrite(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count_prefix_sum,
    uint32_t *relation_u,
    uint32_t *relation_uu
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < new_cand_size; i += num_warps)
    {
        const uint32_t& v = new_cand[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        for (uint32_t j = 0; j < DIV_CEIL(data.sizes_[idx][v], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < data.sizes_[idx][v])
            {
                nbr = data.nbrs_[idx][v][j * WARP_SIZE + lane_id];
                if ((other_cand_bits[nbr / 32] & (1 << (nbr % 32))) > 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[cand_e_count_prefix_sum[i] + write_pos[warp_id] + rank] = v;
                relation_uu[cand_e_count_prefix_sum[i] + write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void filterRelevantCount(
    const CSR_GPU input,
    CSR_GPU output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            output.offs_[i] = 0u;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                const uint32_t jj = j * WARP_SIZE + lane_id;
                const uint32_t warp_sum = __popc(__ballot_sync(
                    0xffffffff, 
                    jj < input.offs_[i + 1] - input.offs_[i] &&
                    (second_candidates[input.nbrs_[jj + input.offs_[i]] / 32] & 1 << (input.nbrs_[jj + input.offs_[i]] % 32)) > 0
                ));
                if (lane_id == 0u)
                {
                    atomicAdd(&temp_num[warp_id], warp_sum);
                }
                __syncwarp();
            }
            __syncwarp();
            if (lane_id == 0) output.offs_[i] = temp_num[warp_id];
            __syncwarp();
        }
    }
}

__global__ void filterRelevantWrite(
    const CSR_GPU input,
    CSR_GPU output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            continue;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                bool found = false;
                if (j * WARP_SIZE + lane_id < input.offs_[i + 1] - input.offs_[i])
                {
                    nbr = input.nbrs_[j * WARP_SIZE + lane_id + input.offs_[i]];
                    if ((second_candidates[nbr / 32] & 1 << (nbr % 32)) > 0)
                    {
                        found = true;
                    }
                }
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    output.nbrs_[output.offs_[i] + write_pos[warp_id] + rank] = nbr;
                    if (rank == 0)
                    {
                        write_pos[warp_id] += __popc(found_mask);
                    }
                }
                __syncwarp();
            }
            __syncwarp();
        }
    }
}

// for building local index

__global__ void edgeList2RelationCount(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            output.sizes_[idx][dv] = 0u;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                const uint32_t jj = j * WARP_SIZE + lane_id;
                const uint32_t warp_sum = __popc(__ballot_sync(
                    0xffffffff, 
                    jj < input.offs_[i + 1] - input.offs_[i] &&
                    (second_candidates[input.nbrs_[jj + input.offs_[i]] / 32] & 1 << (input.nbrs_[jj + input.offs_[i]] % 32)) > 0
                ));
                if (lane_id == 0u)
                {
                    atomicAdd(&temp_num[warp_id], warp_sum);
                }
                __syncwarp();
            }
            __syncwarp();
            if (lane_id == 0) output.sizes_[idx][dv] = temp_num[warp_id];
            __syncwarp();
        }
    }
}

__global__ void edgeList2RelationWrite(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            continue;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                bool found = false;
                if (j * WARP_SIZE + lane_id < input.offs_[i + 1] - input.offs_[i])
                {
                    nbr = input.nbrs_[j * WARP_SIZE + lane_id + input.offs_[i]];
                    if ((second_candidates[nbr / 32] & 1 << (nbr % 32)) > 0)
                    {
                        found = true;
                    }
                }
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    output.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                    if (rank == 0)
                    {
                        write_pos[warp_id] += __popc(found_mask);
                    }
                }
                __syncwarp();
            }
            __syncwarp();
        }
    }
}

__global__ void edgeList2RelationCount_v2(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            output.sizes_[idx][dv] = 0u;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                const uint32_t jj = j * WARP_SIZE + lane_id;
                const uint32_t warp_sum = __popc(__ballot_sync(
                    0xffffffff, 
                    jj < input.offs_[i + 1] - input.offs_[i] &&
                    (second_candidates[input.nbrs_[jj + input.offs_[i]] / 32] & 1 << (input.nbrs_[jj + input.offs_[i]] % 32)) > 0
                ));
                if (lane_id == 0u)
                {
                    atomicAdd(&temp_num[warp_id], warp_sum);
                }
                __syncwarp();
            }
            __syncwarp();
            if (lane_id == 0) output.sizes_[idx][dv] = temp_num[warp_id];
            __syncwarp();
        }
    }
}

__global__ void setLocalBitmap(
    RelationsGPU output,
    const uint8_t idx,
    uint32_t *local_candidates
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < DIV_CEIL(C_DV_COUNT, WARP_SIZE); i += num_warps)
    {
        bool found = false;
        if (i * WARP_SIZE + lane_id < C_DV_COUNT && output.sizes_[idx][i * WARP_SIZE + lane_id] > 0)
        {
            found = true;
        }
        uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0) local_candidates[i] = found_mask;
        __syncwarp();
    }
}

__global__ void edgeList2RelationWrite_v2(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *second_candidates,
    const uint32_t *local_first_candidates
) {
    // lgh: idx is edge_idx
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        if ((local_first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            continue;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                bool found = false;
                if (j * WARP_SIZE + lane_id < input.offs_[i + 1] - input.offs_[i])
                {
                    nbr = input.nbrs_[j * WARP_SIZE + lane_id + input.offs_[i]];
                    if ((second_candidates[nbr / 32] & 1 << (nbr % 32)) > 0)
                    {
                        found = true;
                    }
                }
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    output.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                    if (rank == 0)
                    {
                        write_pos[warp_id] += __popc(found_mask);
                    }
                }
                __syncwarp();
            }
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesBackward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t found[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[backward_index][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) found[warp_id] = false;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[backward_index][dv], WARP_SIZE); j++)
        {
            bool local_found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[backward_index][dv])
            {
                nbr = global_index.nbrs_[backward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    local_found = true;
                }
            }
            uint32_t found_total = __ballot_sync(0xffffffff, local_found);
            if (lane_id == 0 && found_total) found[warp_id] = true;
            __syncwarp();
            if (found[warp_id]) break;
        }

        if (found[warp_id])
        {
            if (lane_id == 0) cum_bn[dv] += 1u;
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesForward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            continue;
        }
        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[forward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[forward_index][dv])
            {
                auto nbr = global_index.nbrs_[forward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                // Doubt: If we us atomicCAS below, seems that we do not need `&& cum_bn[nbr] == pre_max_cum`
                if (global_index.sizes_[backward_index][nbr] != 0u && cum_bn[nbr] == pre_max_cum)
                {
                    //atomicMax(&cum_bn[nbr], pre_max_cum + 1u);
                    atomicCAS(&cum_bn[nbr], pre_max_cum, pre_max_cum + 1u);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesBackward_v2(
    const RelationsGPU global_index,
    const uint32_t *local_backward_candidates,
    const uint8_t backward_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t found[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[backward_index][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) found[warp_id] = false;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[backward_index][dv], WARP_SIZE); j++)
        {
            bool local_found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[backward_index][dv])
            {
                nbr = global_index.nbrs_[backward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if ((local_backward_candidates[nbr / 32] & (1u << (nbr % 32))) != 0)
                {
                    local_found = true;
                }
            }
            uint32_t found_total = __ballot_sync(0xffffffff, local_found);
            if (lane_id == 0 && found_total) found[warp_id] = true;
            __syncwarp();
            if (found[warp_id]) break;
        }

        if (found[warp_id])
        {
            if (lane_id == 0) cum_bn[dv] += 1u;
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesForward_v2(
    const RelationsGPU global_index,
    const uint32_t *local_backward_candidates,
    const uint8_t forward_index,
    const uint8_t backward_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((local_backward_candidates[dv / 32] & (1u << (dv % 32))) == 0)
        {
            continue;
        }
        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[forward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[forward_index][dv])
            {
                auto nbr = global_index.nbrs_[forward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (global_index.sizes_[backward_index][nbr] != 0u && cum_bn[nbr] == pre_max_cum)
                {
                    //atomicMax(&cum_bn[nbr], pre_max_cum + 1u);
                    atomicCAS(&cum_bn[nbr], pre_max_cum, pre_max_cum + 1u);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void setLocalBitmap(
    const uint32_t *cum_bn,
    uint32_t *local_candidates,
    uint32_t pre_max_cum
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < DIV_CEIL(C_DV_COUNT, WARP_SIZE); i += num_warps)
    {
        bool found = false;
        if (i * WARP_SIZE + lane_id < C_DV_COUNT && cum_bn[i * WARP_SIZE + lane_id] == pre_max_cum)
        {
            found = true;
        }
        uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0) local_candidates[i] = found_mask;
        __syncwarp();
    }
}

__global__ void buildLocalRelationNew2OldCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[idx][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (rank == 0)
                {
                    temp_num[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationNew2OldWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[idx][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capacity_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void buildLocalRelationOld2NewCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (cum_bn[nbr] == pre_max_cum)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (rank == 0)
                {
                    temp_num[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationOld2NewWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (cum_bn[nbr] == pre_max_cum)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capacity_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void buildLocalRelationNew2OldCount_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((local_forward_candidates[dv / 32] & (1u << (dv % 32))) == 0u)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if ((local_backward_candidates[nbr / 32] & (1u << (nbr % 32))) != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (rank == 0)
                {
                    temp_num[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationNew2OldWrite_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((local_forward_candidates[dv / 32] & (1u << (dv % 32))) == 0u)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if ((local_backward_candidates[nbr / 32] & (1u << (nbr % 32))) != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capacity_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void buildLocalRelationOld2NewCount_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates
) {
    __shared__ uint32_t temp_num[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((local_backward_candidates[dv / 32] & (1u << (dv % 32))) == 0u)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if ((local_forward_candidates[nbr / 32] & (1u << (nbr % 32))) != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (rank == 0)
                {
                    temp_num[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationOld2NewWrite_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NUM_WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((local_backward_candidates[dv / 32] & (1u << (dv % 32))) == 0u)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if ((local_forward_candidates[nbr / 32] & (1u << (nbr % 32))) != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capacity_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void mapTrieToRelation(
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *vs,
    const uint32_t *offs,
    const uint32_t num_items
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < num_items; i += num_threads)
    {
        local_index.sizes_[idx][vs[i]] = offs[i];
    }
}