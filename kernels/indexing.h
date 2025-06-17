#ifndef KERNELS_INDEXING
#define KERNELS_INDEXING

#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/match_gpu.h"

__global__ void setCapacities(
    const CSR_GPU csr_gpu,
    uint32_t *capacity
);

__global__ void roundCapacities(
    uint32_t *array,
    const uint32_t size
);

__global__ void roundCapacities(
    uint32_t *array,
    const uint32_t size,
    const uint32_t *label_array,
    const uint32_t q_label
);

__global__ void setNeighborPointers(
    uint32_t *base,
    const uint32_t *offsets,
    const uint32_t size,
    uint32_t **ptr
);

__global__ void addTriesToGraph(
    const CSR_GPU csr_gpu,
    RelationsGPU data,
    const uint32_t idx,
    MemPool<uint32_t> nbr_mem_pool
);

__global__ void statisticIndex(
    const RelationsGPU data,
    const uint32_t idx,
    uint32_t *sum,
    uint32_t *count
);

__global__ void getGlobalCandidates(
    const RelationsGPU data,
    const uint32_t idx,
    const CSR_GPU csr_gpu,
    uint32_t *cand_bits,
    bool *cand_flag,
    const uint32_t u
);

__global__ void getGlobalCandidateEdgesCount(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count
);

__global__ void getGlobalCandidateEdgesWrite(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count_prefix_sum,
    uint32_t *relation_u,
    uint32_t *relation_uu
);

__global__ void filterRelevantCount(
    const CSR_GPU input,
    CSR_GPU output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void filterRelevantWrite(
    const CSR_GPU input,
    CSR_GPU output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void edgeList2RelationCount(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void edgeList2RelationWrite(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void edgeList2RelationCount_v2(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void setLocalBitmap(
    RelationsGPU output,
    const uint8_t idx,
    uint32_t *local_candidates
);

__global__ void edgeList2RelationWrite_v2(
    const CSR_GPU input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *second_candidates,
    const uint32_t *local_first_candidates
);

__global__ void getLocalCandidatesBackward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void getLocalCandidatesForward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void getLocalCandidatesBackward_v2(
    const RelationsGPU global_index,
    const uint32_t *local_backward_candidates,
    const uint8_t backward_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void getLocalCandidatesForward_v2(
    const RelationsGPU global_index,
    const uint32_t *local_backward_candidates,
    const uint8_t forward_index,
    const uint8_t backward_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void setLocalBitmap(
    const uint32_t *cum_bn,
    uint32_t *local_candidates,
    uint32_t pre_max_cum
);

__global__ void buildLocalRelationNew2OldCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void buildLocalRelationNew2OldWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
);

__global__ void buildLocalRelationOld2NewCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void buildLocalRelationOld2NewWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
);

__global__ void buildLocalRelationNew2OldCount_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates
);

__global__ void buildLocalRelationNew2OldWrite_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates,
    uint32_t *relation_u
);

__global__ void buildLocalRelationOld2NewCount_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates
);

__global__ void buildLocalRelationOld2NewWrite_v2(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *local_forward_candidates,
    const uint32_t *local_backward_candidates,
    uint32_t *relation_u
);

__global__ void mapTrieToRelation(
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *vs,
    const uint32_t *offs,
    const uint32_t num_items
);

#endif
