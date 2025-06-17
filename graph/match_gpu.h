#ifndef GRAPH_MATCH_GPU
#define GRAPH_MATCH_GPU

#include <cstdint>
#include <vector>

#include "graph/graph.h"
#include "graph/plan.h"
#include "utils/automorphism.h"
#include "utils/config.h"
#include "utils/mem_pool.h"
#include "utils/types.h"

extern __constant__ OrderPerEdge C_INDEXING_ORDERS[MAX_QE_COUNT];
extern __constant__ OrderPerEdge C_ORDERS[MAX_QE_COUNT];
extern __constant__ uint8_t C_EQUIV_EDGE_COUNT[MAX_QE_COUNT];
extern __constant__ uint8_t C_NUM_EQUIV_GROUPS;
extern __constant__ uint8_t C_NON_TAIL_LEAF_DEPTHS[MAX_QE_COUNT];

struct CSR_GPU {
    uint32_t vs_size_;
    uint32_t es_size_;
    uint32_t *vs_;
    uint32_t *offs_;
    uint32_t *nbrs_;
};

struct CSR_GPU_Capacity {
    uint32_t vs_capacity_;
    uint32_t off_capacity_;
    uint32_t es_capacity_;
};

class RelationsGPU {
  public:
    // edge_idx -> an array of pointers, e.g. *nbr_[DV_COUNT + 1], 
    // each pointer points to a segment in MatchGPU::data_graph_all_nbrs_ or MatchGPU::index_all_nbrs_
    uint32_t **nbrs_[MAX_QE_COUNT * 2];
    bool **nbrs_is_update_[MAX_QE_COUNT * 2];

    // edge_idx -> an array, e.g. capacity_xxx[DV_COUNT + 1]
    uint32_t *capacity_[MAX_QE_COUNT * 2];
    // edge_idx -> an array, e.g. sizes_xxx[DV_COUNT + 1]
    uint32_t *sizes_[MAX_QE_COUNT * 2];

    RelationsGPU();
};

class RelationsGPUAllVersions {
  public:
    RelationsGPUAllVersions();
    __host__ __device__ RelationsGPU& GetDataGraphByIdx(size_t idx) {
        return array_data_graphs_[idx];
    }
    // __host__ void AllocateAllRelations(const QueryGraph &query_);

  private:
    RelationsGPU array_data_graphs_[POSSIBLE_NUM_QE];
};

class CandidatesGPU {
  public:
    // query vertex ID -> an array of uint32_t (the length of array: sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u))
    uint32_t *candidate_bits_[MAX_QV_COUNT];
    CandidatesGPU();
};

class CandidatesGPUAllVersions {
  public:
    CandidatesGPUAllVersions();
    __host__ __device__ CandidatesGPU& GetByIdx(size_t idx) {
        return array_candidates_gpu_[idx];
    }
    // __host__ void AllocateAllRelations(const QueryGraph &query_);

  private:
    CandidatesGPU array_candidates_gpu_[POSSIBLE_NUM_QE];
};

class MatchGPU {
  public:
    MatchGPU(const GraphFileIoManager& graph_file_io, 
             const QueryGraph& query,
             const PlanManager& plan);
    MatchGPU(const GraphFileIoManager& graph_file_io, 
             const QueryGraph& query,
             const PlanManager& plan,
             const AutomorphismManager *am_ptr);
    ~MatchGPU();

    void LoadQuery();
    void LoadPlan();

    void GammaLoadPlan();
    
    void BuildTries(const EdgeBatch& edge_lists, CSR_GPU csr_gpu[], const uint8_t i);
    void DeallocTries(CSR_GPU csr_gpu[]);

    void AllocRelations(const DataGraphManager& data_graph, const CSR_GPU csr_gpu[],
                        RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
                        CandidatesGPU& global_bitmap_gpu);

    void DeallocRelations(RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
                          CandidatesGPU& global_bitmap_gpu);

    void AllocOnline(RelationsGPU& local_index_base_gpu_gpu, CandidatesGPU& local_bitmap_gpu);
    void DeallocOnline(RelationsGPU& local_index_base_gpu_gpu, CandidatesGPU& local_bitmap_gpu);

    void SetGraphPtrs(RelationsGPU& data_graph_gpu);

    void GetSummary(const RelationsGPU& global_index_gpu, 
                    uint32_t *cardinalities, float *degrees);

    void UpdateGlobalIndex(RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
                           CandidatesGPU& global_bitmap_gpu, const CSR_GPU csr_gpu[], const uint8_t i);
  
    bool GammaBuildLocalIndex(const RelationsGPU& global_index_gpu, CandidatesGPU& global_bitmap_gpu,
                              RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu,
                              RelationsGPU& local_index,
                              CSR_GPU csr_gpu[], const uint8_t cur_i, const float *avg_degrees);

    void GammaMatching(RelationsGPU& global_index, RelationsGPU& local_index, unsigned long long int& num_matches,
                       const bool materialize_results=false, uint32_t **result_ptr=nullptr, uint32_t *result_size_ptr=nullptr);

  private:
    const QueryGraph& query_;
    const PlanManager& plan_;
    const AutomorphismManager *am_ptr_;
    uint32_t *data_graph_all_nbrs_[MAX_QE_COUNT * 2];
    uint32_t *index_all_nbrs_[MAX_QE_COUNT * 2];

    // uint32_t *data_graph_all_nbrs_[MAX_QE_COUNT * 2];
    // uint32_t *index_all_nbrs_[MAX_QE_COUNT * 2];

    // For CUB
    void *d_temp_storage_;
    size_t temp_storage_bytes_;
    size_t temp_storage_capacity_;

    // For indexing
    bool *cand_flag_;
    uint32_t cand_flag_capacity_;
    uint32_t *d_new_cand_count_[2];
    CSR_GPU temp_tries_[2];
    CSR_GPU_Capacity temp_tries_capacity_[2];
    uint32_t *helper_relation_[2];
    uint32_t helper_relation_capacity_[2];

    // Segments in the array `local_nbr_[edge_index]` are pointed to by the pointers 
    // in (RelationsGPU)local_index_base_gpu.nbrs_[edge_index] (an array of pointers).
    uint32_t *local_nbr_[MAX_QE_COUNT * 2];
    uint32_t local_nbr_capacity_[MAX_QE_COUNT * 2]; 

    uint32_t *cum_bn_;

    // For Enumeration
    unsigned long long int res_;
    unsigned long long int res_size_;

    unsigned long long int new_res_;
    unsigned long long int *new_res_size_;
    unsigned long long int h_new_res_size_;
    unsigned long long int h_max_new_res_size_;

    uint8_t cur_depth_;
    uint8_t new_depth_;

    // For memory allocation
    MemPool<uint32_t> nbr_mem_pool_;
    CyclicQueue<uint32_t> res_queue_;
    unsigned long *res_size_cartesian_product_;
    unsigned long *max_res_size_cartesian_product_;

    void BuildTriesFromEdgeList(const EdgeList& edge_list, 
                                CSR_GPU& csr_gpu, CSR_GPU& rcsr_gpu);
    void AllocRelation(const uint32_t idx, const CSR_GPU& csr_gpu,
                       const uint32_t u, const uint32_t *dvlabels, 
                       RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
                       uint32_t *capacity_prefix_sum,
                       uint32_t *data_graph_all_nbrs,
                       uint32_t *index_all_nbrs,
                       bool build_graph);
    void SelectEdgesFromTrie(const CSR_GPU& input,
                             CSR_GPU& output,
                             const uint32_t *first_candidates,
                             const uint32_t *second_candidates);
};

#endif  // GRAPH_MATCH_GPU