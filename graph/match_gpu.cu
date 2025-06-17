#include <iostream>
#include "cub/cub.cuh"
// #include <cub/cub.cuh>

#include "graph/graph.h"
#include "graph/match_gpu.h"
#include "kernels/cartesian_product.h"
#include "kernels/gamma_enumeration.h"
#include "kernels/indexing.h"
#include "utils/config.h"
#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "utils/globals.h"
#include "utils/types.h"

// #define IS_DEBUGGING_MATCH_GPU

__constant__ OrderPerEdge C_INDEXING_ORDERS[MAX_QE_COUNT];
__constant__ OrderPerEdge C_ORDERS[MAX_QE_COUNT];
__constant__ uint8_t C_EQUIV_EDGE_COUNT[MAX_QE_COUNT];
__constant__ uint8_t C_NUM_EQUIV_GROUPS;
__constant__ uint8_t C_NON_TAIL_LEAF_DEPTHS[MAX_QE_COUNT];

RelationsGPU::RelationsGPU(): nbrs_(), nbrs_is_update_(), capacity_(), sizes_() {}

__global__ void AccessdCandCount(uint32_t *d_new_cand_count[]) {
    uint32_t uint1 = *(d_new_cand_count[0]);
    uint32_t uint2 = *(d_new_cand_count[1]);
    printf("d_new_cand_count_[1]: %d\n", uint1);
    printf("d_new_cand_count_[0]: %d\n", uint2);
}

RelationsGPUAllVersions::RelationsGPUAllVersions() : array_data_graphs_() {}

CandidatesGPUAllVersions::CandidatesGPUAllVersions(): array_candidates_gpu_() {}

CandidatesGPU::CandidatesGPU()
: candidate_bits_()
{}

MatchGPU::MatchGPU(
    const GraphFileIoManager& graph_file_io, 
    const QueryGraph& query,
    const PlanManager& plan)
: query_(query)
, plan_(plan)
, data_graph_all_nbrs_{NULL}
, index_all_nbrs_{NULL}

, d_temp_storage_(NULL)
, temp_storage_bytes_(0ul)
, temp_storage_capacity_(0ul)
, cand_flag_(NULL)
, cand_flag_capacity_(0u)
, d_new_cand_count_{NULL, NULL}
, temp_tries_()
, temp_tries_capacity_{{0u,0u,0u}, {0u,0u,0u}}
, helper_relation_{NULL, NULL}
, helper_relation_capacity_{0u}
, local_nbr_()
, local_nbr_capacity_{0u}
, cum_bn_()

, res_(0ul)
, res_size_(0ul)
, new_res_(0ul)
, new_res_size_(NULL)
, h_new_res_size_(0ul)
, h_max_new_res_size_(0ul)
, cur_depth_(0u)
, new_depth_(0u)

, nbr_mem_pool_()
, res_queue_()
, res_size_cartesian_product_(NULL)
, max_res_size_cartesian_product_(NULL)
{
    nbr_mem_pool_.Alloc(NBR_SPACE);
    cudaErrorCheck(cudaMalloc(&max_res_size_cartesian_product_, sizeof(unsigned long)));

    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[0], sizeof(uint32_t)));
    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[1], sizeof(uint32_t)));

    cudaErrorCheck(cudaMalloc(&new_res_size_, sizeof(unsigned long long int)));
}

MatchGPU::MatchGPU(
    const GraphFileIoManager& graph_file_io, 
    const QueryGraph& query,
    const PlanManager& plan,
    const AutomorphismManager *am_ptr)
: query_(query)
, plan_(plan)
, am_ptr_(am_ptr)
, data_graph_all_nbrs_{NULL}
, index_all_nbrs_{NULL}

, d_temp_storage_(NULL)
, temp_storage_bytes_(0ul)
, temp_storage_capacity_(0ul)
, cand_flag_(NULL)
, cand_flag_capacity_(0u)
, d_new_cand_count_{NULL, NULL}
, temp_tries_()
, temp_tries_capacity_{{0u,0u,0u}, {0u,0u,0u}}
, helper_relation_{NULL, NULL}
, helper_relation_capacity_{0u}
, local_nbr_()
, local_nbr_capacity_{0u}
, cum_bn_()

, res_(0ul)
, res_size_(0ul)
, new_res_(0ul)
, new_res_size_(NULL)
, h_new_res_size_(0ul)
, h_max_new_res_size_(0ul)
, cur_depth_(0u)
, new_depth_(0u)

, nbr_mem_pool_()
, res_queue_()
, res_size_cartesian_product_(NULL)
, max_res_size_cartesian_product_(NULL)
{
    nbr_mem_pool_.Alloc(NBR_SPACE);
    cudaErrorCheck(cudaMalloc(&max_res_size_cartesian_product_, sizeof(unsigned long)));

    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[0], sizeof(uint32_t)));
    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[1], sizeof(uint32_t)));

    cudaErrorCheck(cudaMalloc(&new_res_size_, sizeof(unsigned long long int)));
}

MatchGPU::~MatchGPU()
{
    nbr_mem_pool_.Free();

    cudaErrorCheck(cudaFree(new_res_size_));

    cudaErrorCheck(cudaFree(d_new_cand_count_[0]));
    cudaErrorCheck(cudaFree(d_new_cand_count_[1]));

    if (temp_storage_capacity_ > 0u) cudaErrorCheck(cudaFree(d_temp_storage_));
    if (cand_flag_capacity_ > 0u) cudaErrorCheck(cudaFree(cand_flag_));

    for (auto i = 0u; i < 2u; i++)
    {
        if (temp_tries_capacity_[i].vs_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].vs_));
        if (temp_tries_capacity_[i].off_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].offs_));
        if (temp_tries_capacity_[i].es_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].nbrs_));

        if (helper_relation_capacity_[i] > 0u) cudaErrorCheck(cudaFree(helper_relation_[i]));
    }

    for (auto i = 0u; i < QE_COUNT; i++)
    {
        if (local_nbr_capacity_[query_.qe_eidx_[i].first] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].first]));
        if (local_nbr_capacity_[query_.qe_eidx_[i].second] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].second]));
    }
}

void MatchGPU::LoadQuery()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_COUNT, &QV_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QE_COUNT, &QE_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_OFFS, query_.qv_offs_.data(), sizeof(uint8_t) * (QV_COUNT + 1u)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NLF, query_.NLF_.data(), sizeof(uint8_t) * QE_COUNT * 2u));
    cudaErrorCheck(cudaMemcpyToSymbol(C_EIDX, query_.eidx_.data(), sizeof(uint8_t) * QV_COUNT * QV_COUNT));
}

void MatchGPU::LoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
}

void MatchGPU::GammaLoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_EQUIV_EDGE_COUNT: ";
    const uint8_t *equiv_edge_count_array_hptr = am_ptr_->getEquivEdgeCountArrayPointer();
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        std::cout << " " << (uint32_t)equiv_edge_count_array_hptr[i] << " ";
    }
    std::cout << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_EQUIV_EDGE_COUNT, am_ptr_->getEquivEdgeCountArrayPointer(), sizeof(uint8_t) * MAX_QE_COUNT));
    
    uint8_t num_groups = am_ptr_->getNumEquivGroups();
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_EQUIV_GROUPS, &num_groups, sizeof(uint8_t)));
    
    std::vector<uint8_t> non_tail_leaf_depths = plan_.getNonTailLeafDepths();
    // std::vector<uint8_t> non_tail_leaf_depths = std::vector<uint8_t>(query_.getEdgeCount(), 3u);

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_NON_TAIL_LEAF_DEPTHS: ";
    for (int i = 0; i < non_tail_leaf_depths.size(); i++) {
        std::cout << " " << (uint32_t)non_tail_leaf_depths[i] << " ";
    }
    std::cout << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_NON_TAIL_LEAF_DEPTHS, non_tail_leaf_depths.data(), sizeof(uint8_t) * non_tail_leaf_depths.size()));
}

void MatchGPU::BuildTries(const EdgeBatch& edge_lists, CSR_GPU csr_gpu[], const uint8_t i)
{
    // build csr_gpu based on the initial edge lists

    // uint32_t temp;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "inside BuildTries - before BuildTriesFromEdgeList, d_new_cand_count_[0]: " << d_new_cand_count_[0] << std::endl;  // 250216
    // cout << "inside BuildTries - before BuildTriesFromEdgeList, d_new_cand_count_[1]: " << d_new_cand_count_[1] << std::endl;  // 250216
    
    BuildTriesFromEdgeList(
        edge_lists[query_.qe_eidx_[i].first], 
        csr_gpu[query_.qe_eidx_[i].first], csr_gpu[query_.qe_eidx_[i].second]);
        
    // cout << "inside BuildTries - after BuildTriesFromEdgeList, d_new_cand_count_[0]: " << d_new_cand_count_[0] << std::endl;  // 250216
    // cout << "inside BuildTries - after BuildTriesFromEdgeList, d_new_cand_count_[1]: " << d_new_cand_count_[1] << std::endl;  // 250216
    // uint32_t temp2;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "temp2: " << temp2 << std::endl;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "temp2: " << temp2 << std::endl;  // 250216
}

void MatchGPU::DeallocTries(CSR_GPU csr_gpu[])
{
    for (auto i = 0u; i < QE_COUNT * 2u; i++)
    {
        csr_gpu[i].vs_size_ = 0u;
        csr_gpu[i].es_size_ = 0u;
        cudaErrorCheck(cudaFree(csr_gpu[i].vs_));
        cudaErrorCheck(cudaFree(csr_gpu[i].offs_));
        cudaErrorCheck(cudaFree(csr_gpu[i].nbrs_));
    }
}

void MatchGPU::AllocRelations(
    const DataGraphManager& data_graph, const CSR_GPU csr_gpu[],
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu
) {
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &DV_COUNT, sizeof(uint32_t)));
    uint32_t *dvlabels, *capacity_prefix_sum;

    // copy the vertex label array to gpu
    cudaErrorCheck(cudaMalloc(&dvlabels, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemcpy(dvlabels, data_graph.vlabels_.data(), sizeof(uint32_t) * DV_COUNT, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&capacity_prefix_sum, sizeof(uint32_t) * (DV_COUNT + 1u)));

    // allocate memory for each relation in the data graph and the index
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        AllocRelation(query_.qe_eidx_[i].first, csr_gpu[query_.qe_eidx_[i].first], 
            query_.qe_list_[i].first, dvlabels, 
            data_graph_gpu, global_index_gpu, capacity_prefix_sum,
            data_graph_all_nbrs_[query_.qe_eidx_[i].first],
            index_all_nbrs_[query_.qe_eidx_[i].first],
            query_.first_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first ||
            query_.last_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first);
        AllocRelation(query_.qe_eidx_[i].second, csr_gpu[query_.qe_eidx_[i].second], 
            query_.qe_list_[i].second, dvlabels, 
            data_graph_gpu, global_index_gpu, capacity_prefix_sum,
            data_graph_all_nbrs_[query_.qe_eidx_[i].second],
            index_all_nbrs_[query_.qe_eidx_[i].second],
            query_.first_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second ||
            query_.last_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second);
    }

    cudaErrorCheck(cudaFree(dvlabels));
    cudaErrorCheck(cudaFree(capacity_prefix_sum));

    // allocate the candidate bits arrays
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaMalloc(&global_bitmap_gpu.candidate_bits_[i], sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(global_bitmap_gpu.candidate_bits_[i], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
    }
}

void MatchGPU::DeallocRelations(
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu
) {
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaFree(global_bitmap_gpu.candidate_bits_[i]));
    }
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            if (query_.first_NL_[idx] == idx || query_.last_NL_[idx] == idx)
            {
                cudaErrorCheck(cudaFree(data_graph_all_nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.capacity_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.sizes_[idx]));
            }

            cudaErrorCheck(cudaFree(index_all_nbrs_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.nbrs_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.capacity_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.sizes_[idx]));
        }
    }
}

void MatchGPU::AllocOnline(
    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu
) {
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT)));
        }
    }
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaMalloc(&local_bitmap_gpu.candidate_bits_[i], sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[i], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
    }

    cudaErrorCheck(cudaMalloc(&cum_bn_, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

    cudaErrorCheck(cudaMalloc(&res_size_cartesian_product_, SIZE_SPACE * sizeof(unsigned long)));
    res_queue_.Alloc(RES_SPACE);
    cudaErrorCheck(cudaMemcpyToSymbol(C_RES_QUEUE, &res_queue_, sizeof(CyclicQueue<uint32_t>)));
}

void MatchGPU::DeallocOnline(
    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu
) {
    cudaErrorCheck(cudaFree(res_size_cartesian_product_));
    res_queue_.Free();
    cudaErrorCheck(cudaFree(cum_bn_));
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaFree(local_bitmap_gpu.candidate_bits_[i]));
    }
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            cudaErrorCheck(cudaFree(local_index_base_gpu.sizes_[idx]));
            cudaErrorCheck(cudaFree(local_index_base_gpu.capacity_[idx]));
            cudaErrorCheck(cudaFree(local_index_base_gpu.nbrs_[idx]));
        }
    }
}

void MatchGPU::SetGraphPtrs(RelationsGPU& data_graph_gpu)
{
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (auto j = 0u; j < 2u; j++)
        {
            const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
            if (idx != query_.first_NL_[idx])
            {
                data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.last_NL_[idx]];
                data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.last_NL_[idx]];
            }
            else
            {
                // Doubt: Seems no-op, as idx == query_.first_NL_[idx].
                data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.first_NL_[idx]];
                data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.first_NL_[idx]];
            }
        }
    }
}

void MatchGPU::GetSummary(
    const RelationsGPU& global_index_gpu, uint32_t *cardinalities, float *degrees
) {
    uint32_t h_temp[2], *d_temp;
    // cout << "before cudaMalloc" << std::endl;
    cudaErrorCheck(cudaMalloc(&d_temp, sizeof(uint32_t) * 2u));
    // cout << "before for-loop" << std::endl;
    for (auto i = 0u; i < QE_COUNT * 2; i++)
    {
        cudaErrorCheck(cudaMemset(d_temp, 0u, sizeof(uint32_t) * 2u));
        // i is the current edge_idx
        // *d_temp <- edge_count; *(d_temp + 1) <- num_source_vertices in global_index_gpu.xxx[i]
        statisticIndex<<<GRID_DIM, BLOCK_DIM>>>(
            global_index_gpu, i, d_temp, d_temp + 1
        );
        cudaErrorCheck(cudaMemcpy(h_temp, d_temp, sizeof(uint32_t) * 2u, cudaMemcpyDeviceToHost));
        // cout << "inside" << i << "-th loop" << "before cardinalities assignment" << std::endl;
        cardinalities[i] = h_temp[0];
        degrees[i] = h_temp[0] / (float) h_temp[1];
    }
}

void MatchGPU::UpdateGlobalIndex(
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu, const CSR_GPU csr_gpu[], const uint8_t i
) {
    // update the candidate edges of the query edges adjacent u or uu
    for (auto j = 0u; j < 2u; j++)
    {
        // cout << "inside UpdateGlobalIndex - first for loop, j: " << j << std::endl;  // 250216
        const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
        const auto& u = j == 0u ? query_.qe_list_[i].first : query_.qe_list_[i].second;
        const auto& uu = j == 0u ? query_.qe_list_[i].second : query_.qe_list_[i].first;

        // find new candidates of u (satisfies NLF after the update), and write to the trie
        ReAlloc(cand_flag_, csr_gpu[idx].vs_size_, cand_flag_capacity_, bool);
        cudaErrorCheck(cudaMemset(cand_flag_, false, sizeof(bool) * cand_flag_capacity_));
        cudaErrorCheck(cudaDeviceSynchronize());

        if (query_.NLF_[idx] == 0) continue;
        // Apply NLFs (considering the initial graph data_graph_gpu and new edges in csr_gpu[idx]). 
        // Write values to global_bitmap_gpu.candidate_bits_[u] and cand_flag_. (Add NLF-valid data vertices into candidates.)
        getGlobalCandidates<<<GRID_DIM, BLOCK_DIM>>>(
            data_graph_gpu, idx, csr_gpu[idx], global_bitmap_gpu.candidate_bits_[u], cand_flag_, u
        );
        cudaErrorCheck(cudaDeviceSynchronize());

        // cout << "inside UpdateGlobalIndex - after getGlobalCandidates, d_new_cand_count_[0]: " << d_new_cand_count_[0] << std::endl;  // 250216
        // cout << "inside UpdateGlobalIndex - after getGlobalCandidates, d_new_cand_count_[1]: " << d_new_cand_count_[1] << std::endl;  // 250216
        // uint32_t temp2;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << std::endl;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << std::endl;  // 250216

        // allocate temp_tries.vs_ and temp_tries.offs_
        ReAlloc(temp_tries_[0].vs_, csr_gpu[idx].vs_size_, temp_tries_capacity_[0].vs_capacity_, uint32_t);
        // cudaErrorCheck(cudaDeviceSynchronize());  // 250216

        // cout << "inside UpdateGlobalIndex - before cub::DeviceSelect::Flagged" << std::endl;  // 250216
        // cout << "csr_gpu[idx].vs_size_: " << csr_gpu[idx].vs_size_ << std::endl;  // 250216
        // cout << "cand_flag_capacity_: " << cand_flag_capacity_ << std::endl;  // 250216
        // cout << "bool* cand_flag_: " << cand_flag_ << std::endl;  // 250216
        // cout << "uint32_t* (temp_tries_[0].vs_): " << temp_tries_[0].vs_ << std::endl;  // 250216
        // cout << "idx: " << idx << std::endl;  // 250216
        // cout << "csr_gpu[idx].vs_: " << csr_gpu[idx].vs_ << std::endl;  // 250216
        // cout << "d_temp_storage_: " << d_temp_storage_ << std::endl;  // 250216

        uint32_t *temp_dest = new uint32_t[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_dest, csr_gpu[idx].vs_, sizeof(uint32_t) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        bool *temp_bool_dest = new bool[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_bool_dest, cand_flag_, sizeof(bool) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        uint32_t *temp_dest2 = new uint32_t[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_dest2, temp_tries_[0].vs_, sizeof(uint32_t) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        // for (int i = 0; i < csr_gpu[idx].vs_size_; i++) {
        //     if (temp_bool_dest[i] == false) {
        //         cout << "temp_dest[" << i << "]: " << temp_dest[i] << ", temp_bool_dest[" << i << "]: " << temp_bool_dest[i] << ", temp_dest2[" << i << "]: " << temp_dest2[i] << std::endl;
        //     }
        //     // cout << "temp_dest[" << i << "]: " << temp_dest[i] << ", temp_bool_dest[" << i << "]: " << temp_bool_dest[i] << std::endl;
        // }

        // masked_select: csr_gpu[idx].vs_[cand_flag_] -> temp_tries_[0].vs_, 
        // *d_new_cand_count_[0] is the result item count.
        CUB(cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_));

        // // cout << "before void *pre_d_temp_storage_ = d_temp_storage_, d_temp_storage_: " << d_temp_storage_ << std::endl;  // 250216
        // void *pre_d_temp_storage_ = d_temp_storage_;  // 250216
        // d_temp_storage_ = NULL;    // 250216
        // cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_);    // 250216
        // cudaErrorCheck(cudaDeviceSynchronize());  // 250216
        // // cout << "before if, temp_storage_bytes_: " << temp_storage_bytes_ << std::endl;  // 250216
        // if (temp_storage_bytes_ > temp_storage_capacity_)    // 250216
        // {
        //     if (temp_storage_capacity_ != 0ul)    // 250216
        //     {
        //         cudaErrorCheck(cudaFree(pre_d_temp_storage_));    // 250216
        //     }
        //     cout << "inside if" << std::endl;  // 250216
        //     temp_storage_bytes_ = temp_storage_capacity_ = 
        //         (size_t)exp2(ceil(log2(temp_storage_bytes_)));    // 250216
        //     cudaErrorCheck(cudaMalloc(
        //         &d_temp_storage_, temp_storage_bytes_));    // 250216
        //     cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_);    // 250216
        //     cudaErrorCheck(cudaDeviceSynchronize());  // 250216
        
        // }
        // else
        // {
        //     cout << "inside else" << std::endl;  // 250216
        //     d_temp_storage_ = pre_d_temp_storage_;    // 250216
        // }
        // // cout << "temp_storage_bytes_: " << temp_storage_bytes_ << std::endl;  // 250216
        // // cout << "temp_storage_capacity_: " << temp_storage_capacity_ << std::endl;  // 250216
        // // cout << "d_temp_storage_: " << d_temp_storage_ << std::endl;  // 250216
        
        // size_t freeMem, totalMem;  // 250216
        // cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem));  // 250216
        // std::cout << "Total GPU memory: " << totalMem / 1024 / 1024 << " MB" << std::endl;  // 250216
        // std::cout << "Free GPU memory: " << freeMem / 1024 / 1024 << " MB" << std::endl;  // 250216

        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << std::endl;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << std::endl;  // 250216

        // temp_tries_[0].vs_size_, i.e. CSR_GPU::vs_size_ is in CPU memory.
        cudaErrorCheck(cudaMemcpy(&temp_tries_[0].vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (temp_tries_[0].vs_size_ == 0)
        {
            continue;
        }
        ReAlloc(temp_tries_[0].offs_, csr_gpu[idx].vs_size_ + 1, temp_tries_capacity_[0].off_capacity_, uint32_t);

        // check each relation adjacent to u
        for (auto k = query_.qv_offs_[u]; k < query_.qv_offs_[u + 1]; k++)
        {
            // k is the current edge_idx
            const auto& u_other = query_.qv_nbrs_[k];

            // allocate temp_tries.es_
            // Calculate the number of valid data graph edges for each data graph source vertex in temp_tries_[0].vs_ and store the numbers in temp_tries_[0].offs_
            // A valid data graph edge means that the destination vertex is in `global_bitmap_gpu.candidate_bits_[u_other]`
            // Doubt: What if `global_bitmap_gpu.candidate_bits_[u_other]` has not been written (This may happen when invoked at initialization steps).
            getGlobalCandidateEdgesCount<<<GRID_DIM, BLOCK_DIM>>>(
                data_graph_gpu, k, temp_tries_[0].vs_, temp_tries_[0].vs_size_,
                global_bitmap_gpu.candidate_bits_[u_other], temp_tries_[0].offs_
            );
            cudaErrorCheck(cudaDeviceSynchronize());

            // exclusive_sum(temp_tries_[0].offs_0)
            CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, temp_tries_[0].offs_, temp_tries_[0].offs_, temp_tries_[0].vs_size_ + 1));
            cudaErrorCheck(cudaMemcpy(&temp_tries_[0].es_size_, &temp_tries_[0].offs_[temp_tries_[0].vs_size_], sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if (temp_tries_[0].es_size_ == 0)
            {
                continue;
            }

            temp_tries_[1].es_size_ = temp_tries_[0].es_size_;

            ReAlloc(temp_tries_[0].nbrs_, temp_tries_[0].es_size_, temp_tries_capacity_[0].es_capacity_, uint32_t);
            ReAlloc(helper_relation_[0], temp_tries_[0].es_size_, helper_relation_capacity_[0], uint32_t);
            ReAlloc(temp_tries_[1].nbrs_, temp_tries_[0].es_size_, temp_tries_capacity_[1].es_capacity_, uint32_t);
            ReAlloc(helper_relation_[1], temp_tries_[0].es_size_, helper_relation_capacity_[1], uint32_t);

            // Seems that the size of temp_tries_[1].vs_ and temp_tries_[1].offs_ can be more than needed.
            ReAlloc(temp_tries_[1].vs_, temp_tries_[0].es_size_, temp_tries_capacity_[1].vs_capacity_, uint32_t);
            ReAlloc(temp_tries_[1].offs_, temp_tries_[0].es_size_ + 1, temp_tries_capacity_[1].off_capacity_, uint32_t);

            // fill in temp_tries.es_
            // Fill the source and destination vertices of valid data graph edges into helper_relation_[0] and temp_tries_[0].nbrs_, respectively
            // Discussion: Checking the bitmap again here. (getGlobalCandidateEdgesCount has checked the bitmap)
            getGlobalCandidateEdgesWrite<<<GRID_DIM, BLOCK_DIM>>>(
                data_graph_gpu, k, temp_tries_[0].vs_, temp_tries_[0].vs_size_,
                global_bitmap_gpu.candidate_bits_[u_other], temp_tries_[0].offs_,
                helper_relation_[0], temp_tries_[0].nbrs_
            );
            cudaErrorCheck(cudaDeviceSynchronize());

            // add the temp_tries to the index
            // Question (Solved): Why add the edges in `temp_tries_[0]` to `global_index_gpu`? Answer: temp_tries[0] contains updated edges.
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[0], global_index_gpu, k, nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }

            // reverse temp_tries_[0] into temp_tries_[1]
            CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
                temp_tries_[0].nbrs_, helper_relation_[1], helper_relation_[0], temp_tries_[1].nbrs_, temp_tries_[0].es_size_));

            CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, helper_relation_[1], temp_tries_[1].vs_, temp_tries_[1].offs_, d_new_cand_count_[1], temp_tries_[0].es_size_));
            cudaErrorCheck(cudaMemcpy(&temp_tries_[1].vs_size_, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));

            CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, temp_tries_[1].offs_, temp_tries_[1].offs_, temp_tries_[1].vs_size_ + 1));

            // add the temp_rcsr_gpu to the index
            // Question (Solved): Why add the edges in `temp_tries_[1]` to `global_index_gpu`? Answer: temp_tries[1] contains (reversed) udpated edges.
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[1], global_index_gpu, query_.eidx_[u_other * QV_COUNT + u], nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }
        }
    }
    for (auto j = 0u; j < 2u; j++)
    {
        const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
        const auto& u = j == 0u ? query_.qe_list_[i].first : query_.qe_list_[i].second;
        const auto& uu = j == 0u ? query_.qe_list_[i].second : query_.qe_list_[i].first;
        
        // cout << "inside UpdateGlobalIndex - second for loop, j: " << j << ", idx: " << idx << ", u: " << u << ", uu: " << uu << std::endl;  // 250216
        
        if (query_.first_NL_[idx] == idx || query_.last_NL_[idx] == idx)
        {
            // add all edges mapped to the current query edge to the data graph
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(csr_gpu[idx], data_graph_gpu, idx, nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }
        }
        else
        {
            data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.first_NL_[idx]];
            data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.first_NL_[idx]];
        }

        // add all edges with both endpoints being candidate to the index
        ReAlloc(temp_tries_[0].vs_, csr_gpu[idx].vs_size_, temp_tries_capacity_[0].vs_capacity_, uint32_t);
        ReAlloc(temp_tries_[0].offs_, csr_gpu[idx].vs_size_ + 1, temp_tries_capacity_[0].off_capacity_, uint32_t);
        ReAlloc(temp_tries_[0].nbrs_, csr_gpu[idx].es_size_, temp_tries_capacity_[0].es_capacity_, uint32_t);

        SelectEdgesFromTrie(csr_gpu[idx], temp_tries_[0], global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);

        addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[0], global_index_gpu, idx, nbr_mem_pool_);
        cudaErrorCheck(cudaDeviceSynchronize());
        if (nbr_mem_pool_.OutOfMemory())
        {
            exit(-1);
        }
    }
}

bool MatchGPU::GammaBuildLocalIndex(const RelationsGPU& global_index_gpu, CandidatesGPU& global_bitmap_gpu,
                                    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu,
                                    RelationsGPU& local_index, CSR_GPU csr_gpu[], 
                                    const uint8_t edge_list_idx, const float *avg_degrees) {
    // lgh: cur_i is current idx_in_qe_list_
    // Visit the query vertices in plan_.indexing_orders_[cur_i] one by one.

    std::pair<uint32_t, uint32_t> cur_edge = query_.getEdgeByEdgeListIdx(edge_list_idx);
    const uint32_t u0 = cur_edge.first;
    const uint32_t u1 = cur_edge.second;
    for (auto j = 0u; j < 2u; j++)
    {
        const auto& u = j == 0u ? u0 : u1;
        const auto& uu = j == 0u ? u1 : u0;
        // lgh: index is edge_idx
        const auto& edge_idx = query_.eidx_[u * QV_COUNT + uu];

        // build the relation from trie
        cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[u], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[edge_idx], 0u, sizeof(uint32_t) * DV_COUNT));
        cudaErrorCheck(cudaDeviceSynchronize());
        // index is the current edge_idx, inputs are csr_gpu[index], global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]
        // the output is local_index_base_gpu. Only fill values in output.sizes_[idx] (an array of length DV_COUNT)
        edgeList2RelationCount_v2<<<GRID_DIM, BLOCK_DIM>>>(
            csr_gpu[edge_idx], local_index_base_gpu, edge_idx,
            global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);

        // Input: local_index_base_gpu.sizes_[index], Output: local_bitmap_gpu.candidate_bits_[u] (a bitmap).
        // Directly copy the information in local_index_base_gpu.sizes_[index] to the bitmap (size > 0 --> is a candidate)
        setLocalBitmap<<<GRID_DIM, BLOCK_DIM>>>(
            local_index_base_gpu, edge_idx, local_bitmap_gpu.candidate_bits_[u]);
        cudaErrorCheck(cudaDeviceSynchronize());

        // Fill values in local_index_base_gpu.capacity_[index]
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[edge_idx], local_index_base_gpu.capacity_[edge_idx], DV_COUNT + 1));
        
        auto total_size = 0u;
        cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[edge_idx] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (total_size == 0u) {
            return false;
        }

        ReAlloc(local_nbr_[edge_idx], total_size, local_nbr_capacity_[edge_idx], uint32_t*);
        // Fill values (neighbor pointers) in local_index_base_gpu.nbrs_[index].
        // After this function,
        // segments of local_nbr_[edge_index] (an continuous array) are pointed to by pointers in local_index_base_gpu.nbrs_[edge_index] (an array of pointers).
        setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[edge_idx], local_index_base_gpu.capacity_[edge_idx], DV_COUNT, local_index_base_gpu.nbrs_[edge_idx]);
        cudaErrorCheck(cudaDeviceSynchronize());

        // Fill the neighbor IDs.
        edgeList2RelationWrite_v2<<<GRID_DIM, BLOCK_DIM>>>(
            csr_gpu[edge_idx], local_index_base_gpu, edge_idx,
            global_bitmap_gpu.candidate_bits_[uu],
            local_bitmap_gpu.candidate_bits_[u]);
        cudaErrorCheck(cudaDeviceSynchronize());

        local_index.nbrs_[edge_idx] = local_index_base_gpu.nbrs_[edge_idx];
        local_index.sizes_[edge_idx] = local_index_base_gpu.sizes_[edge_idx];
    }
    return true;
}

void MatchGPU::GammaMatching(RelationsGPU& global_index, RelationsGPU& local_index, unsigned long long int& num_matches,
                             const bool materialize_results, uint32_t ** result_ptr_ptr, uint32_t * result_size_ptr) {
    uint8_t num_query_vertices = query_.getVerticesCount();

    // lgh: i is current idx_in_qe_list_
    res_queue_.Reset();
    bool oom = false;

    unsigned long long int *effective_num_results_dptr;
    cudaErrorCheck(cudaMalloc(&effective_num_results_dptr, sizeof(unsigned long long int)));
    cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(unsigned long long int)));

    /************************************ Step 1 ************************************/
    // initialize the partial results as all data edges mapping to the first two query vertices
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
    // new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

    // uint8_t num_groups = am_ptr_->getNumEquivGroups();
    std::vector<uint8_t> rep_edge_list_idxs = am_ptr_->getRepEdgeListIdxVector();
    uint8_t num_query_edges = query_.getEdgeCount();
    
    bool enumerate_cartesian_product = false;
    // bool &write_results_to_global_mem = enumerate_cartesian_product;
    for (uint8_t i = 0; i < num_query_edges; i++) {
        uint8_t cur_non_tail_leaf_count = plan_.getNonTailLeafDepthByEdgeListIdx(i);

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "edge_list_idx: " << (uint32_t)i << " cur_non_tail_leaf_count: " << (uint32_t)cur_non_tail_leaf_count << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        if (cur_non_tail_leaf_count < num_query_vertices) {
            enumerate_cartesian_product = true;
            break;
        }
    }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    h_new_res_size_ = 0;
    for (uint8_t cur_edge_list_idx : rep_edge_list_idxs) {
        uint8_t cur_edge_idx = query_.getEdgeIdxByEdgeListIdx(cur_edge_list_idx);
        gamma_write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
            local_index, cur_edge_idx, cur_edge_list_idx, num_query_vertices,
            new_res_, new_res_size_, h_max_new_res_size_);
        cudaErrorCheck(cudaDeviceSynchronize());
// #ifdef IS_DEBUGGING_MATCH_GPU
        // cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        // std::cout << "after gamma_write_initial_partial_results for the representative edge " << (uint32_t)cur_edge_list_idx << std::endl;
        // std::cout << "h_new_res_sizew_: " << h_new_res_size_ << std::endl;
        // std::cout << "num_query_vertices: " << (uint32_t)num_query_vertices << std::endl;
        // std::cout << "new_res_: " << new_res_ << std::endl;
// #endif  // IS_DEBUGGING_MATCH_GPU
    }
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
// #endif  // IS_DEBUGGING_MATCH_GPU

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_ = new_res_;  // preserve the old value of `new_res_`, which is the starting point of existing results.
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        return;
    }
    // cur_depth_ = new_depth_;

    /************************************ Step 2 ************************************/

    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
    // new_depth_ = cur_depth_ + 1;
    new_res_ = res_queue_.TryMax();
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "res_: " << res_ << ", new_res_: " << new_res_ << ", before gamma_enumerate." << std::endl;
    std::cout << "# initial results: " << res_size_ << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    uint32_t grid_dim = DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK);
    uint32_t num_threads = grid_dim * BLOCK_DIM;

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "launching " << grid_dim << " blocks, " << num_threads << " threads, " << num_threads / 32 << " warps." << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cur_depth_ = 2;
    bool write_results_to_global_memory = enumerate_cartesian_product || materialize_results;
    gamma_enumerate<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
        res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, global_index, cur_depth_, num_query_vertices, 
        /*enale_work_stealing=*/true, enumerate_cartesian_product, write_results_to_global_memory,
        effective_num_results_dptr
    );

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));

    unsigned long long int effective_num_results_v0 = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results_v0, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "gamma_enumerate finished. (after synchronization), current h_new_res_size_: " << h_new_res_size_ << ", effective_num_results_v0: " << effective_num_results_v0 << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    if (h_new_res_size_ >= h_max_new_res_size_)
    {
        //std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
        std::cout << "OOM: after gamma_enumerate, h_new_res_size_ >= h_max_new_res_size_ !" << std::endl;
        oom = true;
        // break;
    }

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_queue_.Pop(res_size_ * (num_query_vertices + 1));

    res_ = new_res_;
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        return;
    }

    /************************************ Step 4 ************************************/
    // if the workload is imbalanced, perform cartesian product
    
    // bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    
    if (enumerate_cartesian_product)
    {

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "inside if(enemerate_cartesian_product)." << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        // generate max_num_matches array
        GammaGetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, global_index, num_query_vertices
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        
        // prefix sum
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
        cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(*effective_num_results_dptr)));
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long max_result_size = 123u;
        cudaMemcpy(&max_result_size, res_size_cartesian_product_ + res_size_, sizeof(*res_size_cartesian_product_), cudaMemcpyDeviceToHost);

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "max_result_size: " << max_result_size << ", gammaEnumerateCartesianProductTree." << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        gammaEnumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
                res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, 
                global_index, num_query_vertices, new_res_size_, effective_num_results_dptr, materialize_results);

        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));
    }

    if (materialize_results) {
        if (h_new_res_size_ < h_max_new_res_size_) {
            *result_ptr_ptr = res_queue_.CopyToHost(new_res_, h_new_res_size_ * (num_query_vertices + 1));
            *result_size_ptr = h_new_res_size_;
        } else {
            oom = true;
            std::cout << "OOM: h_new_res_size_(" << h_new_res_size_ << ") >= h_max_new_res_size_.(" << h_max_new_res_size_ << ")" << std::endl;
        }
    }

    unsigned long long int effective_num_results = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Matching finished, h_new_res_size_: " << h_new_res_size_ << ", effective_num_results: " << effective_num_results << "." << std::endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaFree(effective_num_results_dptr));
    num_matches += effective_num_results;
}  // void MatchGPU::GammaMatching

void MatchGPU::BuildTriesFromEdgeList(
    const EdgeList& edge_list, 
    CSR_GPU& csr_gpu, CSR_GPU& rcsr_gpu
) {
    const auto& ecount = edge_list.first.size();

    // 1. allocate csr_gpu
    cudaErrorCheck(cudaMalloc(&csr_gpu.vs_, sizeof(uint32_t) * ecount));
    cudaErrorCheck(cudaMalloc(&csr_gpu.offs_, sizeof(uint32_t) * (ecount + 1)));
    cudaErrorCheck(cudaMalloc(&csr_gpu.nbrs_, sizeof(uint32_t) * ecount));
    ReAlloc(helper_relation_[0], ecount, helper_relation_capacity_[0], uint32_t);

    cudaErrorCheck(cudaMalloc(&rcsr_gpu.vs_, sizeof(uint32_t) * ecount));
    cudaErrorCheck(cudaMalloc(&rcsr_gpu.offs_, sizeof(uint32_t) * (ecount + 1)));
    cudaErrorCheck(cudaMalloc(&rcsr_gpu.nbrs_, sizeof(uint32_t) * ecount));
    ReAlloc(helper_relation_[1], ecount, helper_relation_capacity_[1], uint32_t);

    cudaErrorCheck(cudaMemcpy(helper_relation_[0], edge_list.first.data(), sizeof(uint32_t) * ecount, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(csr_gpu.nbrs_, edge_list.second.data(), sizeof(uint32_t) * ecount, cudaMemcpyHostToDevice));

    // 2. sort the neighbor list
    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.nbrs_, helper_relation_[1], helper_relation_[0], rcsr_gpu.nbrs_, ecount));

    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        rcsr_gpu.nbrs_, helper_relation_[0], helper_relation_[1], csr_gpu.nbrs_, ecount));

    // Doubt: This step seems redundant.
    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.nbrs_, helper_relation_[1], helper_relation_[0], rcsr_gpu.nbrs_, ecount));

    csr_gpu.es_size_ = edge_list.first.size();
    rcsr_gpu.es_size_ = edge_list.first.size();

    // 3. run length encoding of the neighbor list
    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, 
        helper_relation_[0], csr_gpu.vs_, csr_gpu.offs_, d_new_cand_count_[0], ecount));
    cudaErrorCheck(cudaMemcpy(&csr_gpu.vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, 
        helper_relation_[1], rcsr_gpu.vs_, rcsr_gpu.offs_, d_new_cand_count_[1], ecount));
    cudaErrorCheck(cudaMemcpy(&rcsr_gpu.vs_size_, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 4. exclusive sum
    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.offs_, csr_gpu.offs_, csr_gpu.vs_size_ + 1));
    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
        rcsr_gpu.offs_, rcsr_gpu.offs_, rcsr_gpu.vs_size_ + 1));
}

void MatchGPU::AllocRelation(
    const uint32_t idx, const CSR_GPU& csr_gpu,
    const uint32_t u, const uint32_t *dvlabels, 
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    uint32_t *capacity_prefix_sum,
    uint32_t *data_graph_all_nbrs,
    uint32_t *index_all_nbrs,
    bool build_graph
) {
    // lgh: idx is edge_idx
    // a. malloc
    // cout << "DV_COUNT: " << DV_COUNT << std::endl;
    if (build_graph)
    {
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT + 1)));
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    cudaErrorCheck(cudaMalloc(&global_index_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT + 1)));
    cudaErrorCheck(cudaMalloc(&global_index_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
    cudaErrorCheck(cudaMalloc(&global_index_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));

    // b. set sizes
    if (build_graph)
    {
        cudaErrorCheck(cudaMemset(data_graph_gpu.sizes_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    cudaErrorCheck(cudaMemset(global_index_gpu.sizes_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));

    // c. set capacities
    if (build_graph)
    {
        cudaErrorCheck(cudaMemset(data_graph_gpu.capacity_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    // auto cur_capacity = global_index_gpu.capacity_[idx][0];  // 250216
    // cudaErrorCheck(cudaDeviceSynchronize());  // 250216
    cudaErrorCheck(cudaMemset(global_index_gpu.capacity_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));

    // Set global_index_gpu.capacity_[idx] according to csr_gpu.offs_
    setCapacities<<<GRID_DIM, BLOCK_DIM>>>(csr_gpu, global_index_gpu.capacity_[idx]);
    cudaErrorCheck(cudaDeviceSynchronize());
    // Round each capacity value to the smallest 2's power that is larger than the current capacity value.
    // Set the capacity value of `s` to 0 if the label of the source vertex `s` (a data graph vertex) does not match 
    // the label of `u` (the source vertex of current query edge, `idx` is the edge_idx of current query edge).
    // Doubt: Explicitly setting the capacity value of label-unmatched source vertex seems redundant, 
    // as all edges in `csr_gpu` should be label-matched (which implies the label-matching of source vertices, edge labels and destination vertices).
    roundCapacities<<<GRID_DIM, BLOCK_DIM>>>(global_index_gpu.capacity_[idx], DV_COUNT, dvlabels, query_.vlabels_[u]);
    cudaErrorCheck(cudaDeviceSynchronize());

    if (build_graph)
    {
        cudaErrorCheck(cudaMemcpy(data_graph_gpu.capacity_[idx], global_index_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
    }

    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, global_index_gpu.capacity_[idx], capacity_prefix_sum, DV_COUNT + 1));
    
    // d. set nbrs
    uint32_t total_size;
    cudaErrorCheck(cudaMemcpy(&total_size, &capacity_prefix_sum[DV_COUNT], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // cout << "edge_idx: " << idx << ", total_size: " << total_size << std::endl;
    if (build_graph)
    {
        cudaErrorCheck(cudaMalloc(&data_graph_all_nbrs, sizeof(uint32_t) * total_size));
        cudaErrorCheck(cudaMemset(data_graph_all_nbrs, 0u, sizeof(uint32_t) * total_size));
        cudaErrorCheck(cudaDeviceSynchronize());
        setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(data_graph_all_nbrs, capacity_prefix_sum, DV_COUNT, data_graph_gpu.nbrs_[idx]);
    }
    cudaErrorCheck(cudaMalloc(&index_all_nbrs, sizeof(uint32_t) * total_size));
    cudaErrorCheck(cudaMemset(index_all_nbrs, 0u, sizeof(uint32_t) * total_size));
    cudaErrorCheck(cudaDeviceSynchronize());
    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(index_all_nbrs, capacity_prefix_sum, DV_COUNT, global_index_gpu.nbrs_[idx]);
}

void MatchGPU::SelectEdgesFromTrie(
    const CSR_GPU& input,
    CSR_GPU& output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    output.vs_size_ = input.vs_size_;
    cudaErrorCheck(cudaMemcpy(output.vs_, input.vs_, sizeof(uint32_t) * input.vs_size_, cudaMemcpyDeviceToDevice));

    filterRelevantCount<<<GRID_DIM, BLOCK_DIM>>>(
        input, output, first_candidates, second_candidates
    );
    cudaErrorCheck(cudaDeviceSynchronize());

    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, output.offs_, output.offs_, output.vs_size_ + 1));

    filterRelevantWrite<<<GRID_DIM, BLOCK_DIM>>>(
        input, output, first_candidates, second_candidates
    );
    cudaErrorCheck(cudaDeviceSynchronize());
}