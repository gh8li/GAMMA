#ifndef GAMMA_H
#define GAMMA_H

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "utils/automorphism.h"
#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "graph/graph.h"
#include "graph/match_gpu.h"
#include "graph/plan.h"

uint32_t get_matching_result_vertex(const uint32_t *result_ptr, const uint32_t num_query_vertices, 
                                    const uint32_t result_idx, const uint32_t vertex_idx) {
    return result_ptr[result_idx * (num_query_vertices + 1) + (vertex_idx + 1)];
}

void print_matching_results(const uint32_t *result_ptr, const uint32_t result_size, const uint32_t num_query_vertices) {
    for (uint32_t i = 0; i < result_size; i++) {
        for (uint32_t j = 0; j < num_query_vertices; j++) {
            std::cout << get_matching_result_vertex(result_ptr, num_query_vertices, i, j) << ",";
        }
        std::cout << std::endl;
    }
}

void gammaProcess(const std::string &query_path, const std::string &data_path, const std::string &update_path,
                  uint32_t batch_size, const bool print_results=false) {

    std::cout << "----------- Read Graphs from Files ------------\n";

    TIME_INIT();
    LTIME_INIT();

    QueryGraph query_graph;
    DataGraphManager data_graph;
    AutomorphismManager am;
    PlanManager plan(query_graph);

    TIME_START();

    GraphFileIoManager graph_file_io(query_path, query_graph);
    // Set the metadata and set NLF.
    graph_file_io.SetQueryMeta();
    graph_file_io.LoadInitial(data_path, data_graph);
    graph_file_io.LoadUpdate(update_path, data_graph, batch_size);
    // For each query edge, generate an indexing order (the BFS order starting from the query edge).
    // For edge (u0, u1), u0 is before u1 if u0 < u1 (comparison between vertex IDs).
    plan.GenerateIndexingOrders();

    am.detect_automorphism_edges(&query_graph);

    // query_graph.loadFromFile(query_path);
    // query_graph.setQueryMeta();

    TIME_END();
    PRINT_LOCAL_TIME("Read Graphs from Files");

    std::cout << "----------- Preprocessing ------------\n";

    MatchGPU match_gpu(graph_file_io, query_graph, plan, &am);
    RelationsGPU data_graph_gpu;
    RelationsGPU global_index_gpu;
    CandidatesGPU global_bitmap_gpu;
    // edge_idx -> CSR_GPU
    CSR_GPU csr_gpu[MAX_QE_COUNT * 2];

    TIME_START();
    match_gpu.LoadQuery();
    for (uint8_t i = 0u; i < QE_COUNT; i++) {
        match_gpu.BuildTries(data_graph.initial_edges_, csr_gpu, i);
    }
    TIME_END();
    PRINT_LOCAL_TIME("Load Initial Graph");

    TIME_START();

    match_gpu.AllocRelations(data_graph, csr_gpu, data_graph_gpu, global_index_gpu, global_bitmap_gpu);

    match_gpu.SetGraphPtrs(data_graph_gpu);
    TIME_END();
    PRINT_LOCAL_TIME("Allocate Relations");

    TIME_START();
    for (uint8_t i = 0u; i < QE_COUNT; i++) {
        // Add edges in csr_gpu into global_index_gpu.
        match_gpu.UpdateGlobalIndex(data_graph_gpu, global_index_gpu, global_bitmap_gpu, csr_gpu, i);
    }
    TIME_END();
    PRINT_LOCAL_TIME("Build Global Index Offline");

    TIME_START();
    // std::cout << "before cardinalities" << endl;
    uint32_t cardinalities[QE_COUNT * 2];
    float avg_degrees[QE_COUNT * 2];
    // std::cout << "before GetSummary" << endl;
    // cardinalities: edge_idx -> edge_count in data graph (global_index_gpu)
    // avg_degrees: edge_idx -> edge_count / src_vertex_count in data graph (global_index_gpu)
    match_gpu.GetSummary(global_index_gpu, cardinalities, avg_degrees);
    // std::cout << "before GenerateGammaMatchingOrder" << endl;
    plan.GenerateGammaMatchingOrders(cardinalities, avg_degrees);
    // std::cout << "before GammaLoadPlan" << endl;
    match_gpu.GammaLoadPlan();
    TIME_END();
    PRINT_LOCAL_TIME("Build Matching Order");

    plan.PrintOrders();

    std::cout << "--------- Incremental Matching --------\n";
    RelationsGPU local_index_base_gpu;
    RelationsGPU local_index;
    CandidatesGPU local_bitmap_gpu;
    match_gpu.AllocOnline(local_index_base_gpu, local_bitmap_gpu);
    unsigned long long int num_positive_matches = 0ull;

    size_t num_batches = data_graph.updated_edges_.size();
    std::cout << "Batch size = " << batch_size << ", " << num_batches << " batches\n";

    TIME_START();
    auto batch_idx = 0u;
    // If split into many many batches, the performance may not good.
    // Larger batch size is better for us.

    uint32_t **result_ptr_array = new uint32_t*[num_batches];
    uint32_t *result_size_array = new uint32_t[num_batches];
    for (const auto &batch : data_graph.updated_edges_) {
        std::cout << std::endl;
        std::cout << "Batch #" << batch_idx << " --------" << std::endl;
        //  Doubt: Why every time?
        match_gpu.SetGraphPtrs(data_graph_gpu);
        for (uint8_t i = 0u; i < QE_COUNT; i++) {
            // LTIME_START();
            std::cout << "Query Edge #" + std::to_string(i) << '\n';
            //  lgh: i is current idx_in_qe_list_
            match_gpu.BuildTries(batch, csr_gpu, i);
            match_gpu.UpdateGlobalIndex(data_graph_gpu, global_index_gpu, global_bitmap_gpu, csr_gpu, i);
            // LTIME_END();
            // LPRINT_LOCAL_TIME("Finish Query Edge #" + std::to_string(i));
        }

        // std::cout << "before build local index" << endl;
        bool local_index_nonempty = false;
        for (uint8_t i = 0u; i < QE_COUNT; i++) {
            if (match_gpu.GammaBuildLocalIndex(global_index_gpu, global_bitmap_gpu, local_index_base_gpu, local_bitmap_gpu, local_index, 
                                               csr_gpu, i, avg_degrees)){
                local_index_nonempty = true;
            }
        }
        // std::cout << "before GammaMatching" << endl;
        if (local_index_nonempty) {
            match_gpu.GammaMatching(global_index_gpu, local_index, num_positive_matches, 
                                    /*materialize_results=*/print_results, &(result_ptr_array[batch_idx]), &(result_size_array[batch_idx]));
        } else {
            std::cout << "All local indexes are empty. Skip matching." << '\n';
        }

        batch_idx++;
        match_gpu.DeallocTries(csr_gpu);
    }
    TIME_END();
    PRINT_LOCAL_TIME("Incremental Matching");

    match_gpu.DeallocOnline(local_index_base_gpu, local_bitmap_gpu);
    match_gpu.DeallocRelations(data_graph_gpu, global_index_gpu, global_bitmap_gpu);

    std::cout << "Num Positive Matches: " << num_positive_matches << '\n';

    if (print_results) {
        std::cout << "\n--------- Matching Results--------\n";
        for (uint32_t i = 0; i < num_batches; i++) {
            std::cout << "Batch #" << i << ":\n";
            print_matching_results(result_ptr_array[i], result_size_array[i], query_graph.getVerticesCount());
        }
    }
}

#endif  // GAMMA_H