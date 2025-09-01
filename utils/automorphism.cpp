#include <cstdint>
#include <set>

#include "graph/graph.h"
#include "utils/automorphism.h"
// #include "utils/sparsepp/spp.h"

void compute_automorphism(const QueryGraph *graph, std::vector<std::vector<uint8_t>> &embeddings) {
    // Note that this method is working on small graphs (tens of vertices) only.
    // Initialize resource.
    uint32_t n = graph->getVerticesCount();
    std::vector<bool> visited(n, false);
    std::vector<uint32_t> idx(n);
    std::vector<uint8_t> mapping(n);
    std::vector<std::vector<uint8_t>> local_candidates(n);
    std::vector<std::vector<uint8_t>> global_candidates(n);
    std::vector<std::vector<uint8_t>> backward_neighbors(n);

    // Initialize global candidates.
    for (uint8_t u = 0; u < n; ++u) {
        uint32_t u_label = graph->getVertexLabel(u);
        uint8_t u_degree = graph->getVertexDegree(u);

        for (uint8_t v = 0; v < n; ++v) {
            uint32_t v_label = graph->getVertexLabel(v);
            uint8_t v_degree = graph->getVertexDegree(v);

            if (v_label == u_label && v_degree >= u_degree)
                global_candidates[u].push_back(v);
        }
    }

    // Generate a matching order.
    std::vector<uint8_t> matching_order;
    uint8_t selected_vertex = 0;
    uint32_t selected_vertex_selectivity = global_candidates[selected_vertex].size();
    for (uint8_t u = 1; u < n; ++u) {
        if (global_candidates[u].size() < selected_vertex_selectivity){
            selected_vertex = u;
            selected_vertex_selectivity = global_candidates[u].size();
        }
    }

    matching_order.push_back(selected_vertex);
    visited[selected_vertex] = true;

    for (uint32_t i = 1; i < n; ++i) {
        selected_vertex_selectivity = n + 1;
        for (uint32_t u = 0; u < n; ++u) {
            if (!visited[u]) {
                bool is_feasible = false;

                uint8_t u_nbr_count;
                auto u_nbr = graph->getVertexNeighbors(u, u_nbr_count);
                for (uint8_t j = 0; j < u_nbr_count; ++j) {
                    uint8_t uu = u_nbr[j];

                    if (visited[uu]) {
                        is_feasible = true;
                        break;
                    }
                }

                if (is_feasible && global_candidates[u].size() < selected_vertex_selectivity) {
                    selected_vertex = u;
                    selected_vertex_selectivity = global_candidates[u].size();
                }
            }
        }
        matching_order.push_back(selected_vertex);
        visited[selected_vertex] = true;
    }

    std::fill(visited.begin(), visited.end(), false);

    // Set backward neighbors to compute local candidates.
    for (uint32_t i = 1; i < n; ++i) {
        uint8_t u = matching_order[i];
        for (uint32_t j = 0; j < i; ++j) {
            uint8_t uu = matching_order[j];

            if (graph->checkEdgeExistence(uu, u)) {
                backward_neighbors[u].push_back(uu);
            }
        }
    }

    // Recursive search along the matching order.
    int cur_level = 0;
    local_candidates[cur_level] = global_candidates[matching_order[0]];

    while (true) {
        while (idx[cur_level] < local_candidates[cur_level].size()) {
            uint32_t u = matching_order[cur_level];
            uint32_t v = local_candidates[cur_level][idx[cur_level]];
            idx[cur_level] += 1;

            if (cur_level == n - 1) {
                // Find an embedding
                mapping[u] = v;
                embeddings.push_back(mapping);
            } else {
                mapping[u] = v;
                visited[v] = true;
                cur_level += 1;
                idx[cur_level] = 0;

                {
                    // Compute local candidates.
                    u = matching_order[cur_level];
                    for (auto temp_v: global_candidates[u]) {
                        if (!visited[temp_v]) {
                            bool is_feasible = true;
                            for (auto uu: backward_neighbors[u]) {
                                uint32_t edge_label = graph->getEdgeLabelByVertex(u, uu);

                                uint32_t temp_vv = mapping[uu];
                                // TODO: check edge label
                                if (!graph->checkEdgeExistence(temp_v, temp_vv)
                                    || edge_label != graph->getEdgeLabelByVertex(temp_v, temp_vv)) {
                                    is_feasible = false;
                                    continue;
                                }
                            }
                            if (is_feasible)
                                local_candidates[cur_level].push_back(temp_v);
                        }
                    }
                }
            }
        }

        local_candidates[cur_level].clear();
        cur_level -= 1;
        if (cur_level < 0) {
            break;
        }
        visited[mapping[matching_order[cur_level]]] = false;
    }
}

std::pair<uint32_t, uint32_t> construct_vertex_pair(uint32_t a, uint32_t b) {
    if (a < b) {
        return {a, b};
    } else {
        return {b, a};
    }
}

void AutomorphismManager::detect_automorphism_edges(const QueryGraph *query_graph) {
    // Divide the vertex into disjoint sets based on automorphisms.
    compute_automorphism(query_graph, automorphisms_);

    uint32_t num_query_edges = query_graph->getEdgeCount();
    edge_list_idx_map_.resize(num_query_edges, UINT8_MAX);
    // equiv_edge_count_.resize(num_query_edges);
    equiv_edge_count_.resize(MAX_QE_COUNT);
    std::fill(equiv_edge_count_.begin(), equiv_edge_count_.end(), 0u);

    uint32_t n = query_graph->getVerticesCount();
    // spp::sparse_hash_set<std::pair<uint32_t, uint32_t>> selected;
    std::set<std::pair<uint32_t, uint32_t>> selected;

    const std::vector<std::pair<uint32_t, uint32_t>> & edge_list = query_graph->getEdgeList();

    representative_edge_list_idxs_.clear();
    for (uint8_t i = 0; i < num_query_edges; ++i) {
        uint32_t u = static_cast<uint32_t>(edge_list[i].first);
        uint32_t uu = static_cast<uint32_t>(edge_list[i].second);
        std::pair<uint32_t, uint32_t> e = {u, uu};
        // if (!selected.contains(e)) {
        if (selected.find(e) == selected.end()) {
            selected.insert(e);
            edge_list_idx_map_[i] = i;
            automorphism_edges_.push_back({e});
            uint8_t count = 1;
            for (auto &embedding: automorphisms_) {
                std::pair<uint32_t, uint32_t> mapped_e = construct_vertex_pair(embedding[u], embedding[uu]);
                
                if (selected.find(e) == selected.end()) {
                    selected.insert(mapped_e);

                    auto idx = query_graph->getEdgeListIdxByVertex(mapped_e.first, mapped_e.second);

                    edge_list_idx_map_[idx] = i;
                    automorphism_edges_.back().push_back(mapped_e);

                    count += 1u;
                }
            }
            equiv_edge_count_[i] = count;
            representative_edge_list_idxs_.push_back(i);
        }
    }

    for (uint8_t i = 0; i < num_query_edges; ++i) {
        if (equiv_edge_count_[i] == 0) {
            equiv_edge_count_[i] = equiv_edge_count_[edge_list_idx_map_[i]];
        }
    }
}