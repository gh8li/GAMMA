#include <algorithm>
#include <array>
#include <bitset>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <tuple>
#include <vector>

#include "graph/plan.h"
#include "graph/graph.h"
#include "utils/automorphism.h"

void OrderPerEdge::CopyFrom(const OrderPerEdge &other) {
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        vs_[i] = other.vs_[i];
    }
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        bni_offs_[i] = other.bni_offs_[i];
    }
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        bni_[i] = other.bni_[i];
    }
}

PlanManager::PlanManager(const QueryGraph& query_graph)
: query_(query_graph)
, num_query_vertices_not_tail_leaf_{}
, rebuild_flags_{}
, rebuild_v_flags_{}
, rebuild_R_flags_{}
, rebuild_B_flags_{}
, cartesian_product_info_{}
, indexing_orders_{}
, orders_{}
{}

// Plan::Plan(const QueryGraph& query_graph)
// : query_(query_graph)
// , am_ptr_(NULL)
// , rebuild_flags_{}
// , rebuild_v_flags_{}
// , rebuild_R_flags_{}
// , rebuild_B_flags_{}
// , cartesian_product_info_{}
// , indexing_orders_{}
// , orders_{}
// {}

// Plan::Plan(const QueryGraph& query_graph, const AutomorphismManager *am_ptr)
// : query_(query_graph)
// , am_ptr_(am_ptr)
// , rebuild_flags_{}
// , rebuild_v_flags_{}
// , rebuild_R_flags_{}
// , rebuild_B_flags_{}
// , cartesian_product_info_{}
// , indexing_orders_{}
// , orders_{}
// {}

void PlanManager::ComputeNumQueryVerticesNonTailLeaf() {
    uint32_t num_query_edges = query_.getEdgeCount();
    uint32_t num_query_vertices = query_.getVerticesCount();
    num_query_vertices_not_tail_leaf_.resize(num_query_edges, 0u);
    for (uint32_t i = 0; i < num_query_edges; i++) {
        int depth_count = 0;
        // while (depth_count < num_query_edges &&  // WRONG!
        while (depth_count < num_query_vertices &&
            cartesian_product_info_[i][depth_count] != PlanManager::CartesianProductType::TreeCartesianProduct &&
            cartesian_product_info_[i][depth_count] != PlanManager::CartesianProductType::TreeSingle ) {
            depth_count++;
        }
        num_query_vertices_not_tail_leaf_[i] = depth_count;
    }
}

void PlanManager::GenerateIndexingOrders()
{
    // first build indexing orders
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        BuildIndexingOrder(u0, u1, indexing_orders_[i]);
    }
}

void PlanManager::GenerateMatchingOrders_v1(uint32_t *cardinalities, float *avg_degrees, const bool enable_relation_switch) {
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_QV_COUNT> visited;

    // build matching order for each query edge based on the local index
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        cur_order.clear();
        visited.reset();

        // 1. ******************** find the starting vertex ********************
        std::vector<uint8_t> selected_vertices, further_selected_vertices;

        // choose the vertex with the maximum degree
        auto max_degree = 0u;
        for (auto i = 0u; i < query_.vcount_; i++)
        {
            auto degree = query_.nbrs_[i].size();
            if (degree > max_degree)
            {
                max_degree = degree;
                selected_vertices.clear();
                selected_vertices.emplace_back(i);
            }
            else if (degree == max_degree)
            {
                selected_vertices.emplace_back(i);
            }
        }
        // set the starting two vertices of the matching order
        visited[selected_vertices[0]] = true;
        cur_order.push_back(selected_vertices[0]);

        // 2. ******************** add other vertices to the order ********************
        for (auto i = 1u; i < query_.vcount_; i++)
        {
            std::vector<uint8_t> selected, further_selected;

            // 1. select the vertices with the maximum number of backward neighbors
            {
                auto max_num_bn = 0u;
                for (auto u = 0u; u < query_.vcount_; u++)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    auto cur_num_bns = 0u;
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn]) cur_num_bns += 1;
                    }
                    if (cur_num_bns > max_num_bn)
                    {
                        max_num_bn = cur_num_bns;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_bns == max_num_bn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u
            if (selected.size() > 1)
            {
                auto max_num_v = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_v = 0u;
                    std::vector<bool> temp_visited(query_.vcount_, false);
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn] && !temp_visited[bn])
                                {
                                    temp_visited[bn] = true;
                                    cur_num_v++;
                                }
                            }
                        }
                    }
                    if (cur_num_v > max_num_v)
                    {
                        max_num_v = cur_num_v;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_v == max_num_v)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 3. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u
            if (selected.size() > 1)
            {
                auto max_num_fn = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_fn = 0u;
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            auto no_visited_nbr_of_fn = true;
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn])
                                {
                                    no_visited_nbr_of_fn = false;
                                    break;
                                }
                            }
                            if (no_visited_nbr_of_fn)
                            {
                                cur_num_fn++;
                            }
                        }
                    }
                    if (cur_num_fn > max_num_fn)
                    {
                        max_num_fn = cur_num_fn;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_fn == max_num_fn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // insert the first selected vertex to the matching order
            cur_order.push_back(selected[0]);
            visited[selected[0]] = true;
        }

        if (enable_relation_switch) {
            // generate rebuild_flags_
            GenerateRebuildFlagsWithOrder(u0, u1, {cur_order[0], cur_order[1]}, rebuild_flags_[i]);
            GenerateRebuildVFlagsWithOrder(u0, u1, i);
            GenerateCartesianProductInfo(cur_order, i);
        } else {
            for (auto u = 0u; u < query_.getVerticesCount(); u++) {
                rebuild_B_flags_[i][u] = true;
                rebuild_R_flags_[i][u] = true;
            }
        }

        // generate meta for the matching order
        auto cum_offs = 0u;
        orders_[i].bni_offs_[0] = cum_offs;
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            const auto& u = cur_order[j];
            orders_[i].vs_[j] = u;
            for (auto k = j - 1; k < query_.vcount_; k--)
            {
                const auto& uu = orders_[i].vs_[k];
                auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
                if (it != query_.nbrs_[u].end() && it->first == uu)
                {
                    orders_[i].bni_[cum_offs++] = k;
                }
            }
            orders_[i].bni_offs_[j + 1] = cum_offs;
            std::sort(
                orders_[i].bni_ + orders_[i].bni_offs_[j],
                orders_[i].bni_ + orders_[i].bni_offs_[j + 1],
                [this, i, u, avg_degrees](const auto& bni1, const auto& bni2){
                    return avg_degrees[this->query_.eidx_[this->orders_[i].vs_[bni1] * this->query_.vcount_ + u]]
                    < avg_degrees[this->query_.eidx_[this->orders_[i].vs_[bni2] * this->query_.vcount_ + u]];
                }
            );
        }
    }
}

void PlanManager::GenerateMatchingOrders_v2(uint32_t *cardinalities, float *avg_degrees, const bool enable_relation_switch)
{
    // Values in cur_order is copied to orders_[edge].vs_ (edge is `index` in the for loop below)
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_QV_COUNT> visited;

    // build matching order for each query edge based on the local index
    for (auto index = 0u; index < query_.ecount_; index++)
    {
        const auto& [u0, u1] = query_.qe_list_[index];
        cur_order.clear();
        visited.reset();

        // 1. ******************** find the starting edge ********************
        std::vector<std::pair<uint8_t, uint8_t>> selected_edges, further_selected_edges;

        // choose the edges whose endpoints have a maximum sum degree
        auto max_sum_degree = 0u;
        for (auto i = 0u; i < query_.ecount_; i++)
        {
            auto sum_degree = query_.nbrs_[query_.qe_list_[i].first].size() + query_.nbrs_[query_.qe_list_[i].second].size();
            if (sum_degree > max_sum_degree)
            {
                max_sum_degree = sum_degree;
                selected_edges.clear();
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
            else if (sum_degree == max_sum_degree)
            {
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
        }
        // if there is a tie, choose the edges that form the greatest number of triangles
        if (selected_edges.size() > 1)
        {
            auto max_num_triangles = 0u;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto num_triangles = 0u;
                for (auto nbr = 0u; nbr < query_.vcount_; nbr++)
                {
                    auto found_uu0 = false, found_uu1 = false;
                    for (const auto& [u, _]: query_.nbrs_[nbr])
                    {
                        if (u == uu0) found_uu0 = true;
                        else if (u == uu1) found_uu1 = true;
                    }
                    if (found_uu0 && found_uu1) num_triangles++;
                }
                if (num_triangles > max_num_triangles)
                {
                    max_num_triangles = num_triangles;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (num_triangles == max_num_triangles)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if one of the selected edge is the updated edge, choose it as the starting edge of the order
        if (selected_edges.size() > 1)
        {
            for (const auto& [uu0, uu1]: selected_edges)
            {
                if ((uu0 == u0 && uu1 == u1) || (uu0 == u1 && uu1 == u0))
                {
                    further_selected_edges.emplace_back(u0, u1);
                    std::swap(selected_edges, further_selected_edges);
                    further_selected_edges.clear();
                    break;
                }
            }
        }
        // if there is a tie, choose the edges with the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_cardinality = UINT32_MAX;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto cardinality = cardinalities[query_.eidx_[uu0 * query_.vcount_ + uu1]];
                if (cardinality < min_cardinality)
                {
                    min_cardinality = cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (cardinality == min_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if there is a tie, choose the edges whose neighbor have the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_sum_cardinality = std::numeric_limits<float>::max();
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto sum_cardinality = 0.0f;
                for (const auto& [nbr, _]: query_.nbrs_[uu0])
                {
                    if (nbr == uu1) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu0 * query_.vcount_ + nbr]];
                }
                for (const auto& [nbr, _]: query_.nbrs_[uu1])
                {
                    if (nbr == uu0) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu1 * query_.vcount_ + nbr]];
                }
                if (sum_cardinality < min_sum_cardinality)
                {
                    min_sum_cardinality = sum_cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (sum_cardinality == min_sum_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        
        // set the starting two vertices of the matching order
        bool reverse = false;
        for (auto i = 0u; i < query_.vcount_; i++)
        {
            if (indexing_orders_[index].vs_[i] == selected_edges[0].first) break;
            if (indexing_orders_[index].vs_[i] == selected_edges[0].second) {reverse = true; break;}
        }
        visited[selected_edges[0].first] = true;
        visited[selected_edges[0].second] = true;
        if (!reverse)
        {
            cur_order.push_back(selected_edges[0].first);
            cur_order.push_back(selected_edges[0].second);
        }
        else
        {
            cur_order.push_back(selected_edges[0].second);
            cur_order.push_back(selected_edges[0].first);
        }

        // 2. ******************** add other vertices to the order ********************
        for (auto i = 2u; i < query_.vcount_; i++)
        {
            std::vector<uint8_t> selected, further_selected;

            // 0. gather all extendable vertices
            {
                std::vector<std::pair<uint8_t, uint32_t>> selected_with_avg_degree;
                // add valid ("valid" means unvisited) u into `selected_with_avg_degree`
                // for each item in `selected_with_avg_degree`, item.first is `u`, item.second stores the minimum avg_degree among all [$bn, u]
                // see main.cpp for the definition of avg_degree
                // $bn can be any neighbors of u that have been added to the matching order before u.
                for (auto u = 0u; u < query_.vcount_; u++)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            if (selected_with_avg_degree.empty() || selected_with_avg_degree.back().first != u)
                            {
                                selected_with_avg_degree.emplace_back(u, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                            }
                            else
                            {
                                if (avg_degrees[query_.eidx_[bn * query_.vcount_ + u]] < selected_with_avg_degree.back().second)
                                {
                                    selected_with_avg_degree.back().second = avg_degrees[query_.eidx_[bn * query_.vcount_ + u]];
                                }
                            }
                        }
                    }
                }
                // sort
                std::sort(
                    selected_with_avg_degree.begin(),
                    selected_with_avg_degree.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );
                // remove vertices with extreme large avg degrees
                auto new_end = selected_with_avg_degree.size();
                for (auto j = 0u; j < selected_with_avg_degree.size() - 1; j++)
                {
                    if (selected_with_avg_degree[j + 1].second >= selected_with_avg_degree[j].second * (2 << 9))
                    {
                        new_end = j + 1;
                        break;
                    }
                }
                for (auto j = 0u; j < new_end; j++)
                {
                    selected.push_back(selected_with_avg_degree[j].first);
                }
                std::sort(selected.begin(), selected.end());
            }

            // 1. select the vertices with the maximum number of backward neighbors
            if (selected.size() > 1)
            {
                auto max_num_bn = 0u;
                for (const auto& u: selected)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    auto cur_num_bns = 0u;
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn]) cur_num_bns += 1;
                    }
                    if (cur_num_bns > max_num_bn)
                    {
                        max_num_bn = cur_num_bns;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_bns == max_num_bn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one neighbor
            // that is not in the matching order and is connected with u.
            if (selected.size() > 1)
            {
                auto max_num_v = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_v = 0u;
                    std::vector<bool> temp_visited(query_.vcount_, false);
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn] && !temp_visited[bn])
                                {
                                    temp_visited[bn] = true;
                                    cur_num_v++;
                                }
                            }
                        }
                    }
                    if (cur_num_v > max_num_v)
                    {
                        max_num_v = cur_num_v;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_v == max_num_v)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 3. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 3. if there is a tie, compute the number of dangling neighbors of u.
            // a dangling neighbor `fn` means `fn` does not has neighbors that are already in the matching order 
            if (selected.size() > 1)
            {
                auto max_num_fn = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_fn = 0u;
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            auto no_visited_nbr_of_fn = true;
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn])
                                {
                                    no_visited_nbr_of_fn = false;
                                    break;
                                }
                            }
                            if (no_visited_nbr_of_fn)
                            {
                                cur_num_fn++;
                            }
                        }
                    }
                    if (cur_num_fn > max_num_fn)
                    {
                        max_num_fn = cur_num_fn;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_fn == max_num_fn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // 4. if there is a tie, select a vertex with the minimum number of candiates on extension,
            // where the number of candidates is estimated by the min property

            // 4. if there is a tie, select a vertex with the minimum value of "minimum avg_degree among all [$bn, u]"
            // see main.cpp for the definition of avg_degree
            // $bn can be any neighbors of u that have been added to the matching order before u.
            if (selected.size() > 1)
            {
                auto min_num_candidates = std::numeric_limits<float>::max();
                for (const auto& u: selected)
                {
                    auto cur_num_candidates = std::numeric_limits<float>::max();
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            cur_num_candidates = std::min(cur_num_candidates, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                        }
                    }
                    if (cur_num_candidates < min_num_candidates)
                    {
                        min_num_candidates = cur_num_candidates;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_candidates == min_num_candidates)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // insert the first selected vertex to the matching order
            cur_order.push_back(selected[0]);
            visited[selected[0]] = true;
        }

        SetRebuildFlags(enable_relation_switch, u0, u1, cur_order, index);

        GenerateCartesianProductInfo(cur_order, index);

        // generate meta for the matching order
        auto cum_offs = 0u;
        // fills in orders_[index].bni_ and orders_[index].bni_offs_ (two arrays, backward_neighbor_index and backward_neighbor_index_offsets)
        // for each u, sort the orders_[index].bni_ according to the avg_degree of [bn, u] (ascending)
        orders_[index].bni_offs_[0] = cum_offs;
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            const auto& u = cur_order[j];
            orders_[index].vs_[j] = u;
            // Doubt: Does this mean k >= 0?
            for (auto k = j - 1; k < query_.vcount_; k--)
            {
                const auto& uu = orders_[index].vs_[k];
                auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
                if (it != query_.nbrs_[u].end() && it->first == uu)
                {
                    orders_[index].bni_[cum_offs++] = k;
                }
            }
            orders_[index].bni_offs_[j + 1] = cum_offs;
            std::sort(
                orders_[index].bni_ + orders_[index].bni_offs_[j],
                orders_[index].bni_ + orders_[index].bni_offs_[j + 1],
                [this, index, u, avg_degrees](const auto& bni1, const auto& bni2){
                    return avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni1] * this->query_.vcount_ + u]]
                    < avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni2] * this->query_.vcount_ + u]];
                }
            );
        }
    }

    ComputeNumQueryVerticesNonTailLeaf();
}

void PlanManager::SetRebuildFlags(const bool enable_relation_switch, const uint32_t &u0, const uint32_t &u1, 
                                  std::vector<uint8_t> &cur_order, unsigned int edge_list_idx) {
    if (enable_relation_switch) {
        // generate rebuild_flags_
        GenerateRebuildFlagsWithOrder(u0, u1, {cur_order[0], cur_order[1]}, rebuild_flags_[edge_list_idx]);
        // GenerateRebuildVFlagsWithOrder(u0, u1, index);  // Commented by xsunax
        GenerateRebuildVFlagsWithOrder_v2(u0, u1, edge_list_idx, cur_order);
    }
    else {
        for (auto u = 0u; u < query_.getVerticesCount(); u++) {
            rebuild_B_flags_[edge_list_idx][u] = true;
            rebuild_R_flags_[edge_list_idx][u] = true;
        }
    }
}

void PlanManager::GenerateOursMatchingOrders(uint32_t *cardinalities, float *avg_degrees)
{
    // Values in cur_order is copied to orders_[edge].vs_ (edge is `index` in the for loop below)
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_QV_COUNT> visited;

    // build matching order for each query edge based on the local index
    for (auto index = 0u; index < query_.ecount_; index++)
    {
        const auto& [u0, u1] = query_.qe_list_[index];
        cur_order.clear();
        visited.reset();

        // 1. ******************** find the starting edge ********************
        std::vector<std::pair<uint8_t, uint8_t>> selected_edges, further_selected_edges;

        // choose the edges whose endpoints have a maximum sum degree
        auto max_sum_degree = 0u;
        for (auto i = 0u; i < query_.ecount_; i++)
        {
            auto sum_degree = query_.nbrs_[query_.qe_list_[i].first].size() + query_.nbrs_[query_.qe_list_[i].second].size();
            if (sum_degree > max_sum_degree)
            {
                max_sum_degree = sum_degree;
                selected_edges.clear();
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
            else if (sum_degree == max_sum_degree)
            {
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
        }
        // if there is a tie, choose the edges that form the greatest number of triangles
        if (selected_edges.size() > 1)
        {
            auto max_num_triangles = 0u;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto num_triangles = 0u;
                for (auto nbr = 0u; nbr < query_.vcount_; nbr++)
                {
                    auto found_uu0 = false, found_uu1 = false;
                    for (const auto& [u, _]: query_.nbrs_[nbr])
                    {
                        if (u == uu0) found_uu0 = true;
                        else if (u == uu1) found_uu1 = true;
                    }
                    if (found_uu0 && found_uu1) num_triangles++;
                }
                if (num_triangles > max_num_triangles)
                {
                    max_num_triangles = num_triangles;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (num_triangles == max_num_triangles)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if one of the selected edge is the updated edge, choose it as the starting edge of the order
        if (selected_edges.size() > 1)
        {
            for (const auto& [uu0, uu1]: selected_edges)
            {
                if ((uu0 == u0 && uu1 == u1) || (uu0 == u1 && uu1 == u0))
                {
                    further_selected_edges.emplace_back(u0, u1);
                    std::swap(selected_edges, further_selected_edges);
                    further_selected_edges.clear();
                    break;
                }
            }
        }
        // if there is a tie, choose the edges with the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_cardinality = UINT32_MAX;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto cardinality = cardinalities[query_.eidx_[uu0 * query_.vcount_ + uu1]];
                if (cardinality < min_cardinality)
                {
                    min_cardinality = cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (cardinality == min_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if there is a tie, choose the edges whose neighbor have the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_sum_cardinality = std::numeric_limits<float>::max();
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto sum_cardinality = 0.0f;
                for (const auto& [nbr, _]: query_.nbrs_[uu0])
                {
                    if (nbr == uu1) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu0 * query_.vcount_ + nbr]];
                }
                for (const auto& [nbr, _]: query_.nbrs_[uu1])
                {
                    if (nbr == uu0) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu1 * query_.vcount_ + nbr]];
                }
                if (sum_cardinality < min_sum_cardinality)
                {
                    min_sum_cardinality = sum_cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (sum_cardinality == min_sum_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        
        // set the starting two vertices of the matching order
        bool reverse = false;
        for (auto i = 0u; i < query_.vcount_; i++)
        {
            if (indexing_orders_[index].vs_[i] == selected_edges[0].first) break;
            if (indexing_orders_[index].vs_[i] == selected_edges[0].second) {reverse = true; break;}
        }
        visited[selected_edges[0].first] = true;
        visited[selected_edges[0].second] = true;
        if (!reverse)
        {
            cur_order.push_back(selected_edges[0].first);
            cur_order.push_back(selected_edges[0].second);
        }
        else
        {
            cur_order.push_back(selected_edges[0].second);
            cur_order.push_back(selected_edges[0].first);
        }

        // 2. ******************** add other vertices to the order ********************
        for (auto i = 2u; i < query_.vcount_; i++)
        {
            std::vector<uint8_t> selected, further_selected;

            // 0. gather all extendable vertices
            {
                std::vector<std::pair<uint8_t, uint32_t>> selected_with_avg_degree;
                // add valid ("valid" means unvisited) u into `selected_with_avg_degree`
                // for each item in `selected_with_avg_degree`, item.first is `u`, item.second stores the minimum avg_degree among all [$bn, u]
                // see main.cpp for the definition of avg_degree
                // $bn can be any neighbors of u that have been added to the matching order before u.
                for (auto u = 0u; u < query_.vcount_; u++)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            if (selected_with_avg_degree.empty() || selected_with_avg_degree.back().first != u)
                            {
                                selected_with_avg_degree.emplace_back(u, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                            }
                            else
                            {
                                if (avg_degrees[query_.eidx_[bn * query_.vcount_ + u]] < selected_with_avg_degree.back().second)
                                {
                                    selected_with_avg_degree.back().second = avg_degrees[query_.eidx_[bn * query_.vcount_ + u]];
                                }
                            }
                        }
                    }
                }
                // sort
                std::sort(
                    selected_with_avg_degree.begin(),
                    selected_with_avg_degree.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );
                // remove vertices with extreme large avg degrees
                auto new_end = selected_with_avg_degree.size();
                for (auto j = 0u; j < selected_with_avg_degree.size() - 1; j++)
                {
                    if (selected_with_avg_degree[j + 1].second >= selected_with_avg_degree[j].second * (2 << 9))
                    {
                        new_end = j + 1;
                        break;
                    }
                }
                for (auto j = 0u; j < new_end; j++)
                {
                    selected.push_back(selected_with_avg_degree[j].first);
                }
                std::sort(selected.begin(), selected.end());
            }

            // 1. select the vertices with the maximum number of backward neighbors
            if (selected.size() > 1)
            {
                auto max_num_bn = 0u;
                for (const auto& u: selected)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    auto cur_num_bns = 0u;
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn]) cur_num_bns += 1;
                    }
                    if (cur_num_bns > max_num_bn)
                    {
                        max_num_bn = cur_num_bns;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_bns == max_num_bn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one neighbor
            // that is not in the matching order and is connected with u.
            if (selected.size() > 1)
            {
                auto max_num_v = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_v = 0u;
                    std::vector<bool> temp_visited(query_.vcount_, false);
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn] && !temp_visited[bn])
                                {
                                    temp_visited[bn] = true;
                                    cur_num_v++;
                                }
                            }
                        }
                    }
                    if (cur_num_v > max_num_v)
                    {
                        max_num_v = cur_num_v;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_v == max_num_v)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 3. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 3. if there is a tie, compute the number of dangling neighbors of u.
            // a dangling neighbor `fn` means `fn` does not has neighbors that are already in the matching order 
            if (selected.size() > 1)
            {
                auto max_num_fn = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_fn = 0u;
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            auto no_visited_nbr_of_fn = true;
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn])
                                {
                                    no_visited_nbr_of_fn = false;
                                    break;
                                }
                            }
                            if (no_visited_nbr_of_fn)
                            {
                                cur_num_fn++;
                            }
                        }
                    }
                    if (cur_num_fn > max_num_fn)
                    {
                        max_num_fn = cur_num_fn;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_fn == max_num_fn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // 4. if there is a tie, select a vertex with the minimum number of candiates on extension,
            // where the number of candidates is estimated by the min property

            // 4. if there is a tie, select a vertex with the minimum value of "minimum avg_degree among all [$bn, u]"
            // see main.cpp for the definition of avg_degree
            // $bn can be any neighbors of u that have been added to the matching order before u.
            if (selected.size() > 1)
            {
                auto min_num_candidates = std::numeric_limits<float>::max();
                for (const auto& u: selected)
                {
                    auto cur_num_candidates = std::numeric_limits<float>::max();
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            cur_num_candidates = std::min(cur_num_candidates, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                        }
                    }
                    if (cur_num_candidates < min_num_candidates)
                    {
                        min_num_candidates = cur_num_candidates;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_candidates == min_num_candidates)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // insert the first selected vertex to the matching order
            cur_order.push_back(selected[0]);
            visited[selected[0]] = true;
        }
        // generate rebuild_flags_
        for (auto u = 0u; u < query_.getVerticesCount(); u++) {
            rebuild_B_flags_[index][u] = true;
            rebuild_R_flags_[index][u] = true;
        }
        GenerateCartesianProductInfo(cur_order, index);

        // generate meta for the matching order
        auto cum_offs = 0u;
        // fills in orders_[index].bni_ and orders_[index].bni_offs_ (two arrays, backward_neighbor_index and backward_neighbor_index_offsets)
        // for each u, sort the orders_[index].bni_ according to the avg_degree of [bn, u] (ascending)
        orders_[index].bni_offs_[0] = cum_offs;
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            const auto& u = cur_order[j];
            orders_[index].vs_[j] = u;
            // Doubt: Does this mean k >= 0?
            for (auto k = j - 1; k < query_.vcount_; k--)
            {
                const auto& uu = orders_[index].vs_[k];
                auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
                if (it != query_.nbrs_[u].end() && it->first == uu)
                {
                    orders_[index].bni_[cum_offs++] = k;
                }
            }
            orders_[index].bni_offs_[j + 1] = cum_offs;
            std::sort(
                orders_[index].bni_ + orders_[index].bni_offs_[j],
                orders_[index].bni_ + orders_[index].bni_offs_[j + 1],
                [this, index, u, avg_degrees](const auto& bni1, const auto& bni2){
                    return avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni1] * this->query_.vcount_ + u]]
                    < avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni2] * this->query_.vcount_ + u]];
                }
            );
        }
    }

    ComputeNumQueryVerticesNonTailLeaf();
}

void PlanManager::PrintOrders()
{
    std::cout << "Orders:\n";
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        std::cout << '[' << u0 << ',' << u1 << "]:\n    indexing order:";
        /*for (uint32_t j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[j]) << " (";
            for (uint8_t k = indexing_orders_[i].bni_offs_[j]; k < indexing_orders_[i].bni_offs_[j + 1]; k++)
            {
                std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[indexing_orders_[i].bni_[k]]) << '@'
                    <<static_cast<uint32_t>(indexing_orders_[i].bni_[k]);
                if (rebuild_flags_[i][indexing_orders_[i].vs_[j] * query_.vcount_ + indexing_orders_[i].vs_[indexing_orders_[i].bni_[k]]])
                {
                    std::cout << '+';
                }
                if (k != indexing_orders_[i].bni_offs_[j + 1] - 1)
                {
                    std::cout << ' ';
                }
            }
            std::cout << ") ";
        }
        std::cout << "\n    |- local index: ";
        for (uint8_t ii = 0u; ii < query_.vcount_; ii++)
        {
            for (uint8_t jj = 0u; jj < query_.vcount_; jj++)
            {
                if (rebuild_flags_[i][ii * query_.vcount_ + jj])
                {
                    std::cout << '(' << static_cast<uint32_t>(ii) << ',' << static_cast<uint32_t>(jj) << ") ";
                }
            }
        }
        std::cout << "\n    |- local index: build v ";
        for (uint8_t ii = 0u; ii < query_.vcount_; ii++)
        {
            const uint8_t& u = indexing_orders_[i].vs_[ii];
            if (rebuild_v_flags_[i][u])
            {
                std::cout << static_cast<uint32_t>(u) << ' ';
            }
        }*/
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[j]);
            std::cout << (rebuild_R_flags_[i][indexing_orders_[i].vs_[j]] ? "-buildR" : "");
            std::cout << (rebuild_B_flags_[i][indexing_orders_[i].vs_[j]] ? "-buildB-(" : "-(");
            for (auto k = indexing_orders_[i].bni_offs_[j]; k < indexing_orders_[i].bni_offs_[j + 1]; k++)
            {
                std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[indexing_orders_[i].bni_[k]]) << '@'
                    <<static_cast<uint32_t>(indexing_orders_[i].bni_[k]);
                if (k != indexing_orders_[i].bni_offs_[j + 1] - 1)
                {
                    std::cout << ' ';
                }
            }
            std::cout << ") ";
        }
        std::cout << "\n    order: ";
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(orders_[i].vs_[j]) << "-(";
            for (auto k = orders_[i].bni_offs_[j]; k < orders_[i].bni_offs_[j + 1]; k++)
            {
                std::cout << static_cast<uint32_t>(orders_[i].vs_[orders_[i].bni_[k]]) << '@'
                    << static_cast<uint32_t>(orders_[i].bni_[k]);
                if (k != orders_[i].bni_offs_[j + 1] - 1)
                {
                    std::cout << ' ';
                }
            }
            std::cout << ") ";
        }
        std::cout << "\n    cartesian product info: ";
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(orders_[i].vs_[j]);
            if (j == 0u)
            {
                std::cout << ' ';
            }
            else
            {
                std::cout << "-(" << static_cast<uint32_t>(cartesian_product_info_[i][j]) << ") ";
            }
        }
        std::cout << "\n";
    }
}

void PlanManager::AddVertices(
    std::vector<uint8_t>& order,
    std::bitset<MAX_QV_COUNT>& visited,
    const std::vector<uint8_t>& vrole,
    const uint8_t group_id
) {
    // 1. find all initial extendable vertices
    std::bitset<MAX_QV_COUNT> extendable;
    auto extendable_count = 0u;
    std::vector<uint8_t> extendable_score(query_.vcount_, 0);
    for (auto v = 0u; v < query_.vcount_; v++)
    {
        if (visited[v] || vrole[v] != group_id) continue;
        // check if v is connected with one vertex in the current order
        for (const auto& [nbr, label]: query_.nbrs_[v])
        {
            if (!visited[nbr]) continue;
            if (!extendable[v])
            {
                extendable_count += 1;
                extendable[v] = true;
            }
            extendable_score[v] += 1;
        }
    }

    // 2. move an extendable vertex to the matching order and update the extendable vertices
    while (extendable_count != 0)
    {
        auto selected_v = 0u, selected_v_score = 0u;
        for (auto i = 0u; i < query_.vcount_; i++)
        {
            if (extendable[i] && extendable_score[i] > selected_v_score)
            {
                selected_v_score = extendable_score[i];
                selected_v = i;
            }
        }
        order.push_back(selected_v);
        visited[selected_v] = true;
        extendable[selected_v] = false;
        extendable_count --;
        for (const auto& [nbr, label]: query_.nbrs_[selected_v])
        {
            if (visited[nbr] || vrole[nbr] != group_id) continue;
            if (!extendable[nbr])
            {
                extendable_count += 1;
                extendable[nbr] = true;
            }
            extendable_score[nbr] += 1;
        }
    }
}

void PlanManager::AddVertices(
    std::vector<uint8_t>& order,
    std::bitset<MAX_QV_COUNT>& visited,
    const std::vector<uint8_t>& vrole
) {
    // 1. find all initial extendable vertices
    std::bitset<MAX_QV_COUNT> extendable;
    auto extendable_count = 0u;
    std::vector<std::pair<uint8_t, uint8_t>> extendable_score(query_.vcount_, {0,0});
    for (auto v = 0u; v < query_.vcount_; v++)
    {
        // check if v is connected with on vertex in the current order
        if (visited[v]) continue;
        for (const auto& [nbr, label]: query_.nbrs_[v])
        {
            if (!visited[nbr]) continue;
            if (!extendable[v])
            {
                extendable_count += 1;
                extendable[v] = true;
                extendable_score[v].first = vrole[v] > 0 ? 1 : 0;
            }
            extendable_score[v].second += 1;
        }
    }

    // 2. move an extendable vertex to the matching order and update the extendable vertices
    while (extendable_count != 0)
    {
        uint8_t selected_v;
        std::pair<uint8_t, uint8_t> selected_v_score {0u, 0u};
        for (uint32_t i = 0; i < query_.vcount_; i++)
        {
            if (extendable[i] && extendable_score[i] > selected_v_score)
            {
                selected_v_score = extendable_score[i];
                selected_v = i;
            }
        }
        order.push_back(selected_v);
        visited[selected_v] = true;
        extendable[selected_v] = false;
        extendable_count --;
        for (const auto& [nbr, label]: query_.nbrs_[selected_v])
        {
            if (visited[nbr]) continue;
            if (!extendable[nbr])
            {
                extendable_count += 1;
                extendable[nbr] = true;
                extendable_score[nbr].first = vrole[nbr] > 0 ? 1 : 0;
            }
            extendable_score[nbr].second += 1;
        }
    }
}

uint8_t PlanManager::ChooseStartingVertex(
    const std::vector<uint8_t>& vrole,
    std::bitset<MAX_QV_COUNT * MAX_QV_COUNT>& rebuild_flags,
    uint8_t u0, uint8_t u1
) {
    // find a vertex in the densest region with the mininum number of relations
    // to build in the local index
    std::tuple<uint8_t, uint8_t, uint32_t> max_score {0u,0u,0u};
    uint8_t selected_v = 0u;
    for (uint8_t v = 0u; v < query_.vcount_; v++)
    {
        // compute the number of relations to build
        std::bitset<MAX_QV_COUNT * MAX_QV_COUNT> temp_rebuild_flags;
        GenerateRebuildFlagsWithOrder(u0, u1, {v}, temp_rebuild_flags);
        auto n = temp_rebuild_flags.count();

        // update the first vertex in the matching order
        std::tuple<uint8_t, uint8_t, uint32_t> temp_score {vrole[v] >= 2 ? 2 : vrole[v], UINT8_MAX - n, query_.nbrs_[v].size()};
        if (temp_score > max_score)
        {
            max_score = temp_score;
            selected_v = v;
            std::swap(rebuild_flags, temp_rebuild_flags);
        }
    }
    return selected_v;
}

/*uint8_t Plan::GetNumBuildRelations(
    uint8_t update_u0, uint8_t update_u1,
    uint8_t starting_vertex,
    std::bitset<MAX_VCOUNT * MAX_VCOUNT>& temp_rebuild_flags
) {
    rebuild_flags[update_u0 * query_.vcount_ + update_u1] = true;
    rebuild_flags[update_u1 * query_.vcount_ + update_u0] = true;

    std::bitset<MAX_VCOUNT> visited;
    visited[starting_vertex] = true;
    std::vector<uint8_t> path(query_.vcount_, UINT8_MAX);
    path[0] = starting_vertex;

    // DFS call-stack
    std::vector<uint8_t> candidate_index(query_.vcount_, 0u);
    auto depth = 1u;

    while (true)
    {
        while (candidate_index[depth] < query_.nbrs_[path[depth - 1]].size())
        {
            const auto& next_v = query_.nbrs_[path[depth - 1]][candidate_index[depth]].first;
            if (visited[next_v])
            {
                candidate_index[depth]++;
                continue;
            }
            if (next_v == update_u0 || next_v == update_u1)
            {
                // a path is found, set the rebuild flag
                path[depth] = next_v;
                for (auto i = 0u; i < depth; i++)
                {
                    temp_rebuild_flags[path[i] * query_.vcount_ + path[i + 1]] = true;
                    temp_rebuild_flags[path[i + 1] * query_.vcount_ + path[i]] = true;
                }
                path[depth] = UINT8_MAX;
                candidate_index[depth]++;
                continue;
            }

            visited[next_v] = true;
            path[depth] = next_v;
            depth++;
        }
        if (candidate_index[depth] >= query_.nbrs_[path[depth - 1]].size())
        {
            candidate_index[depth] = 0u;
            depth--;
            if (depth == 0u) break;
            visited[path[depth]] = false;
            candidate_index[depth] ++;
        }
    }
    return temp_rebuild_flags.count();
}*/

void PlanManager::BuildIndexingOrder(
    const uint8_t u0, const uint8_t u1,
    OrderPerEdge& indexing_orders_
) {
    // get the bfs order of the query vertices
    std::vector<uint8_t> cur_order {u0, u1};

    std::vector<uint8_t> pre_level {u0, u1};
    std::vector<uint8_t> cur_level;
    std::bitset<MAX_QV_COUNT> visited;
    visited[u0] = true;
    visited[u1] = true;

    // Starting from u0 and u1, do breadth-first search.
    while (!pre_level.empty())
    {
        for (const auto& pre_v: pre_level)
        {
            for (const auto& [nbr, label]: query_.nbrs_[pre_v])
            {
                if (!visited[nbr])
                {
                    cur_level.push_back(nbr);
                    visited[nbr] = true;
                    cur_order.push_back(nbr);
                }
            }
        }
        pre_level.clear();
        std::swap(pre_level, cur_level);
    }

    // fill in the indexing_order
    indexing_orders_.vs_[0] = cur_order[0];
    indexing_orders_.bni_offs_[0] = 0u;
    indexing_orders_.bni_[0] = 1u;
    auto cum_offs = 1u;
    indexing_orders_.bni_offs_[1] = 1u;
    for (auto j = 1u; j < query_.vcount_; j++)
    {
        const auto& u = cur_order[j];
        indexing_orders_.vs_[j] = u;
        for (auto k = 0u; k < j; k++)
        {
            const auto& uu = indexing_orders_.vs_[k];
            // Doubt: What if the edge label of (u, uu) in query_.nbrs_ is larger than 0 ?
            auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
            if (it != query_.nbrs_[u].end() && it->first == uu)
            {
                indexing_orders_.bni_[cum_offs++] = k;
            }
        }
        indexing_orders_.bni_offs_[j + 1] = cum_offs;
    }
}

// Find all paths from u0 and u1 (respectively) to cur_order[0] or cur_order[1], using depth-first search. 
// Set the flags of all edges in the paths to true. (In the Bitset rebuild_flags_[index])
void PlanManager::GenerateRebuildFlagsWithOrder(
    uint8_t update_u0, uint8_t update_u1,
    std::initializer_list<uint8_t>&& starting_vertices,
    std::bitset<MAX_QV_COUNT * MAX_QV_COUNT>& rebuild_flags
) {
    rebuild_flags[update_u0 * query_.vcount_ + update_u1] = true;
    rebuild_flags[update_u1 * query_.vcount_ + update_u0] = true;

    std::vector<uint8_t> path(query_.vcount_, UINT8_MAX);
    for (const auto& starting_vertex: starting_vertices)
    {
        std::bitset<MAX_QV_COUNT> visited;
        visited[starting_vertex] = true;
        path[0] = starting_vertex;
        std::vector<uint8_t> candidate_index(query_.vcount_, 0u);
        auto depth = 1u;

        // Find all paths from starting_vertex to update_u0 or update_u1, using depth-first search. 
        // Set the flags of all edges in the paths to true. (rebuild_flags)
        while (true)
        {
            while (candidate_index[depth] < query_.nbrs_[path[depth - 1]].size())
            {
                const auto& next_v = query_.nbrs_[path[depth - 1]][candidate_index[depth]].first;
                if (visited[next_v])
                {
                    candidate_index[depth]++;
                    continue;
                }
                if (next_v == update_u0 || next_v == update_u1)
                {
                    // a path is found, set the rebuild flag
                    path[depth] = next_v;
                    for (auto i = 0u; i < depth; i++)
                    {
                        rebuild_flags[path[i] * query_.vcount_ + path[i + 1]] = true;
                        rebuild_flags[path[i + 1] * query_.vcount_ + path[i]] = true;
                    }
                    path[depth] = UINT8_MAX;
                    candidate_index[depth]++;
                    continue;
                }

                visited[next_v] = true;
                path[depth] = next_v;
                depth++;
            }
            if (candidate_index[depth] >= query_.nbrs_[path[depth - 1]].size())
            {
                candidate_index[depth] = 0u;
                depth--;
                if (depth == 0u) break;
                visited[path[depth]] = false;
                candidate_index[depth] ++;
            }
        }
    }
}

void PlanManager::GenerateRebuildVFlagsWithOrder(
    uint8_t update_u0, uint8_t update_u1,
    uint8_t index
) {
    // iterate over vertices on the indexing order
    for (auto i = 0u; i < query_.vcount_; i++)
    {
        const auto& u = indexing_orders_[index].vs_[i];
        if (i == 0u)
        {
            rebuild_v_flags_[index][u] = true;
            continue;
        }
        uint8_t sum = std::transform_reduce(
            &indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i]],
            &indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i + 1]],
            0,
            [](const auto& v1, const auto& v2){return v1 + v2;},
            [this, u, index](const auto& bni){
                return this->rebuild_flags_[index][this->indexing_orders_[index].vs_[bni] * this->query_.vcount_ + u] ? 1u : 0u;
            }
        );
        if (sum == 0u)
        {
            rebuild_v_flags_[index][u] = false;
        }
        else if (sum == indexing_orders_[index].bni_offs_[i + 1] - indexing_orders_[index].bni_offs_[i])
        {
            rebuild_v_flags_[index][u] = true;
        }
        else
        {
            std::cout << "Error in order generation!\n";
            exit(-1);
        }
    }
}

// Fill values in rebuild_R_flags_[index] and rebuild_B_flags_[index]
void PlanManager::GenerateRebuildVFlagsWithOrder_v2(
    uint8_t update_u0, uint8_t update_u1,
    uint8_t index, std::vector<uint8_t>& cur_order
) {
    // iterate over vertices on the indexing order and construct rebuild_R_flags_[index]
    // for edge(u0, u1), indexing_orders_[edge] always starts with u0 and u1.
    // rebuild_R_flags_[edge][u] (u is not updated_u0 and updated_u1) is true, if u has more than one backward neighbors,
    // or if the only backward neighbor of u appears after u in the cur_order (the matching order)
    //
    // rebuild_R_flags_[edge][u] is false, only if u has only one backward neighbor `bn` and
    // `bn` appears before u in cur_order (the matching order) 
    for (auto i = 2u; i < query_.vcount_; i++)
    {
        const auto& u = indexing_orders_[index].vs_[i];
        if (indexing_orders_[index].bni_offs_[i + 1] - indexing_orders_[index].bni_offs_[i] == 1)
        {
            const auto& u_backward = indexing_orders_[index].vs_[indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i]]];
            bool found_u_backward_first = false;
            for (auto j = 0u; j < query_.vcount_; j++)
            {
                if (cur_order[j] == u) break;
                if (cur_order[j] == u_backward)
                {
                    found_u_backward_first = true;
                    break;
                }
            }
            if (found_u_backward_first) rebuild_R_flags_[index][u] = false;
            else rebuild_R_flags_[index][u] = true;
        }
        else
        {
            rebuild_R_flags_[index][u] = true;
        }
    }
    // iterate over vertices on the indexing order and construct rebuild_B_flags_[index]
    std::vector<uint32_t> degrees(query_.vcount_);
    for (auto i = 0u; i < query_.vcount_; i++)
    {
        degrees[i] = query_.nbrs_[i].size();
    }
    // rebuild_B_flags_[edge][u] is true, if the degree of u is larger than 1 (after removing dangling neighbors) or 
    // the only neighbor of u appears after u in indexing_orders_[index] or
    // the only neighbor of u appears after u in cur_order (the matching order).
    // the dangling neighbor `dn` of u means that, u is the only neighbor of `dn` and
    // `dn` appears after u in indexing_orders_[index] and `dn` appears after u in cur_order.

    // rebuild_B_flags_[edge][u] is false, only if the degree of u is 1 (after removing dangling neighbors) and 
    // the only neighbor of u appears before u in indexing_orders_[index] and
    // the only neighbor of u appears before u in cur_order (the matching order).
    for (auto i = query_.vcount_ - 1; i >= 2; i--)
    {
        const auto& u = indexing_orders_[index].vs_[i];
        if (degrees[u] == 1u && indexing_orders_[index].bni_offs_[i + 1] - indexing_orders_[index].bni_offs_[i] == 1)
        {
            const auto& u_backward = indexing_orders_[index].vs_[indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i]]];
            bool found_u_backward_first = false;
            for (auto j = 0u; j < query_.vcount_; j++)
            {
                if (cur_order[j] == u) break;
                if (cur_order[j] == u_backward)
                {
                    found_u_backward_first = true;
                    break;
                }
            }
            if (found_u_backward_first)
            {
                rebuild_B_flags_[index][u] = false;
                degrees[u]--;
                degrees[u_backward]--;
            }
            else rebuild_B_flags_[index][u] = true;
        }
        else
        {
            rebuild_B_flags_[index][u] = true;
        }
    }
}

void PlanManager::GenerateCartesianProductInfo(
    std::vector<uint8_t>& cur_order,
    uint8_t index
) {
    CartesianProductType type;
    uint8_t vertex = cur_order[query_.vcount_ - 1];
    if (query_.nbrs_[vertex].size() == 1)
    {
        cartesian_product_info_[index][query_.vcount_ - 1] = CartesianProductType::TreeSingle;
        type = CartesianProductType::TreeCartesianProduct;
    }
    else
    {
        cartesian_product_info_[index][query_.vcount_ - 1] = CartesianProductType::NonTreeSingle;
        type = CartesianProductType::NonTreeCartesianProduct;
    }
    for (auto i = 1u; i < query_.vcount_ - 1; i++)
    {
        if (type == CartesianProductType::None)
        {
            cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::None;
        }
        else
        {
            vertex = cur_order[query_.vcount_ - 1 - i];
            bool valid = true;
            // check if there exist a forward neighbor
            for (auto j = 0u; j < i; j++)
            {
                uint8_t v_other = cur_order[query_.vcount_ - 1 - j];
                auto it = std::lower_bound(query_.nbrs_[vertex].begin(), query_.nbrs_[vertex].end(), std::make_pair<uint32_t, uint32_t>(v_other, 0));
                if (it != query_.nbrs_[vertex].end() && it->first == v_other)
                {
                    valid = false;
                    break;
                }
            }
            if (valid) // `vertex` has no forward neighbor.
            {
                if (query_.nbrs_[vertex].size() == 1 && type == CartesianProductType::TreeCartesianProduct)
                {
                    cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::TreeCartesianProduct;
                }
                else
                {
                    cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::NonTreeCartesianProduct;
                    type = CartesianProductType::NonTreeCartesianProduct;
                }
            }
            else // `vertex` has at least one forward neighbor
            {
                cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::None;
                type = CartesianProductType::None;
            }
        }
    }
}

void PlanManager::GenerateGammaMatchingOrders(const uint32_t *cardinalities, const float *avg_degrees, 
                                              const bool enable_relation_switch) {
    // Values in cur_order is copied to orders_[edge].vs_ (edge is `index` in the for loop below)
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_QV_COUNT> visited;

    // std::cout << "before for-loop" << endl;

    // build matching order for each query edge
    for (auto index = 0u; index < query_.ecount_; index++)
    {
        const auto& [u0, u1] = query_.qe_list_[index];
        cur_order.clear();
        visited.reset();

        // 1. ******************** set the starting edge ********************

        cur_order.push_back(u0);
        visited[u0] = true;

        cur_order.push_back(u1);
        visited[u1] = true;

        // 2. ******************** add other vertices to the order ********************
        for (auto i = 2u; i < query_.vcount_; i++)
        {
            std::vector<uint8_t> selected, further_selected;

            // 0. gather all extendable vertices
            {
                std::vector<std::pair<uint8_t, uint32_t>> selected_with_avg_degree;
                // add valid ("valid" means unvisited) u into `selected_with_avg_degree`
                // for each item in `selected_with_avg_degree`, item.first is `u`, item.second stores the minimum avg_degree among all [$bn, u]
                // see main.cpp for the definition of avg_degree
                // $bn can be any neighbors of u that have been added to the matching order before u.
                for (auto u = 0u; u < query_.vcount_; u++)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            if (selected_with_avg_degree.empty() || selected_with_avg_degree.back().first != u)
                            {
                                selected_with_avg_degree.emplace_back(u, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                            }
                            else
                            {
                                if (avg_degrees[query_.eidx_[bn * query_.vcount_ + u]] < selected_with_avg_degree.back().second)
                                {
                                    selected_with_avg_degree.back().second = avg_degrees[query_.eidx_[bn * query_.vcount_ + u]];
                                }
                            }
                        }
                    }
                }
                // sort
                std::sort(
                    selected_with_avg_degree.begin(),
                    selected_with_avg_degree.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );
                // remove vertices with extreme large avg degrees
                auto new_end = selected_with_avg_degree.size();
                for (auto j = 0u; j < selected_with_avg_degree.size() - 1; j++)
                {
                    if (selected_with_avg_degree[j + 1].second >= selected_with_avg_degree[j].second * (2 << 9))
                    {
                        new_end = j + 1;
                        break;
                    }
                }
                for (auto j = 0u; j < new_end; j++)
                {
                    selected.push_back(selected_with_avg_degree[j].first);
                }
                std::sort(selected.begin(), selected.end());
            }

            // 1. select the vertices with the maximum number of backward neighbors
            if (selected.size() > 1)
            {
                auto max_num_bn = 0u;
                for (const auto& u: selected)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    auto cur_num_bns = 0u;
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn]) cur_num_bns += 1;
                    }
                    if (cur_num_bns > max_num_bn)
                    {
                        max_num_bn = cur_num_bns;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_bns == max_num_bn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one neighbor
            // that is not in the matching order and is connected with u.
            if (selected.size() > 1)
            {
                auto max_num_v = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_v = 0u;
                    std::vector<bool> temp_visited(query_.vcount_, false);
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn] && !temp_visited[bn])
                                {
                                    temp_visited[bn] = true;
                                    cur_num_v++;
                                }
                            }
                        }
                    }
                    if (cur_num_v > max_num_v)
                    {
                        max_num_v = cur_num_v;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_v == max_num_v)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 3. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u

            // 3. if there is a tie, compute the number of dangling neighbors of u.
            // a dangling neighbor `fn` means `fn` does not has neighbors that are already in the matching order 
            if (selected.size() > 1)
            {
                auto max_num_fn = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_fn = 0u;
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            auto no_visited_nbr_of_fn = true;
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn])
                                {
                                    no_visited_nbr_of_fn = false;
                                    break;
                                }
                            }
                            if (no_visited_nbr_of_fn)
                            {
                                cur_num_fn++;
                            }
                        }
                    }
                    if (cur_num_fn > max_num_fn)
                    {
                        max_num_fn = cur_num_fn;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_fn == max_num_fn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // 4. if there is a tie, select a vertex with the minimum number of candiates on extension,
            // where the number of candidates is estimated by the min property

            // 4. if there is a tie, select a vertex with the minimum value of "minimum avg_degree among all [$bn, u]"
            // see main.cpp for the definition of avg_degree
            // $bn can be any neighbors of u that have been added to the matching order before u.
            if (selected.size() > 1)
            {
                auto min_num_candidates = std::numeric_limits<float>::max();
                for (const auto& u: selected)
                {
                    auto cur_num_candidates = std::numeric_limits<float>::max();
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            cur_num_candidates = std::min(cur_num_candidates, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                        }
                    }
                    if (cur_num_candidates < min_num_candidates)
                    {
                        min_num_candidates = cur_num_candidates;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_candidates == min_num_candidates)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // insert the first selected vertex to the matching order
            cur_order.push_back(selected[0]);
            visited[selected[0]] = true;
        }

        SetRebuildFlags(enable_relation_switch, u0, u1, cur_order, index);
        
        // std::cout << "before GenerateCartesianProductInfo" << endl;

        GenerateCartesianProductInfo(cur_order, index);

        // generate meta for the matching order
        auto cum_offs = 0u;
        // fills in orders_[index].bni_ and orders_[index].bni_offs_ (two arrays, backward_neighbor_index and backward_neighbor_index_offsets)
        // for each u, sort the orders_[index].bni_ according to the avg_degree of [bn, u] (ascending)
        orders_[index].bni_offs_[0] = cum_offs;
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            const auto& u = cur_order[j];
            orders_[index].vs_[j] = u;
            // Doubt: Does this mean k >= 0?
            for (auto k = j - 1; k < query_.vcount_; k--)
            {
                const auto& uu = orders_[index].vs_[k];
                auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
                if (it != query_.nbrs_[u].end() && it->first == uu)
                {
                    orders_[index].bni_[cum_offs++] = k;
                }
            }
            orders_[index].bni_offs_[j + 1] = cum_offs;
            std::sort(
                orders_[index].bni_ + orders_[index].bni_offs_[j],
                orders_[index].bni_ + orders_[index].bni_offs_[j + 1],
                [this, index, u, avg_degrees](const auto& bni1, const auto& bni2){
                    return avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni1] * this->query_.vcount_ + u]]
                    < avg_degrees[this->query_.eidx_[this->orders_[index].vs_[bni2] * this->query_.vcount_ + u]];
                }
            );
        }
    }

    // std::cout << "before depths_ assignment" << endl;
    ComputeNumQueryVerticesNonTailLeaf();
}