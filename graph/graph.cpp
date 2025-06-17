
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <sys/stat.h> /* For stat() */
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/config.h"
#include "utils/constants.h"
#include "graph/graph.h"
#include "graph.h"

size_t FileExist(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

void QueryGraph::loadFromFile(const std::string& file_path) {
    loadHelper.clear();

    if (!FileExist(file_path.c_str())) {
        std::cout << "Failed to open: " << file_path << std::endl;
        exit(-1);
    }

    std::ifstream ifs(file_path);

    try {
        char type;
        while (ifs >> type) {
            if (type == 't') {
                char temp1;
                uint temp2;
                ifs >> temp1 >> temp2;
            }
            else if (type == 'v') {
                uint vertex_id, label;
                ifs >> vertex_id >> label;

                if (loadHelper.vlabel_map_.find(label) == loadHelper.vlabel_map_.end()) {
                    loadHelper.vlabel_map_[label] = loadHelper.vlabel_map_.size();
                }
                label = loadHelper.vlabel_map_.at(label);

                if (vertex_id >= vlabels_.size())
                {
                    vlabels_.resize(vertex_id + 1, NOT_EXIST);
                    vlabels_[vertex_id] = label;
                    nbrs_.resize(vertex_id + 1);
                }
                else if (vlabels_[vertex_id] == NOT_EXIST)
                {
                    vlabels_[vertex_id] = label;
                }

                vcount_ += 1;
            }
            else {
                uint from_id, to_id, label;
                ifs >> from_id >> to_id >> label;

                if (loadHelper.elabel_map_.find(label) == loadHelper.elabel_map_.end()) {
                    loadHelper.elabel_map_[label] = loadHelper.elabel_map_.size();
                }
                label = loadHelper.elabel_map_.at(label);

                // In the query graph, an adjacency array is sorted
                std::pair insert_pair(to_id, label);
                auto lower = std::lower_bound(nbrs_[from_id].begin(), nbrs_[from_id].end(), insert_pair);
                nbrs_[from_id].insert(lower, insert_pair);

                insert_pair = std::make_pair(from_id, label);
                lower = std::lower_bound(nbrs_[to_id].begin(), nbrs_[to_id].end(), insert_pair);
                nbrs_[to_id].insert(lower, insert_pair);

                ecount_ += 1;
            }
        }
    } catch (std::runtime_error err) {
        std::cout << err.what();
        ifs.close();
        exit(-1);
    }

    ifs.close();

    if (vcount_ < 2)
    {
        std::cout << "the number of query verticex should not be less than 2!" << std::endl;
        exit(-1);
    }
}

void QueryGraph::setQueryMeta() {

    QV_COUNT = vcount_;
    QE_COUNT = ecount_;

    if (QV_COUNT > MAX_QV_COUNT || QE_COUNT > MAX_QE_COUNT)
    {
        std::cout << "The query graph should have at most " << MAX_QV_COUNT
        << " vertices and " << MAX_QE_COUNT << " edges.\n";
        exit(-1);
    }

    uint32_t edge_pos = 0u;
    for (auto u = 0u; u < vcount_; u++)
    {
        qv_offs_[u] = edge_pos;
        for (const auto& [uu, _]: nbrs_[u])
        {
            eidx_[to_edge(u, uu)] = edge_pos;
            qv_nbrs_[edge_pos] = uu;
            edge_pos++;
        }
    }
    qv_offs_[vcount_] = edge_pos;

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        // (edge_label, dst_label) -> (edge_idx_first, edge_idx_last, count) [edge_idx is the index in nbrs_]
        // Doubt (solved): If there are edges of the same (edge_label, dst_label) among the neighborhood of two query vertices, there will be interference.
        std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t>> distinct_NLs;
        for (const auto& [uu, label]: nbrs_[u])
        {
            if (distinct_NLs.find(label * QV_COUNT + vlabels_[uu]) == distinct_NLs.end())
            {
                distinct_NLs[label * QV_COUNT + vlabels_[uu]] = {eidx_[u * QV_COUNT + uu], eidx_[u * QV_COUNT + uu], 1};
            }
            else
            {
                std::get<1>(distinct_NLs.at(label * QV_COUNT + vlabels_[uu])) = eidx_[u * QV_COUNT + uu];
                std::get<2>(distinct_NLs.at(label * QV_COUNT + vlabels_[uu])) += 1;
            }
        }
        for (const auto& [uu, label]: nbrs_[u])
        {
            // Doubt: Why > 2 rather than > 1 ? (solved)
            if (std::get<2>(distinct_NLs.at(label * QV_COUNT + vlabels_[uu])) > 2)
            {
                first_NL_[eidx_[u * QV_COUNT + uu]] = std::get<0>(distinct_NLs.at(label * QV_COUNT + vlabels_[uu]));
                last_NL_[eidx_[u * QV_COUNT + uu]] = std::get<1>(distinct_NLs.at(label * QV_COUNT + vlabels_[uu]));
            }
            else // No need to optimize.
            {
                first_NL_[eidx_[u * QV_COUNT + uu]] = eidx_[u * QV_COUNT + uu];
                last_NL_[eidx_[u * QV_COUNT + uu]] = eidx_[u * QV_COUNT + uu];
            }
        }
        for (const auto& [NL, t]: distinct_NLs)
        {
            NLF_[std::get<0>(t)] = std::get<2>(t);
        }
    }

    edge_pos = 0u;
    for (auto u = 0u; u < vcount_; u++)
    {
        for (const auto& [uu, label]: nbrs_[u])
        {
            if (u > uu) continue;

            qe_list_[edge_pos] = {u, uu};
            qe_labels_[edge_pos] = {vlabels_[u], label, vlabels_[uu]};
            qe_reversed_labels_[edge_pos] = {vlabels_[uu], label, vlabels_[u]};
            qe_eidx_[edge_pos] = {
                eidx_[u * vcount_ + uu], eidx_[uu * vcount_ + u]
            };
            edge_pos++;
        }
    }

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        for (const auto& [uu, _]: nbrs_[u])
        {
            std::cout << "[" << u << "," << uu << "]: " << static_cast<uint32_t>(eidx_[u * QV_COUNT + uu]) << ' ';
        }
        std::cout << '\n';
    }
}


DataGraphManager::DataGraphManager(): vcount_(0u), vlabels_(), 
                                      initial_edges_(), updated_edges_() {}

// initialize `query_` with `query_graph`, load query_graph from `query_path`
GraphFileIoManager::GraphFileIoManager(const std::string& query_path, 
                                       QueryGraph& query_graph): 
                                           query_(query_graph),
                                           vlabel_count_(0u),
                                           elabel_count_(0u),
                                           vlabel_map_(),
                                           elabel_map_() {
    if (!FileExist(query_path.c_str()))
    {
        std::cout << "Failed to open: " << query_path << std::endl;
        exit(-1);
    }

    // create the query graph
    std::ifstream ifs(query_path);

    char type;
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;

            if (vlabel_map_.find(label) == vlabel_map_.end())
            {
                vlabel_map_[label] = vlabel_map_.size();
            }
            label = vlabel_map_.at(label);

            if (vertex_id >= query_.vlabels_.size())
            {
                query_.vlabels_.resize(vertex_id + 1, NOT_EXIST);
                query_.vlabels_[vertex_id] = label;
                query_.nbrs_.resize(vertex_id + 1);
            }
            else if (query_.vlabels_[vertex_id] == NOT_EXIST)
            {
                query_.vlabels_[vertex_id] = label;
            }

            query_.vcount_ += 1;
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;

            if (elabel_map_.find(label) == elabel_map_.end())
            {
                elabel_map_[label] = elabel_map_.size();
            }
            label = elabel_map_.at(label);

            // In the query graph, an adjacency array is sorted
            std::pair insert_pair(to_id, label);
            auto lower = std::lower_bound(query_.nbrs_[from_id].begin(), query_.nbrs_[from_id].end(), insert_pair);
            query_.nbrs_[from_id].insert(lower, insert_pair);

            insert_pair = std::make_pair(from_id, label);
            lower = std::lower_bound(query_.nbrs_[to_id].begin(), query_.nbrs_[to_id].end(), insert_pair);
            query_.nbrs_[to_id].insert(lower, insert_pair);

            query_.ecount_ += 1;
        }
    }
    ifs.close();
    if (query_.vcount_ < 2)
    {
        std::cout << "the number of query verticex should not be less than 2!" << std::endl;
        exit(-1);
    }
}

void GraphFileIoManager::SetQueryMeta() {
    // set other meta data of the class
    QV_COUNT = query_.vcount_;
    QE_COUNT = query_.ecount_;
    if (QV_COUNT > MAX_QV_COUNT || QE_COUNT > MAX_QE_COUNT)
    {
        std::cout << "The query graph should have at most " << MAX_QV_COUNT
        << " vertices and " << MAX_QE_COUNT << " edges.\n";
        exit(-1);
    }

    uint32_t edge_pos = 0u;
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        query_.qv_offs_[u] = edge_pos;
        for (const auto& [uu, _]: query_.nbrs_[u])
        {
            query_.eidx_[u * query_.vcount_ + uu] = edge_pos;
            query_.qv_nbrs_[edge_pos] = uu;
            edge_pos++;
        }
    }
    query_.qv_offs_[query_.vcount_] = edge_pos;

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        // (edge_label, dst_label) -> (edge_idx_first, edge_idx_last, count) [edge_idx is the index in query_.nbrs_]
        // Doubt (solved): If there are edges of the same (edge_label, dst_label) among the neighborhood of two query vertices, there will be interference.
        std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t>> distinct_NLs;
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            if (distinct_NLs.find(label * QV_COUNT + query_.vlabels_[uu]) == distinct_NLs.end())
            {
                distinct_NLs[label * QV_COUNT + query_.vlabels_[uu]] = {query_.eidx_[u * QV_COUNT + uu], query_.eidx_[u * QV_COUNT + uu], 1};
            }
            else
            {
                std::get<1>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) = query_.eidx_[u * QV_COUNT + uu];
                std::get<2>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) += 1;
            }
        }
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            // Doubt: Why > 2 rather than > 1 ? (solved)
            if (std::get<2>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) > 2)
            {
                query_.first_NL_[query_.eidx_[u * QV_COUNT + uu]] = std::get<0>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu]));
                query_.last_NL_[query_.eidx_[u * QV_COUNT + uu]] = std::get<1>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu]));
            }
            else // No need to optimize.
            {
                query_.first_NL_[query_.eidx_[u * QV_COUNT + uu]] = query_.eidx_[u * QV_COUNT + uu];
                query_.last_NL_[query_.eidx_[u * QV_COUNT + uu]] = query_.eidx_[u * QV_COUNT + uu];
            }
        }
        for (const auto& [NL, t]: distinct_NLs)
        {
            query_.NLF_[std::get<0>(t)] = std::get<2>(t);
        }
    }

    edge_pos = 0u;
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            if (u > uu) continue;

            query_.qe_list_[edge_pos] = {u, uu};
            query_.qe_labels_[edge_pos] = {query_.vlabels_[u], label, query_.vlabels_[uu]};
            query_.qe_reversed_labels_[edge_pos] = {query_.vlabels_[uu], label, query_.vlabels_[u]};
            query_.qe_eidx_[edge_pos] = {
                query_.eidx_[u * query_.vcount_ + uu], query_.eidx_[uu * query_.vcount_ + u]
            };
            edge_pos++;
        }
    }

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        for (const auto& [uu, _]: query_.nbrs_[u])
        {
            std::cout << "[" << u << "," << uu << "]: " << static_cast<uint32_t>(query_.eidx_[u * QV_COUNT + uu]) << ' ';
        }
        std::cout << '\n';
    }
}

void GraphFileIoManager::LoadInitial(const std::string& data_path, DataGraphManager& data_graph)
{
    if (!FileExist(data_path.c_str()))
    {
        std::cout << "Failed to open: " << data_path << std::endl;
        exit(-1);
    }

    data_graph.initial_edges_.resize(QE_COUNT * 2);
    char type;
    // create the data graph
    std::ifstream ifs(data_path);
    bool has_printed_reading_edges = false;  // 250216
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;
            // data_graph.vlabels_.resize(vertex_id + 1, NOT_EXIST);  // WRONG: The resize here may shrink the vector. by gli945
            if (vertex_id >= data_graph.vlabels_.size())
            {
                data_graph.vlabels_.resize(vertex_id + 1, NOT_EXIST);
            }
            if (vlabel_map_.find(label) == vlabel_map_.end())
            {
                continue;
            }
            label = vlabel_map_[label];

            // Doubt: Why check this after we have performed data_graph.vlabels.resize ?
            // Doubt Partially Solved: In fact, this is to check vertex_id == data_graph.vlabels_.size().
            // Solved: The previous resize may shrink the vector. Code has been corrected. by gli945
            if (vertex_id >= data_graph.vlabels_.size())
            {
                data_graph.vlabels_[vertex_id] = label;
            }
            else if (data_graph.vlabels_[vertex_id] == NOT_EXIST)
            {
                data_graph.vlabels_[vertex_id] = label;
            }
            // if (vertex_id >= 34278877) {
            //     std::cout << "vertex_id: " << vertex_id << 
            //                  ", label: " << label << 
            //                  ", data_graph.vlabels_.size()" << data_graph.vlabels_.size() << std::endl;
            // }
        }
        else
        {
            if (!has_printed_reading_edges)  // 250216
            {
                has_printed_reading_edges = true;  // 250216
                std::cout << "reading edges..." << std::endl;  // 250216
            }
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;
            if (data_graph.vlabels_[from_id] == NOT_EXIST)
            {
                continue;
            }
            if (data_graph.vlabels_[to_id] == NOT_EXIST)
            {
                continue;
            }
            if (elabel_map_.find(label) == elabel_map_.end())
            {
                continue;
            }
            label = elabel_map_[label];

            std::tuple elabals = {data_graph.vlabels_[from_id], label, data_graph.vlabels_[to_id]};
            // For each data graph edge, assign it to one group.
            // The group is defined by the elabel, i.e. {source_label, edge_label, dest_label}
            for (auto i = 0u; i < QE_COUNT; i++)
            {
                if (query_.qe_labels_[i] == elabals)
                {
                    // (from_id, to_id) matches qe_list_[i]
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].first.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].second.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].first.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].second.push_back(from_id);
                }
                // Doubt: Seems that one data graph edge is added to initial_edges_[edge_idx] twice (for each edge_idx).
                // Doubt Solved: If for any query edge, qe_label != qe_reverse_label, the data edge will not be collected twice.
                // Doubt Solved: But if for some query edge, qe_label == qe_reverse_label, corresponding data edges will be collected twice.s
                if (query_.qe_reversed_labels_[i] == elabals)
                {
                    // (to_id, from_id) matches qe_list_[i]
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].first.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].second.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].first.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].second.push_back(to_id);
                }
            }
        }
    }
    ifs.close();
    std::cout << "reading edges done." << std::endl;  // 250216
    DV_COUNT = data_graph.vcount_ = data_graph.vlabels_.size();
    std::cout << "#data graph vertices: " << DV_COUNT << std::endl;
    std::cout << "#initial data graph edges: " << std::endl;
    PrintNumDataEdges(data_graph.initial_edges_);
}

void GraphFileIoManager::PrintNumDataEdges(const EdgeBatch &edge_batch)
{
    size_t num_total_edges = 0;
    for (auto cur_edge_idx = 0u; cur_edge_idx < QE_COUNT * 2; cur_edge_idx++)
    {
        size_t cur_num_edges = edge_batch[cur_edge_idx].first.size();
        std::cout << "edge_idx: " << cur_edge_idx << ", #edges: " << cur_num_edges << std::endl;
        num_total_edges += cur_num_edges;
    }
    std::cout << "total #edges: " << num_total_edges << std::endl;
}

void GraphFileIoManager::LoadUpdate(const std::string &update_path, DataGraphManager &data_graph, const uint32_t batch_size)
{
    if (!FileExist(update_path.c_str()))
    {
        std::cout << "Failed to open: " << update_path << std::endl;
        exit(-1);
    }

    uint32_t num_updated_edges = 0u;
    // create the update stream
    std::ifstream ifs(update_path);
    char type;
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            std::cout << "vertex update.\n";
            exit(-1);
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;
            if (num_updated_edges % batch_size == 0)
            {
                data_graph.updated_edges_.emplace_back();
                data_graph.updated_edges_.back().resize(QE_COUNT * 2);
            }
            num_updated_edges++;
            if (data_graph.vlabels_[from_id] == NOT_EXIST)
            {
                continue;
            }
            if (data_graph.vlabels_[to_id] == NOT_EXIST)
            {
                continue;
            }
            if (elabel_map_.find(label) == elabel_map_.end())
            {
                continue;
            }
            label = elabel_map_[label];

            std::tuple elabals = {data_graph.vlabels_[from_id], label, data_graph.vlabels_[to_id]};
            // lgh: data_graph.updated_edges_ are grouped by elabel, i.e. {source_label, edge_label, dest_label}
            for (auto i = 0u; i < QE_COUNT; i++)
            {
                if (query_.qe_labels_[i] == elabals)
                {
                    // (from_id, to_id) matches qe_list_[i]
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].first.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].second.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].first.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].second.push_back(from_id);
                }
                if (query_.qe_reversed_labels_[i] == elabals)
                {
                    // (to_id, from_id) matches qe_list_[i]
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].first.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].second.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].first.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].second.push_back(to_id);
                }
            }
        }
    }
    ifs.close();
    std::cout << "#updated data graph edges: " << std::endl;
    int batch_idx = 0;
    for (auto& edge_batch: data_graph.updated_edges_)
    {
        std::cout << "--batch #" << batch_idx << std::endl;
        PrintNumDataEdges(edge_batch);
        batch_idx++;
    }
}

size_t GraphFileIoManager::FileExist(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}
