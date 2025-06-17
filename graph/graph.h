#ifndef RAPIDCSM_GRAPH_GRAPH_H_
#define RAPIDCSM_GRAPH_GRAPH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/config.h"
#include "utils/types.h"

class GraphFileIoManager;
class MatchGPU;
class PlanManager;

struct GraphFileIOHelper {
    std::unordered_map<uint32_t, uint32_t> vlabel_map_;
    std::unordered_map<uint32_t, uint32_t> elabel_map_;
    void clear() {
        vlabel_map_.clear();
        elabel_map_.clear();
    }
};

class QueryGraph {
  public:
    friend class GraphFileIoManager;
    friend class MatchGPU;
    friend class MatchCPU;
    friend class PlanManager;

    QueryGraph(): vcount_(0u),
                  ecount_(0u),
                  vlabels_(),
                  nbrs_(),
                  qv_offs_(MAX_QV_COUNT + 1),
                  qv_nbrs_(MAX_QE_COUNT * 2),
                  NLF_(MAX_QE_COUNT * 2, 0u),
                  first_NL_(MAX_QE_COUNT * 2, 0u),
                  last_NL_(MAX_QE_COUNT * 2, 0u),
  
                  qe_list_(MAX_QE_COUNT),
                  qe_labels_(MAX_QE_COUNT),
                  qe_reversed_labels_(MAX_QE_COUNT),
  
                  eidx_(MAX_QV_COUNT * MAX_QV_COUNT, UINT8_MAX),
                  qe_eidx_(MAX_QE_COUNT) {}
    
    QueryGraph(const QueryGraph& query_graph): vcount_(query_graph.vcount_),
                                               ecount_(query_graph.ecount_),
                                               vlabels_(query_graph.vlabels_),
                                               nbrs_(query_graph.nbrs_),
                                               qv_offs_(query_graph.qv_offs_),
                                               qv_nbrs_(query_graph.qv_nbrs_),
                                               NLF_(query_graph.NLF_),
                                               first_NL_(query_graph.first_NL_),
                                               last_NL_(query_graph.last_NL_),

                                               qe_list_(query_graph.qe_list_),
                                               qe_labels_(query_graph.qe_labels_),
                                               qe_reversed_labels_(query_graph.qe_reversed_labels_),
                                               
                                               eidx_(query_graph.eidx_),
                                               qe_eidx_(query_graph.qe_eidx_) {}

    void loadFromFile(const std::string& file_path);
    void setQueryMeta();

    uint32_t getVerticesCount() const {
        return vcount_;
    }
    uint32_t getVertexLabel(uint32_t vertex) const {
        return vlabels_[vertex];
    }
    uint8_t getVertexDegree(uint32_t vertex) const {
        return qv_offs_[vertex + 1] - qv_offs_[vertex];
    }
    const uint8_t * getVertexNeighbors(const uint32_t vertex, uint8_t &count) const {
        count = qv_offs_[vertex + 1] - qv_offs_[vertex];
        return qv_nbrs_.data() + qv_offs_[vertex];
    }
    uint32_t getEdgeCount() const {
        return ecount_;
    }
    const std::vector<std::pair<uint32_t, uint32_t>> & getEdgeList() const {
        return qe_list_;
    }
    uint8_t getEdgeListIdxByVertex(uint8_t src, uint8_t dst) const {
        std::pair<uint32_t, uint32_t> vertex_pair = {UINT32_MAX, UINT32_MAX};
        if (src < dst) {
            vertex_pair = {static_cast<uint32_t>(src), static_cast<uint32_t>(dst)};
        } else {
            vertex_pair = {static_cast<uint32_t>(dst), static_cast<uint32_t>(src)};
        }
        // We assume the edge (src, dst) exists.
        auto it = std::find(qe_list_.begin(), qe_list_.end(), vertex_pair);
        auto idx = it - qe_list_.begin();
        return static_cast<uint8_t>(idx);
    }
    uint8_t getEdgeIdxByEdgeListIdx(uint8_t edge_list_idx) const {
        return qe_eidx_[edge_list_idx].first;
    }
    uint8_t getTheOtherEdgeIdxByEdgeListIdx(uint8_t edge_list_idx) const {
        return qe_eidx_[edge_list_idx].second;
    }
    std::pair<uint8_t, uint8_t> getEdgeIdxPairByEdgeListIdx(uint8_t edge_list_idx) const {
        return qe_eidx_[edge_list_idx];
    }
    uint8_t getEdgeIdxByVertex(uint8_t src, uint8_t dst) const {
        return eidx_[src * vcount_ + dst];
    }
    bool checkEdgeExistence(uint32_t u, uint32_t v) const {
        if (getVertexDegree(u) < getVertexDegree(v)) {
            std::swap(u, v);
        }
        uint8_t count = 0;
        const uint8_t* neighbors =  getVertexNeighbors(v, count);

        int begin = 0;
        int end = count - 1;
        while (begin <= end) {
            int mid = begin + ((end - begin) >> 1);
            if (neighbors[mid] == u) {
                return true;
            }
            else if (neighbors[mid] > u)
                end = mid - 1;
            else
                begin = mid + 1;
        }

        return false;
    }
    uint32_t getEdgeLabelByVertex(uint8_t src, uint8_t dst) const {
        auto idx = getEdgeListIdxByVertex(src, dst);
        return std::get<1>(qe_labels_[idx]);
    }
    std::pair<uint32_t, uint32_t> getEdgeByEdgeListIdx(uint8_t edge_list_idx) const {
        return qe_list_[edge_list_idx];
    }

  private:
    uint32_t vcount_;
    uint32_t ecount_;

    std::vector<uint32_t> vlabels_;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> nbrs_;  // src -> list of (dst, edge_label)
   
    std::vector<uint8_t> qv_offs_;  // CSR of the query graph
    std::vector<uint8_t> qv_nbrs_;
    
    std::vector<uint8_t> NLF_;
    std::vector<uint8_t> first_NL_;  // edge_idx -> edge_idx_first
    std::vector<uint8_t> last_NL_;  // edge_idx -> edge_idx_last

    /* 
      idx_in_qe_list_ -> (u, uu), collected in the order of `for(u in range(0, vcount_)){for(uu in nbrs_[u])}`
      only contains u <= uu 
    */
    std::vector<std::pair<uint32_t, uint32_t>> qe_list_;
    
    // idx_in_qe_list_ -> (src_label, edge_label, dst_label)
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> qe_labels_;
    
    // idx_in_qe_list_ -> (dst_label, edge_label, src_label)
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> qe_reversed_labels_;

    /* 
      find the relation index (direction considered) based on the two end-vertices
      (u, uu) -> edge_idx [u * vcount_ + uu] 
    */
    std::vector<uint8_t> eidx_;

    // idx_in_qe_list_ -> (edge_idx, edge_idx of the reverse edge)
    std::vector<std::pair<uint32_t, uint32_t>> qe_eidx_;

    GraphFileIOHelper loadHelper;

    uint32_t to_edge(uint32_t src, uint32_t dst) {
        return src * vcount_ + dst;
    }
};


class DataGraphManager {
  public:
    EdgeBatch initial_edges_;  // edge_idx -> EdgeList
    std::vector<EdgeBatch> updated_edges_;

    DataGraphManager();
    friend class GraphFileIoManager;
    friend class MatchGPU;
    friend class MatchCPU;

  private:
    uint32_t vcount_;
    std::vector<uint32_t> vlabels_;
};

class GraphFileIoManager {
  public:
    GraphFileIoManager(const std::string& query_path, QueryGraph& query_graph);

    // further set the values of members in query_
    void SetQueryMeta();

    void LoadInitial(const std::string &data_path, DataGraphManager &data_graph);
    void PrintNumDataEdges(const EdgeBatch &edge_batch);
    void LoadUpdate(const std::string &update_path, DataGraphManager &data_graph,
                    const uint32_t batch_size);

  private:
    friend class MatchGPU;
    friend class MatchCPU;

    QueryGraph& query_;

    uint32_t vlabel_count_;
    uint32_t elabel_count_;
    std::unordered_map<uint32_t, uint32_t> vlabel_map_;
    std::unordered_map<uint32_t, uint32_t> elabel_map_;

    size_t FileExist(const char *path);
};


#endif  // RAPIDCSM_GRAPH_GRAPH_H_
