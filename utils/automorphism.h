#ifndef UTILS_AUTOMORPHISM_H_
#define UTILS_AUTOMORPHISM_H_

#include <cstdint>
#include <vector>

#include "graph/graph.h"

typedef std::pair<uint8_t, uint8_t> VertexPair;

class AutomorphismManager {
    std::vector<std::vector<uint8_t>> automorphisms_;  // each element vector is an automorphism match
    std::vector<std::vector<VertexPair>> automorphism_edges_;
    
    /*
      edge_list_idx of e -> the edge_list_idx of the representative edge for the equivalent group containing e;
      the representative edge in the group is the edge with the smallest (src_id, dst_id).
    */
    std::vector<uint8_t> edge_list_idx_map_;

    // edge_list_idx of e -> the number of edges in the equivalent group containing e;
    std::vector<uint8_t> equiv_edge_count_;

    std::vector<uint8_t> representative_edge_list_idxs_;

public:
    AutomorphismManager() {}
    ~AutomorphismManager() {}
    
    void detect_automorphism_edges(const QueryGraph *query_graph);

    uint8_t getEquivEdgeByEdgeListIdx(uint8_t edge_list_idx) const {
      return edge_list_idx_map_[edge_list_idx];
    }

    uint8_t getEquivEdgeCountByEdgeListIdx(uint8_t edge_list_idx) const {
      return edge_list_idx_map_[edge_list_idx];
    }

    const uint8_t * getEquivEdgeCountArrayPointer() const {
      return equiv_edge_count_.data();
    }

    uint8_t getNumEquivGroups() const {
      return representative_edge_list_idxs_.size();
    }

    std::vector<uint8_t> getRepEdgeListIdxVector() const {
      return representative_edge_list_idxs_;
    }
};

#endif  // UTILS_AUTOMORPHISM_H_