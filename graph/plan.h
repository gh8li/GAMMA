#ifndef GRAPH_PLAN_H
#define GRAPH_PLAN_H

#include <array>
#include <bitset>
#include <cstdint>
#include <vector>

#include "graph/graph.h"
#include "utils/config.h"

class OrderPerEdge {
  public:
    uint8_t vs_[MAX_QV_COUNT];
    uint8_t bni_offs_[MAX_QV_COUNT + 1];
    // backward_neighbor_index in vs_
    uint8_t bni_[MAX_QE_COUNT]; // sorted by the reversed order of index

    void CopyFrom(const OrderPerEdge &other);
};

class PlanManager {
  public:
    enum CartesianProductType : uint8_t {
        None = 0,
        TreeSingle,
        NonTreeSingle,
        TreeCartesianProduct,
        NonTreeCartesianProduct
    };
    std::bitset<MAX_QV_COUNT * MAX_QV_COUNT> rebuild_flags_[MAX_QE_COUNT];
    std::bitset<MAX_QV_COUNT> rebuild_v_flags_[MAX_QE_COUNT];
    // Rebuild-Relation Flags
    std::bitset<MAX_QV_COUNT> rebuild_R_flags_[MAX_QE_COUNT];
    // Rebuild-Bitmap Flags
    // If Rebuild-Bitmap Flags[dv] if false, Rebuild-Relation Flags[dv] must be false.
    std::bitset<MAX_QV_COUNT> rebuild_B_flags_[MAX_QE_COUNT];
    std::array<CartesianProductType, MAX_QV_COUNT> cartesian_product_info_[MAX_QE_COUNT];
    // Generated using breadth-first search from the updated edge.
    OrderPerEdge indexing_orders_[MAX_QE_COUNT];
    OrderPerEdge orders_[MAX_QE_COUNT];

    PlanManager(const QueryGraph& query_graph);
    // Plan(const QueryGraph& query_graph, const AutomorphismManager *am_ptr);

    // // generate orders with minimum number of relations to rebuild
    // void GenerateOrders_v0();

    void GenerateIndexingOrders();

    // generate orders with the RI's ordering method
    void GenerateMatchingOrders_v1(uint32_t *cardinalities, float *avg_degrees, 
                                   const bool enable_relation_switch=true);

    // update-optimized matching order
    void GenerateMatchingOrders_v2(uint32_t *cardinalities, float *avg_degrees, 
                                   const bool enable_relation_switch=true);

    void GenerateGammaMatchingOrders(const uint32_t *cardinalities, const float *avg_degrees, 
                                     const bool enable_relation_switch=true);

    void GenerateOursMatchingOrders(uint32_t *cardinalities, float *avg_degrees);

    std::vector<uint8_t> getNonTailLeafDepths() const {
        return num_query_vertices_not_tail_leaf_;
    }

    uint8_t getNonTailLeafDepthByEdgeListIdx(uint8_t edge_list_idx) const {
        return num_query_vertices_not_tail_leaf_[edge_list_idx];
    }

    void PrintOrders();

  private:
    const QueryGraph& query_;
    // const AutomorphismManager *am_ptr_;
    std::vector<uint8_t> num_query_vertices_not_tail_leaf_;
    
    void SetRebuildFlags(const bool enable_relation_switch, 
                         const uint32_t &u0, const uint32_t &u1, 
                         std::vector<uint8_t> &cur_order, 
                         unsigned int edge_list_idx);

    void AddVertices(std::vector<uint8_t>& order,
                     std::bitset<MAX_QV_COUNT>& visited,
                     const std::vector<uint8_t>& vrole,
                     const uint8_t group_id);

    void AddVertices(std::vector<uint8_t>& order,
                     std::bitset<MAX_QV_COUNT>& visited,
                     const std::vector<uint8_t>& vrole);

    uint8_t ChooseStartingVertex(const std::vector<uint8_t>& vrole,
                                 std::bitset<MAX_QV_COUNT * MAX_QV_COUNT>& rebuild_flags,
                                 uint8_t u0, uint8_t u1);

    /*uint8_t GetNumBuildRelations(
        uint8_t update_u0, uint8_t update_u1,
        uint8_t starting_vertex,
        std::bitset<MAX_VCOUNT * MAX_VCOUNT>& temp_rebuild_flags
    );*/
    void BuildIndexingOrder(const uint8_t u0, const uint8_t u1,
                            OrderPerEdge& indexing_orders_);

    void GenerateRebuildFlagsWithOrder(uint8_t update_u0, uint8_t update_u1,
                                       std::initializer_list<uint8_t>&& starting_vertices,
                                       std::bitset<MAX_QV_COUNT * MAX_QV_COUNT>& rebuild_flags);

    void GenerateRebuildVFlagsWithOrder(uint8_t update_u0, uint8_t update_u1,
                                        uint8_t index);

    void GenerateRebuildVFlagsWithOrder_v2(uint8_t update_u0, uint8_t update_u1,
                                           uint8_t index, std::vector<uint8_t>& cur_order);

    void GenerateCartesianProductInfo(std::vector<uint8_t>& cur_order, uint8_t index);

    void ComputeNumQueryVerticesNonTailLeaf();
};

#endif  // GRAPH_PLAN_H
