#ifndef UTILS_TYPES_H
#define UTILS_TYPES_H

#include <cstdint>
#include <vector>

using EdgeList = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

using EdgeBatch = std::vector<EdgeList>;

#endif