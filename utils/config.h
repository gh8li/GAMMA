#ifndef UTILS_CONFIG_H_
#define UTILS_CONFIG_H_

#include <cstdint>

#define NOT_EXIST UINT32_MAX
#define MAX_QV_COUNT 16u
#define MAX_QE_COUNT 42u
#define MIN_NBR_SIZE 8u

#define GRID_DIM 1024u
#define BLOCK_DIM 512u
#define WARP_SIZE 32u
#define NUM_WARP_PER_BLOCK (BLOCK_DIM / WARP_SIZE)

// #define NBR_SPACE (256ul * 1024 * 1024 / sizeof(uint32_t)) // 256 MB of uint32_t  // default
// #define NBR_SPACE (4ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 4 GB of uint32_t
// #define NBR_SPACE (1ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 1 GB of uint32_t  // 250217
// #define NBR_SPACE (2ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 2 GB of uint32_t  // 250218
#define NBR_SPACE (1536 * 1024 * 1024 / sizeof(uint32_t)) // 1.5 GB of uint32_t  // 250218

#define NBR_IS_UPDATE_SPACE (384 * 1024 * 1024 / sizeof(bool)) // roughly 0.38 GB of bool

// #define RES_SPACE (2ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 2 GB of uint32_t  // 250218
#define RES_SPACE (3ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 3 GB of uint32_t  // 250218
// #define RES_SPACE (4ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 4 GB of uint32_t  // default
// #define RES_SPACE (6ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 6 GB of uint32_t  // 250217
// #define RES_SPACE (8ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 8 GB of uint32_t
// #define RES_SPACE (10ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 10 GB of uint32_t  // 250217
// #define RES_SPACE (12ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 12 GB of uint32_t
#define SIZE_SPACE (2ul * 1024 * 1024 * 1024 / sizeof(long)) // 2 GB of long  //default
// #define SIZE_SPACE (1ul * 1024 * 1024 * 1024 / sizeof(long)) // 1 GB of long
#define MIN_NUM_RESULTS_TO_GPU (1 << 16)

#define MATERIALIZE false

#define POSSIBLE_NUM_QE 16u

const uint32_t MaxNumQV = 16u;
const uint32_t MaxNumQE = 42u;
const uint32_t MinNbrSize = 8u;

const uint32_t GridDim = 1024u;
const uint32_t BlockDim = 512u;
const uint32_t WarpSize = 32u;
const uint32_t NumWarpPerBlock = BlockDim / WarpSize;

const uint64_t NbrSpace = 256ul * 1024 * 1024 / sizeof(uint32_t);
const uint64_t ResultSpace = 4ul * 1024 * 1024 * 1024 / sizeof(uint32_t);
const uint64_t SizeSpace = 2ul * 1024 * 1024 * 1024 / sizeof(long);
const uint64_t MinMunResultsToGPU = 1 << 16;

const bool Materialize = false;

const uint32_t PossibleNumQE = 16u;

// // uint32_t QE_COUNT;
// // uint32_t QV_COUNT;
// // uint32_t DV_COUNT;

#endif  // UTILS_CONFIG_H_