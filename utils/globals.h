#ifndef UTILS_GLOBAL_H_
#define UTILS_GLOBAL_H_

#include "utils/config.h"
#include "utils/mem_pool.h"

extern __constant__ uint32_t C_QE_COUNT;
extern __constant__ uint32_t C_QV_COUNT;
extern __constant__ uint32_t C_DV_COUNT;
extern __constant__ uint8_t C_NLF[MAX_QE_COUNT * 2];
extern __constant__ uint8_t C_QV_OFFS[MAX_QV_COUNT + 1];

extern __constant__ uint8_t C_EIDX[MAX_QV_COUNT * MAX_QV_COUNT];

extern __constant__ CyclicQueue<uint32_t> C_RES_QUEUE;

#endif  // UTILS_GLOBAL_H_