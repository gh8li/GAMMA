
#include <cstdint>

#include "utils/config.h"
#include "utils/mem_pool.h"
#include "utils/globals.h"

__constant__ uint32_t C_QE_COUNT;
__constant__ uint32_t C_QV_COUNT;
__constant__ uint32_t C_DV_COUNT;
__constant__ uint8_t C_NLF[MAX_QE_COUNT * 2];
__constant__ uint8_t C_QV_OFFS[MAX_QV_COUNT + 1];

__constant__ uint8_t C_EIDX[MAX_QV_COUNT * MAX_QV_COUNT];

__constant__ CyclicQueue<uint32_t> C_RES_QUEUE;
