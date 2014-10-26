// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _RAINBOW_CUDA_H
#define _RAINBOW_CUDA_H

#include "Common.h"
using namespace utils;

#define BLOCK_LENGTH        1024
#define MAX_THREAD			256
#define ALL                 (1024*256)
#define CHAINLEN            4096

namespace rainbowcrack
{

#ifndef TX
#if (__CUDA_ARCH__ < 200)
#define TX (__umul24(blockIdx.x,blockDim.x) + threadIdx.x)
#else
#define TX (blockIdx.x * blockDim.x + threadIdx.x)
#endif
#endif
cudaError_t cudaerrno;
#define _CUDA(call) {																	\
	call;				                                												\
	cudaerrno=cudaGetLastError();																	\
	if(cudaSuccess!=cudaerrno) {                                       					         						\
		fprintf(stderr, "CUDA ERROR %d in file '%s' in line %i: %s.\n",cudaerrno,__FILE__,__LINE__,cudaGetErrorString(cudaerrno));	\
		exit(EXIT_FAILURE);                                                  											\
    } }

__device__ uint64_t totalSpace = (1ull << 63) - 1 + (1ull << 63);
uint64_t totalSpace_Global = (1ull << 63) - 1 + (1ull << 63);

__device__ void U64_2_CHAR(uint64_t message, uint8_t *pPlain)
{
    for(int i = 0; i < 8; i++) pPlain[i] = (message >> (i*8)) & ((1 << 8) - 1);
}

__device__ void CHAR_2_U64(uint64_t &message, uint8_t *pPlain)
{
    message = 0;
    for(int i = 0; i < 8; i++)
    {
        uint64_t value = pPlain[i];
        message |= (value << (i * 8));
    }
}

};

#endif