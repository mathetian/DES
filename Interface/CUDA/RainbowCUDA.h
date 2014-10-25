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
		fprintf(stderr, "Cuda error %d in file '%s' in line %i: %s.\n",cudaerrno,__FILE__,__LINE__,cudaGetErrorString(cudaerrno));	\
		exit(EXIT_FAILURE);                                                  											\
    } }

};

#endif