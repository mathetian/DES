#ifndef _common_h
#define _common_h

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <sys/time.h>
#include <openssl/evp.h>
#include <openssl/des.h>
#include <cuda_runtime_api.h>

#define BLOCK_LENGTH        1024
#define MAX_THREAD			256
#define ALL                 1024*256

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

typedef long long int64;

#endif
