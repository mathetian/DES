#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#include "cuda_common.h"
#include "common.h"

float time_elapsed;
cudaEvent_t time_start,time_stop;

void checkCUDADevice(struct cudaDeviceProp *deviceProp, int output_verbosity) {
	int deviceCount;
	cudaError_t cudaerrno;

	_CUDA(cudaGetDeviceCount(&deviceCount));

	if (!deviceCount) {
		if (output_verbosity!=OUTPUT_QUIET) 
			fprintf(stderr,"There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	if (output_verbosity>=OUTPUT_NORMAL) 
		fprintf(stdout,"Successfully found %d CUDA devices (CUDART_VERSION %d).\n",deviceCount, CUDART_VERSION);

	_CUDA(cudaSetDevice(6));
	_CUDA(cudaGetDeviceProperties(deviceProp, 6));
	
	if (output_verbosity==OUTPUT_VERBOSE) {
        	fprintf(stdout,"\nDevice %d: \"%s\"\n", 6, deviceProp->name);
      	 	fprintf(stdout,"  CUDA Compute Capability:                       %d.%d\n", deviceProp->major,deviceProp->minor);
#if CUDART_VERSION >= 2000
        	fprintf(stdout,"  Number of multiprocessors (SM):                %d\n", deviceProp->multiProcessorCount);
#endif
#if CUDART_VERSION >= 2020
		fprintf(stdout,"  Integrated:                                    %s\n", deviceProp->integrated ? "Yes" : "No");
        	fprintf(stdout,"  Support host page-locked memory mapping:       %s\n", deviceProp->canMapHostMemory ? "Yes" : "No");
#endif
		fprintf(stdout,"\n");
	}
}
