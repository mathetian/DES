#include "common.h"
#include "rainbow.h"


void DES_cuda_crypt() 
{
	uint64_t *deviceKeyIn, *deviceKeyOut;
	uint64_t keys[ALL];struct timeval tstart, tend;
	int round,size,index;round=0;
	printf("Starting DES kernel\n");
	size=ALL*sizeof(uint64_t);
    _cuda(cudaMalloc((void**)&deviceKeyIn,size));
	_cuda(cudaMalloc((void**)&deviceKeyOut,size));	
	while(1)
	{
		printf("Begin Round: %d\n",round);
		gettimeofday(&tstart, NULL);
	    for(i=0;i<ALL;i++) keys[i]=rand();
	    /*for(i=0;i<ALL;i++) write file*/
	    _cuda(cudaMemcpy(deviceKeyIn,key,size,cudaMemcpyHostToDevice));
		_cuda(DESencKernel<<<BLOCK_LENGTH,MAX_THREAD>>>(device_data_in));
		_cuda(cudaMemcpy(key,deviceKeyOut,size,cudaMemcpyDeviceToHost));
		 /*for(i=0;i<ALL;i++) write file*/
		gettimeofday(&tend, NULL);
		int64 uses=1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
		printf("round time: %lld us\n", uses);
		printf("End Round: %d\n",round);round++;
	}
	printf("Ending DES kernel\n");
}

int main()
{
	struct timeval tstart, tend;
	gettimeofday(&tstart, NULL);
	DES_cuda_crypt();
	gettimeofday(&tend, NULL);
	int64 uses = 1000000 * (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec);
	printf("total time: %lld us\n", uses);
	return 0;
}