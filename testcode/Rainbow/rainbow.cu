#include "common.h"
#include "rainbow.h"

__device__ int generateKey(uint64_t key,uint64_t *store)
{
	uint32_t c,d,t,s,t2;
	const unsigned char *in;
	int i; uint32_t k[32];
	const_DES_cblock key1;uint64_t aa=(1<<9)-1;
	for(i=0;i<8;i++){ key1[i]=(key&aa); key>>=8;}
	in = &(key1)[0];c2l(in,c);c2l(in,d);
	
	PERM_OP (d,c,t,4,0x0f0f0f0fL);
	HPERM_OP(c,t,-2,0xcccc0000L);
	HPERM_OP(d,t,-2,0xcccc0000L);
	PERM_OP (d,c,t,1,0x55555555L);
	PERM_OP (c,d,t,8,0x00ff00ffL);
	PERM_OP (d,c,t,1,0x55555555L);
	d=	(((d&0x000000ffL)<<16L)| (d&0x0000ff00L)     |
		 ((d&0x00ff0000L)>>16L)|((c&0xf0000000L)>>4L));
	c&=0x0fffffffL;
	//one round, 0.25s*16=4s
	RoundKey0(0);RoundKey0(1);RoundKey1(2);RoundKey1(3);
	RoundKey1(4);RoundKey1(5);RoundKey1(6);RoundKey1(7);
	RoundKey0(8);RoundKey1(9);RoundKey1(10);RoundKey1(11);
    RoundKey1(12);RoundKey1(13);RoundKey1(14);RoundKey0(15);
	return 0;
}

__device__ uint64_t desOneTime(uint64_t*roundKeys)
{
	uint64_t rs;
	uint32_t right = plRight;uint32_t left = plLeft;

	IP(right,left);
	
	left=ROTATE(left,29);
	right=ROTATE(right,29);

	D_ENCRYPT(left,right, 0);
	D_ENCRYPT(right,left, 1);
	D_ENCRYPT(left,right, 2);
	D_ENCRYPT(right,left, 3);
	D_ENCRYPT(left,right, 4);
	D_ENCRYPT(right,left, 5);
	D_ENCRYPT(left,right, 6);
	D_ENCRYPT(right,left, 7);
	D_ENCRYPT(left,right, 8);
	D_ENCRYPT(right,left, 9);
	D_ENCRYPT(left,right,10);
	D_ENCRYPT(right,left,11);
	D_ENCRYPT(left,right,12);
	D_ENCRYPT(right,left,13);
	D_ENCRYPT(left,right,14);
	D_ENCRYPT(right,left,15);
	D_ENCRYPT(right,left,15);

	left=ROTATE(left,3);
	right=ROTATE(right,3);

	FP(right,left);
	rs = left|((uint64_t)right)<<32;
	
	return rs;
}

__global__ void desEncrypt(uint64_t *data) 
{
	((uint64_t *)des_SP)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];
	#if MAX_THREAD == 128
		((uint64_t *)des_SP)[threadIdx.x+128] = ((uint64_t *)des_d_sp_c)[threadIdx.x+128];
	#endif
	__syncthreads();
	register uint64_t key=data[TX];
	for(int i=0;i<(1<<8);i++)
	{
		uint64_t roundKeys[16];
		generateKey(key,roundKeys);
		key=desOneTime(roundKeys);
	}
	data[TX]=key;
}

void desCrypt() 
{
	uint64_t *deviceKeyIn, *deviceKeyOut;int i;
	uint64_t keys[ALL];struct timeval tstart, tend;
	int round,size;FILE*f1,*f2;
	round=0;
	if((f1=fopen("start.in","w"))==NULL)
	{
		printf("desCrypt: fopen start.in error\n");
		exit(0);
	}
	if((f2=fopen("end.in","w"))==NULL)
	{
		printf("desCrypt: fopen end.out error\n");
		exit(0);
	}
	printf("Starting DES kernel\n");
	size=ALL*sizeof(uint64_t);
    _CUDA(cudaMalloc((void**)&deviceKeyIn,size));
	_CUDA(cudaMalloc((void**)&deviceKeyOut,size));	
	while(1)
	{
		printf("Begin Round: %d\n",round);
		fprintf(f1,"Begin Round: %d\n",round);
		fprintf(f2,"Begin Round: %d\n",round);

		gettimeofday(&tstart, NULL);
	    for(i=0;i<ALL;i++) keys[i]=rand();
	    for(i=0;i<ALL;i++) fprintf(f1,"%lu\n",keys[i]);
	    _CUDA(cudaMemcpy(deviceKeyIn,keys,size,cudaMemcpyHostToDevice));
		desEncrypt<<<BLOCK_LENGTH,MAX_THREAD>>>(deviceKeyIn);
		_CUDA(cudaMemcpy(keys,deviceKeyOut,size,cudaMemcpyDeviceToHost));
		for(i=0;i<ALL;i++) fprintf(f2,"%lu\n",keys[i]);
		gettimeofday(&tend, NULL);
		int64 uses=1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
		
		printf("round time: %lld us\n", uses);
		fprintf(f1,"round time: %lld us\n",uses);
		fprintf(f2,"round time: %lld us\n",uses);
		
		printf("End Round: %d\n",round);
		fprintf(f1,"End Round: %d\n",round);
		fprintf(f2,"End Round: %d\n",round);

		round++;
	}
	fclose(f1);fclose(f2);
	printf("Ending DES kernel\n");
}

int main()
{
	struct timeval tstart, tend;
	gettimeofday(&tstart, NULL);
	desCrypt();
	gettimeofday(&tend, NULL);
	int64 uses = 1000000 * (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec);
	printf("total time: %lld us\n", uses);
	return 0;
}