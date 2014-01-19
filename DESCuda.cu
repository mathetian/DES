#include "DESCuda.h"

#include <iostream>
using namespace std;

#include <assert.h>

#define RoundKey0(S) { \
	c=((c>>1L)|(c<<27L)); d=((d>>1L)|(d<<27L));\
	c&=0x0fffffffL;d&=0x0fffffffL;\
	s=	des_skb[0][ (c    )&0x3f                ]|\
		des_skb[1][((c>> 6L)&0x03)|((c>> 7L)&0x3c)]|\
		des_skb[2][((c>>13L)&0x0f)|((c>>14L)&0x30)]|\
		des_skb[3][((c>>20L)&0x01)|((c>>21L)&0x06) |\
					  ((c>>22L)&0x38)];\
	t=	des_skb[4][ (d    )&0x3f                ]|\
		des_skb[5][((d>> 7L)&0x03)|((d>> 8L)&0x3c)]|\
		des_skb[6][ (d>>15L)&0x3f                ]|\
		des_skb[7][((d>>21L)&0x0f)|((d>>22L)&0x30)];\
	t2=((t<<16L)|(s&0x0000ffffL))&0xffffffffL;\
	store[S]  = ROTATE(t2,30)&0xffffffffL;\
	t2=((s>>16L)|(t&0xffff0000L));\
	store[S] |= ((ROTATE(t2,26)&0xffffffffL) << 32);\
}

#define RoundKey1(S) { \
	c=((c>>2L)|(c<<26L)); d=((d>>2L)|(d<<26L));\
	c&=0x0fffffffL;d&=0x0fffffffL;\
	s=	des_skb[0][ (c    )&0x3f                ]|\
		des_skb[1][((c>> 6L)&0x03)|((c>> 7L)&0x3c)]|\
		des_skb[2][((c>>13L)&0x0f)|((c>>14L)&0x30)]|\
		des_skb[3][((c>>20L)&0x01)|((c>>21L)&0x06) |\
					  ((c>>22L)&0x38)];\
	t=	des_skb[4][ (d    )&0x3f                ]|\
		des_skb[5][((d>> 7L)&0x03)|((d>> 8L)&0x3c)]|\
		des_skb[6][ (d>>15L)&0x3f                ]|\
		des_skb[7][((d>>21L)&0x0f)|((d>>22L)&0x30)];\
	t2=((t<<16L)|(s&0x0000ffffL))&0xffffffffL;\
	store[S]  = ROTATE(t2,30)&0xffffffffL;\
	t2=((s>>16L)|(t&0xffff0000L));\
	store[S] |= ((ROTATE(t2,26)&0xffffffffL) << 32);\
}

__device__ int GenerateKey(uint64_t key, uint64_t * store)
{
	uint32_t c, d, t, s, t2;

	/**c: low 32 bits, d high 32 bits**/
	c = (1ull << 32) & key; d = (key >> 32);
	
	PERM_OP (d,c,t,4,0x0f0f0f0fL);
	HPERM_OP(c,t, -2,0xcccc0000L);
	HPERM_OP(d,t, -2,0xcccc0000L);
	PERM_OP (d,c,t,1,0x55555555L);
	PERM_OP (c,d,t,8,0x00ff00ffL);
	PERM_OP (d,c,t,1,0x55555555L);

	d =	(((d&0x000000ffL)<<16L)| (d&0x0000ff00L)     |
		 ((d&0x00ff0000L)>>16L)|((c&0xf0000000L)>>4L));
	c&=0x0fffffffL;

	//one round, 0.25s*16=4s

	RoundKey0(0); RoundKey0(1) ;RoundKey1(2) ;RoundKey1(3);
	RoundKey1(4); RoundKey1(5) ;RoundKey1(6) ;RoundKey1(7);
	RoundKey0(8); RoundKey1(9) ;RoundKey1(10);RoundKey1(11);
    RoundKey1(12);RoundKey1(13);RoundKey1(14);RoundKey0(15);
	
	return 0;
}

__device__ uint64_t DESOneTime(uint64_t * roundKeys)
{
	uint64_t rs;
	uint32_t right = plRight, left = plLeft;

	IP(right, left);
	
	left  = ROTATE(left,29); right = ROTATE(right,29);

	D_ENCRYPT(left,right, 0); D_ENCRYPT(right,left, 1);
	D_ENCRYPT(left,right, 2); D_ENCRYPT(right,left, 3);
	D_ENCRYPT(left,right, 4); D_ENCRYPT(right,left, 5);
	D_ENCRYPT(left,right, 6); D_ENCRYPT(right,left, 7);
	D_ENCRYPT(left,right, 8); D_ENCRYPT(right,left, 9);
	D_ENCRYPT(left,right,10); D_ENCRYPT(right,left,11);
	D_ENCRYPT(left,right,12); D_ENCRYPT(right,left,13);
	D_ENCRYPT(left,right,14); D_ENCRYPT(right,left,15);
	D_ENCRYPT(right,left,15);

	left  = ROTATE(left,3); right = ROTATE(right,3);

	FP(right, left);

	rs = (((uint64_t)left) << 32)|right;
	
	return rs;
}

__global__ void DESEncrypt(uint64_t *data) 
{
	/**Don't know why should use it.**/
	((uint64_t *)des_SP)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];
	#if MAX_THREAD == 128
		((uint64_t *)des_SP)[threadIdx.x+128] = ((uint64_t *)des_d_sp_c)[threadIdx.x+128];
	#endif

	__syncthreads();

	register uint64_t key = data[TX];
	uint64_t roundKeys[16];
	
	for(int i = 0;i < (1<<8);i++)
	{
		GenerateKey(key,roundKeys);
		key = DESOneTime(roundKeys);
	}
	data[TX]=key;
}

uint64_t rand64()
{
	uint64_t rs = 0;
	for(uint64_t i = 0;i < 8;i++)
		rs |= ((rand() % 256) << (8*i));
	return rs;
}

#define FF(i, n) for(i = 0;i < n;i++)

void DESCrypt() 
{
	struct timeval tstart, tend;

	uint64_t * deviceKeyIn, *deviceKeyOut;
	uint64_t   keys[ALL]; int i;

	int round = 0, size; FILE * f1; FILE * f2;

	f1 = fopen("start.in","w");
	f2 = fopen("end.in","w");
	assert(f1 && f2);

	printf("Starting DES kernel\n");
	
	size = ALL * sizeof(uint64_t);

    _CUDA(cudaMalloc((void**)&deviceKeyIn , size));
	_CUDA(cudaMalloc((void**)&deviceKeyOut, size));	
	
	while(1)
	{
		printf("Begin Round: %d\n",round);
		
		fprintf(f1,"Begin Round: %d\n",round);
		fprintf(f2,"Begin Round: %d\n",round);

		gettimeofday(&tstart, NULL);

	    FF(i, ALL) keys[i] = rand64();
	    FF(i, ALL) fprintf(f1,"%lld",(long long)keys[i]);
	    	    
	    _CUDA(cudaMemcpy(deviceKeyIn, keys, size, cudaMemcpyHostToDevice));

		DESEncrypt<<<BLOCK_LENGTH, MAX_THREAD>>>(deviceKeyIn);

		_CUDA(cudaMemcpy(keys, deviceKeyOut, size, cudaMemcpyDeviceToHost));
		
		FF(i, ALL) fprintf(f2,"%lld\n", (long long)keys[i]);

		gettimeofday(&tend, NULL);

		long long uses=1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
		
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
	return 0;
}