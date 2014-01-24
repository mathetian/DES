#include "DESCuda.h"

#include <openssl/rand.h>
#include <openssl/des.h>

#include <iostream>
using namespace std;


__device__ int GenerateKey(uint64_t key, uint64_t * store)
{
	uint32_t c, d, t, s, t2;
	uint64_t tmp;
	/**c: low 32 bits, d high 32 bits**/
	c = ((1ull << 32) - 1) & key; d = (key >> 32);
	
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

__global__ void Gee(uint64_t * store)
{
	//14969965219234971648 0xcfc0000d78740000L
	//14897907633854087168 0xcec0000f7c740000L
	//uint64_t key=0x0E0E0E0E0E0E0E02;
	uint64_t key=0x02080E0E0E0E0E0E;
	GenerateKey(key,store);
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

/**
	DESEncrypt was used to conduct basic experiment
**/
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

	__syncthreads();
}

/**
	DESGeneratorCUDA, the really entrance function
**/
__global__ void  DESGeneratorCUDA(uint64_t * data)
{
	/**Don't know why should use it.**/
	((uint64_t *)des_SP)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];
	#if MAX_THREAD == 128
		((uint64_t *)des_SP)[threadIdx.x+128] = ((uint64_t *)des_d_sp_c)[threadIdx.x+128];
	#endif

	__syncthreads();

	register uint64_t m_nIndex = data[TX]; uint64_t roundKeys[16];
	
	/**
		Sorry, I didn't find how to change the device 
		value in general CODE, so centainly for each time
	**/
	for(int nPos = 0;nPos < 1;nPos++)
	{	
		/**First Step(Cipher Function)**/
		GenerateKey(m_nIndex,roundKeys);
		m_nIndex = DESOneTime(roundKeys);

		/**Second Step(Reduction Function)**/
		m_nIndex &= totalSpace;
		m_nIndex = (m_nIndex + nPos) & totalSpace;	
		m_nIndex = (m_nIndex + (nPos << 8)) & totalSpace;
		m_nIndex = (m_nIndex + ((nPos << 8) << 8)) & totalSpace;

	}

	data[TX] = m_nIndex;

	__syncthreads();
}

__global__ void OneTime(uint64_t * roundKeys)
{
	__syncthreads();

	//uint64_t plain = 0x305532286D6F295A;
	//uint64_t key   = 0xF1F1F1F1F1F1F1F1;
	uint64_t key = 0x0E0E0E0E0E0E0E02;
	GenerateKey(key, roundKeys);
}

void OneTimeTest()
{
	uint64_t incuda[16];

	_CUDA(cudaMalloc((void**)&incuda , sizeof(uint64_t)*16));
	DESEncrypt<<<1,1>>>(incuda);

	FILE*file=fopen("OneTimeTest.txt","wb");
	assert(file);
	for(int i = 0;i<16;i++)
	{
		fwrite((char*)&(incuda[i]),sizeof(uint64_t),1,file);
	}
	//assert(fwrite((char*)incuda,sizeof(uint64_t),16,file) == 16);
	//fclose(file);

	des_key_schedule ks;//const uint64_t key   = 0xF1F1F1F1F1F1F1F1;
	const_DES_cblock key = {0xF1,0xF1,0xF1,0xF1,0xF1,0xF1,0xF1,0xF1};

	DES_set_key_unchecked(&key,&ks);

	for(int i = 0;i< 16;i++)
	{
		fwrite((char*)&(ks.ks[i].cblock),sizeof(uint64_t),1,file);
	}
}

uint64_t rand64()
{
	uint64_t rs = 0;
	for(uint64_t i = 0;i < 8;i++)
		rs |= ((rand() % 256) << (8*i));
	return rs;
}

#define FF(i, n) for(i = 0;i < n;i++)

/**
	Combined with DESEncrypt to conduct simple performance test
**/
void DESCrypt() 
{
	struct timeval tstart, tend;

	uint64_t * deviceKeyIn, *deviceKeyOut;
	uint64_t   keys[ALL]; int i;

	int round = 0, size; FILE * f1; FILE * f2;

	f1 = fopen("start.in" ,"w");
	f2 = fopen("end.in"   ,"w");

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

	//fclose(f1);fclose(f2);
	
	//printf("Ending DES kernel\n");
}

/**
Combined with DESGeneratorCUDA to generate data
**/

void Logo()
{
	printf("DESRainbowCrack 1.0\n 	Make an implementation of DES Time-and-Memory Tradeoff Technology\n 	By Tian Yulong(mathetian@gmail.com)\n\n");
}

void Usage()
{
	Logo();
	printf("Usage: gencuda   chainLen chainCount suffix\n");
	printf("                 benchmark\n");
	printf("                 onetimetest\n");
	printf("                 keystest\n\n");
	printf("example 1: gencuda 1000 10000 suffix\n");
	printf("example 2: gencuda benchmark\n");
}

struct RainbowChain_t
{
	uint64_t nStartKey;
	uint64_t nEndKey;
};	

typedef struct RainbowChain_t RainbowChain;

uint64_t GetFileLen(FILE* file)
{
    unsigned int pos = ftell(file);
    fseek(file, 0, SEEK_END);
    uint64_t len = ftell(file);
    fseek(file, pos, SEEK_SET);

    return len;
}

void DESGenerator(uint64_t chainLen, uint64_t chainCount, const char * suffix)
{
	char fileName[100];

	sprintf(fileName,"DES_%lld-%lld_%s-cuda", (long long)chainLen, (long long)chainCount,suffix);

	FILE * file = fopen(fileName, "a+");
	
	assert(file);

	uint64_t nDatalen = GetFileLen(file);
	
	assert((nDatalen & ((1 << 4) - 1)) == 0);

	int remainCount =  chainCount - (nDatalen >> 4);
	
	int time1 = remainCount/ALL;
	if(remainCount % ALL != 0) time1++;

	/**Start Preparation**/
	
	uint64_t size = sizeof(uint64_t)*ALL;

	uint64_t * cudaIn;
	uint64_t starts[ALL], ends[ALL];
	
	_CUDA(cudaMalloc((void**)&cudaIn , size));

	/**End Preparation**/
	printf("Need to compute %d rounds %lld\n", time1, (long long)remainCount);
	
	for(int round = 0;round < time1;round++)
	{
		printf("Begin compute the %d round\n", round+1);
		for(uint64_t i = 0;i < ALL;i++)
		{
			RAND_bytes((unsigned char*)(&(starts[i])),sizeof(uint64_t));
			starts[i] &= totalSpaceT;
		}
		/**Belong to CUDA logic**/
		_CUDA(cudaMemcpy(cudaIn,starts,size,cudaMemcpyHostToDevice));

		DESGeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

		_CUDA(cudaMemcpy(ends,cudaIn,size,cudaMemcpyDeviceToHost));
		/**End of CUDA logic**/
		
		for(uint64_t i = 0;i < ALL;i++)
		{
			/**Soooory for the sad expression**/
			int flag1 = fwrite((char*)&(starts[i]),sizeof(uint64_t),1,file);
			int flag2 = fwrite((char*)&(ends[i]),sizeof(uint64_t),1,file);
			assert((flag1 == 1) && (flag2 == 1));
		}

		printf("End compute the %d round\n", round+1);
	}
}	

void KeyTest()
{
	//uint64_t key=0xFEFEFEFEFEFEFEFE;
	uint64_t key=0x0E0E0E0E0E0E0E02;
	uint64_t * cudaIn; uint64_t starts[16];
	_CUDA(cudaMalloc((void**)&cudaIn , sizeof(uint64_t)*16));
	Gee<<<1, 1>>>(cudaIn); cout << "hello" << endl;
	_CUDA(cudaMemcpy(starts,cudaIn,sizeof(uint64_t)*16,cudaMemcpyDeviceToHost));
	for(int i=0;i<16;i++)
	{
		cout<<starts[i]<<endl;
	}
}

int main(int argc, char * argv[])
{
	if(argc != 2 && argc != 4)
	{
		Usage();
		return 1;
	}

	if(argc == 2)
	{
		if(strcmp(argv[1],"benchmark") == 0)
			DESCrypt();
		else if(strcmp(argv[1],"onetimetest") == 0)
			OneTimeTest();
		else if(strcmp(argv[1],"keystest")==0)
			KeyTest();
		else Usage();
		return 1;
	}

	uint64_t chainLen, chainCount; 
	char suffix[100]; 
	
	memset(suffix,0,sizeof(suffix));

	chainLen   = atoll(argv[1]);
	chainCount = atoll(argv[2]);
	memcpy(suffix,argv[3],strlen(argv[3]));

	DESGenerator(chainLen, chainCount, suffix);
	return 0;
}