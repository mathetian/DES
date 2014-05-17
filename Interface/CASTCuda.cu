
#include "../Include/CASTCuda.h"

__device__ int GenerateKey(uint64_t key, uint64_t * store)
{
    return 0;
}

__device__ uint64_t CASTGeneratorCUDA(uint64_t *roundKeys) 
{
	register uint32_t l,r,t;
	register uint64_t block = 0x5A296F6D28325530;

	nl2i(block,l,r);

	__syncthreads();

	E_CAST( 0,roundKeys,l,r,+,^,-);
	E_CAST( 2,roundKeys,r,l,^,-,+);
	E_CAST( 4,roundKeys,l,r,-,+,^);
	E_CAST( 6,roundKeys,r,l,+,^,-);
	E_CAST( 8,roundKeys,l,r,^,-,+);
	E_CAST(10,roundKeys,r,l,-,+,^);
	E_CAST(12,roundKeys,l,r,+,^,-);
	E_CAST(14,roundKeys,r,l,^,-,+);
	E_CAST(16,roundKeys,l,r,-,+,^);
	E_CAST(18,roundKeys,r,l,+,^,-);
	E_CAST(20,roundKeys,l,r,^,-,+);
	E_CAST(22,roundKeys,r,l,-,+,^);
	E_CAST(24,roundKeys,l,r,+,^,-);
	E_CAST(26,roundKeys,r,l,^,-,+);
	E_CAST(28,roundKeys,l,r,-,+,^);
	E_CAST(30,roundKeys,r,l,+,^,-);

	block = ((uint64_t)r) << 32 | l;

	flip64(block);
	return block;
}

__global__ void CASTGeneratorCUDA(uint64_t *data) 
{
	#if MAX_THREAD == 128
		((uint64_t *)CAST_S_table0)[threadIdx.x] = ((uint64_t *)CAST_S_table_constant)[threadIdx.x];
		((uint64_t *)CAST_S_table1)[threadIdx.x] = ((uint64_t *)CAST_S_table_constant)[threadIdx.x+128];
		((uint64_t *)CAST_S_table2)[threadIdx.x] = ((uint64_t *)CAST_S_table_constant)[threadIdx.x+256];
		((uint64_t *)CAST_S_table3)[threadIdx.x] = ((uint64_t *)CAST_S_table_constant)[threadIdx.x+384];
	#elif MAX_THREAD == 256
		((uint32_t *)CAST_S_table0)[threadIdx.x] = ((uint32_t *)CAST_S_table_constant)[threadIdx.x];
		((uint32_t *)CAST_S_table1)[threadIdx.x] = ((uint32_t *)CAST_S_table_constant)[threadIdx.x+256];
		((uint32_t *)CAST_S_table2)[threadIdx.x] = ((uint32_t *)CAST_S_table_constant)[threadIdx.x+512];
		((uint32_t *)CAST_S_table3)[threadIdx.x] = ((uint32_t *)CAST_S_table_constant)[threadIdx.x+768];
	#endif

    register uint64_t m_nIndex = data[TX];
    uint64_t roundKeys[16];

    for(int nPos = 0; nPos < CHAINLEN; nPos++)
    {
        GenerateKey(m_nIndex,roundKeys);
        m_nIndex  = CASTOneTime(roundKeys);
        m_nIndex &= totalSpace;
    }

    data[TX] = m_nIndex;
}

void CASTGenerator(uint64_t chainLen, uint64_t chainCount, const char * suffix)
{
    char fileName[100];
    memset(fileName, 0, 100);

    sprintf(fileName,"CAST_%lld-%lld_%s-cuda", (long long)chainLen, (long long)chainCount,suffix);

    FILE * file = fopen(fileName, "ab+");
    assert(file);

    uint64_t nDatalen = GetFileLen(file);

    assert((nDatalen & ((1 << 4) - 1)) == 0);

    int remainCount =  chainCount - (nDatalen >> 4);

    int time1 = (remainCount + ALL - 1)/ALL;
    /**Start Preparation**/

    uint64_t size = sizeof(uint64_t)*ALL;

    uint64_t * cudaIn;
    uint64_t starts[ALL], ends[ALL];

    _CUDA(cudaMalloc((void**)&cudaIn , size));

    /**End Preparation**/
    printf("Need to compute %d rounds %lld\n", time1, (long long)remainCount);

    for(int round = 0; round < time1; round++)
    {
        printf("Begin compute the %d round\n", round+1);

        TimeStamp tms;
        tms.StartTime();

        for(uint64_t i = 0; i < ALL; i++)
        {
            RAND_bytes((unsigned char*)(&(starts[i])),sizeof(uint64_t));
            starts[i] &= totalSpaceT;
        }

        _CUDA(cudaMemcpy(cudaIn,starts,size,cudaMemcpyHostToDevice));

        CASTGeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends,cudaIn,size,cudaMemcpyDeviceToHost));
        /**End of CUDA logic**/

        for(uint64_t i = 0; i < ALL; i++)
        {
            /**Soooory for the sad expression**/
            int flag1 = fwrite((char*)&(starts[i]),sizeof(uint64_t),1,file);
            int flag2 = fwrite((char*)&(ends[i]),sizeof(uint64_t),1,file);
            assert((flag1 == 1) && (flag2 == 1));
        }

        printf("End compute the %d round\n", round+1);
        tms.StopTime("StopTime: ");
    }
}

int main(int argc, char * argv[])
{
    CASTGenerator(1024, 250000, "Hello");

    return 0;
}