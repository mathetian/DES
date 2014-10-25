// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "TimeStamp.h"
using namespace utils;

#include "RainbowMD5CUDA.h"
#include "RainbowCipherSet.h"
using namespace rainbowcrack;

void Usage()
{
    Logo();
    printf("Usage:     md5crackcuda chainLen chainCount suffix\n\n");

    printf("example 1: md5crackcuda 1000 10000 suffix\n");
}

__global__ void MD5CrackCUDA(uint64_t *data)
{
    __syncthreads();

    uint64_t tx = TX / 4096;
    uint64_t st = tx*4096*2;
    uint64_t ix = st + (TX % 4096);

    register uint64_t value = data[ix];
    for(int nPos = (TX % 4096) + 1; nPos < 4096; nPos++)
        value = Cipher2MSG(MSG2Ciper(value), nPos);

    data[ix] = value;

    ix = st + 2*4096 - TX % 4096 - 1;
    st = st + 4096;

    value = data[ix];
    for(int nPos = 4096 - (TX % 4096); nPos < 4096; nPos++)
        value = Cipher2MSG(MSG2Ciper(value), nPos);

    data[ix] = value;

    __syncthreads();
}

void MD5Crack(RainbowCipherSet *p_cs, uint64_t chainLen)
{
    assert(chainLen == 4096);

    uint64_t starts[ALL*2], ends[ALL*2];

    uint64_t *cudaIn;
    int size = ALL*sizeof(uint64_t)*2;
    _CUDA(cudaMalloc((void**)&cudaIn , size));

    FILE *result = fopen("Result.txt", "wb+");
    assert(result);

    while(p_cs -> GetRemainCount() < 128)
    {
        uint64_t keys[128];

        for(int i = 0; i < 128; i++)
        {
            keys[i] = p_cs -> GetLastKey();
            p_cs -> Done();
        }

        for(int i = 0; i < 128; i++)
        {
            for(uint64_t nPos = 0; nPos < chainLen ; nPos++)
            {
                starts[i*chainLen + nPos] = (keys[i] & totalSpace_Global);
                if(nPos >= 1300)
                {
                    starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + nPos) & totalSpace_Global;
                    starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + (nPos << 8)) & totalSpace_Global;
                    starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + ((nPos << 8) << 8)) & totalSpace_Global;
                }
            }
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        MD5CrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(int i = 0; i < 128; i++)
        {
            for(uint64_t pos = 0; pos < chainLen ; pos++)
            {
                int flag1 = fwrite((char*)&(ends[i*chainLen + pos]), sizeof(uint64_t), 1, result);
                assert(flag1 == 1);
            }
        }
    }

    fclose(result);
}

int main(int argc, char * argv[])
{
    uint64_t chainLen;
    FILE *file;
    RainbowCipherSet  *p_cs = RainbowCipherSet::GetInstance();

    if(argc != 4 || strcmp(argv[1], "file") != 0)
    {
        Usage();
        return 0;
    }

    chainLen = atoll(argv[2]);
    file     = fopen(argv[3], "rb");
    assert(file);

    RainbowChain chain;
    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        p_cs -> AddKey(chain.nEndKey);

    MD5Crack(p_cs, chainLen);
    fclose(file);

    return 0;
}