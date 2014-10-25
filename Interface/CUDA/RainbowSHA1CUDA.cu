// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "TimeStamp.h"
using namespace utils;

#include "RainbowSHA1CUDA.h"
using namespace rainbowcrack;

void Usage()
{
    Logo();
    printf("Usage:     sha1cuda chainLen chainCount suffix\n\n");

    printf("example 1: sha1cuda 1000 10000 suffix\n");
}

__global__ void SHA1GeneratorCUDA(uint64_t *data)
{
    __syncthreads();

    register uint64_t value = data[TX];

    for(int nPos = 0; nPos < CHAINLEN; nPos++)
        value = Cipher2MSG(MSG2Ciper(value), nPos);

    data[TX] = value;

    __syncthreads();
}

void SHA1Generator(uint64_t chainLen, uint64_t chainCount, const char *suffix)
{
    char fileName[100];
    sprintf(fileName,"SHA1_%lld-%lld_%s-cuda", (long long)chainLen, (long long)chainCount, suffix);
    FILE * file = fopen(fileName, "ab+");
    assert(file);
    uint64_t nDatalen = GetFileLen(file);

    assert((nDatalen & ((1 << 4) - 1)) == 0);

    int remainCount = chainCount - (nDatalen >> 4);

    int time_0 = (remainCount + ALL - 1)/ALL;

    uint64_t size = sizeof(uint64_t)*ALL;

    uint64_t *cudaIn;
    uint64_t starts[ALL], ends[ALL];

    _CUDA(cudaMalloc((void**)&cudaIn , size));

    printf("Need to compute %d rounds %lld\n", time_0, (long long)remainCount);

    for(int round = 0; round < time_0; round++)
    {
        printf("Begin compute the %d round\n", round+1);

        TimeStamp tms;
        tms.StartTime();

        for(uint64_t i = 0; i < ALL; i++)
        {
            starts[i]  = round*ALL + i;
            starts[i] &= totalSpace_Global;
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        SHA1GeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(uint64_t i = 0; i < ALL; i++)
        {
            int flag1 = fwrite((char*)&(starts[i]), sizeof(uint64_t), 1, file);
            int flag2 = fwrite((char*)&(ends[i]), sizeof(uint64_t), 1, file);
            assert((flag1 == 1) && (flag2 == 1));
        }

        printf("End compute the %d round\n", round+1);

        tms.StopTime("StopTime: ");
    }
}

int main(int argc, char * argv[])
{
    if(argc != 4)
    {
        Usage();
        return 0;
    }

    uint64_t chainLen = atoll(argv[1]);
    uint64_t chainCount = atoll(argv[2]);
    char suffix[100];
    memcpy(suffix, argv[3], strlen(argv[3]));

    SHA1Generator(chainLen, chainCount, suffix);

    return 0;
}