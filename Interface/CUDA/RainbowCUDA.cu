// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowCUDA.h"
#include "RainbowMD5CUDA.h"
#include "RainbowDESCUDA.h"
#include "RainbowHMACCUDA.h"
#include "RainbowSHA1CUDA.h"
using namespace rainbowcrack;

#include "TimeStamp.h"
using namespace utils;

void Usage()
{
    Logo();
    printf("Usage:     cuda type chainLen chainCount suffix\n\n");

    printf("example 1: cuda des/md5/sha1/hmac 1000 10000 suffix\n");
}

void CUDAGenerator(uint64_t chainLen, uint64_t chainCount, const char *suffix, const char *type)
{
    char fileName[100];
    memset(fileName, 0, 100);

    sprintf(fileName,"%s_%lld-%lld_%s", type, (long long)chainLen, (long long)chainCount, suffix);

    FILE * file = fopen(fileName, "ab+");
    assert(file);

    uint64_t nDatalen = GetFileLen(file);

    assert((nDatalen & ((1 << 4) - 1)) == 0);

    int remainCount =  chainCount - (nDatalen >> 4);

    int time_0 = (remainCount + ALL - 1)/ALL;

    uint64_t size = sizeof(uint64_t)*ALL;

    uint64_t * cudaIn;
    uint64_t starts[ALL], ends[ALL];

    _CUDA(cudaMalloc((void**)&cudaIn, size));

    printf("Need to compute %d rounds %lld\n", time_0, (long long)remainCount);

    int i_type = 0;

    if(strcmp(type, "des") == 0)       i_type = 0;
    else if(strcmp(type, "md5") == 0)  i_type = 1;
    else if(strcmp(type, "sha1") == 0) i_type = 2;
    else if(strcmp(type, "hmac") == 0) i_type = 3;
    
    for(int round = 0; round < time_0; round++)
    {
        printf("Begin compute the %d round\n", round + 1);

        TimeStamp tms; tms.StartTime();

        for(uint64_t i = 0; i < ALL; i++)
        {
            starts[i] = round*ALL + i;
            if(i_type == 0) starts[i] = Convert(starts[i], 6) & totalSpace_Global_DES;
            else starts[i] &= totalSpace_Global;
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        if(i_type == 0)
            DESCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 1)
            MD5CUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 2)
            SHA1CUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 3)
            HMACCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(uint64_t i = 0; i < ALL; i++)
        {
            assert(fwrite((char*)&(starts[i]), sizeof(uint64_t), 1, file) == 1);
            assert(fwrite((char*)&(ends[i]), sizeof(uint64_t), 1, file) == 1);
        }

        printf("End compute the %d round\n", round + 1);

        tms.StopTime("StopTime: ");
    }
}

int main(int argc, char * argv[])
{
    if(argc != 5)
    {
        Usage();
        return 1;
    }

    uint64_t chainLen, chainCount;
    char suffix[256], type[256];

    strcpy(type, argv[1]);
    chainLen   = atoll(argv[2]);
    chainCount = atoll(argv[3]);
    strcpy(suffix, argv[4]);

    CUDAGenerator(chainLen, chainCount, suffix, type);

    return 0;
}