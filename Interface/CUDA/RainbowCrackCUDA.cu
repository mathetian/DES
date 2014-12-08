// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowCUDA.h"
#include "RainbowDESCUDA.h"
#include "RainbowHMACCUDA.h"
#include "RainbowCipherSet.h"
using namespace rainbowcrack;

#include "TimeStamp.h"
using namespace utils;

#include <sys/resource.h>

void Usage()
{
    Logo();
    printf("Usage: crackcuda type file chainLen encryptedFile\n\n");

    printf("example 1: crackcuda des/hmac_md5 file chainLen encryptedFile\n\n");
}

void increase_stack_size()
{
    const rlim_t kStackSize = 64L * 1024L * 1024L;   // min stack size = 64 Mb
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }
        }
    }
}

void CUDACrack(uint64_t key, uint64_t chainLen, const char *type, uint64_t *ends)
{
    assert(chainLen == 4096);

    uint64_t starts[ALL];

    uint64_t *cudaIn;
    int size = ALL*sizeof(uint64_t);
    _CUDA(cudaMalloc((void**)&cudaIn, size));

    int i_type = 0;

    if(strcmp(type, "des") == 0)       i_type = 0;
    else assert(0);

    int total_value = totalSpace_Global;
    if(i_type == 0) total_value = totalSpace_Global_DES;

    TimeStamp tmps; tmps.StartTime();

    for(uint64_t nPos = 0; nPos < chainLen; nPos++)
    {
        uint64_t value = key;
        value = value & total_value;
        value = (value + nPos) & total_value;
        value = (value + (nPos << 8)) & total_value;
        value = (value + ((nPos << 8) << 8)) & total_value;
        starts[nPos] = value;
    }

    _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

    int block_length = 2048/MAX_THREAD;
    if(i_type == 0) 
        DES_CrackCUDA<<<block_length, MAX_THREAD>>>(cudaIn);

    _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

    tmps.StopTime("Initialization Time: ");
}

int main(int argc, char *argv[])
{
    increase_stack_size();

    if(argc != 5 || strcmp(argv[2], "file") != 0)
    {
        Usage(); return 0;
    }

    uint64_t chainLen = atoll(argv[3]);
    FILE *file        = fopen(argv[4], "rb");
    assert(file != NULL);

    FILE *result = fopen("Result.txt", "wb+");
    assert(result != NULL); uint64_t ends[ALL];

    RainbowChain chain;
    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
    {
        CUDACrack(chain.nEndKey, chainLen, argv[1], ends);
        for(int i = 0;i < ALL;i++)
            assert(fwrite((char*)&ends[i], sizeof(uint64_t), 1, result) == 1);
    }

    fclose(file); fclose(result);

    return 0;
}
