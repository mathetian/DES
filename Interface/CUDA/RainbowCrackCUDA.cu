// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowCUDA.h"
#include "RainbowMD5CUDA.h"
#include "RainbowDESCUDA.h"
#include "RainbowHMACCUDA.h"
#include "RainbowSHA1CUDA.h"
#include "RainbowCipherSet.h"
using namespace rainbowcrack;

#include "TimeStamp.h"
using namespace utils;

void Usage()
{
    Logo();
    printf("Usage: crackcuda type file chainLen encryptedFile\n\n");

    printf("example 1: crackcuda type file chainLen encryptedFile\n\n");
}

void CUDACrack(RainbowCipherSet *p_cs, uint64_t chainLen, const char *type)
{
    assert(chainLen == 4096);

    uint64_t starts[ALL*2], ends[ALL*2];

    uint64_t *cudaIn;
    int size = ALL*sizeof(uint64_t)*2;
    _CUDA(cudaMalloc((void**)&cudaIn, size));

    FILE *file = fopen("Result.txt", "wb+");
    assert(file);

    int i_type = 0;
    if(strcmp(type, "des") == 0)      i_type = 0;
    else if(strcmp(type, "md5") == 0) i_type = 1;

    int total_value = totalSpace_Global;
    if(i_type == 0)  total_value = totalSpace_Global_DES;

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
            for(uint64_t nPos = 0; nPos < chainLen; nPos++)
            {
                starts[i*chainLen + nPos] = keys[i] & total_value;

                if(nPos >= 1300)
                {
                    starts[i*chainLen + nPos] = \
                                                (starts[i*chainLen + nPos] + nPos) & total_value;
                    starts[i*chainLen + nPos] = \
                                                (starts[i*chainLen + nPos] + (nPos << 8)) & total_value;
                    starts[i*chainLen + nPos] = \
                                                (starts[i*chainLen + nPos] + ((nPos << 8) << 8)) & total_value;
                }
            }
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        if(i_type == 0)
            DESCrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 1)
            MD5CrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(int i = 0; i < 128; i++)
        {
            for(uint64_t pos = 0; pos < chainLen ; pos++)
            {
                assert(fwrite((char*)&(ends[i*chainLen + pos]), sizeof(uint64_t), 1, file) == 1);
            }
        }
    }

    fclose(file);
}

int main(int argc, char *argv[])
{
    uint64_t chainLen;
    RainbowCipherSet  *p_cs = RainbowCipherSet::GetInstance();

    if(argc != 5 || strcmp(argv[1], "file") != 0)
    {
        Usage();
        return 0;
    }

    char type[256];
    FILE *file = NULL;

    strcpy(type, argv[1]);
    chainLen = atoll(argv[2]);
    file     = fopen(argv[3], "rb");
    assert(file);

    RainbowChain chain;
    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        p_cs -> AddKey(chain.nEndKey);

    CUDACrack(p_cs, chainLen, type);

    fclose(file);

    return 0;
}