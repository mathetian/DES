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

    // 1024*256*2 = 2^19
    uint64_t starts[ALL*2], ends[ALL*2];

    uint64_t *cudaIn;
    int size = ALL*sizeof(uint64_t)*2;
    _CUDA(cudaMalloc((void**)&cudaIn, size));

    FILE *file = fopen("Result.txt", "wb+");
    assert(file != NULL);

    int i_type = 0;

    if(strcmp(type, "des") == 0)       i_type = 0;
    else if(strcmp(type, "md5") == 0)  i_type = 1;
    else if(strcmp(type, "sha1") == 0) i_type = 2;
    else if(strcmp(type, "hmac") == 0) i_type = 3;
    
    int total_value = totalSpace_Global;
    if(i_type == 0)  total_value = totalSpace_Global_DES;

    while(p_cs -> GetRemainCount() >= 128)
    {
        uint64_t keys[128];
        for(int i = 0; i < 128; i++)
        {
            keys[i] = p_cs -> GetLastKey(); p_cs -> Done();
        }

        /// 2^12*2^7 -> 2^19
        for(int i = 0; i < 128; i++)
        {
            for(uint64_t nPos = 0; nPos < chainLen; nPos++)
            {
                uint64_t &value = starts[i*chainLen + nPos];
                value = keys[i] & total_value;
                if(nPos >= 1300)
                {
                    value = (value + nPos) & total_value;
                    value = (value + (nPos << 8)) & total_value;
                    value = (value + ((nPos << 8) << 8)) & total_value;
                }
            }
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        if(i_type == 0)
            DESCrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 1)
            MD5CrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        else if(i_type == 2)
            SHA1CrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);
        // else if(i_type == 3)
        //     HMACCrackCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

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
    if(argc != 5 || strcmp(argv[2], "file") != 0) { Usage(); return 0; }

    RainbowCipherSet  *p_cs = RainbowCipherSet::GetInstance();

    uint64_t chainLen = atoll(argv[3]);
    FILE *file        = fopen(argv[4], "rb");
    assert(file != NULL);

    RainbowChain chain;
    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        p_cs -> AddKey(chain.nEndKey);

    CUDACrack(p_cs, chainLen, argv[1]);

    fclose(file);

    return 0;
}