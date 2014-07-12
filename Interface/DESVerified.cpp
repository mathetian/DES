// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
using namespace utils;

#include "DESChainWalkContext.h"
using namespace descrack;

void Usage()
{
    Logo();
    printf("Usage  : verified filename chainLen\n");

    printf("example: verified hello.rt 1000\n\n");
}

int main(int argc,char*argv[])
{
    FILE * file;
    RainbowChain chain;

    uint64_t chainLen, chainCount;
    uint64_t fileLen, index;

    DESChainWalkContext cwc;

    if(argc != 3)
    {
        Usage();
        return 0;
    }

    if((file  = fopen(argv[1],"rb")) == NULL)
    {
        printf("fopen error\n");
        return 0;
    }

    fseek(file, 0, SEEK_SET);

    chainLen = atoll(argv[2]);
    fileLen  = GetFileLen(file);

    if(fileLen % 16 != 0)
    {
        printf("verified failed, error length\n");
        return 0;
    }

    chainCount = fileLen >> 4;

    printf("FileLen: %lld, ChainCount: %lld\n", (long long)fileLen, (long long)chainCount);

    DESChainWalkContext::SetChainInfo(chainLen, chainCount);

    for(index = 0; index < chainCount; index++)
    {
        assert(fread(&chain, sizeof(RainbowChain), 1, file) == 1);

        cwc.SetKey(chain.nStartKey);

        for(uint32_t j = 0; j < chainLen; j++)
        {
            cwc.KeyToCipher();
            cwc.KeyReduction(j);
        }

        if(cwc.GetKey() != chain.nEndKey)
            printf("warning: integrity check fail, index: %lld \n", (long long)index);

        if(index % 5000 == 0)
            printf("Have check %lld chains\n", (long long)(index + 1));
    }

    fclose(file);

    return 0;
}