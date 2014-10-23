// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
using namespace utils;

#include "RainbowChainWalk.h"
using namespace rainbowcrack;

void Usage()
{
    Logo();
    printf("Usage  : verified type filename chainLen\n");
    printf("example: verified des/md5 hello.rt 1000\n\n");
}

int main(int argc,char*argv[])
{
    FILE * file;
    RainbowChain chain;

    uint64_t chainLen, chainCount;
    uint64_t fileLen, index;

    RainbowChainWalk cwc;
    char type[256];
    memset(type, 0, sizeof(type));

    if(argc != 4)
    {
        Usage();
        return 0;
    }

    strcpy(type, argv[1]);

    if((file  = fopen(argv[2],"rb")) == NULL)
    {
        printf("fopen error\n");
        return 0;
    }

    fseek(file, 0, SEEK_SET);

    chainLen = atoll(argv[3]);
    fileLen  = GetFileLen(file);

    if(fileLen % 16 != 0)
    {
        printf("verified failed, error length\n");
        return 0;
    }

    chainCount = fileLen >> 4;

    printf("FileLen: %lld, ChainCount: %lld\n", (long long)fileLen, (long long)chainCount);

    RainbowChainWalk::SetChainInfo(chainLen, chainCount, type);

    for(index = 0;index < chainCount; index++)
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