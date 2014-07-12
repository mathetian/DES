// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
using namespace utils;

#include "DESCrackEngine.h"
#include "DESChainWalkContext.h"
using namespace descrack;

void Usage()
{
    Logo();
    printf("Usage: crack   text chainLen chainCount encryptedText \n");
    printf("               file chainLen chainCount encryptedFile \n\n");

    printf("example 1: crack text chainLen chainCount 12345 7831224 541234 3827427\n");
    printf("example 2: crack file chainLen chainCount fileName\n\n");
}

int main(int argc,char*argv[])
{
    int keyNum, index, num = 0;
    DESCrackEngine ce;
    DESCipherSet  *p_cs = DESCipherSet::GetInstance();

    if(argc <= 3)
    {
        Usage();
        return 0;
    }
    else if(strcmp(argv[1],"file") == 0)
    {
        if(argc != 4)
        {
            Usage();
            return 0;
        }

        FILE * file = fopen(argv[3],"rb");
        assert(file && "main fopen error\n");
        RainbowChain chain;

        while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
            p_cs -> AddKey(chain.nEndKey);

        fclose(file);
    }
    else if(strcmp(argv[1],"text") == 0)
    {
        keyNum = argc - 3;

        for(index = 0; index < keyNum; index++)
            p_cs -> AddKey(atoll(argv[index+3]));
    }
    else
    {
        Usage();
        return 0;
    }

    ce.Run(argv[2]);

    printf("Statistics\n");
    printf("-------------------------------------------------------\n");

    int foundNum = p_cs -> GetKeyFoundNum();
    struct timeval diskTime  = ce.GetDiskTime();
    struct timeval totalTime = ce.GetTotalTime();

    printf("Key found: %d\n", foundNum);
    printf("Total disk access time: %lld s, %lld us\n",(long long)diskTime.tv_sec,(long long)diskTime.tv_usec);
    printf("Total spend time      : %lld s, %lld us\n",(long long)totalTime.tv_sec,(long long)totalTime.tv_usec);
    printf("Total chains step     : %lld\n", (long long)ce.GetTotalChains());
    printf("Total false alarm     : %lld\n", (long long)ce.GetFalseAlarms());
    printf("\n");

    FILE * file = fopen(argv[3],"rb");

    assert(file && "main fopen error\n");

    RainbowChain chain;

    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        num += p_cs -> Detect(chain);

    fclose(file);

    printf("Detected %d numbers\n", num);
}
