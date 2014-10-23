// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
using namespace utils;

#include "RainbowCrackEngine.h"
#include "RainbowChainWalk.h"
using namespace rainbowcrack;

void Usage()
{
    Logo();
    printf("Usage: crack  type text chainLen chainCount encryptedText \n");
    printf("              type file RainbowFiles encryptedFile \n\n");

    printf("example 1: crack md5/sha1 text RainbowFiles 12345 7831224 541234 3827427\n");
    printf("example 2: crack md5/sha1 file RainbowFiles encryptedFile\n\n");
}

int main(int argc,char*argv[])
{
    int keyNum, index, num = 0;
    RainbowCrackEngine ce;
    RainbowCipherSet  *p_cs = RainbowCipherSet::GetInstance();
    char type[256];
    memset(type, 0, sizeof(type));

    strcpy(type, argv[1]);

    if(argc <= 4)
    {
        Usage();
        return 0;
    }
    else if(strcmp(argv[2],"file") == 0)
    {
        if(argc != 5)
        {
            Usage();
            return 0;
        }

        FILE * file = fopen(argv[4],"rb");
        assert(file && "main fopen error\n");
        RainbowChain chain;

        while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
            p_cs -> AddKey(chain.nEndKey);

        fclose(file);
    }
    else if(strcmp(argv[2], "text") == 0)
    {
        keyNum = argc - 4;

        for(index = 0; index < keyNum; index++)
            p_cs -> AddKey(atoll(argv[index+4]));
    }
    else
    {
        Usage();
        return 0;
    }

    ce.Run(argv[3]);

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

    FILE * file = fopen(argv[4],"rb");

    assert(file && "main fopen error\n");

    RainbowChain chain;

    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        num += p_cs -> Detect(chain);

    fclose(file);

    printf("Detected %d numbers\n", num);
}
