// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
using namespace utils;

#include "RainbowChainWalk.h"
#include "RainbowCrackEngine.h"
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
    int index = 4, num = 0;
    RainbowCrackEngine ce; RainbowChain chain;
    RainbowCipherSet  *p_cs = RainbowCipherSet::GetInstance();

    if(argc < 5 || (strcmp(argv[2], "file") != 0 && strcmp(argv[2], "text") != 0))
    {
        Usage(); return 0;
    }
    else if(strcmp(argv[2],"file") == 0)
    {
        if(argc != 5) { Usage(); return 0; }

        FILE * file = fopen(argv[4],"rb");
        assert(file && "main fopen error\n");
        
        while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
            p_cs -> AddKey(chain.nEndKey);

        fclose(file);
    }
    else
    {
        for(; index < argc; index++) p_cs -> AddKey(atoll(argv[index]));
    }
   
    ce.Run(argv[3], argv[1]);

    printf("Statistics\n");
    printf("-------------------------------------------------------\n");

    int foundNum = p_cs -> GetKeyFoundNum();
    struct timeval diskTime  = ce.GetDiskTime();
    struct timeval totalTime = ce.GetTotalTime();

    cout << "Key found: " << foundNum << endl;
    cout << "Total disk access time: " << diskTime.tv_sec << " s, " << diskTime.tv_usec << " us" << endl;
    cout << "Total dspend time     : " << totalTime.tv_sec << " s, " << totalTime.tv_usec << " us" << endl;
    cout << "Total chains step     : " << ce.GetTotalChains()  << endl;
    cout << "Total false alarm     : " << ce.GetFalseAlarms() << endl;
    cout << endl;

    FILE *file = fopen(argv[4],"rb");

    assert(file && "main fopen error\n");

    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        num += p_cs -> Detect(chain);

    fclose(file);

    printf("Detected %d numbers\n", num);
}
