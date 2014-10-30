// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"
#include "TimeStamp.h"
using namespace utils;

#include "RainbowChainWalk.h"
using namespace rainbowcrack;

void Usage()
{
    Logo();
    printf("Usage  :   sort number fileName\n");
    printf("example 1: sort 1 DES_100_100_test\n");
}

typedef pair<RainbowChain, int> PPR;

struct cmp
{
    bool operator()(const PPR &a, const PPR &b)
    {
        return a.first < b.first ? false : true;
    }
};

void QuickSort(RainbowChain *pChain, uint64_t length)
{
    sort(pChain, pChain + length);
}

void ExternalSort(FILE *file, vector <FILE*>tmpFiles)
{
    int index = 0;
    RainbowChain chain;

    fseek(file, 0, SEEK_SET);

    vector <uint64_t> tmpLens(tmpFiles.size(), 0);

    int ss = (int)tmpFiles.size();

    for(; index < ss; index++)
    {
        fseek(tmpFiles[index], 0, SEEK_SET);
        tmpLens[index] = GetFileLen(tmpFiles[index]) >> 4;
    }

    priority_queue<PPR, vector<PPR>, cmp> chainPQ;

    for(index = 0; index < ss; index++)
    {
        assert(fread((char*)&chain,sizeof(RainbowChain),1,tmpFiles[index]) == 1);
        chainPQ.push(make_pair(chain,index));
    }

    while(!chainPQ.empty())
    {
        chain = chainPQ.top().first;
        index = chainPQ.top().second;

        chainPQ.pop();

        fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
        tmpLens[index]--;
        if(tmpLens[index] == 0) continue;
        assert(fread((char*)&chain, sizeof(RainbowChain), 1, tmpFiles[index]) == 1);

        chainPQ.push(make_pair(chain, index));
    }
}

void ExternalSort(FILE *file)
{
    uint64_t nAvailPhys, fileLen, memoryCount;

    int tmpNum, index = 0;
    char str[256];

    nAvailPhys = GetAvailPhysMemorySize();

    fileLen    = GetFileLen(file);

    memoryCount = nAvailPhys >> 4;

    uint64_t eachLen = memoryCount << 4;
    uint64_t lastLen = fileLen % eachLen;

    if(lastLen == 0) lastLen = eachLen;

    tmpNum      = fileLen/nAvailPhys;

    if(fileLen % nAvailPhys != 0) tmpNum++;

    assert((nAvailPhys <= fileLen) && "Error ExternalSort type\n");

    RainbowChain * chains =  (RainbowChain*)new uint8_t[eachLen];

    fseek(file, 0, SEEK_SET);

    vector <FILE*> tmpFiles(tmpNum, NULL);

    for(; index < tmpNum; index++)
    {
        sprintf(str,"tmpFiles-%d",index);
        tmpFiles[index] = fopen(str, "wb");

        assert(tmpFiles[index] &&("tmpFiles fopen error\n"));

        if(index < tmpNum - 1)
        {
            assert(fread((char*)chains, sizeof(RainbowChain), memoryCount, file) == memoryCount);
            QuickSort(chains, memoryCount);
            fwrite((char*)chains, sizeof(RainbowChain), memoryCount, tmpFiles[index]);
        }
        else
        {
            assert(fread((char*)chains, lastLen, 1, file) == 1);
            assert((lastLen % 16 == 0) && ("Error lastLen"));
            QuickSort(chains, lastLen >> 4);
            fwrite((char*)&chains, lastLen, 1, tmpFiles[index]);
        }
    }

    ExternalSort(file, tmpFiles);

    for(index = 0; index < tmpNum; index++)
    {
        fclose(tmpFiles[index]);
    }
}

void printMemory(const char * str, long long nAvailPhys)
{
    long long a = 1000, b = 1000*1000;
    long long c = b * 1000;
    printf("%s %lld GB, %lld MB, %lld KB, %lld B\n", str, nAvailPhys/c, (nAvailPhys%c)/b, (nAvailPhys%b)/a, nAvailPhys%1000);
}

void SortFiles(vector<string> fileNames, vector<FILE*> files, const char *prefix)
{
    int index = 0;
    uint64_t nAvailPhys;
    char str[256];
    vector <uint64_t> fileLens(fileNames.size(), 0);
    FILE * targetFile = NULL;
    nAvailPhys = GetAvailPhysMemorySize();

    sprintf(str, "Available free physical memory: ");
    printMemory(str, nAvailPhys);

    int ss = (int)fileNames.size();

    for(; index < ss; index++)
    {
        uint64_t & fileLen = fileLens[index];
        fileLen = GetFileLen(files[index]);

        assert((fileLen % 16 ==0) && ("Rainbow table size check failed\n"));

        printf("%s FileLen %lld bytes\n", fileNames[index].c_str(), (long long)fileLen);

        if(nAvailPhys > fileLen)
        {
            uint64_t nRainbowChainCount = fileLen >> 4;

            RainbowChain * pChain = (RainbowChain*)new uint8_t[fileLen];

            if(pChain != NULL)
            {
                printf("%d, Loading rainbow table...\n", index);

                fseek(files[index], 0, SEEK_SET);

                if(fread(pChain, 1, fileLen, files[index]) != fileLen)
                {
                    printf("%d, disk read fail\n", index);
                    goto ABORT;
                }

                printf("%d, Sorting the rainbow table...\n", index);

                QuickSort(pChain, nRainbowChainCount);

                printf("%d, Writing sorted rainbow table...\n", index);

                fseek(files[index], 0, SEEK_SET);
                fwrite(pChain, 1, fileLen, files[index]);
                delete [] pChain;
            }
        }
        else ExternalSort(files[index]);
    }

    targetFile = fopen(prefix,"wb");
    fclose(targetFile);

    targetFile = fopen(prefix,"rb+");
    assert(targetFile && ("targetFile fopen error\n"));

    printf("Begin Actually ExternalSort\n");
    ExternalSort(targetFile, files);
    printf("End Actually ExternalSort\n");
    fclose(targetFile);

ABORT:
    for(index = 0; index < ss; index++)
        fclose(files[index]);

}

void SortOneFile(const char *prefix)
{
    uint64_t nAvailPhys;
    char str[256];
    FILE *targetFile;
    uint64_t fileLen;

    nAvailPhys = GetAvailPhysMemorySize();
    sprintf(str, "Available free physical memory: ");
    printMemory(str, nAvailPhys);

    targetFile = fopen(prefix,"rb+");
    assert(targetFile && "fopen error\n");

    fileLen = GetFileLen(targetFile);

    assert((fileLen % 16 ==0) && ("Rainbow table size check failed\n"));

    printf("%s FileLen %lld bytes\n", prefix, (long long)fileLen);

    if(nAvailPhys > fileLen)
    {
        uint64_t nRainbowChainCount = fileLen >> 4;

        RainbowChain * pChain = (RainbowChain*)new uint8_t[fileLen];

        if(pChain != NULL)
        {
            fseek(targetFile, 0, SEEK_SET);

            if(fread(pChain, 1, fileLen, targetFile) != fileLen)
            {
                printf("disk read fail\n");
                goto ABORT;
            }

            printf("Sorting the rainbow table...\n");

            QuickSort(pChain, nRainbowChainCount);

            printf("Writing sorted rainbow table...\n");

            fseek(targetFile, 0, SEEK_SET);
            assert(fwrite(pChain, 1, fileLen, targetFile)==fileLen);

            delete [] pChain;
        }
    }
    else
        ExternalSort(targetFile);

ABORT:
    fclose(targetFile);
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        Usage();
        return 0;
    }

    int num =  atoi(argv[1]);
    assert((num >= 1) && ("sorry number must be less than ten and more than zero\n"));

    if(num == 1)
    {
        printf("Begin Sort One File\n");
        SortOneFile(argv[2]);
        printf("End Sort One File\n");
    }
    else
    {
        vector<string> fileNames(num, "");
        vector<FILE*>  files(num, NULL);

        for(int index = 0; index < num; index++)
        {
            stringstream ss;
            ss << argv[2] << "_" << index;
            fileNames[index] = ss.str();
            files[index] = fopen(fileNames[index].c_str(),"rb+");
            cout << fileNames[index] << endl;
            assert(files[index] && "fopen error\n");
        }

        printf("Begin SortFiles\n");
        SortFiles(fileNames, files, argv[2]);
        printf("End SortFiles\n");
    }

    return 0;
}
