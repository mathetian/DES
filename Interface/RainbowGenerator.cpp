// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <mpi.h>

#include "Common.h"
#include "TimeStamp.h"
using namespace utils;

#include "RainbowChainWalk.h"
using namespace rainbowcrack;

#define TRUE 1
#define FALSE 0

void Usage()
{
    Logo();
    printf("Usage: generator type chainLen chainCount suffix\n");
    printf("                 type rand\n");

    printf("example 1: generator des/md5 1000 10000 suffix\n");
    printf("example 2: generator des/md5 rand\n\n");
}

typedef long long ll;

void Rand(const char *type)
{
    RainbowChain     chain;
    RainbowChainWalk cwc;

    srand((uint32_t)time(0));

    char fileName[100];
    sprintf(fileName, "%s.txt", type);
    FILE *file = fopen(fileName, "wb");

    assert(file && "fopen error\n");

    cwc.SetChainInfo(1, 1, type);

    for(int index = 0; index < 50; index++)
    {
        chain.nStartKey = cwc.GetRandomKey();
        chain.nEndKey   = cwc.Crypt(chain.nStartKey);

        fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
    }

    fclose(file);
}

uint64_t Convert(uint64_t num, int time)
{
    assert(time < 8);

    uint64_t rs = 0, tmp = 0;

    for(int i = 0; i < time; i++)
    {
        tmp = num & ((1ull << 7) - 1);
        tmp <<= 1;
        tmp <<= (8*i);
        rs |= tmp;
        num >>= 7;
    }

    return rs;
}

void Generator(char *szFileName, uint64_t chainLen, uint64_t totalChainCount, int rank, int numproc, const char *type)
{
    RainbowChainWalk cwc;
    char str[256];

    uint64_t nDatalen, index, nChainStart;

    RainbowChain chain;

    uint64_t chainCount = totalChainCount / numproc;

    MPI_File fh;
    MPI_Status  status;

    bool my_file_open_error = FALSE, my_write_error = FALSE;

    char error_string[BUFSIZ];
    int length_of_error_string, error_class;

    my_file_open_error = MPI_File_open(MPI_COMM_SELF, szFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    if (my_file_open_error != MPI_SUCCESS)
    {

        MPI_Error_class(my_file_open_error, &error_class);
        MPI_Error_string(error_class, error_string, &length_of_error_string);
        printf("%3d: %s\n", rank, error_string);

        MPI_Error_string(my_file_open_error, error_string,
                         &length_of_error_string);
        printf("%3d: %s\n", rank, error_string);

        my_file_open_error = TRUE;
    }


    printf("rank %d of %d, succeed to create %s\n", rank, numproc, szFileName);
    printf("%lld %lld %lld %d %d\n", (long long)chainCount, (long long)chainLen, (long long)totalChainCount, rank, numproc);
    nDatalen = 0;

    if(nDatalen == (chainCount << 4))
    {
        printf("rank %d of %d, precompute has finised\n",rank, numproc);
        return;
    }

    if(nDatalen > 0)
    {
        printf("rank %d of %d, continuing from interrupted precomputing, ", rank, numproc);
        printf("have computed %lld chains\n", (ll)(nDatalen >> 4));
    }

    nChainStart += (nDatalen >> 4);

    index = nDatalen >> 4;

    cwc.SetChainInfo(chainLen, chainCount, type);

    TimeStamp tms;
    tms.StartTime();

    for(; index < chainCount; index++)
    {
        if(strcmp(type, "des") == 0)
            cwc.SetKey(Convert(rank*chainCount + index, 6));
        else
            cwc.SetKey(rank*chainCount + index);

        chain.nStartKey = cwc.GetKey();

        for(uint32_t nPos = 0; nPos < chainLen; nPos++)
        {
            cwc.KeyToCipher();
            cwc.KeyReduction(nPos);
        }

        chain.nEndKey = cwc.GetKey();

        MPI_File_write(fh, (char*)(&(chain)), 2, MPI_UINT64_T, &status);

        if (my_write_error != MPI_SUCCESS)
        {
            MPI_Error_class(my_write_error, &error_class);
            MPI_Error_string(error_class, error_string, &length_of_error_string);
            printf("%3d: %s\n", rank, error_string);
            MPI_Error_string(my_write_error, error_string, &length_of_error_string);
            printf("%3d: %s\n", rank, error_string);
            my_write_error = TRUE;
        }

        if((index + 1)%10000 == 0 || index + 1 == chainCount)
        {
            sprintf(str,"rank %d of %d, generate: nChains: %lld, chainLen: %lld: total time:", rank, numproc, (long long)index, (long long)chainLen);
            tms.StopTime(str);
            tms.StartTime();
        }
    }

    MPI_File_close(&fh);
}

int main(int argc,char * argv[])
{
    long long chainLen, chainCount;
    char suffix[256], szFileName[256], type[256];

    int numproc,rank;

    if(argc == 3)
    {
        strcpy(type, argv[1]);
        if(strcmp(argv[2], "rand") == 0) Rand(type);
        else Usage();

        return 0;
    }
    else if(argc != 5)
    {
        Usage();
        return 0;
    }

    strcpy(type, argv[1]);
    chainLen   = atoll(argv[2]);
    chainCount = atoll(argv[3]);

    strcpy(suffix, argv[4]);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sprintf(szFileName,"%s_%lld-%lld_%s_%d", type, chainLen, chainCount, suffix, rank);

    Generator(szFileName, chainLen, chainCount, rank, numproc, type);

    MPI_Finalize();

    return 0;
}
