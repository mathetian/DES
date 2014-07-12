// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <mpi.h>

#include "Common.h"
#include "TimeStamp.h"
using namespace utils;

#include "DESChainWalkContext.h"
using namespace descrack;

#define BOOLEAN int
#define MASTER_RANK 0
#define TRUE 1
#define FALSE 0
#define BOOLEAN int
#define BLOCK_SIZE 1048576
#define MBYTE 1048576
#define SYNOPSIS printf ("synopsis: %s -f <file> -l <blocks>\n", argv[0])

void Usage()
{
    Logo();
    printf("Usage: generator chainLen chainCount suffix\n");
    printf("                 benchmark\n");
    printf("                 testcasegenerator\n");

    printf("example 1: generator 1000 10000 suffix\n");
    printf("example 2: generator benchmark\n");
    printf("example 7: generator testcasegenerator\n\n");
}

typedef long long ll;

void Benchmark()
{
    int index, nLoop = 1 << 21;

    char str[256];
    memset(str, 0, sizeof(str));

    DESChainWalkContext cwc;
    cwc.GetRandomKey();

    TimeStamp tmps;
    tmps.StartTime();

    for(index = 0; index < nLoop; index++)
        cwc.KeyToCipher();

    sprintf(str, "Benchmark: nLoop %d: keyToHash time:", nLoop);

    tmps.StopTime(str);

    cwc.GetRandomKey();

    tmps.StartTime();

    for(index = 0; index < nLoop; index++)
    {
        cwc.KeyToCipher();
        cwc.KeyReduction(index);
    }

    sprintf(str, "Benchmark: nLoop %d: total time:    ", nLoop);

    tmps.StopTime(str);
}

void TestCaseGenerator()
{
    RainbowChain chain;
    DESChainWalkContext cwc;

    srand((uint32_t)time(0));

    FILE *file = fopen("TestCaseGenerator.txt","wb");

    assert(file && "TestCaseGenerator fopen error\n");

    for(int index = 0; index < 100; index++)
    {
        chain.nStartKey = cwc.GetRandomKey();
        chain.nEndKey   = cwc.Crypt(chain.nStartKey);

        fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
    }

    fclose(file);
}

#ifdef _WIN32
typedef struct
{
    char szFileName[256];
    uint64_t chainLen;
    uint64_t chainCount;
    int rank;
    int numproc;
} DATA;

DWORD WINAPI MyThreadFunction( LPVOID lpParam )
{
    DATA * data = (DATA*)lpParam;
    const char * szFileName = data -> szFileName;
    uint64_t chainLen = data -> chainLen;
    uint64_t totalChainCount = data -> chainCount;

    int rank = data -> rank;
    int numproc =  data -> numproc;

    srand(rank);

    FILE * file;
    DESChainWalkContext cwc;
    char str[256];

    uint64_t nDatalen, index, nChainStart;

    RainbowChain chain;

    uint64_t chainCount = totalChainCount / numproc;

    if((file = fopen(szFileName,"ab+")) == NULL)
    {
        printf("rank %d of %d, failed to create %s\n", rank, numproc, szFileName);
        return 0;
    }
    printf("open successfully\n");
    nDatalen = GetFileLen(file);
    nDatalen = (nDatalen >> 4) << 4;

    if(nDatalen == (chainCount << 4))
    {
        printf("rank %d of %d, precompute has finised\n",rank, numproc);
        return 0;
    }

    if(nDatalen > 0)
    {
        printf("rank %d of %d, continuing from interrupted precomputing, ", rank, numproc);
        printf("have computed %lld chains\n", (ll)(nDatalen >> 4));
    }

    fseek(file, (long)nDatalen, SEEK_SET);
    nChainStart = (nDatalen >> 4);

    index = nDatalen >> 4;

    cwc.SetChainInfo(chainLen, chainCount);

    TimeStamp tmps, parts;
    tmps.StartTime();
    for(; index < chainCount; index++)
    {
        chain.nStartKey = cwc.GetRandomKey();

        for(int nPos = 0; nPos < chainLen; nPos++)
        {
            cwc.KeyToCipher();
            cwc.KeyReduction(nPos);
        }

        chain.nEndKey = cwc.GetKey();
        parts.StartTime();
        if(fwrite((char*)&chain, sizeof(RainbowChain), 1, file) != 1)
        {
            printf("rank %d of %d, disk write error\n", rank, numproc);
            return 0;
        }
        parts.StopTime();
        parts.AddTime(m_disktime);
        if((index + 1)%10000 == 0||index + 1 == chainCount)
        {
            sprintf(str,"rank %d of %d, generate: nChains: %lld, chainLen: %lld: total time:", rank, numproc, (long long)index, (long long)chainLen);
            tmps.StopTime(str);
            tmps.StartTime();
        }
    }
    fclose(file);
    return 0;
}

int main(int argc,char * argv[])
{
    uint64_t chainLen, chainCount;
    char suffix[256];

    if(argc == 2)
    {
        if(strcmp(argv[1], "benchmark") == 0)
            Benchmark();
        else if(strcmp(argv[1],"testcasegenerator") == 0)
            TestCaseGenerator();
        else
            Usage();

        return 0;
    }
    else if(argc != 4)
    {
        Usage();
        return 0;
    }

    chainLen   = atoll(argv[1]);
    chainCount = atoll(argv[2]);

    memset(suffix, 0, 256);
    memcpy(suffix, argv[3], strlen(argv[3]));

#define THRNUM 8
    DATA datas[THRNUM];
    HANDLE  hThreadArray[THRNUM];
    DWORD   dwThreadIdArray[THRNUM];

    for(int i = 0; i < THRNUM; i++)
    {
        sprintf(datas[i].szFileName,"DES_%lld-%lld_%s_%d", (long long)chainLen, (long long)chainCount, suffix, i);
        datas[i].chainLen = chainLen;
        datas[i].chainCount = chainCount;
        datas[i].rank = i;
        datas[i].numproc = THRNUM;
        hThreadArray[i] = CreateThread( NULL,0, MyThreadFunction, &datas[i],0,&dwThreadIdArray[i]);
        assert(hThreadArray[i]);
    }
    WaitForMultipleObjects(THRNUM, hThreadArray, TRUE, INFINITE);
    return 0;
}
#else

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

void Generator(char * szFileName, uint64_t chainLen, uint64_t totalChainCount, int rank, int numproc)
{
    DESChainWalkContext cwc;
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


    printf("rank %d of %d, succeed to create %s\n",rank, numproc, szFileName);
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

    cwc.SetChainInfo(chainLen, chainCount);

    TimeStamp tms;
    tms.StartTime();

    for(; index < chainCount; index++)
    {
        cwc.SetKey(Convert(rank*chainCount + index, 6));
        chain.nStartKey = cwc.GetKey();

        uint32_t nPos;

        for(nPos = 0; nPos < chainLen; nPos++)
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
    char suffix[256], szFileName[256];

    int numproc,rank;

    if(argc == 2)
    {
        if(strcmp(argv[1],"benchmark") == 0)
            Benchmark();
        else if(strcmp(argv[1],"testcasegenerator") == 0)
            TestCaseGenerator();
        else
            Usage();

        return 0;
    }
    else if(argc != 4)
    {
        Usage();

        return 0;
    }

    chainLen   = atoll(argv[1]);
    chainCount = atoll(argv[2]);

    memcpy(suffix, argv[3], sizeof(argv[3]));
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sprintf(szFileName,"DES_%lld-%lld_%s_%d", chainLen, chainCount, suffix, rank);

    Generator(szFileName, chainLen, chainCount, rank, numproc);

    MPI_Finalize();

    return 0;
}

#endif
