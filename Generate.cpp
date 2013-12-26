#include <iostream>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#include "ChainWalkContext.h"
#include "common.h"

void Usage()
{
	Logo();
	printf("Usage: generate chainLen chainCount\\\n");
	printf("				 suffix\\\n");

	printf("\n");
	printf("example: generate 1000 10000 suffix");
}

void Benchmark()
{
	ChainWalkContext cwc;
	struct timeval tstart, tend;
	uint64_t useTimes; 
	int index, nLoop = 1 << 25;	

	cwc.GetRandomKey();
	
	gettimeofday(&tstart, NULL);

	for(index = 0;index < nLoop;index++) 
		cwc.KeyToHash();

	gettimeofday(&tend, NULL);

	useTimes = 1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
    printf("Benchmark: nLoop %d: keyToHash time: %lld us\n", nLoop, (long long)useTimes);

	cwc.GetRandomKey();

	gettimeofday(&tstart, NULL);
	for(index = 0;index < nLoop;index++)
	{
		cwc.KeyToHash();
		cwc.HashToKey(index);
	}

	gettimeofday(&tend, NULL);

	useTimes = 1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
    printf("Benchmark: nLoop %d: total time: %lld us\n", nLoop, (long long)useTimes);
}

int main(int argc,char*argv[])
{
	int chainLen, chainCount, index;
	char suffix[256], szFileName[256];
	FILE * file; ChainWalkContext cwc;
	uint64_t nDatalen, nChainStart, useTimes;
	struct timeval tstart, tend;

	if(argc == 2)
	{
		if(strcmp(argv[1],"benchmark") == 0)
			Benchmark();
		return 0;
	}

	if(argc != 4)
	{
		Usage();
		return 0;
	}
	
	chainLen   = atoi(argv[2]);
	chainCount = atoi(argv[3]);
	memcpy(suffix,argv[4],sizeof(argv[4]));
	sprintf(szFileName,"DES_%d-%d_%s",chainLen,chainCount,suffix);
	if((file = fopen(szFileName,"r+b")) == NULL)
	{
		printf("failed to create %s\n",szFileName);
		return 0;
	}

	nDatalen = GetFileLen(file);
	nDatalen = (nDatalen >> 4) << 4;

	if(nDatalen == (chainCount << 4))
	{
		printf("precompute has finised\n");
		return 0;
	}

	if(nDatalen > 0) printf("continuing from interrupted precomputing\n");
	
	fseek(file, nDatalen, SEEK_SET);
	nChainStart += (nDatalen >> 4);

	index = nDatalen >> 4;

	cwc.SetChainInfo(chainLen, chainCount);
	for(;index < chainCount; index++)
	{
		uint64_t nKey = cwc.GetRandomKey();
		if(fwrite(&nKey,1,8,file)!=8)
		{
			printf("disk write error\n");
			break;
		}

		int nPos;
		for(nPos = 0;nPos < chainLen - 1;nPos++)
		{
			cwc.KeyToHash();
			cwc.HashToKey(nPos);
		}

		nKey = cwc.GetKey();

		if(fwrite(&nKey,1,8,file)!=8)
		{
			printf("disk write error\n");
			break;
		}

		if((index + 1)%100000 == 0||index + 1 == chainCount)
		{
			gettimeofday(&tend, NULL);
			useTimes = 1000000*(tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec);
    		printf("Generate: nLoop %d: total time: %lld us\n", 100000, (long long)useTimes);
			gettimeofday(&tstart, NULL);
		}
	}
	fclose(file);
	return 0;
}