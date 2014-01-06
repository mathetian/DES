#include "Common.h"
#include "TimeStamp.h"
#include "ChainWalkContext.h"

void Usage()
{
	Logo();
	printf("Usage: generator chainLen chainCount suffix\n");
	printf("                 benchmark\n\n");
	printf("example 1: generator 1000 10000 suffix\n");
	printf("example 2: generator benchmark\n\n");
}

typedef long long ll;

void Benchmark()
{
	ChainWalkContext cwc;
	int index, nLoop = 1 << 21;	
	char str[256]; 
	memset(str, 0, sizeof(str));

	cwc.GetRandomKey();
	
	TimeStamp::StartTime();

	for(index = 0;index < nLoop;index++) 
		cwc.KeyToCipher();

	sprintf(str, "Benchmark: nLoop %d: keyToHash time:", nLoop);

	TimeStamp::StopTime(str);
	
	cwc.GetRandomKey();

	TimeStamp::StartTime();

	for(index = 0;index < nLoop;index++)
	{
		cwc.KeyToCipher();
		cwc.KeyReduction(index);
	}
	sprintf(str, "Benchmark: nLoop %d: total time:    ", nLoop);
	TimeStamp::StopTime(str);
}

int main(int argc,char*argv[])
{
	long long chainLen, chainCount, index;
	char suffix[256], szFileName[256];

	FILE * file; ChainWalkContext cwc;
	uint64_t nDatalen, nChainStart;
	RainbowChain chain;


	char str[256];

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
	chainLen   = atoll(argv[1]);
	chainCount = atoll(argv[2]);

	memcpy(suffix, argv[3], sizeof(argv[3]));
	sprintf(szFileName,"DES_%lld-%lld_%s", chainLen, chainCount,suffix);
	
	if((file = fopen(szFileName,"a+")) == NULL)
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

	if(nDatalen > 0)
	{
		printf("continuing from interrupted precomputing\n");
		printf("have computed %lld chains\n", (ll)(nDatalen >> 4));
	} 
	
	fseek(file, nDatalen, SEEK_SET);
	nChainStart += (nDatalen >> 4);

	index = nDatalen >> 4;

	cwc.SetChainInfo(chainLen, chainCount);
	
	TimeStamp::StartTime();

	for(;index < chainCount;index++)
	{
		chain.nStartKey = cwc.GetRandomKey();

		int nPos;
		for(nPos = 0;nPos < chainLen;nPos++)
		{
			cwc.KeyToCipher();
			cwc.KeyReduction(nPos);
			uint64_t key1 = cwc.GetKey();
			/*if(index == 0)
			fwrite((char*)&key1,sizeof(uint64_t),1,file);*/
		}

		chain.nEndKey = cwc.GetKey();

		if(fwrite((char*)&chain, sizeof(RainbowChain), 1, file) != 1)
		{
			printf("disk write error\n");
			break;
		}

		if((index + 1)%10000 == 0||index + 1 == chainCount)
		{
			sprintf(str,"Generate: nChains: %d, chainLen: %lld: total time:", 10000, chainLen);
			TimeStamp::StopTime(str);
			TimeStamp::StartTime();
		}
	}
	fclose(file);
	return 0;
}