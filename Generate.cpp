#include "Common.h"
#include "TimeStamp.h"
#include "ChainWalkContext.h"

#include <iostream>
using namespace std;

void Usage()
{
	Logo();
	printf("Usage: generator chainLen chainCount suffix\n");
	printf("                 benchmark\n");
	printf("				 single startKey\n");
	printf("				 testrandom\n\n");

	printf("example 1: generator 1000 10000 suffix\n");
	printf("example 2: generator benchmark\n");
	printf("example 3: generator single 563109\n");
	printf("example 4: generator testrandom\n\n");
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

void Single(int startKey)
{
	ChainWalkContext cwc; int index;
	cwc.SetKey(startKey); uint64_t key = cwc.GetKey();
	fwrite((char*)&key,sizeof(uint64_t),1,stdout);
	fflush(stdout);
	for(index = 0;index < 1024;index++)
	{
		cwc.KeyToCipher();
		cwc.KeyReduction(index);
		key = cwc.GetKey();
		fwrite((char*)&key,sizeof(uint64_t),1,stdout);
		fflush(stdout);
	}
}

void TestRandom()
{
	ChainWalkContext cwc; RainbowChain chain;
	
	FILE * file;

	if((file = fopen("TestRandom.txt","w")) == NULL)
	{
		fprintf(stderr,"TestRandom.txt open error\n");
		return;
	}
	
	printf("Begin TestRandom\n");
	
	for(int index = 0;index < (1 << 20);index++)
	{
		chain.nStartKey = cwc.GetRandomKey();
		chain.nEndKey   = cwc.Crypt(chain.nStartKey);
		fwrite((char*)&chain,sizeof(RainbowChain),1,file);
	}
	
	printf("End TestRandom\n");

	fclose(file);
}

void clear(unsigned char * key, int type)
{
	if(type ==  20)
	{
		int index   = 2;
		key[index] &= 63;
		for(index++;index < 8;index++)
			key[index] = 0;
	}
	else if(type == 24)
	{
		for(int index = 3;index < 8;index++)
			key[index] = 0;
	}
	else if(type == 28)
	{
		int index   = 3;
		key[index] &= 15;
		for(index++;index < 8;index++)
			key[index] = 0;
	}
}

unsigned char plainText[8] = {0x6B,0x05,0x6E,0x18,0x75,0x9F,0x5C,0xCA};

void TestNativeRandom()
{
	unsigned char key[8], out[8]; des_key_schedule ks;
	RainbowChain chain; FILE * file;

	if((file = fopen("TestNativeRandom.txt","w")) == NULL)
	{
		printf("TestNativeRandom open error\n");
		return;
	}
	
	int type = 20;
	for(int index = 0;index < (1<<type);index++)
	{
		RAND_bytes(key, 8); clear(key, type);
		chain.nStartKey = *(int*)key;
		memset(out, 0, 8);

		DES_set_key_unchecked(&key, &ks);
		
		des_ecb_encrypt(&plainText,&out,ks,DES_ENCRYPT);

		clear(out, type);
		chain.nEndKey = *(int*)out;
		fwrite((char*)&chain, sizeof(RainbowChain), 1, file);

		if(index % 1000000 == 0)
			cout << index << endl;
	}

	fclose(file);
}

void clear(uint64_t & start)
{
	uint64_t a = (1 << 22) - 4;
	start &= a;
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
		else if(strcmp(argv[1],"testrandom") == 0)
			TestRandom();
		else if(strcmp(argv[1],"testnativerandom") == 0)
			TestNativeRandom();
		else  Usage();
		return 0;
	}
	else if(argc == 3)
	{
		if(strcmp(argv[1],"single") == 0)
			Single(atoi(argv[2]));
		else Usage();
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