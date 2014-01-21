#include "Common.h"
#include "TimeStamp.h"
#include "ChainWalkContext.h"

#include <iostream>
using namespace std;

#include <assert.h>

void Usage()
{
	Logo();
	printf("Usage: generator chainLen chainCount suffix\n");
	printf("                 benchmark\n");
	printf("                 single startKey\n");
	printf("                 testrandom\n");
	printf("                 testnativerandom\n");
	printf("                 testkeyschedule\n");
	printf("                 testcasegenerator\n");

	printf("example 1: generator 1000 10000 suffix\n");
	printf("example 2: generator benchmark\n");
	printf("example 3: generator single 563109\n");
	printf("example 4: generator testrandom\n");
	printf("example 5: generator testnativerandom\n");
	printf("example 6: generator testkeyschedule\n");
	printf("example 7: generator testcasegenerator\n\n");
}

typedef long long ll;

void Benchmark()
{
	ChainWalkContext cwc;
	int index, nLoop = 1 << 21;	
	char str[256]; 
	memset(str, 0, sizeof(str));

	cwc.GetRandomKey();
	
	TimeStamp tmps;
	tmps.StartTime();

	for(index = 0;index < nLoop;index++) 
		cwc.KeyToCipher();

	sprintf(str, "Benchmark: nLoop %d: keyToHash time:", nLoop);

	tmps.StopTime(str);
	
	cwc.GetRandomKey();

	tmps.StartTime();

	for(index = 0;index < nLoop;index++)
	{
		cwc.KeyToCipher();
		cwc.KeyReduction(index);
	}

	sprintf(str, "Benchmark: nLoop %d: total time:    ", nLoop);
	tmps.StopTime(str);
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

	if((file = fopen("TestRandom.txt","wb")) == NULL)
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

int get(unsigned char cc)
{
	int a = cc; int f = 0;
	while(a)
	{
		if((a & 1) == 1)
			f++;
		a >>= 1;
	}
	if(f % 2 == 1)
		return 0;
	return 1;
}

void clear(unsigned char * key, int type)
{
	if(type ==  20)
	{
		key[0] &= ((1<<8) - 2);
		key[0] |= get(key[0]);
		key[1] &= ((1<<8) - 2);
		key[1] |= get(key[1]);
		key[2] &= ((1<<8) - 4);
		key[2] |= get(key[2]);

		for(int index = 3;index < 8;index++)
			key[index] = 1;
	}
	else if(type == 24)
	{
		for(int index = 3;index < 8;index++)
			key[index] = 0;
	}
	else if(type == 26)
	{
		//unsigned char rkey[8];
	}
	else if(type == 28)
	{
		/*int index   = 3;
		key[index] &= 15;
		for(index++;index < 8;index++)
			key[index] = 0;*/
	}
}

unsigned char plainText[8] = {0x6B,0x05,0x6E,0x18,0x75,0x9F,0x5C,0xCA};

void Generate(unsigned char * key, int type)
{
	if(type == 20)
	{
		int rr = rand() % (1 << 20);
		int ff = (1 << 7) - 1;
		key[0] = (rr & ff) << 1;
		rr >>= 7;
		key[1] = (rr & ff) << 1;
		rr >>= 7;
		key[2] = (rr) << 2;
		int index = 3;
		for(;index < 8;index++)
			key[index] = 0;
	}
}

void TestNativeRandom()
{
	unsigned char key[8], out[8]; des_key_schedule ks;
	RainbowChain chain; FILE * file;

	if((file = fopen("TestNativeRandom.txt","wb")) == NULL)
	{
		printf("TestNativeRandom open error\n");
		return;
	}
	
	int type = 20; srand((uint32_t)time(0));
	for(int index = 0;index < (1<<10);index++)
	{
		Generate(key,20);
		chain.nStartKey = *(uint64_t*)key; 

		memset(out, 0, 8);

		DES_set_key_unchecked(&key, &ks);
		des_ecb_encrypt(&plainText,&out,ks,DES_ENCRYPT);

		clear(out, type);
		chain.nEndKey = *(uint64_t*)out;
		fwrite((char*)&chain, sizeof(RainbowChain), 1, file);

		if(index % 1000000 == 0)
			cout << index << endl;
	}

	fclose(file);
}

unsigned char keyData  [8] = {0x01,0x70,0xF1,0x75,0x46,0x8F,0xB5,0xE6};

void TestKeySchedule()
{
	des_key_schedule ks; FILE * file; int index = 0;
	DES_set_key_unchecked(&keyData, &ks);
	if((file = fopen("TestKeySchedule.txt","wb")) == NULL)
	{
		printf("TestKeySchedule fopen error\n");
		return;
	}
	
	for(;index < 16;index++)
	{
		fwrite(ks.ks[index].cblock,8,1,file);
	}
	fclose(file);
}

void TestCaseGenerator()
{
	FILE * file; RainbowChain chain;
	ChainWalkContext cwc;
	srand((uint32_t)time(0));

	file = fopen("TestCaseGenerator.txt","wb");
	
	assert(file && "TestCaseGenerator fopen error\n");

	for(int index = 0;index < 100;index++)
	{
		chain.nStartKey = cwc.GetRandomKey();
		chain.nEndKey   = cwc.Crypt(chain.nStartKey);
		fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
	}

	fclose(file);
}

typedef struct{
	char szFileName[256];
	uint64_t chainLen;
	uint64_t chainCount;
	int rank;
	int numproc;
}DATA;

DWORD WINAPI MyThreadFunction( LPVOID lpParam )
{
	DATA * data = (DATA*)lpParam;
	const char * szFileName = data -> szFileName;
	uint64_t chainLen = data -> chainLen;
	uint64_t totalChainCount = data -> chainCount;
	
	int rank = data -> rank;
	int numproc =  data -> numproc;
	
	srand(rank);
	
	FILE * file; ChainWalkContext cwc; char str[256];

	uint64_t nDatalen, index, nChainStart;

	RainbowChain chain;

	uint64_t chainCount = totalChainCount / numproc;

	if((file = fopen(szFileName,"ab+")) == NULL)
	{
		printf("rank %d of %d, failed to create %s\n", rank, numproc, szFileName);
		return 0;
	}

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
	
	TimeStamp tmps;
	tmps.StartTime();
	
	for(;index < chainCount;index++)
	{
		chain.nStartKey = cwc.GetRandomKey();

		for(int nPos = 0;nPos < chainLen;nPos++)
		{
			cwc.KeyToCipher();
			cwc.KeyReduction(nPos);
		}

		chain.nEndKey = cwc.GetKey();
		if(fwrite((char*)&chain, sizeof(RainbowChain), 1, file) != 1)
		{
			printf("rank %d of %d, disk write error\n", rank, numproc);
			return 0;
		}
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
	//long long chainLen, chainCount, index;
	uint64_t chainLen, chainCount;
	//char suffix[256], szFileName[256];
	char suffix[256];
	//int numproc,rank;
    
        if(argc == 2)
	{
		if(strcmp(argv[1],"benchmark") == 0)
			Benchmark();
		else if(strcmp(argv[1],"testrandom") == 0)
			TestRandom();
		else if(strcmp(argv[1],"testnativerandom") == 0)
			TestNativeRandom();
		else if(strcmp(argv[1],"testkeyschedule") == 0)
			TestKeySchedule();
		else if(strcmp(argv[1],"testcasegenerator") == 0)
			TestCaseGenerator();
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

    DATA datas[8];  HANDLE  hThreadArray[8]; DWORD   dwThreadIdArray[8];
    
    for(int i = 0;i < 1;i++)
    {	
    	sprintf(datas[i].szFileName,"DES_%lld-%lld_%s_%d", chainLen, chainCount, suffix, i);
    	
    	datas[i].chainLen = chainLen; datas[i].chainCount = chainCount;
    	datas[i].rank = i; datas[i].numproc = 8;
    	
    	hThreadArray[i] = CreateThread( NULL,0, MyThreadFunction, &datas[i],0,&dwThreadIdArray[i]);
    }

   	WaitForMultipleObjects(1, hThreadArray, TRUE, INFINITE);

	return 0;
}