#include "Common.h"
#include "ChainWalkContext.h"
#include "CrackEngine.h"

#include <iostream>
using namespace std;
#include <assert.h>

void Usage()
{
	Logo();
	printf("Usage: crack   text hashListFileName encryptedText \n");
	printf("               file hashListFileName encryptedFile \n\n");

	printf("example 1: crack text DES_1024-30000_test 12345 7831224 541234 3827427\n");
	printf("example 2: crack file DES_1024-30000_test fileName\n\n");
}

int main(int argc,char*argv[])
{
	int keyNum, index;
	const char * fileName;
	CrackEngine  ce;
	CipherSet  * p_cs = CipherSet::GetInstance();

	if(argc <= 3)
	{
		Usage();
		return 0;
	}
	if(strcmp(argv[1],"file") == 0)
	{
		if(argc != 4)
		{
			Usage();return 0;
		}
		FILE * file = fopen(argv[3],"r");
		assert(file && "main fopen error\n");
		RainbowChain chain;
		while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
		{
			p_cs -> AddKey(chain.nEndKey);
		}
	}
	else if(strcmp(argv[1],"text") == 0)
	{
		keyNum = argc - 3;
		for(index = 0;index < keyNum;index++)
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
	
	p_cs -> PrintAllFound();
	
	printf("Key found: %d\n", foundNum);
	printf("Total disk access time: %lld s, %lld us\n",(long long)diskTime.tv_sec,(long long)diskTime.tv_usec);
	printf("Total spend time    : %lld s, %lld us\n",(long long)totalTime.tv_sec,(long long)totalTime.tv_usec);
	printf("Total chains step: %lld\n", (long long)ce.GetTotalChains());
	printf("Total false alarm: %lld\n", (long long)ce.GetFalseAlarms());
	printf("\n");
}