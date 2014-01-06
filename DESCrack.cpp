#include "Common.h"
#include "ChainWalkContext.h"
#include "CrackEngine.h"

#include <iostream>
using namespace std;

void Usage()
{
	Logo();
	printf("Usage: crack   hashListFileName encryptedText \n");
	printf("example: crack DES_1024-30000_test 12345 7831224 541234 3827427\n\n");
}

int main(int argc,char*argv[])
{
	int keyNum, index;
	const char * fileName;
	CrackEngine  ce;
	CipherSet  * p_cs = CipherSet::GetInstance();

	if(argc <= 2)
	{
		Usage();
		return 0;
	}

	fileName   = argv[1];

	keyNum = argc - 2;

	for(index = 0;index < keyNum;index++)
		p_cs -> AddKey(atoll(argv[index+2]));

	ce.Run(fileName);

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