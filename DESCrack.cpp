#include "Common.h"
#include "ChainWalkContext.h"
#include "CrackEngine.h"

void Usage()
{
	Logo();
	printf("Usage: crack   encryptedText hashListFileName\n");
	printf("example: crack 0x305532286D6F295A DES_100-10_test\n\n");
}

int main(int argc,char*argv[])
{
	uint64_t     cipherKey;
	const char * fileName;
	CrackEngine  ce;
	CipherSet    cs;

	if(argc!=3)
	{
		Usage();
		return 0;
	}

	cipherKey  = atoll(argv[1]);
	fileName   = argv[2];
	
	cs.AddKey(cipherKey);
	ce.Run(fileName, cs);

	printf("Statistics\n");
	printf("-------------------------------------------------------\n");
	
	int foundNum = cs.GetKeyFoundNum();
	struct timeval diskTime  = ce.GetDiskTime();	
	struct timeval totalTime = ce.GetTotalTime();
	
	cs.PrintAllFound();
	
	printf("Key found: %d\n", foundNum);
	printf("Total disk access time: %lld s, %lld us\n",(long long)diskTime.tv_sec,(long long)diskTime.tv_usec);
	printf("Total spend time    : %lld s, %lld us\n",(long long)totalTime.tv_sec,(long long)totalTime.tv_usec);
	printf("Total chains step: %lld\n", (long long)ce.GetTotalChains());
	printf("Total false alarm: %lld\n", (long long)ce.GetFalseAlarms());
	printf("\n");
}