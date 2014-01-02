#include "Common.h"
#include "ChainWalkContext.h"
#include "CrackEngine.h"

void Usage()
{
	Logo();
	printf("Usage: crack   encryptedText hashListFileName\n");
	printf("example: crack 0x305532286D6F295A hello.txt");
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
	
	cs.AddHash(cipherKey);
	ce.Run(fileName, cs);

	printf("Statistics\n");
	printf("-------------------------------------------------------\n");
	
	printf("Key found: %d\n", cs.GetKeyFoundNum());
	printf("Total disk access time: %d us\n", ce.GetDiskTime());
	printf("Total spend time: %d us\n", ce.GetTotalTime());
	printf("Total chain walk step: %d\n", ce.GetTotalSteps());
	printf("Total false alarm: %d\n", ce.GetFalseAlarms());
	
	printf("\n");
}