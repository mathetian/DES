#include "common.h"
#include "ChainWalkContext.h"

void Usage()
{
	Logo();
	printf("Usage: Crack    encryptedText hashListFileName\\\n");

	printf("\n");
	printf("example: Crack 0x305532286D6F295A hello.txt");
}

int main(int argc,char*argv[])
{
	if(argc!=3)
	{
		Usage();
		return 0;
	}
	string encryptedText=argv[1];
	string fileName=argv[2];
	CrackEngine ce;
	ce.Run(encryptedText,fileName);

	printf("statistics\n");
	printf("--------------------\n");
	HashRoute re;
	printf("key found: %d\n",re.GetStatHashFound());
	printf("total disk access time: %f s\n",ce.GetDiskAccessTime());
	printf("total spend time: %f s\n",ce.GetTotalTime());
	printf("total chain walk step: %d\n",ce.GetTotalSteps());
	printf("total false alarm: %d\n",ce.GetFalseAlarm());
	printf("\n");
}