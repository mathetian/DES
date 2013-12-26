#include "Common.h"
#include "ChainWalkContext.h"

void Usage()
{
	Logo();
	printf("Usage: verified filename chainLen");

	printf("\n");
	printf("example: verified hello.rt 111111");
}

int main(int argc,char*argv[])
{
	FILE * file;
	int chainLen, chainCount, fileLen;
	RainbowChain chain; int index;
	ChainWalkContext cwc;

	if(argc != 3)
	{
		Usage();
		return 0;
	}
	if((file  = fopen(argv[1],"r")) == NULL)
	{
		printf("fopen error\n");
		return 0;
	}
	
	chainLen = atoi(argv[2]);
	fileLen  = GetFileLen(file);
	
	if(fileLen % 16 != 0)
	{
		printf("verified failed, error length\n");
		return 0;
	}
	chainCount = fileLen >> 4;
	
	ChainWalkContext::SetChainInfo(chainLen, chainCount);
	
	for(index = 0;index < chainCount;index++)
	{
		fread(&chain, sizeof(RainbowChain), 1, file);

		for(int j = 0;j < chainLen;j++)
		{
			cwc.KeyToHash();
			cwc.HashToKey(j);
		}

		if(cwc.GetKey() != chain.nEndKey)
			printf("\n warning: integrity check fail, index: %d \n",index);
		if(index % 10000 == 0)
			printf("\n Have check %d chains\n",index);

	}
	fclose(file);
	return 0;
}