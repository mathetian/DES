#include "Common.h"
#include "ChainWalkContext.h"

void Usage()
{
	Logo();
	printf("Usage  : verified filename chainLen\n");

	printf("example: verified hello.rt 1000\n\n");
}

int main(int argc,char*argv[])
{
	FILE * file;
	long long chainLen, chainCount, fileLen;
	RainbowChain chain; long long index;
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

	fseek(file, 0, SEEK_SET);

	chainLen = atoll(argv[2]);
	fileLen  = GetFileLen(file);
	
	if(fileLen % 16 != 0)
	{
		printf("verified failed, error length\n");
		return 0;
	}

	chainCount = fileLen >> 4;
	
	printf("FileLen: %lld, ChainCount: %lld\n", fileLen, chainCount);

	ChainWalkContext::SetChainInfo(chainLen, chainCount);
	
	for(index = 0;index < chainCount;index++)
	{
		fread(&chain, sizeof(RainbowChain), 1, file);

		cwc.SetKey(chain.nStartKey);
		for(int j = 0;j < chainLen;j++)
		{
			cwc.KeyToCipher();
			cwc.KeyReduction(j);
		}
		if(cwc.GetKey() != chain.nEndKey)
			printf("warning: integrity check fail, index: %lld \n",index);
		
		if(index % 10000 == 0)
			printf("Have check %lld chains\n",index + 1);

	}
	fclose(file);
	return 0;
}