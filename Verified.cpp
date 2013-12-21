#include "RainbowChain.cpp"

void Logo()
{
	printf("DESRainbowCuda 1.0 - Make an implementation of DES Time-and-Memory Tradeoff Technology\n");
	printf("by Tian Yulong(mathetian@gmail.com)\n");
}

void Usage()
{
	Logo();
	printf("Usage: verified filename chainLen");

	printf("\n");
	printf("example: verified hello.rt 111111");
}


int main(int argc,char*argv[])
{
	if(argc!=3)
	{
		Usage();
		return 0;
	}
	FILE*file=fopen(argv[1],"r");
	int chainLen=atoi(argv[2]);
	if(!file)
	{
		printf("fopen error\n");
		return 0;
	}
	int fileLen=GetFileLen(file);
	if(fileLen%16!=0)
	{
		printf("verified failed, error length\n");
		return 0;
	}
	int chainCount=fileLen/16;
	RainbowChain chain;
	for(int i=0;i<chainCount;i++)
	{
		fread(&chain,sizeof(RainbowChain),1,file);
		ChainWalkContext cwc;
		cwc.setStartIndex(chain.nStartIndex);
		for(int j=0;j<chainLen;j++)
		{
			cwc.PlainToHash();
			cwc.HashToIndex();
		}
		printf("%d\b",cwc.GetCurrentIndex());
		if(cwc.GetCurrentIndex()!=chain.nEndIndex)
			printf("\n warning: integrity check fail\n");
	}
	fclose(file);
	return 0;
}