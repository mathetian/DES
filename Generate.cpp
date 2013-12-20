#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "ChainWalkContext.h"
using namespace std;

void Logo()
{
	printf("DESRainbowCuda 1.0 - Make an implementation of DES Time-and-Memory Tradeoff Technology\n");
	printf("by Tian Yulong(mathetian@gmail.com)\n");
}

void Usage()
{
	Logo();
	printf("Usage: generate plaintext\\\n");
	printf("	   			chainLen chainCount\\\n");
	printf("				fileNamePrefix\\\n");

	printf("\n");
	printf("example: generate 0x305532286D6F295A 1000 10000");
}

void Bench()
{
	ChainWalkContext cwc;
	cwc.GenerateRandomIndex();
	cwc.IndexToPlain();
	clock_t t1=clock();
	int nLoop=25000000;
	int i;
	for(i=0;i<nLoop;i++)
		cwc.PlainToHash();
	clock_t t2=clock();int nSecond=(t2-t1)/CLOCK_PER_SEC;
	printf("%d of %d rainbow chains generated (%d m %d s)\n",nSecond/60,nSecond%60);

	cwc.GenerateRandomIndex();
	t1=clock();
	for(i=0;i<nLoop;i++)
	{
		cwc.IndexToPlain();
		cwc.PlainToHash();
		cwc.HashToIndex(i);
	}
	t2=clock();nSecond=(t2-t1)/CLOCK_PER_SEC;
	printf("%d of %d rainbow chains generated (%d m %d s)\n",nSecond/60,nSecond%60);
}

int main(int argc,char*argv[])
{
	if(argc!=5)
	{
		Usage();
		return 0;
	}
	const char*plainText=argv[1];
	int chainLen=atoi(argv[2]);
	int chainCount=atoi(argv[3]);
	const char*fileNamePrefix=argv[4];
	char szFileName[256];
	sprintf(szFileName,"DES_%s_%d-%d_%s",plainText,chainLen,chainCount,fileNamePrefix);
	FILE*file=fopen(szFileName,"r+b");
	if(file==NULL)
	{
		printf("failed to create %s\n",szFileName);
		return 0;
	}
	unsigned int nDatalen=GetFileLen(file);
	nDatalen=nDatalen/16*16;
	if(nDatalen==chainCount*16)
	{
		printf("precompute has finised\n");
		return 0;
	}
	if(nDatalen>0) printf("continuing from interrupted precomputing\n");
	fseek(file,nDatalen,SEEK_SET);
	ChainWalkContext cwc;
	nChainStart+=nDatalen/16;
	clock_t t1=clock();
	int i;
	for(i=nDatalen/16;i<chainCount;i++)
	{
		cwc.setIndex(nChainStart++);
		uint64 nIndex=cwc.GetIndex();
		if(fwrite(&nIndex,1,8,file)!=8)
		{
			printf("disk write error\n");
			break;
		}
		int npos;
		for(npos=0;npos<chainLen-1;npos++)
		{
			cwc.IndexToPlain();
			cwc.PlainToHash();
			cwc.HashToIndex(npos);
		}
		nIndex=cwc.GetIndex();
		if(fwrite(&nIndex,1,8,file)!=8)
		{
			printf("disk write error\n");
			break;
		}
		if((i+1)%100000==0||i+1==chainCount)
		{
			clock_t t2=clock();
			int nSecond=(t2-t1)/CLOCK_PER_SEC;
			printf("%d of %d rainbow chains generated (%d m %d s)\n",i+1,chainCount,nSecond/60,nSecond%60);
			t1=clock();
		}
	}
	fclose(file);
	return 0;
}