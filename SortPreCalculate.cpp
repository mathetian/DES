#include <algorithm>
#include <vector>
#include <queue>
using namespace std;

#include "Common.h"
#include <assert.h>

void Usage()
{
	Logo();
	printf("Usage  : sort fileName\n");
	printf("         sort distinct fileName\n");
	printf("example 1: sort DES_100_100_test\n");
	printf("example 2: sort distinct DES_100_100_test\n\n");
}

int QuickSort(RainbowChain * pChain,int length)
{ sort(pChain,pChain+length); }

void ExternalSort(FILE*file,const char * tmpfileName)
{
	FILE*tmpFile;int nAvailPhys, chainCount;
    int memoryCount, segNums, lastCount, fileLen;
    vector<SortedSegment*> svec; int index;
	priority_queue<pair<RainbowChain*,int> > chainPQ;

    if((tmpFile = fopen(tmpfileName,"w+r")) == NULL)
    {
    	printf("ExternalSort: fopen error\n");
		return;
    }

	SortedSegment::file=file;
	SortedSegment::tmpFile=file;

	nAvailPhys  = GetAvailPhysMemorySize();
	fileLen     = GetFileLen(file);
	chainCount  = fileLen >> 4;
	memoryCount = nAvailPhys >> 4;

	segNums = chainCount/memoryCount;
	
	if(chainCount % memoryCount != 0) 
		segNums++;
	
	lastCount = chainCount % memoryCount;
	
	if(lastCount == 0) lastCount = memoryCount;

	svec = vector<SortedSegment*>(segNums,NULL);
	
	for(index = 0;index < segNums;index++)
	{
		SortedSegment * seg = new SortedSegment();
		svec[index] = seg;
		if(index < segNums - 1)
		seg -> setProperty(memoryCount*index,memoryCount,0);
		else
		seg -> setProperty(memoryCount*index,lastCount,0);
	}

	fseek(file,0,SEEK_SET);
	fseek(tmpFile,0,SEEK_SET);

	for(index = 0;index < segNums;index++)
	{
		RainbowChain * chains = svec.at(index) -> getAll();
		QuickSort(chains, svec.at(index) -> getLength());
		fwrite(chains,sizeof(RainbowChain),svec.at(index) -> getLength(),tmpFile);
	}

	fseek(file   ,0,SEEK_SET);
	fseek(tmpFile,0,SEEK_SET);

	for(index = 0;index < segNums;index++)
		chainPQ.push(make_pair(svec.at(index)->getNext(),index));

	while(!chainPQ.empty())
	{
		RainbowChain * chain = chainPQ.top().first;
		int id = chainPQ.top().second;
		chainPQ.pop();
		fread(chain,sizeof(RainbowChain),1,file);
		RainbowChain * next = svec.at(id)->getNext();
		if(!next) continue;
		chainPQ.push(make_pair(next,id));
	}
}

void printMemory(const char * str, long long nAvailPhys)
{
	long long a = 1000, b = 1000*1000;
	long long c = b * 1000;
	printf("%s %lld GB, %lld MB, %lld KB, %lld B\n", str, nAvailPhys/c, (nAvailPhys%c)/b, (nAvailPhys%b)/a, nAvailPhys%1000);
}

void Distinct(const char * sPathName)
{
	FILE * file; int fileLen;
	
	if((file = fopen(sPathName,"r")) == NULL)
	{
		printf("Failed to open: %s\n",sPathName);
		return;
	}

	fileLen = GetFileLen(file);

	int nRainbowChainCount = fileLen >> 4;

	RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];
	RainbowChain * tmpChain = (RainbowChain*)new unsigned char[fileLen];
	
	fseek(file, 0, SEEK_SET);

	printf("Begin Read file\n");
	if(fread(pChain, 1, fileLen, file) != fileLen)
	{
		printf("disk read fail\n");
		return;
	}
	printf("End Read file\n");
	fclose(file);
	QuickSort(pChain, nRainbowChainCount);
	printf("Begin Distinct\n");
	int index = 0, num =0;
	while(index < nRainbowChainCount)
	{
		tmpChain[num++] = pChain[index];
		/*while(index + 1 < nRainbowChainCount && \
				pChain[index].nStartKey == pChain[index+1].nStartKey)
			index++;*/
		while(index + 1 < nRainbowChainCount && \
				pChain[index].nEndKey == pChain[index+1].nEndKey)
			index++;
		index ++;
	}
	FILE * file2 = fopen("Distinct.txt","wb");
	assert(file2 && "fopen error");
	printf("End Distinct\n");
	fwrite((char*)tmpChain, sizeof(RainbowChain), num, file2);
	fclose(file2);
}

int main(int argc,char*argv[])
{
	const char * sPathName; FILE * file, * file2;
	long long fileLen, nAvailPhys;
	char str[256];

	if(argc == 3)
	{
		if(strcmp(argv[1],"distinct") == 0)
			Distinct(argv[2]);
		else
			Usage();
		return 0;
	}
	if(argc!=2)
	{
		Usage();
		return 0;
	}

	sPathName = argv[1];

	if((file = fopen(sPathName,"r")) == NULL)
	{
		printf("Failed to open: %s\n",sPathName);
		return 0;
	}

	fileLen = GetFileLen(file);
	
	if(fileLen % 16 != 0)
	{
		printf("Rainbow table size check failed\n");
		return 0;
	}
	printf("%s FileLen %lld bytes\n", argv[1], fileLen);

	nAvailPhys = GetAvailPhysMemorySize();	
	sprintf(str, "Available free physical memory: ");
	printMemory(str, nAvailPhys);

	if(nAvailPhys >= fileLen)
	{
		int nRainbowChainCount = fileLen >> 4;
		RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];
		if(pChain!=NULL)
		{
			printf("Loading rainbow table...\n");
			fseek(file, 0, SEEK_SET);
			if(fread(pChain, 1, fileLen, file) != fileLen)
			{
				printf("disk read fail\n");
				goto ABORT;
			}
			printf("Sorting the rainbow table...\n");
			QuickSort(pChain, nRainbowChainCount);
			printf("writing sorted rainbow table...\n");
			fclose(file);
			file = fopen(argv[1],"w");
			fwrite(pChain, 1, fileLen, file);
			delete [] pChain;
		}
	}
	else ExternalSort(file,"tmpFile");
ABORT:
	fclose(file);
	return 0;
}