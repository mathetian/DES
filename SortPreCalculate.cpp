#include <algorithm>
#include <vector>
#include <queue>
using namespace std;

#include <stdio.h>
#include "common.h"

void Usage()
{
	Logo();
	printf("Usage: sort fileName\n");

	printf("\n");
	printf("example: sort fileName");
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

int main(int argc,char*argv[])
{
	const char * sPathName; 
	FILE * file;
	unsigned int fileLen, nAvailPhys;

	if(argc!=2)
	{
		Usage();
		return 0;
	}

	sPathName = argv[1];
	if((file = fopen(sPathName,"r+b")) == NULL)
	{
		printf("failed to open: %s\n",sPathName);
		return 0;
	}

	fileLen = GetFileLen(file);
	
	if(fileLen % 16 != 0)
	{
		printf("rainbow table size check failed\n");
		return 0;
	}
	nAvailPhys = GetAvailPhysMemorySize();	
	printf("available physical memory: %u bytes\n",nAvailPhys);

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
			printf("Sorting the rainbow table\n");
			QuickSort(pChain, nRainbowChainCount);
			printf("writing sorted rainbow table...\n");
			fseek(file, 0, SEEK_SET);
			fwrite(pChain, 1, fileLen, file);
			delete [] pChain;
		}
	}
	else ExternalSort(file,"tmpFile");
ABORT:
	fclose(file);
	return 0;
}