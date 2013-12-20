#include <stdio.h>

int QuickSort(RainbowChain*pChain,int length)
{
	sort(pChain,pChain+length);
}

#define CHAIN_IN_MEMORY_MAX 1024 
class SortedSegment{
public:
	RainbowChain * getFirst();
	RainbowChain * getAll();
	RainbowChain * getNext();
	void setProperty(int offset,int length,int curOffset);
	int  getLength();
private:
	static FILE*file;
	static FILE*tmpFile;
private:
	int offset;
	int length;
	int curOffset;
	RainbowChain chains[CHAIN_IN_MEMORY_MAX];
};

void SortedSegment::setProperty(int offset,int length,int curOffset)
{
	this->offset=offset;
	this->length=length;
	this->curOffset=curOffset;
}

int SortedSegment::getLength()
{
	return length;
}

RainbowChain * getFirst()
{
	fseek(file,offset+curOffset,SEEK_SET);
	fread(chains,sizeof(RainbowChain),1,file);
	return chains;
}

RainbowChain * getAll()
{
	fseek(file,offset,SEEK_SET);
	fread(chains,sizeof(RainbowChain),length,file);
	return chains;
}

RainbowChain * getNext()
{
	if(curOffset==length*sizeof(RainbowChain))
		return NULL;
	fread(chains,sizeof(RainbowChain),1,tmpFile);
	curOffset+=sizeof(RainbowChain);
}

void ExternalSort(FILE*file,const string&tmpfileName)
{
	FILE*tmpFile=fopen(tmpfileName,"w+r");
	if(!tmpFile)
	{
		printf("ExternalSort: fopen error\n");
		return;
	}
	SortedSegment::file=file;
	SortedSegment::tmpFile=file;

	unsigned int nAvailPhys=GetAvailPhysMemorySize();
	
	int chainCount=nFileLen/16;

	int totalCount=nAvailPhys/16;
	int segNums=chainCount/totalCount;
	if(chainCount%totalCount!=0) segNums++;
	int remainCount=totalCount;
	if(chainCount%totalCount!=0) remainCount=chainCount%totalCount;
	vector<SortedSegment*> svec(segNums,NULL);
	
	for(int i=0;i<segNums-1;i++)
	{
		SortedSegment*seg=new SortedSegment;svec[i]=seg;
		seg->setProperty(totalCount*(i-1),totalCount,0);
	}
	seg->setProperty(totalCount*(segNums-2),remainCount,0);
	fseek(file,0,SEEK_SET);
	fseek(tmpFile,0,SEEK_SET);
	for(int i=0;i<segNums;i++)
	{
		RainbowChain*chains=svec.at(i)->getAll();
		QuickSort(chains,svec.at(i)->getLength());
		fwrite(chains,sizeof(RainbowChain),svec.at(i)->getLength(),tmpFile);
	}

	fseek(file,0,SEEK_SET);fseek(tmpFile,0,SEEK_SET);
	priority_queue<pair<RainbowChain*,int> > chainPQ;
	for(int i=0;i<segNums;i++)
		chainPQ.push(make_pair(svec.at(i)->getNext(),i));
	while(!chainPQ.empty())
	{
		RainbowChain * chain = chainPQ.top().first;
		int id=chainPQ.top().second;
		chainPQ.pop();
		fread(chain,sizeof(RainbowChain),1,file);
		RainbowChain * next=svec.at(id)->getNext();
		if(!next) continue;
		chainPQ.push(make_pair(next,id));
	}
}

int main(int argc,char*argv[])
{
	if(argc!=2)
	{
		Logo();
		printf("usgage: sort pathname\n");
		return 0;
	}

	string sPathName=argv[1];
	FILE*file=fopen(sPathName.c_str(),"r+b");
	if(file==NULL)
	{
		printf("failed to open: %s\n",sPathName);
		return 0;
	}
	unsigned int nFileLen=GetFileLen(file);
	if(nFileLen%16!=0) printf("rainbow table size check failed\n");
	else
	{
		unsigned int nAvailPhys=GetAvailPhysMemorySize();
		printf("available physical memory: %u bytes\n",nAvailPhys);
		if(nAvailPhys>=nFileLen)
		{
			int nRainbowChainCount=nFileLen/16;
			RainbowChain* pChain = (RainbowChain*)new unsigned char[nFileLen];
			if(pChain!=NULL)
			{
				printf("Loading rainbow table...\n");
				fseek(file,0,SEEK_SET);
				if(fread(pChain,1,nFileLen,file)!=nFileLen)
				{
					printf("disk read fail\n");
					goto AbORT;
				}
				printf("Sorting the rainbow table\n");
				QuickSort(pChain,0,nRainbowChainCount-1);
				printf("writing sorted rainbow table...\n");
				fseek(file,0,SEEK_SET);
				fwrite(pChain,1,nFileLen,file);
				delete [] pChain;
			}
		}
		else ExternalSort(file,"tmpFile");
	}
ABORT:
	fclose(file);
	return 0;
}