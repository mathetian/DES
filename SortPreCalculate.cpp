#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
using namespace std;

#include "Common.h"
#include <assert.h>

void Usage()
{
	Logo();
	printf("Usage  : sort sort number fileName\n");
	printf("         sort distinct number fileName\n");
	printf("example 1: sort sort 1 DES_100_100_test\n");
	printf("example 2: sort distinct 1 DES_100_100_test\n\n");
}

typedef pair<RainbowChain, int> PPR;

struct cmp
{
    int operator()(PPR a,PPR b){
    	RainbowChain  r1 = a.first;
    	RainbowChain  r2 = b.first;
    	if(r1.nEndKey < r2.nEndKey)
    		return 1;
    	else if(r1.nEndKey == r2.nEndKey)
    		return 0;
    }
};

void QuickSort(RainbowChain * pChain, uint64_t length)
{ sort(pChain, pChain + length); }

void ExternalSort(FILE * file, vector <FILE*> tmpFiles)
{
	int index = 0; RainbowChain chain;

	fseek(file, 0, SEEK_SET);

	vector <uint64_t> tmpLens(tmpFiles.size(), 0);

	for(;index < tmpFiles.size();index++)
	{
		fseek(tmpFiles[index], 0, SEEK_SET);
		tmpLens[index] = GetFileLen(tmpFiles[index]) >> 4;
	}

	priority_queue<PPR, vector<PPR>, cmp> chainPQ;

	for(index = 0;index < tmpFiles.size();index++)
	{
		fread((char*)&chain,sizeof(RainbowChain),1,tmpFiles[index]);
		chainPQ.push(make_pair(chain,index));
	}

	while(!chainPQ.empty())
	{
		chain = chainPQ.top().first;
		index = chainPQ.top().second;
		
		chainPQ.pop();

		fwrite((char*)&chain, sizeof(RainbowChain), 1, file);
		tmpLens[index]--;
		if(tmpLens[index] == 0) continue;
		fread((char*)&chain, sizeof(RainbowChain), 1, tmpFiles[index]);

		chainPQ.push(make_pair(chain, index));
	}
}

void ExternalSort(FILE * file)
{
	uint64_t nAvailPhys, fileLen, chainCount;

	uint64_t memoryCount; int tmpNum; int index = 0;

	char str[256];

	nAvailPhys = GetAvailPhysMemorySize();

	fileLen    = GetFileLen(file); 

	chainCount  = fileLen >> 4;
	
	memoryCount = nAvailPhys >> 4;
	
	uint64_t eachLen = memoryCount << 4;
	uint64_t lastLen = fileLen % eachLen;
	if(lastLen == 0) lastLen = eachLen;

	tmpNum      = fileLen/nAvailPhys;
	
	if(fileLen % nAvailPhys != 0) tmpNum++;

	assert((nAvailPhys <= fileLen) && "Error ExternalSort type\n");

	RainbowChain * chains =  (RainbowChain*)new unsigned char[eachLen];

	fseek(file, 0, SEEK_SET);

	vector <FILE*> tmpFiles(tmpNum, NULL);

	for(;index < tmpNum;index++)
	{
		sprintf(str,"tmpFiles-%d",index);
		tmpFiles[index] = fopen(str, "w");
		assert(tmpFiles[index] &&("tmpFiles fopen error\n"));
		if(index < tmpNum - 1)
		{
			fread((char*)chains, sizeof(RainbowChain), memoryCount, file);
			QuickSort(chains, memoryCount);
			fwrite((char*)chains, sizeof(RainbowChain), memoryCount, tmpFiles[index]);
		}
		else
		{
			fread((char*)chains, lastLen, 1, file);
			assert((lastLen % 16 == 0) && ("Error lastLen"));
			QuickSort(chains, lastLen >> 4);
			fwrite((char*)&chains, lastLen, 1, tmpFiles[index]);
		}	
	}

	ExternalSort(file, tmpFiles);

	for(index = 0;index < tmpNum;index++)
	{
		fclose(tmpFiles[index]);
	}
}

void printMemory(const char * str, long long nAvailPhys)
{
	long long a = 1000, b = 1000*1000;
	long long c = b * 1000;
	printf("%s %lld GB, %lld MB, %lld KB, %lld B\n", str, nAvailPhys/c, (nAvailPhys%c)/b, (nAvailPhys%b)/a, nAvailPhys%1000);
}

void Distinct(const char * fileName)
{
	FILE * file; uint64_t fileLen;
	
	if((file = fopen(fileName, "r")) == NULL)
	{
		printf("Failed to open: %s\n",fileName);
		return;
	}

	fileLen = GetFileLen(file);

	uint64_t nRainbowChainCount = fileLen >> 4;

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

	printf("Begin Distinct\n");
	uint64_t index = 0, num =0;
	while(index < nRainbowChainCount)
	{
		tmpChain[num++] = pChain[index];
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

void SortFiles(vector <string> fileNames, vector <FILE*> files, const char * prefix)
{
	uint64_t nAvailPhys; int index = 0;
	char str[256];

	vector <uint64_t> fileLens(fileNames.size(), 0);	

	FILE * targetFile;

	nAvailPhys = GetAvailPhysMemorySize();	
	sprintf(str, "Available free physical memory: ");
	printMemory(str, nAvailPhys);

	for(;index < fileNames.size();index++)
	{

		uint64_t & fileLen = fileLens[index];

		fileLen = GetFileLen(files[index]);

		assert((fileLen % 16 ==0) && ("Rainbow table size check failed\n"));

		printf("%s FileLen %lld bytes\n", fileNames[index].c_str(), (long long)fileLen);

		if(nAvailPhys > fileLen)
		{
			uint64_t nRainbowChainCount = fileLen >> 4;
		
			RainbowChain * pChain = (RainbowChain*)new unsigned char[fileLen];
			
			if(pChain!=NULL)
			{
				printf("%d, Loading rainbow table...\n", index);
				
				fseek(files[index], 0, SEEK_SET);

				if(fread(pChain, 1, fileLen, files[index]) != fileLen)
				{
					printf("%d, disk read fail\n", index);
					goto ABORT;
				}

				printf("%d, Sorting the rainbow table...\n", index);
				
				QuickSort(pChain, nRainbowChainCount);

				printf("%d, Writing sorted rainbow table...\n", index);
				
				fseek(files[index], 0, SEEK_SET);								
				fwrite(pChain, 1, fileLen, files[index]);
				delete [] pChain;
			}
		}
		else ExternalSort(files[index]);
	}

	targetFile = fopen(prefix,"w");
	fclose(targetFile);

	targetFile = fopen(prefix,"r+");
	assert(targetFile && ("targetFile fopen error\n"));

	printf("Begin Actually ExternalSort\n");
	ExternalSort(targetFile, files);
	printf("End Actually ExternalSort\n");
	fclose(targetFile);

ABORT:
	for(index = 0;index < fileNames.size();index++)
		fclose(files[index]);
	
}

int main(int argc,char*argv[])
{
	if(argc != 4)
	{
		Usage();
		return 0;
	}

	if(strcmp(argv[1],"distinct") == 0)
	{
		int num = atoi(argv[2]);
		assert((num == 1) && "Sorry for that, I want to write `less` code");
		Distinct(argv[3]);
	}	
	else if(strcmp(argv[1],"sort") == 0)
	{
		int num =  atoi(argv[2]);

		assert((num < 9) && ("sorry number must be less than ten\n"));

		vector <string> fileNames(num, "");
		vector <FILE*>  files(num, NULL);

		for(int index = 0;index < num;index++)
		{
			fileNames[index] = argv[3];
			fileNames[index] +=  "_";
			fileNames[index].push_back(index + '0');
			files[index] = fopen(fileNames[index].c_str(),"r+");
			assert(files[index] && "fopen error\n");
		}	

		printf("Begin SortFiles\n");
		SortFiles(fileNames, files, argv[3]);
		printf("End SortFiles\n");
	}
	else Usage();

	return 0;
}