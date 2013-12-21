#include "CrackEngine.h"

CrackEngine::CrackEngine()
{
	m_diskTime=0.0;
	m_totalTime=0.0;
	m_totalSteps=0;
	m_falseAlarms=0;
}

CrackEngine::~CrackEngine()
{
}

int CrackEngine::BinarySearch(RainbowChain*pChain,int chainCount,uint64 nIndex)
{
	int low=0,high=chainCount;
	while(low<high)
	{
		int mid=(low+high)/2;
		if(pChain[mid]->nEndIndex==nIndex) return mid;
		else if(pChain[mid]->nEndIndex<nIndex) low=mid+1;
		else high=mid;
	}
	return low;
}

void CrackEngine::GetChainIndexRangeWithSameEndPoint(RainbowChain*pChain,int nRainbowChainCount,int nChainIndex,int&nChainIndexFrom,int&nChainIndexTo)
{
	nChainIndexFrom=nChainIndex;
	nChainIndexTo=nChainIndex;
	while(nChainIndexFrom>0)
	{
		if(pChain[nChainIndexFrom-1].nEndIndex==pChain[nChainIndex].nEndIndex)
			nChainIndexFrom--;
	}
	while(nChainIndexTo<nRainbowChainCount-1)
	{
		if(pChain[nChainIndexTo+1].nEndIndex==pChain[nChainIndex].nEndIndex)
			nChainIndexTo++;
	}
}

bool CrackEngine::CheckAlarm(RainbowChain*pChain,int nGuessPos,unsigned char*pHash,HashSet&hs)
{
	ChainWalkContext cwc;
	cwc.setIndex(pChain->nIndexS);
	int nPos;
	for(nPos=0;nPos<nGuessPos;nPos++)
	{
		cwc.PlainToHash();
		cwc.HashToIndex();
	}
	if(cwc.CheckHash(pHash))
	{
		if(printf("plaintext of %s is %s\n",cwc.GetHash().c_str()));
		hs.SetPlain(cwc.GetHash(),cwc.GetPlain(),cwc.GetBinary());
		return true;
	}
	return false;
}

void CrackEngine::SearchRainbowTable(string sPathName,HashSet&hs)
{
	int nIndex=sPathName.find_last_of('/');
	string sFileName;
	if(nIndex!=-1)
		sFileName=sPathName.substr(nIndex+1);
	else
		sFileName=sPathName;
	int nRainbowChainLen,nRainbowChainCount;
	if(!ChainWalkContext::SetupWithPathName(sPathName,nRainbowChainLen,nRainbowChainCount))
		return;
	if(!hs.AnyHashLeftWithLen(ChainWalkContext::GetHashLen()))
	{
		printf("contains\n");
		return;
	}
	FILE*file=fopen(sPathName.c_str(),"rb");
	if(file==NULL)
	{
		printf("SearchRainbowTable: fopen error\n");
		return;
	}
	unsigned int nFileLen=GetFileLen(file);
	if(uFileLen%16!=0||nRainbowChainCount*16!=nFileLen)
		printf("file length check error\n");
	else
	{
		static MemoryPool mp;
		unsigned int nAllocateSize;
		RainbowChain * pChain=(RainbowChain*)mp.Allocate(nFileLen,nAllocateSize);
		if(pChain==NULL)
		{
			printf("SearchRainbowTable: allocate error\n");
			return;
		}
		nAllocateSize=nAllocateSize/16*16;
		fseek(file,0,SEEK_SET);
		bool fVerified=false;
		while(true)
		{
			if(ftell(file)==nFileLen) break;
			clock_t t1=clock();
			unsigned int nDataRead=fread(pChain,1,nAllocateSize,file);
			clock_t t2=clock();
			float fTime=1.0f*(t2-t1)/CLOCKS_PER_SEC;
			printf("%u bytes read, disk access time: %.2f s\n",nDataRead,fTime);
			m_diskTime+=fTime;
			int nRainbowChainCountRead=nDataRead/16;
			t1=clock();
			SearchTableChunk(pChain,nRainbowChainLen,nRainbowChainCountRead,hs);
			t2=clock();
			fTime=1.0f*(t2-t1)/CLOCKS_PER_SEC;
			printf("cryptanalysis time: %.2f s\n",fTime);
			m_totalTime+=fTime;
			if(!hs.AnyHashLeftWithLen(ChainWalkContext::GetHashLen()))
				break;
		}
	}
	fclose(file);
}

void CrackEngine::SearchTableChunk(RainbowChain*pChain,int nRainbowChainLen,int nRainbowChainCount,HashSet&hs)
{
	vector<string> vHash;
	hs.GetLeftHashWithLen(vHash,ChainWalkContext::GetHashLen());
	printf("searching for %d hash%s...",vhash.size(),vHash.size()>1?"es":"");
	int nChainWalkStep=0;
	int nFalseAlarm=0;
	int nChainWalkStepDueToFalseAlarm=0;
	int nHashIndex;
	for(nHashIndex=0;nHashIndex<vHash.size();nHashIndex++)
	{
		unsigned char TargetHash[MAX_HASH_LEN];
		int nHashLen;
		ParseHash(vHash[nHashIndex],TargetHash,nHashIndex);
		if(nHashLen!=ChainWalkContext::GetHashLen())
			printf("SearchTableChunk: nHashLen mismatch\n");
		bool fNewlyGenerated;
		uint64*pStartPosIndexE=m_cws.RequestWalk();
		int nPos;
		for(nPos=nRainbowChainLen-2;nPos>=0;nPos--)
		{
			if(fNewlyGenerated)
			{
				ChainWalkContext cwc;
				cwc.SetHash(TargetHash);
				int i;
				for(i=nPos+1;i<=nRainbowChainLen-2;i++)
				{
					cwc.PlainToHash();
					cwc.HashToIndex();
				}
				pStartPosIndexE[nPos]=cwc.GetIndex();
				nChainWalkStep+=nRainbowChainLen-2-nPos;
			}
			uint64 nIndexEOfCurPos=pStartPosIndexE[nPos];
			int nMathingIndexE=BinarySearch(pChain,nRainbowChainCount,nIndexEOfCurPos);
			if(nMathingIndexE!=-1)
			{
				int nMathingIndexEFrom,nMathingIndexETo;
				GetChainIndexRangeWithSameEndPoint(pChain,nRainbowChainCount,nMathingIndexE,nMathingIndexEFrom,nMathingIndexETo);
				int i;
				for(i=nMathingIndexEFrom;i<=nMathingIndexETo;i++)
				{
					if(CheckAlarm(pChain+1,nPos,TargetHash,hs))
					{
						cwc.DiscardWalk(pStartPosIndexE);
						goto NEXT_HASH;
					}
					else
					{
						nChainWalkStepDueToFalseAlarm+=nPos+1;
						nFalseAlarm++;
					}
				}
			}
		}
	NEXT_HASH:;
	}
	m_nTotalChainWalkStep+=nChainWalkStep;
	m_falseAlarms+=nFalseAlarm;
	m_nToatalChainWalkStepDueToFalseAlarm+=nChainWalkStepDueToFalseAlarm;
}

void CrackEngine::Run(vector<string> vPathName,HashSet&hs)
{
	int i,j;
	for(i=1;i<vPathName.size();i++)
	{
		string strTmp=vPathName.at(i);
		for(j=i-1;j>=0;j--)
		{
			if(vPathName.at(j)>strTmp)
			{
				vPathName[j+1]=vPathName.at(j);
			}
			else break;
		}	
		vPathName[j+1]=strTmp;
	}
	for(i=0;i<vPathName.size()&&hs.AnyHashLeft();i++)
	{
		SearchRainbowTable(vPathName[i],hs);
		printf("\n");
	}
}

float CrackEngine::GetDiskTime()
{
	return m_diskTime;
}

float CrackEngine::GetTotalTime()
{
	return m_totalTime;
}

int CrackEngine::GetTotalSteps()
{
	return m_totalSteps;
}

int CrackEngine::GetFalseAlarms()
{
	return m_falseAlarms;
}
