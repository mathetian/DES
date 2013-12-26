#include "CrackEngine.h"
#include <sys/time.h>
#include "MemoryPool.h"

CrackEngine::CrackEngine()
{
	m_diskTime    = 0;
	m_totalTime   = 0;
	m_totalSteps  = 0;
	m_falseAlarms = 0;
	m_nTotalChainWalkStep = 0;
	m_nToatalChainWalkStepDueToFalseAlarm = 0;
}

CrackEngine::~CrackEngine()
{
}

int CrackEngine::BinarySearch(RainbowChain * pChain,int pChainCount,uint64_t nIndex)
{
	int low=0, high=pChainCount;
	while(low<high)
	{
		int mid = (low+high)/2;
		if(pChain[mid].nEndKey == nIndex) return mid;
		else if(pChain[mid].nEndKey < nIndex) low=mid+1;
		else high=mid;
	}
	return low;
}

void CrackEngine::GetIndexRange(RainbowChain*pChain,int pChainCount,int nChainIndex,int&nChainIndexFrom,int&nChainIndexTo)
{
	nChainIndexFrom = nChainIndex;
	nChainIndexTo   = nChainIndex;

	while(nChainIndexFrom>0)
	{
		if(pChain[nChainIndexFrom - 1].nEndKey == pChain[nChainIndex].nEndKey)
			nChainIndexFrom--;
	}

	while(nChainIndexTo < pChainCount)
	{
		if(pChain[nChainIndexTo+1].nEndKey==pChain[nChainIndex].nEndKey)
			nChainIndexTo++;
	}
}

bool CrackEngine::CheckAlarm(RainbowChain*pChain,int nGuessPos)
{
	ChainWalkContext cwc;int nPos;
	cwc.SetKey(pChain -> nStartKey);

	uint64_t old = cwc.GetKey();

	for(nPos = 0;nPos < nGuessPos;nPos++)
	{
		old = cwc.GetKey();
		cwc.KeyToHash();
		cwc.HashToKey(nPos);
	}

	if(cwc.GetKey() == m_cs.GetLeftHash())
	{
		printf("plaintext of %lld is %lld\n",(long long)cwc.GetKey(), (long long)old);
		m_cs.AddResult(m_cs.GetLeftHash(), old);
		return true;
	}

	return false;
}

void CrackEngine::SearchRainbowTable(const string&fileName)
{
	int nRainbowChainLen,nRainbowChainCount;
	FILE * file; unsigned int fileLen, nAllocateSize;
	static MemoryPool mp; RainbowChain * pChain;
	struct timeval tstart, tend; bool fVerified;
	uint64_t useTimes; 
	unsigned int nDataRead;
	
	if((file = fopen(fileName.c_str(),"rb")) == NULL)
	{
		printf("SearchRainbowTable: fopen error\n");
		return;
	}

	fileLen = GetFileLen(file);

	if(fileLen % 16 != 0 || nRainbowChainCount*16 != fileLen)
	{
		printf("file length check error\n");
		return;
	}

	if((pChain = (RainbowChain*)mp.Allocate(fileLen, nAllocateSize)) == NULL)
	{
		printf("SearchRainbowTable: allocate error\n");
		return;
	}
	
	nAllocateSize = nAllocateSize / 16 * 16;
	fseek(file, 0, SEEK_SET);

	while(true)
	{
		if(ftell(file) == fileLen) break;
		
		gettimeofday(&tstart, NULL);
		nDataRead = fread(pChain,1,nAllocateSize,file);
		gettimeofday(&tend, NULL);

		useTimes = 1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
    	
    	printf("%u bytes read, disk access time: %lld us\n", nDataRead, (long long)useTimes);
		m_diskTime += useTimes;
				
		gettimeofday(&tstart, NULL);
		SearchTableChunk(pChain,nDataRead >> 4);
		gettimeofday(&tend, NULL);
		
		useTimes = 1000000*(tend.tv_sec-tstart.tv_sec)+(tend.tv_usec-tstart.tv_usec);
    	printf("cryptanalysis time: %lld us\n",(long long)useTimes);
		m_totalTime += useTimes;
		
		if(m_cs.solved())
			break;
	}
	fclose(file);
}

void CrackEngine::SearchTableChunk(RainbowChain * pChain, int pChainCount)
{
	printf("Searching for cipherText: %lld...",(long long)m_cs.GetLeftHash());
	
	int nFalseAlarm, nChainWalkStepDueToFalseAlarm;
	int nHashLen, nPos, nIndex;
	uint64_t cipherText = m_cs.GetLeftHash();
	
	nFalseAlarm = 0; 
	nChainWalkStepDueToFalseAlarm = 0;
	
	vector <int> pEndKeys(ChainWalkContext::m_chainLen, 0);

	for(nPos = ChainWalkContext::m_chainLen - 2;nPos >= 0;nPos--)
	{
		m_cwc.SetKey(cipherText);
		for(nIndex = nPos + 1; nIndex < ChainWalkContext::m_chainLen;nIndex++)
		{
			m_cwc.KeyToHash();
			m_cwc.HashToKey(nIndex);
		}
		pEndKeys[nPos] = m_cwc.GetKey();
	}

	for(nPos = 0; nPos < m_cwc.m_chainLen - 1;nPos++)
	{
		int nMathingIndexE = BinarySearch(pChain, pChainCount, pEndKeys[nPos]);
		
		if(nMathingIndexE != -1)
		{
			int nMathingIndexEFrom, nMathingIndexETo;
			GetIndexRange(pChain,pChainCount,nMathingIndexE,nMathingIndexEFrom,nMathingIndexETo);
			for(nIndex = nMathingIndexEFrom;nIndex <= nMathingIndexETo;nIndex++)
			{
				if(CheckAlarm(pChain+nIndex, nPos))
					goto NEXT_HASH;
				else
				{
					nChainWalkStepDueToFalseAlarm += nPos+1;
					nFalseAlarm ++;
				}
			}
		}
	}
NEXT_HASH:;
	m_nTotalChainWalkStep += pChainCount;
	m_falseAlarms += nFalseAlarm;
	m_nToatalChainWalkStepDueToFalseAlarm += nChainWalkStepDueToFalseAlarm;
}

void CrackEngine::Run(const string & fileName, CipherSet & cs)
{
	this -> m_cs = cs;
	ChainWalkContext::SetupWithPathName(fileName);
	while(cs.AnyHashLeft())
	{
		SearchRainbowTable(fileName);
	}
}

int CrackEngine::GetDiskTime()
{
	return m_diskTime;
}

int CrackEngine::GetTotalTime()
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
