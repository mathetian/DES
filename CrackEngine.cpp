#include "CrackEngine.h"
#include "TimeStamp.h"

MemoryPool CrackEngine::mp;

CrackEngine::CrackEngine() : m_totalChains(0), m_falseAlarms(0)			
{
}

CrackEngine::~CrackEngine()
{
}

uint64_t CrackEngine::BinarySearch(RainbowChain * pChain, uint64_t pChainCount, uint64_t nIndex)
{
	long long low=0, high=pChainCount;
	while(low<high)
	{
		long long mid = (low+high)/2;
		if(pChain[mid].nEndKey == nIndex) return mid;
		else if(pChain[mid].nEndKey < nIndex) low = mid + 1;
		else high=mid;
	}
	return low;
}

void CrackEngine::GetIndexRange(RainbowChain * pChain,uint64_t pChainCount, uint64_t nChainIndex,uint64_t&nChainIndexFrom, uint64_t&nChainIndexTo)
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

bool CrackEngine::CheckAlarm(RainbowChain * pChain, uint64_t nGuessPos)
{
	ChainWalkContext cwc;int nPos; uint64_t old = pChain -> nStartKey;
	
	cwc.SetKey(pChain -> nStartKey);

	for(nPos = 0;nPos < nGuessPos;nPos++)
	{
		old = cwc.GetKey();
		cwc.KeyToCipher();
		cwc.KeyReduction(nPos);
	}

	if(cwc.GetKey() == m_cs.GetLeftKey())
	{
		printf("plaintext of %lld is %lld\n",(long long)cwc.GetKey(), (long long)old);
		m_cs.AddResult(m_cs.GetLeftKey(), old);
		m_cs.Done();
		return true;
	}

	return false;
}

void CrackEngine::SearchRainbowTable(const char * fileName)
{
	char str[256];
	uint64_t nChainLen, nChainCount;
	uint64_t fileLen, nAllocateSize, nDataRead;
	FILE * file; RainbowChain * pChain;
		
	AnylysisFileName(fileName, nChainLen, nChainCount);

	if((file = fopen(fileName,"rb")) == NULL)
	{
		printf("SearchRainbowTable: fopen error\n");
		return;
	}

	fileLen = GetFileLen(file);

	if(fileLen % 16 != 0 || nChainCount*16 != fileLen)
	{
		printf("file length check error\n");
		return;
	}
	/**
		Reuse the space and avoid duplicate allocation
		Allocate at most max(fileLen,memorySize);
	**/
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


		TimeStamp::StartTime();
		
		nDataRead = fread(pChain,1, nAllocateSize,file);
		if(nDataRead != nAllocateSize)
		{
			printf("Warning nDataRead: %lld, nAllocateSize: %lld\n", (long long)nDataRead, (long long)nAllocateSize);
		}

		sprintf(str,"%lld bytes read, disk access time:", (long long)nAllocateSize);
		TimeStamp::StopTime(str);
		TimeStamp::AddTime(m_diskTime);

		TimeStamp::StartTime();
		
		SearchTableChunk(pChain,nDataRead >> 4);		
    	
    	sprintf(str,"cryptanalysis time: ");
    	TimeStamp::StopTime(str);
    	TimeStamp::AddTime(m_totalTime);
		
		if(m_cs.Solved()) break;
	}
	fclose(file);
}

void CrackEngine::SearchTableChunk(RainbowChain * pChain, int pChainCount)
{
	int nFalseAlarm, nChainWalkStepDueToFalseAlarm;
	int nHashLen, nPos, nIndex;
	uint64_t key = m_cs.GetLeftKey();

	printf("Searching for key: %lld...",(long long)key);
	
	nFalseAlarm = 0; 
	nChainWalkStepDueToFalseAlarm = 0;
	
	vector <uint64_t> pEndKeys(ChainWalkContext::m_chainLen, 0);

	for(nPos = ChainWalkContext::m_chainLen - 2;nPos >= 0;nPos--)
	{
		m_cwc.SetKey(key);

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
