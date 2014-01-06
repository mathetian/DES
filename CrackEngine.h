#ifndef _CRACK_ENGINE_H
#define _CRACK_ENGINE_H

#include "Common.h"
#include "CipherSet.h"
#include "ChainWalkContext.h"
#include "MemoryPool.h"

class CrackEngine{
public:
	CrackEngine();
  ~ CrackEngine();

public:
	void  Run(const char * fileName);
	struct timeval   GetDiskTime();
	struct timeval   GetTotalTime();
	uint64_t   		 GetTotalChains();
	uint64_t		 GetFalseAlarms();

private:
	ChainWalkContext m_cwc;
	CipherSet	 *   p_cs;
	struct timeval   m_diskTime;
	struct timeval   m_totalTime;
	uint64_t         m_totalChains;
	uint64_t         m_falseAlarms;

private:
	static MemoryPool mp;

private:
	uint64_t  BinarySearch(RainbowChain * pChain, uint64_t pChainCount, uint64_t nIndex);
	void      GetIndexRange(RainbowChain * pChain, uint64_t pChainCount, uint64_t nChainIndex, uint64_t & nChainIndexFrom, uint64_t & nChainIndexTo);

private:
	bool      CheckAlarm(RainbowChain * pChain, uint64_t nGuessedPos,uint64_t testV);
	void      SearchTableChunk(RainbowChain * pChain,int pChainCount);
	void      SearchRainbowTable(const char * fileName);
};

#endif