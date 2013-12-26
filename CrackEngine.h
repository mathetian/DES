#ifndef _CRACK_ENGINE_H
#define _CRACK_ENGINE_H

#include "CipherSet.h"
#include "ChainWalkContext.h"
#include "common.h"

#include <stdint.h>

class CrackEngine{
public:
	CrackEngine();
	virtual ~ CrackEngine();

public:
	void  Run(const string & fileName, CipherSet & hs);
	int   GetDiskTime();
	int   GetTotalTime();
	int   GetTotalSteps();
	int   GetFalseAlarms();

private:
	ChainWalkContext m_cwc;
	CipherSet		 m_cs;
	int              m_diskTime;
	int          	 m_totalTime;
	int          	 m_totalSteps;
	int          	 m_falseAlarms;
	int 			 m_nTotalChainWalkStep;
	int 			 m_nToatalChainWalkStepDueToFalseAlarm;
private:
	int  BinarySearch(RainbowChain * pChain, int pChainCount, uint64_t nIndex);
	void GetIndexRange(RainbowChain * pChain,int pChainCount,int nChainIndex, int&nChainIndexFrom, int&nChainIndexTo);
	bool CheckAlarm(RainbowChain * pChain,int nGuessedPos);
	void SearchTableChunk(RainbowChain * pChain,int pChainCount);
	void SearchRainbowTable(const string & fileName);
};

#endif