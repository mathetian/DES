#ifndef _CRACK_ENGINE_H
#define _CRACK_ENGINE_H
class CrackEngine{
public:
	CrackEngine();
	virtual ~ CrackEngine();
private:
	ChainWalkSet m_cws;
	float m_diskTime;
	float m_totalTime;
	int m_totalSteps;
	int m_falseAlarms;
private:
	int BinarySearch(RainbowChain*pChain,int nRainbowChainCount,uint64 nIndex);
	void GetChainIndexRangeWithSameEndPoint(RainbowChain*pChain,int nRainbowChainCount,int nChainIndex,int&nChainIndexFrom,int&nChainIndexTo);
	bool checkFalseAlarm(RainbowChain*pChain,int nGuessedPos,unsigned char*pHash,HashSet&hs);
	void SearchTableChunk(RainbowChain*pChain,int nRainbowChainLen,int nRainbowChainCount,CHashSet&hs);
	void SearchRainbowTable(string sPathName,HashSet&hs);
public:
	void run(const string&encryptedText,const string&filename);
	float GetDiskTime();
	float GetTotalTime();
	int GetTotalSteps();
	int GetFalseAlarms();
};
#endif