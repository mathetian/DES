#include "ChainWalkContext.h"

string ChainWalkContext::m_plainText;
int ChainWalkContext::m_chainLen;
int ChainWalkContext::m_chainCount;

ChainWalkContext::ChainWalkContext()
{
}

ChainWalkContext::~ChainWalkContext()
{
}

void ChainWalkContext::setPlainText(const string&plainText)
{
	m_plainText=plainText;
}

void ChainWalkContext::setChainLen(int chainLen)
{
	m_chainLen=chainLen;
}

void ChainWalkContext::setChainCount(int chainCount)
{
	m_chainCount=chainCount;
}

void ChainWalkContext::setProperty(const string&plainText,int chainLen,int chainCount)
{
	setPlainText(plainText);
	setChainLen(chainLen);
	setChainCount(chainCount);
}

void ChainWalkContext::Dump()
{
	printf("plainText: %s, chainLen: %d, chainCount: %d\n",m_plainText,m_chainLen,m_chainCount);
}

void ChainWalkContext::GenerateRandomIndex()
{
	RAND_bytes((unsigned char*)&m_nIndex,8);
	m_nIndex=m_nIndex%m_nPlainSpaceTotal;
}

void ChainWalkContext::setIndex(uint64 nIndex)
{
	m_nIndex=nIndex;
}

void ChainWalkContext::IndexToPlain()
{
	int i;
}

void ChainWalkContext::PlainToHash()
{

}

void ChainWalkContext::HashToIndex()
{
	m_nIndex=(*(uint64*)m_Hash+m_nReduceOffset+nPos)%m_nPlainSpaceTotal;
}

uint64 ChainWalkContext::GetIndex()
{
	return m_nIndex;
}

uint64 ChainWalkContext::GetHashValue()
{
}