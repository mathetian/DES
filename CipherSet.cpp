#include "CipherSet.h"

CipherSet::CipherSet() : index(0), solve(0)
{
}

CipherSet::~CipherSet()
{
}

void CipherSet::AddKey(uint64_t cipherKey)
{
	m_vKeys.push_back(cipherKey);
}

bool CipherSet::AnyKeyLeft()
{
	return index == m_vKeys.size() ? 0 : 1;
}

uint64_t CipherSet::GetLeftKey()
{
	solve = 0;
	return m_vKeys.at(index);
}

void CipherSet::AddResult(uint64_t cipherKey,uint64_t key)
{
	m_vFound.push_back(make_pair(cipherKey, key));
}

void CipherSet::Succeed()
{
	solve = 1;
}

void CipherSet::Done()
{
	index++;
}

bool CipherSet::Solved()
{
	return solve == 1 ? 1 : 0;
}

int CipherSet::GetKeyFoundNum()
{
	return m_vFound.size();
}

void CipherSet::PrintAllFound()
{
	int index = 0;
	for(;index < m_vFound.size();index++)
		printf("Time: %d, %lld %lld\n",index,(long long)m_vFound.at(index).first,(long long)m_vFound.at(index).second);
}