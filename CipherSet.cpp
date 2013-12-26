#include "CipherSet.h"

CipherSet::CipherSet()
{
}

CipherSet::~CipherSet()
{
}

void CipherSet::AddHash(uint64_t sHash)
{
	m_vHash.push_back(sHash);
}

bool CipherSet::AnyHashLeft()
{
	return index == m_vHash.size() ? 0 : 1;
}

uint64_t CipherSet::GetLeftHash()
{
	return m_vHash.at(index);
}

void CipherSet::AddResult(uint64_t cipherKey,uint64_t key)
{
	m_vFound.push_back(make_pair(cipherKey,key));
}

void CipherSet::Done()
{
	index++;solve = 1;
}

bool CipherSet::Solved()
{
	return solve == 1 ? 1 : 0;
}