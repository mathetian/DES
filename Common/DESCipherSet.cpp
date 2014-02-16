#include "DESCipherSet.h"

#include <iostream>
using namespace std;

DESCipherSet * DESCipherSet::p_cs;

DESCipherSet * DESCipherSet::GetInstance()
{
    if(!p_cs)
        p_cs = new DESCipherSet();
    return p_cs;
}

DESCipherSet::DESCipherSet() : index(0), solve(0)
{
}

DESCipherSet::~DESCipherSet()
{
}

void DESCipherSet::AddKey(uint64_t cipherKey)
{
    m_vKeys.push_back(cipherKey);
}

bool DESCipherSet::AnyKeyLeft()
{
    return index == m_vKeys.size() ? 0 : 1;
}

uint64_t DESCipherSet::GetLeftKey()
{
    solve = 0;
    return m_vKeys.at(index);
}

void DESCipherSet::AddResult(uint64_t cipherKey,uint64_t key)
{
    m_vFound.push_back(make_pair(cipherKey, key));
}

void DESCipherSet::Succeed()
{
    solve = 1;
}

void DESCipherSet::Done()
{
    index++;
}

bool DESCipherSet::Solved()
{
    return solve == 1 ? 1 : 0;
}

int DESCipherSet::GetKeyFoundNum()
{
    return m_vFound.size();
}

void DESCipherSet::PrintAllFound()
{
    int index = 0;
    int ss = m_vFound.size();
    for(; index < ss; index++)
    {
        unsigned int high1 = (m_vFound.at(index).first >> 32);
        unsigned int low1  = (m_vFound.at(index).first & ((1ull << 32) - 1));
        unsigned int high2 = (m_vFound.at(index).second >> 32);
        unsigned int low2  = (m_vFound.at(index).second & ((1ull << 32) - 1));
        printf("Time: %d, 0x%x%x 0x%x%x\n", index + 1, high1, low1, high2, low2);

    }
}